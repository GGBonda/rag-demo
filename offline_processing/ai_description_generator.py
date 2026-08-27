"""
RAG 知识库 - AI 描述生成模块
为表格、图片和代码类型的 IndexChunk 生成便于检索的文本描述。
为 RetrieveChunk 生成简短的 IndexChunk 索引简介。
"""

import json

from config import config
from data_class import IndexChunk, RetrieveChunk
from data_class.retrieve_chunk import DATA_IMAGE_RE
from model_request import (
    request_code_description,
    request_image_description,
    request_retrieve_description,
    request_table_description,
)
from prompts import load_prompt

from .chunker import detect_text_type


class AIDescriptionGenerator:

    RETRIEVE_DESC_MAX_RETRIES = 3
    SUPPORTED_TYPES = {"table", "image", "code"}
    _TYPE_REQUESTS = {
        "table": request_table_description,
        "code": request_code_description,
        "image": request_image_description,
    }
    _TYPE_PROMPTS = {
        "table": load_prompt("ai_desc_for_table.txt"),
        "code": load_prompt("ai_desc_for_code.txt"),
        "image": load_prompt("ai_desc_for_image.txt"),
    }
    _RETRIEVE_PROMPT = load_prompt("ai_desc_question_for_retrieve_chunk.txt")

    def generate_image_code_table_desc(self, chunks: list[IndexChunk]) -> None:
        for chunk in chunks:
            chunk_type = detect_text_type(chunk.text)
            if chunk_type not in self.SUPPORTED_TYPES or not chunk.text.strip():
                continue

            content, _ = self._TYPE_REQUESTS[chunk_type](
                self._build_user_content(chunk, chunk_type)
            )

            if not content or not content.strip():
                raise RuntimeError(f"模型未返回 {chunk_type} 分片的描述")
            chunk.text = content.strip()

    def generate_retrieve_desc(
        self,
        retrieve_chunks: list[RetrieveChunk],
        start_index: int=0
    ) -> list[IndexChunk]:
        """将每条简介和每个问题分别生成为独立的 IndexChunk。"""
        description_count = config.chunk.index_chunk_description_count
        if description_count <= 0:
            raise ValueError("INDEX_CHUNK_DESCRIPTION_COUNT 必须为正整数")
        question_count = config.chunk.index_chunk_question_count
        if question_count <= 0:
            raise ValueError("INDEX_CHUNK_QUESTION_COUNT 必须为正整数")

        result: list[IndexChunk] = []
        for retrieve_chunk in retrieve_chunks:
            for retry_count in range(self.RETRIEVE_DESC_MAX_RETRIES + 1):
                try:
                    image_matches = DATA_IMAGE_RE.findall(retrieve_chunk.text)
                    user_content = self._build_index_chunk_user_content(
                        retrieve_chunk,
                        description_count,
                        question_count,
                        image_matches,
                    )
                    content, finish_reason = request_retrieve_description(
                        user_content
                    )

                    try:
                        descriptions, questions = self._parse_index_content(
                            content,
                            description_count,
                            question_count,
                        )
                    except Exception:
                        print(f"[召回文档简介] 响应校验失败: finish_reason={finish_reason}, content={content!r}")
                        raise
                    description_chunks = [
                        IndexChunk(
                            id=int(
                                f"{retrieve_chunk.id}"
                                f"{start_index + description_index:03d}"
                            ),
                            retrieve_id=retrieve_chunk.id,
                            text=description,
                        )
                        for description_index, description in enumerate(
                            descriptions,
                            start=1,
                        )
                    ]
                    question_chunks = [
                        IndexChunk(
                            id=int(
                                f"{retrieve_chunk.id}"
                                f"{start_index + len(descriptions) + question_index:03d}"
                            ),
                            retrieve_id=retrieve_chunk.id,
                            text=question,
                        )
                        for question_index, question in enumerate(
                            questions,
                            start=1,
                        )
                    ]
                    result.extend(description_chunks)
                    result.extend(question_chunks)
                    break
                except Exception as exc:
                    print(
                        f"[召回文档简介] 生成失败: retrieve_chunk_id={retrieve_chunk.id}, "
                        f"异常={type(exc).__name__}: {exc}"
                    )
                    if retry_count >= self.RETRIEVE_DESC_MAX_RETRIES:
                        print(
                            "[召回文档简介] 已达到最大重试次数: "
                            f"retrieve_chunk_id={retrieve_chunk.id}"
                        )
                        raise
                    print(
                        f"[召回文档简介] 即将进行第 {retry_count + 1}/"
                        f"{self.RETRIEVE_DESC_MAX_RETRIES} 次重试"
                    )

        return result

    def _build_index_chunk_user_content(
        self,
        retrieve_chunk: RetrieveChunk,
        description_count: int,
        question_count: int,
        image_matches: list[tuple[str, str]],
    ) -> str | list[dict]:
        if not image_matches:
            return self._RETRIEVE_PROMPT[1].substitute(
                description_count=description_count,
                question_count=question_count,
                title_path=retrieve_chunk.title_path,
                content=retrieve_chunk.text,
            )

        context = DATA_IMAGE_RE.sub(
            lambda match: (
                f"[图片说明：{match.group(1)}]"
                if match.group(1)
                else "[图片]"
            ),
            retrieve_chunk.text,
        ).strip()
        content: list[dict] = [
            {
                "type": "text",
                "text": self._RETRIEVE_PROMPT[1].substitute(
                    description_count=description_count,
                    question_count=question_count,
                    title_path=retrieve_chunk.title_path,
                    content=context,
                ),
            }
        ]
        content.extend(
            {
                "type": "image_url",
                "image_url": {"url": image_url},
            }
            for _, image_url in image_matches
        )
        return content

    @staticmethod
    def _parse_index_content(
        content: str | None,
        expected_description_count: int,
        expected_question_count: int,
    ) -> tuple[list[str], list[str]]:
        if not content or not content.strip():
            raise RuntimeError("模型未返回索引简介和问题")

        index_content = AIDescriptionGenerator._load_json_object(content)

        descriptions = AIDescriptionGenerator._parse_string_list(
            index_content.get("descriptions"),
            expected_description_count,
            "索引简介",
        )
        questions = AIDescriptionGenerator._parse_string_list(
            index_content.get("questions"),
            expected_question_count,
            "可能问题",
        )

        return descriptions, questions

    @staticmethod
    def _load_json_object(content: str) -> dict:
        """解析 JSON 对象，并兼容对象前后的模型说明或 Markdown 代码块。"""
        try:
            index_content = json.loads(content)
        except json.JSONDecodeError as exc:
            decoder = json.JSONDecoder()
            for position, character in enumerate(content):
                if character != "{":
                    continue
                try:
                    candidate, _ = decoder.raw_decode(content, position)
                except json.JSONDecodeError:
                    continue
                if isinstance(candidate, dict):
                    return candidate
            raise RuntimeError("模型返回的索引内容不是有效的 JSON 对象") from exc

        if not isinstance(index_content, dict):
            raise RuntimeError("模型返回的索引内容必须是 JSON 对象")
        return index_content

    @staticmethod
    def _parse_string_list(
        values: object,
        expected_count: int,
        content_name: str,
    ) -> list[str]:
        if not isinstance(values, list) or len(values) != expected_count:
            raise RuntimeError(f"模型必须返回 {expected_count} 条{content_name}")

        normalized_values: list[str] = []
        for value in values:
            if not isinstance(value, str) or not value.strip():
                raise RuntimeError(f"{content_name}必须为非空字符串")
            normalized_values.append(value.strip())
        return normalized_values

    def _build_user_content(
        self,
        chunk: IndexChunk,
        chunk_type: str,
    ) -> str | list[dict]:
        prompt = self._TYPE_PROMPTS[chunk_type][1]
        if chunk_type != "image":
            return prompt.substitute(content=chunk.text)

        image_matches = DATA_IMAGE_RE.findall(chunk.text)
        if not image_matches:
            return prompt.substitute(
                content_heading="原始内容",
                content=chunk.text,
            )

        context = DATA_IMAGE_RE.sub(
            lambda match: f"[图片说明：{match.group(1)}]" if match.group(1) else "[图片]",
            chunk.text,
        ).strip()
        content: list[dict] = [
            {
                "type": "text",
                "text": prompt.substitute(
                    content_heading="图片上下文",
                    content=context,
                ),
            }
        ]
        content.extend(
            {
                "type": "image_url",
                "image_url": {"url": image_url},
            }
            for _, image_url in image_matches
        )
        return content
