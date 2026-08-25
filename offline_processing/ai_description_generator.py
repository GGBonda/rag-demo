"""
RAG 知识库 - AI 描述生成模块
为表格、图片和代码类型的 IndexChunk 生成便于检索的文本描述。
为 RetrieveChunk 生成简短的 IndexChunk 索引简介。
"""

import json
import time
from pathlib import Path
from string import Template

from config import config
from model import IndexChunk, RetrieveChunk
from model.retrieve_chunk import DATA_IMAGE_RE

from .chunker import detect_text_type


PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
SYSTEM_PROMPT_MARKER = "[system]"
USER_PROMPT_MARKER = "[user]"


def _load_prompt(file_name: str) -> tuple[str, Template]:
    content = (PROMPTS_DIR / file_name).read_text(encoding="utf-8").strip()
    system_marker, separator, user_content = content.partition(USER_PROMPT_MARKER)
    if not separator or not system_marker.strip().startswith(SYSTEM_PROMPT_MARKER):
        raise ValueError(
            f"提示词文件 {file_name} 必须包含 [system] 和 [user] 标记"
        )

    system_content = system_marker.strip()[len(SYSTEM_PROMPT_MARKER):].strip()
    user_content = user_content.strip()
    if not system_content or not user_content:
        raise ValueError(f"提示词文件 {file_name} 的 system 和 user 内容不能为空")
    return system_content, Template(user_content)


class AIDescriptionGenerator:

    RETRIEVE_DESC_MAX_RETRIES = 3
    SUPPORTED_TYPES = {"table", "image", "code"}
    _TYPE_PROMPTS = {
        "table": _load_prompt("ai_desc_for_table.txt"),
        "code": _load_prompt("ai_desc_for_code.txt"),
        "image": _load_prompt("ai_desc_for_image.txt"),
    }
    _RETRIEVE_PROMPT = _load_prompt("ai_desc_question_for_retrieve_chunk.txt")

    def __init__(self):
        self.text_model = config.llm.openai_pro_model.strip()
        if not self.text_model:
            raise ValueError("未配置 OPENAI_PRO_MODEL 环境变量")

        self.vision_model = config.vision_llm.openai_model.strip()
        if not self.vision_model:
            raise ValueError("未配置 OPENAI_VISUAL_MODEL 环境变量")

        self._text_client = self._create_client(
            api_key=config.llm.openai_api_key,
            base_url=config.llm.openai_base_url,
            api_key_name="OPENAI_API_KEY",
        )
        self._vision_client = self._create_client(
            api_key=config.vision_llm.openai_api_key,
            base_url=config.vision_llm.openai_base_url,
            api_key_name="OPENAI_VISUAL_API_KEY",
        )

    def generate_image_code_table_desc(self, chunks: list[IndexChunk]) -> None:
        for chunk in chunks:
            chunk_type = detect_text_type(chunk.text)
            if chunk_type not in self.SUPPORTED_TYPES or not chunk.text.strip():
                continue

            if chunk_type == "image":
                client = self._vision_client
                model = self.vision_model
            else:
                client = self._text_client
                model = self.text_model

            print(f"[AI 描述] 开始调用模型: model={model}, chunk_type={chunk_type}, chunk_id={chunk.id}")
            start_time = time.perf_counter()
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": self._TYPE_PROMPTS[chunk_type][0],
                    },
                    {
                        "role": "user",
                        "content": self._build_user_content(chunk, chunk_type),
                    },
                ],
            )
            print(f"  [AI 描述] 模型调用结束: 耗时={time.perf_counter() - start_time:.2f}秒")

            description = response.choices[0].message.content
            if not description or not description.strip():
                raise RuntimeError(f"模型未返回 {chunk_type} 分片的描述")
            chunk.text = description.strip()

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
                    if image_matches:
                        client = self._vision_client
                        model = self.vision_model
                    else:
                        client = self._text_client
                        model = self.text_model

                    print(
                        f"[索引简介] 开始调用模型: model={model}, "
                        f"retrieve_chunk_id={retrieve_chunk.id}, "
                        f"尝试次数={retry_count + 1}/{self.RETRIEVE_DESC_MAX_RETRIES + 1}"
                    )
                    start_time = time.perf_counter()
                    response = client.chat.completions.create(
                        model=model,
                        response_format={"type": "json_object"},
                        messages=[
                            {
                                "role": "system",
                                "content": self._RETRIEVE_PROMPT[0],
                            },
                            {
                                "role": "user",
                                "content": self._build_index_chunk_user_content(
                                    retrieve_chunk,
                                    description_count,
                                    question_count,
                                    image_matches,
                                ),
                            },
                        ],
                    )
                    print(f"[召回文档简介] 模型调用结束: 耗时={time.perf_counter() - start_time:.2f}秒")

                    choice = response.choices[0]
                    content = choice.message.content
                    try:
                        descriptions, questions = self._parse_index_content(
                            content,
                            description_count,
                            question_count,
                        )
                    except Exception:
                        print(f"[召回文档简介] 响应校验失败: finish_reason={getattr(choice, 'finish_reason', None)}, content={content!r}")
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

    @staticmethod
    def _create_client(api_key: str, base_url: str, api_key_name: str):
        resolved_api_key = api_key.strip()
        if not resolved_api_key:
            raise ValueError(f"未配置 {api_key_name} 环境变量")

        from openai import OpenAI

        client_options = {"api_key": resolved_api_key}
        resolved_base_url = base_url.strip()
        if resolved_base_url:
            client_options["base_url"] = resolved_base_url
        return OpenAI(**client_options)

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
