"""
RAG 知识库 - AI 描述生成模块
为表格、图片和代码类型的 IndexChunk 生成便于检索的文本描述。
为 RetrieveChunk 生成简短的 IndexChunk 索引简介。
"""

import json
import time

from config import config
from .chunker import DATA_IMAGE_RE, IndexChunk, RetrieveChunk, detect_text_type


class AIDescriptionGenerator:
    """使用 OpenAI 兼容的大语言模型补充 IndexChunk.ai_desc_text。"""

    SUPPORTED_TYPES = {"table", "image", "code"}
    _TYPE_PROMPTS = {
        "table": (
            "请描述下面表格的主题、字段含义、关键数据和重要结论。"
            "描述应完整、准确，便于后续语义检索，且不超过500字；"
            "只输出描述正文。"
        ),
        "code": (
            "请描述下面代码的用途、主要逻辑、输入输出和关键实现。"
            "描述应完整、准确，便于后续语义检索，且不超过500字；"
            "只输出描述正文。"
        ),
        "image": (
            "请描述图片中的主体、文字、结构、流程和关键结论。"
            "描述应完整、准确，便于后续语义检索，且不超过500字；"
            "只输出描述正文。"
        ),
    }

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

    def generate_descriptions(self, chunks: list[IndexChunk]) -> None:
        """原地为 table、image、code 类型的分片生成 ai_desc_text。"""
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
                        "content": (
                            "你是文档内容分析助手，负责将非纯文本内容转换为"
                            "准确、清晰的中文文本描述。不要虚构原内容中没有的信息。"
                        ),
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
            chunk.ai_desc_text = description.strip()

    def generate_index_chunks(
        self,
        retrieve_chunks: list[RetrieveChunk],
    ) -> list[IndexChunk]:
        """为每个 RetrieveChunk 生成配置数量的简短 IndexChunk。"""
        description_count = config.chunk.index_chunk_description_count
        if description_count <= 0:
            raise ValueError("INDEX_CHUNK_DESCRIPTION_COUNT 必须为正整数")

        result: list[IndexChunk] = []
        for retrieve_chunk in retrieve_chunks:
            image_matches = DATA_IMAGE_RE.findall(retrieve_chunk.text)
            if image_matches:
                client = self._vision_client
                model = self.vision_model
            else:
                client = self._text_client
                model = self.text_model

            print(f"[索引简介] 开始调用模型: model={model}, retrieve_chunk_id={retrieve_chunk.id}")
            start_time = time.perf_counter()
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是文档检索索引生成助手。请严格基于提供的正文和图片，从不同角度生成准确、清晰的中文简介，不要虚构原内容中没有的信息。"
                    },
                    {
                        "role": "user",
                        "content": self._build_index_chunk_user_content(
                            retrieve_chunk,
                            description_count,
                            image_matches,
                        ),
                    },
                ],
            )
            print(f"[召回文档简介] 模型调用结束: 耗时={time.perf_counter() - start_time:.2f}秒")

            content = response.choices[0].message.content
            descriptions = self._parse_index_descriptions(
                content,
                description_count,
            )
            result.extend(
                IndexChunk(
                    id=int(f"{retrieve_chunk.id}{description_index:03d}"),
                    retrieve_id=retrieve_chunk.id,
                    text=description,
                )
                for description_index, description in enumerate(
                    descriptions,
                    start=1,
                )
            )

        return result

    def _build_index_chunk_user_content(
        self,
        retrieve_chunk: RetrieveChunk,
        description_count: int,
        image_matches: list[tuple[str, str]],
    ) -> str | list[dict]:
        prompt = (
            f"请生成 {description_count} 条简介，每条不超过100字。"
            "仅返回 JSON 字符串数组，不要输出其他内容。\n\n"
            f"章节标题：{retrieve_chunk.title_path}\n"
        )
        if not image_matches:
            return f"{prompt}章节正文：\n{retrieve_chunk.text}"

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
                "text": f"{prompt}章节正文：\n{context}",
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
    def _parse_index_descriptions(
        content: str | None,
        expected_count: int,
    ) -> list[str]:
        if not content or not content.strip():
            raise RuntimeError("模型未返回索引简介")

        try:
            descriptions = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError("模型返回的索引简介不是有效的 JSON 数组") from exc

        if not isinstance(descriptions, list) or len(descriptions) != expected_count:
            raise RuntimeError(f"模型必须返回 {expected_count} 条索引简介")

        normalized_descriptions: list[str] = []
        for description in descriptions:
            if not isinstance(description, str) or not description.strip():
                raise RuntimeError("索引简介必须为非空字符串")
            normalized_descriptions.append(description.strip())

        return normalized_descriptions

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
        prompt = self._TYPE_PROMPTS[chunk_type]
        if chunk_type != "image":
            return f"{prompt}\n\n原始内容：\n{chunk.text}"

        image_matches = DATA_IMAGE_RE.findall(chunk.text)
        if not image_matches:
            return f"{prompt}\n\n原始内容：\n{chunk.text}"

        context = DATA_IMAGE_RE.sub(
            lambda match: f"[图片说明：{match.group(1)}]" if match.group(1) else "[图片]",
            chunk.text,
        ).strip()
        content: list[dict] = [
            {
                "type": "text",
                "text": f"{prompt}\n\n图片上下文：\n{context}",
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
