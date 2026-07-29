"""
RAG 知识库 - AI 描述生成模块
为表格、图片和代码类型的 ParagraphChunk 生成便于检索的文本描述。
"""

import re

from config import config
from .chunker import ParagraphChunk


class ParagraphChunkDescriptionGenerator:
    """使用 OpenAI 兼容的大语言模型补充 ParagraphChunk.ai_desc_text。"""

    SUPPORTED_TYPES = {"table", "image", "code"}
    _DATA_IMAGE_RE = re.compile(
        r"!\[([^\]]*)\]\((data:image/[^;]+;base64,[^)]+)\)"
    )
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
        self.model = config.llm.openai_flash_model.strip()
        if not self.model:
            raise ValueError("未配置 OPENAI_FLASH_MODEL 环境变量")

        resolved_api_key = config.llm.openai_api_key
        if not resolved_api_key:
            raise ValueError("未配置 OPENAI_API_KEY 环境变量")

        from openai import OpenAI

        client_options = {"api_key": resolved_api_key}
        resolved_base_url = config.llm.openai_base_url
        if resolved_base_url:
            client_options["base_url"] = resolved_base_url
        self._client = OpenAI(**client_options)

    def generate_descriptions(self, chunks: list[ParagraphChunk]) -> None:
        """原地为 table、image、code 类型的分片生成 ai_desc_text。"""
        for chunk in chunks:
            if chunk.type not in self.SUPPORTED_TYPES or not chunk.text.strip():
                continue

            response = self._client.chat.completions.create(
                model=self.model,
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
                        "content": self._build_user_content(chunk),
                    },
                ],
            )
            description = response.choices[0].message.content
            if not description or not description.strip():
                raise RuntimeError(f"模型未返回 {chunk.type} 分片的描述")
            chunk.ai_desc_text = description.strip()[:500]

    def _build_user_content(self, chunk: ParagraphChunk) -> str | list[dict]:
        prompt = self._TYPE_PROMPTS[chunk.type]
        if chunk.type != "image":
            return f"{prompt}\n\n原始内容：\n{chunk.text}"

        image_matches = self._DATA_IMAGE_RE.findall(chunk.text)
        if not image_matches:
            return f"{prompt}\n\n原始内容：\n{chunk.text}"

        context = self._DATA_IMAGE_RE.sub(
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
