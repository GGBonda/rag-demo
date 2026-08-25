"""召回分片数据模型。"""

import re
from dataclasses import dataclass


# 用于识别并提取 Markdown 中的 base64 图片
DATA_IMAGE_RE = re.compile(
    r"!\[([^\]]*)\]\("
    r"(data:image/[^;]+;base64,[^\s)]+)"
    r"(?:\s+(?:\"[^\"]*\"|'[^']*'|\([^)]*\)))?"
    r"\)"
)


@dataclass
class RetrieveChunk:
    """召回分片数据类"""

    """主键id"""
    id: int = 0
    """所属文档id"""
    doc_id: int = 0
    """章节标题路径"""
    title_path: str = ""
    """章节文本"""
    text: str = ""

    def analyse_rerank(self) -> list[dict[str, str]]:
        """将 Markdown 正文解析为 Qwen3-VL-Rerank 的多模态文档。"""
        images: list[str] = []

        def replace_image(match: re.Match) -> str:
            images.append(match.group(2))
            image_number = len(images)
            alt_text = match.group(1).strip()
            if alt_text:
                return f"[图片{image_number}：{alt_text}]"
            return f"[图片{image_number}]"

        text = DATA_IMAGE_RE.sub(replace_image, self.text)
        if not images:
            return [{"text": text}]

        return [
            {
                "text": f"{text}\n当前视觉输入对应[图片{image_number}]",
                "image": image,
            }
            for image_number, image in enumerate(images, start=1)
        ]
