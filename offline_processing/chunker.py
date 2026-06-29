"""
RAG 知识库 - 文档分片模块
对 Markdown 文本按最小章节（任意级别标题）分片，图片单独分片
"""

import re
from typing import List, Optional

from llama_index.core import Document

from config import config

# 匹配 Markdown 标题行：行首 "# " ~ "###### "
_HEADING_PATTERN = re.compile(r"^#{1,6}\s+\S")
# 匹配行内图片：![alt](url)
_IMAGE_PATTERN = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


class Chunker:
    """文档分片器，按最小章节（任意级别标题）拆分，图片单独成片"""

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        self.chunk_size = chunk_size or config.chunk.chunk_size
        self.min_chunk_len = 128
        self.hard_max_len = self.chunk_size * 4

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def chunk_markdown(
        self,
        markdown_text: str,
        file_name: str = "",
        file_path: str = "",
    ) -> List[Document]:
        """将 Markdown 文本按最小章节拆分，图片单独成片"""
        if not markdown_text.strip():
            return []

        # Step 1: 按标题边界拆分 + 提取图片为独立块
        raw_blocks = self._split_into_raw_blocks(markdown_text)

        # Step 2: 尺寸管理：合并过小文本块、拆分过大文本块；图片块保持原样
        chunks = self._manage_chunk_sizes(raw_blocks)

        # Step 3: 构建 Document 列表
        result: List[Document] = []
        for i, chunk_text in enumerate(chunks):
            if not chunk_text.strip():
                continue

            first_line = chunk_text.strip().split("\n", 1)[0].strip()
            section_title = first_line[:80] if len(first_line) > 80 else first_line
            chunk_type = "image" if self._is_image_chunk(chunk_text) else "text"

            chunk_doc = Document(
                text=chunk_text,
                metadata={
                    "file_name": file_name,
                    "file_path": file_path,
                    "file_type": "pdf",
                    "chunk_type": chunk_type,
                    "section_index": i,
                    "section_count": len(chunks),
                    "section_title": section_title,
                },
            )
            result.append(chunk_doc)

        image_count = sum(1 for c in result if c.metadata.get("chunk_type") == "image")
        text_count = len(result) - image_count
        print(
            f"  [Markdown 分片] {file_name}: {len(result)} 个 chunk"
            f"（文本 {text_count}，图片 {image_count}）"
        )
        return result

    # ------------------------------------------------------------------
    # 内部分片逻辑
    # ------------------------------------------------------------------

    def _split_into_raw_blocks(self, text: str) -> List[str]:
        """按任意级别标题边界切分，再将每段中的图片提取为独立块"""
        lines = text.split("\n")
        sections: List[str] = []
        current: List[str] = []

        for line in lines:
            if _HEADING_PATTERN.match(line):
                if current:
                    sections.append("\n".join(current))
                current = [line]
            else:
                current.append(line)

        if current:
            sections.append("\n".join(current))

        # 从每个 section 中提取图片为独立块
        result: List[str] = []
        for section in sections:
            stripped = section.strip()
            if not stripped:
                continue

            images = list(_IMAGE_PATTERN.finditer(stripped))
            if not images:
                result.append(stripped)
            else:
                # 移除图片后的文本保留为一个块
                text_without_images = _IMAGE_PATTERN.sub("", stripped)
                text_without_images = re.sub(r"\n{3,}", "\n\n", text_without_images)
                text_without_images = text_without_images.strip()
                if text_without_images:
                    result.append(text_without_images)
                # 每张图片独立成块
                for m in images:
                    result.append(m.group(0))

        return result

    # ------------------------------------------------------------------
    # 尺寸管理
    # ------------------------------------------------------------------

    def _manage_chunk_sizes(self, blocks: List[str]) -> List[str]:
        """先合并过小的文本块，再拆分过大的文本块；图片块保持原样"""
        merged = self._merge_small_chunks(blocks)
        result: List[str] = []
        for chunk in merged:
            if self._is_image_chunk(chunk):
                result.append(chunk)
            elif len(chunk) > self.hard_max_len:
                result.extend(self._split_large_chunk(chunk))
            else:
                result.append(chunk)
        return result

    def _is_image_chunk(self, text: str) -> bool:
        """判断是否为纯图片块（整块只有一张图片的 markdown 语法）"""
        stripped = text.strip()
        if not stripped:
            return False
        # 单张图片：整块恰好匹配 ![alt](url)
        if _IMAGE_PATTERN.fullmatch(stripped):
            return True
        # 同行多张图片：整行由图片和空白组成
        no_images = _IMAGE_PATTERN.sub("", stripped).strip()
        return no_images == ""

    def _merge_small_chunks(self, blocks: List[str]) -> List[str]:
        """将过小的文本块与相邻文本块合并；图片块不参与合并"""
        if len(blocks) <= 1:
            return blocks

        result: List[str] = []
        i = 0
        while i < len(blocks):
            current = blocks[i]

            # 图片块不合并，直接保留
            if self._is_image_chunk(current):
                result.append(current)
                i += 1
                continue

            # 持续向后合并，直到大于等于 min_chunk_len 或遇到图片/超大块
            while len(current) < self.min_chunk_len and i + 1 < len(blocks):
                next_block = blocks[i + 1]
                if self._is_image_chunk(next_block):
                    break
                merged = current + "\n\n" + next_block
                if len(merged) > self.hard_max_len:
                    break
                current = merged
                i += 1

            result.append(current)
            i += 1

        return result

    def _split_large_chunk(self, text: str) -> List[str]:
        """将超大文本块按段落边界拆分为多个块"""
        if len(text) <= self.hard_max_len:
            return [text]

        # 先尝试按双换行（段落边界）拆分
        paragraphs = re.split(r"\n\s*\n", text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if len(paragraphs) <= 1:
            return self._split_by_lines(text)

        # 按段落重新组合
        chunks: List[str] = []
        current: List[str] = []
        current_len = 0

        for para in paragraphs:
            para_len = len(para) + 2
            if current_len + para_len > self.hard_max_len and current:
                chunks.append("\n\n".join(current))
                current = []
                current_len = 0
            current.append(para)
            current_len += para_len

        if current:
            chunks.append("\n\n".join(current))

        return chunks if chunks else [text]

    def _split_by_lines(self, text: str) -> List[str]:
        """按行粗暴拆分（段落内无结构时的兜底策略）"""
        lines = text.split("\n")
        chunks: List[str] = []
        current: List[str] = []
        current_len = 0

        for line in lines:
            line_len = len(line) + 1
            if current_len + line_len > self.hard_max_len and current:
                chunks.append("\n".join(current))
                current = []
                current_len = 0
            current.append(line)
            current_len += line_len

        if current:
            chunks.append("\n".join(current))

        return chunks if chunks else [text]
