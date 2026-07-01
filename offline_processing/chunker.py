"""
RAG 知识库 - 文档分片模块
对 Markdown 文本按最小章节（任意级别标题）分片
"""

from typing import List, Optional

from llama_index.core import Document
from markdown_chunker import MarkdownChunkingStrategy

from config import config


class Chunker:
    """文档分片器，基于 markdown-chunker 按最小章节拆分"""

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        self.chunk_size = chunk_size or config.chunk.chunk_size
        self.min_chunk_len = 128
        self.hard_max_len = self.chunk_size * 4

        self._strategy = MarkdownChunkingStrategy(
            min_chunk_len=self.min_chunk_len,
            soft_max_len=self.chunk_size,
            hard_max_len=self.hard_max_len,
            detect_headers_footers=True,
            remove_duplicates=True,
            heading_based_chunking=True,
        )

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def chunk_markdown(
        self,
        markdown_text: str,
        file_name: str = "",
        file_path: str = "",
    ) -> List[Document]:
        """将 Markdown 文本按最小章节拆分为 Document 列表"""
        if not markdown_text.strip():
            return []

        chunks = self._strategy.chunk_markdown(markdown_text)

        result: List[Document] = []
        for i, chunk_text in enumerate(chunks):
            if not chunk_text.strip():
                continue

            first_line = chunk_text.strip().split("\n", 1)[0].strip()
            section_title = first_line[:80] if len(first_line) > 80 else first_line

            chunk_doc = Document(
                text=chunk_text,
                metadata={
                    "file_name": file_name,
                    "file_path": file_path,
                    "file_type": "pdf",
                    "chunk_type": "text",
                    "section_index": i,
                    "section_count": len(chunks),
                    "section_title": section_title,
                },
            )
            result.append(chunk_doc)

        print(f"  [Markdown 分片] {file_name}: {len(result)} 个 chunk")
        return result
