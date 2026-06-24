"""
RAG 知识库 - 文档分片模块
对 Markdown 文本按标题结构进行分片
"""

from typing import List, Optional

from llama_index.core import Document
from markdown_chunker import MarkdownChunkingStrategy

from config import config


class Chunker:
    """文档分片器，对 Markdown 文本按标题边界拆分并生成 Document chunk"""

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        self.chunk_size = chunk_size or config.chunk.chunk_size
        # chunk_overlap 在 markdown-chunker 中无直接对应概念，
        # 保留参数以兼容现有调用方，但不影响分片行为

        self._strategy = MarkdownChunkingStrategy(
            min_chunk_len=128,
            soft_max_len=self.chunk_size,
            hard_max_len=self.chunk_size * 4,
            heading_based_chunking=True,
            detect_headers_footers=False,  # 文档加载器已处理
            remove_duplicates=True,
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
        """将 Markdown 文本按标题结构拆分为 Document chunk 列表"""
        if not markdown_text.strip():
            return []

        chunks: List[str] = self._strategy.chunk_markdown(markdown_text)

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
                    "section_index": i,
                    "section_count": len(chunks),
                    "section_title": section_title,
                },
            )
            result.append(chunk_doc)

        print(f"  [Markdown 分片] {file_name}: {len(result)} 个 chunk")
        return result
