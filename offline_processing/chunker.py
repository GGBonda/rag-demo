"""
RAG 知识库 - 文档分片模块
将解析后的文档元素按章节边界拆分为 chunk
"""

from typing import List, Optional

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

from config import config

from offline_processing.document_loader import ParsedDocument


class Chunker:
    """文档分片器，按章节边界拆分元素并生成 Document chunk"""

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        self.chunk_size = chunk_size or config.chunk.chunk_size
        self.chunk_overlap = chunk_overlap or config.chunk.chunk_overlap

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def chunk_elements(self, parsed_document:ParsedDocument) -> List[Document]:
        """将元素列表按章节边界拆分为 Document chunk 列表"""
        if not parsed_document.elements:
            return []

        chunks = self._split_elements_by_title(parsed_document.elements)

        result: List[Document] = []
        for i, chunk in enumerate(chunks):
            chunk_text = (
                chunk.text.strip() if hasattr(chunk, "text") else str(chunk).strip()
            )
            if not chunk_text:
                continue

            first_line = chunk_text.split("\n", 1)[0].strip()
            section_title = first_line[:80] if len(first_line) > 80 else first_line

            chunk_doc = Document(
                text=chunk_text,
                metadata={
                    "file_name": parsed_document.file_name,
                    "file_path": str(parsed_document.file_path),
                    "file_type": "pdf",
                    "section_index": i,
                    "section_count": len(chunks),
                    "section_title": section_title,
                },
            )

            # 超大章节回退到段落级分片
            if len(chunk_text) > self.chunk_size:
                sub_nodes = self._fallback_paragraph_split(chunk_doc)
                for node in sub_nodes:
                    node.metadata["section_title"] = section_title
                result.extend(sub_nodes)
            else:
                result.append(chunk_doc)

        print(f"  [章节拆分] {parsed_document.file_name}: {len(result)} 个 chunk")
        return result

    # ------------------------------------------------------------------
    # 内部分片方法
    # ------------------------------------------------------------------

    @staticmethod
    def _split_elements_by_title(elements: list) -> list:
        """按标题边界对元素进行分片，字符数量过大的章节先不管，只将字符数量过小的章节进行合并。"""
        from unstructured.chunking.title import chunk_by_title

        return chunk_by_title(
            elements,
            max_characters=100000,
            new_after_n_chars=100000,
            combine_text_under_n_chars=300,
            overlap=0,
        )

    def _fallback_paragraph_split(self, document: Document) -> List[Document]:
        """章节过大时回退到按段落拆分"""
        text = document.text
        raw_paragraphs = text.split("\n\n")
        paragraphs = [p.strip() for p in raw_paragraphs if p.strip()]

        if not paragraphs:
            raw_paragraphs = text.split("\n")
            paragraphs = [p.strip() for p in raw_paragraphs if p.strip()]

        if not paragraphs:
            return [document]

        chunks: List[str] = []
        current_chunk = ""
        for para in paragraphs:
            # 单个段落超过 chunk_size，对该段落回退到句子级拆分
            if len(para) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                splitter = SentenceSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                )
                para_doc = Document(text=para, metadata=document.metadata)
                sub_nodes = splitter.get_nodes_from_documents([para_doc])
                chunks.extend([node.text for node in sub_nodes])
                continue

            if len(current_chunk) + len(para) + 2 <= self.chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
            else:
                chunks.append(current_chunk)
                current_chunk = para

        if current_chunk:
            chunks.append(current_chunk)

        result: List[Document] = []
        for chunk_text in chunks:
            chunk_doc = Document(
                text=chunk_text,
                metadata={**document.metadata},
            )
            result.append(chunk_doc)

        return result
