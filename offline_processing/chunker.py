"""
RAG 知识库 - 文本分片器模块
将文档分割成适当大小的 chunk，支持多种分片策略

chapter 策略使用 unstructured.partition.auto.partition 直接解析 PDF，
保留文档结构信息（标题、正文、字号等），再通过 chunk_by_title 按最小章节边界分片。
"""

import re
from pathlib import Path
from typing import List, Optional

from llama_index.core import Document
from llama_index.core.node_parser import (
    SentenceSplitter,
    TokenTextSplitter,
    SentenceWindowNodeParser,
)

from config import config

# ------------------------------------------------------------------
# 中文章节标题的兜底模式 —— 用于后处理阶段校正元素类型（Title ↔ NarrativeText）
# ------------------------------------------------------------------
_ZH_HEADING_PATTERNS = [
    re.compile(r"^#{1,6}\s+\S"),
    re.compile(r"^第[一二三四五六七八九十百千万\d]+[章节节篇部分]"),
    re.compile(r"^[一二三四五六七八九十]+[、．，,]\s*\S"),
    re.compile(r"^（[一二三四五六七八九十]+）\s*\S"),
    re.compile(r"^\d+(?:\.\d+)+[.、．]?\s+\S"),
    re.compile(r"^(?:Chapter|Section|Part)\s+\d+", re.IGNORECASE),
    re.compile(r"^[一二三四五六七八九十]{1,3}\s{2,}\S"),
]

# 正文误判为标题的长度阈值：超过此长度的 "Title" 很可能是正文
_TITLE_MAX_LENGTH = 15


class Chunker:
    """文本分片器，将 PDF 文档拆分为语义完整的 chunk

    chapter 策略特点：
    - 使用 unstructured.partition.auto.partition 直接解析，保留字体/位置等结构信息
    - 按最小章节标题边界拆分，每个章节成为一个独立 chunk
    - 超大章节自动回退到句子级分片
    """

    STRATEGIES = ["sentence", "token", "sentence_window", "chapter"]

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        strategy: str = "sentence",
    ):
        self.chunk_size = chunk_size or config.chunk.chunk_size
        self.chunk_overlap = chunk_overlap or config.chunk.chunk_overlap

        if strategy not in self.STRATEGIES:
            raise ValueError(
                f"不支持的分片策略: {strategy}，可选: {self.STRATEGIES}"
            )
        self.strategy = strategy
        self._parser = self._create_parser()

    def _create_parser(self):
        """根据策略创建对应的解析器（chapter 策略不使用 parser，返回 None）"""
        if self.strategy == "chapter":
            return None

        if self.strategy == "sentence":
            return SentenceSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                paragraph_separator="\n\n",
                secondary_chunking_regex=r"[^,.;。；]+[,.;。；]?",
            )
        elif self.strategy == "token":
            return TokenTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                backup_separators=["\n\n", "\n", "。", ".", " "],
            )
        elif self.strategy == "sentence_window":
            return SentenceWindowNodeParser(
                sentence_splitter=SentenceSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                ),
                window_size=3,
            )

    def split(self, documents: List[Document]) -> List[Document]:
        """将文档列表分割为 chunk 节点"""
        if not documents:
            print("警告: 没有文档需要分片")
            return []

        if self.strategy == "chapter":
            print(f"使用 'chapter' 策略按最小章节分片 "
                  f"(chunk_size={self.chunk_size} 作为超长章节的兜底)")
            return self._split_by_chapter(documents)

        print(f"使用 '{self.strategy}' 策略进行分片 "
              f"(chunk_size={self.chunk_size}, overlap={self.chunk_overlap})...")

        nodes = self._parser.get_nodes_from_documents(documents)
        print(f"分片完成: {len(documents)} 个文档 → {len(nodes)} 个 chunk")
        return nodes

    # ------------------------------------------------------------------
    # chapter 策略核心
    # ------------------------------------------------------------------

    @staticmethod
    def _is_heading(text: str) -> bool:
        """检查文本是否匹配中/英文章节标题模式（用于后处理校正）"""
        text = text.strip()
        if not text:
            return False
        for pattern in _ZH_HEADING_PATTERNS:
            if pattern.match(text):
                return True
        return False

    def _correct_elements(self, elements: list) -> list:
        """校正元素类型：中文标题提升为 Title，误判的长文本降级为 NarrativeText"""
        from unstructured.documents.elements import Title, NarrativeText

        corrected: list = []
        for elem in elements:
            elem_text = str(elem.text).strip()
            if not elem_text:
                continue

            if self._is_heading(elem_text):
                if not isinstance(elem, Title):
                    corrected.append(Title(elem_text))
                else:
                    corrected.append(elem)
                continue

            if isinstance(elem, Title) and len(elem_text) > _TITLE_MAX_LENGTH:
                corrected.append(NarrativeText(elem_text))
                continue

            corrected.append(elem)

        return corrected

    @staticmethod
    def _get_pdf_elements(file_path: str) -> list:
        """使用 unstructured.partition.auto.partition 解析 PDF，返回 Element 列表"""
        from unstructured.partition.auto import partition

        elements = partition(
            filename=file_path,
            strategy="fast",
            include_page_breaks=False,
        )
        if elements:
            print(f"  [unstructured] {file_path}: 解析出 {len(elements)} 个元素")
        else:
            print(f"  [unstructured] {file_path}: 返回 0 个元素")
        return list(elements)

    @staticmethod
    def _get_file_path(doc: Document) -> Optional[str]:
        """从文档元数据中提取文件路径"""
        metadata = doc.metadata or {}
        return metadata.get("file_path") or metadata.get("file_name")

    def _get_elements_for_document(self, doc: Document) -> list:
        """获取 PDF 文档的结构化元素"""
        file_path = self._get_file_path(doc)

        if file_path and Path(file_path).suffix.lower() == ".pdf" and Path(file_path).is_file():
            return self._get_pdf_elements(file_path)

        return []

    def _fallback_sentence_split(self, document: Document) -> List[Document]:
        """章节过大时回退到句子级拆分"""
        splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        return splitter.get_nodes_from_documents([document])

    def _split_by_chapter(self, documents: List[Document]) -> List[Document]:
        """按最小章节边界拆分 PDF 文档

        流程：
        1. PDF 文件 → unstructured.partition.auto.partition 直接解析
        2. 中文标题模式校正
        3. chunk_by_title 按标题边界分组，生成以最小章节为单位的 chunk
        4. 超大章节回退到句子级分片作为兜底
        """
        from unstructured.chunking.title import chunk_by_title

        result: List[Document] = []

        for doc in documents:
            elements = self._get_elements_for_document(doc)
            elements = self._correct_elements(elements)

            if not elements:
                result.append(doc)
                continue

            chunks = chunk_by_title(
                elements,
                max_characters=self.chunk_size * 4,
                new_after_n_chars=self.chunk_size,
                combine_text_under_n_chars=max(self.chunk_size // 4, 1),
                overlap=self.chunk_overlap,
            )

            for i, chunk in enumerate(chunks):
                chunk_text = chunk.text.strip() if hasattr(chunk, "text") else str(chunk).strip()
                if not chunk_text:
                    continue

                first_line = chunk_text.split("\n", 1)[0].strip()
                section_title = first_line[:80] if len(first_line) > 80 else first_line

                chunk_doc = Document(
                    text=chunk_text,
                    metadata={
                        **doc.metadata,
                        "section_index": i,
                        "section_count": len(chunks),
                        "section_title": section_title,
                    },
                )

                if len(chunk_text) > self.chunk_size * 4:
                    sub_nodes = self._fallback_sentence_split(chunk_doc)
                    for node in sub_nodes:
                        node.metadata["section_title"] = section_title
                    result.extend(sub_nodes)
                else:
                    result.append(chunk_doc)

        print(f"分片完成: {len(documents)} 个文档 → {len(result)} 个 chunk（按章节）")
        return result

    def split_single(self, document: Document) -> List[Document]:
        """分割单个文档"""
        return self.split([document])
