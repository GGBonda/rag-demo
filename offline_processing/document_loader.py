"""
RAG 知识库 - 文档加载器模块
加载 PDF 并按章节边界拆分为 chunk
"""

import os
import re
from enum import Enum
from pathlib import Path
from typing import List, Optional

# Monkey-patch: unstructured 0.20+ 需要 IsExtracted 枚举，
# 但 unstructured-inference 0.7.x 版本中尚未提供该枚举
import unstructured_inference.constants as _uic  # noqa: E402

if not hasattr(_uic, "IsExtracted"):
    class IsExtracted(Enum):
        TRUE = True
        FALSE = False
        PARTIAL = "partial"

    _uic.IsExtracted = IsExtracted

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

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


class DocumentLoader:
    """文档加载器，加载 PDF 并按最小章节边界拆分为 chunk"""

    SUPPORTED_EXTENSIONS = {
        ".pdf": "application/pdf",
    }

    def __init__(
        self,
        input_dir: str = "./documents",
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        self.input_dir = Path(input_dir)
        self.chunk_size = chunk_size or config.chunk.chunk_size
        self.chunk_overlap = chunk_overlap or config.chunk.chunk_overlap
        os.makedirs(self.input_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def _get_supported_files(self) -> List[Path]:
        """获取目录下所有支持的文件"""
        files = []
        for ext in self.SUPPORTED_EXTENSIONS:
            files.extend(self.input_dir.glob(f"**/*{ext}"))
        return sorted(files)

    def load_all(self) -> List[Document]:
        """加载目录下所有支持的文档，按章节拆分为 chunk"""
        all_chunks = []
        files = self._get_supported_files()

        if not files:
            print(f"警告: 目录 '{self.input_dir}' 下未找到支持的文档文件")
            print(f"支持的格式: {', '.join(self.SUPPORTED_EXTENSIONS.keys())}")
            return all_chunks

        print(f"找到 {len(files)} 个文件待加载...")

        for file_path in files:
            try:
                chunks = self._load_single_file(file_path)
                all_chunks.extend(chunks)
                print(f"  ✓ 已加载: {file_path.name} ({len(chunks)} 个章节 chunk)")
            except Exception as e:
                print(f"  ✗ 加载失败: {file_path.name} - {e}")

        print(f"\n总共加载 {len(all_chunks)} 个章节 chunk")
        return all_chunks

    def load_file(self, file_path: str) -> List[Document]:
        """加载单个指定文件，按章节拆分为 chunk"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        return self._load_single_file(path)

    # ------------------------------------------------------------------
    # 章节拆分核心
    # ------------------------------------------------------------------

    def _load_single_file(self, file_path: Path) -> List[Document]:
        """加载单个 PDF 文件，按最小章节边界拆分为 chunk 列表"""
        suffix = file_path.suffix.lower()
        if suffix not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"不支持的文件格式: {suffix}")

        elements = self._get_pdf_elements(str(file_path))
        elements = self._remove_headers_footers(elements)
        elements = self._correct_elements(elements)

        if not elements:
            return []

        chunks = self._split_elements_by_title(elements)

        result: List[Document] = []
        for i, chunk in enumerate(chunks):
            chunk_text = chunk.text.strip() if hasattr(chunk, "text") else str(chunk).strip()
            if not chunk_text:
                continue

            first_line = chunk_text.split("\n", 1)[0].strip()
            section_title = first_line[:80] if len(first_line) > 80 else first_line

            chunk_doc = Document(
                text=chunk_text,
                metadata={
                    "file_name": file_path.name,
                    "file_path": str(file_path.resolve()),
                    "file_type": suffix.lstrip("."),
                    "section_index": i,
                    "section_count": len(chunks),
                    "section_title": section_title,
                },
            )

            # 超大章节回退到句子级分片
            if len(chunk_text) > self.chunk_size * 4:
                sub_nodes = self._fallback_sentence_split(chunk_doc)
                for node in sub_nodes:
                    node.metadata["section_title"] = section_title
                result.extend(sub_nodes)
            else:
                result.append(chunk_doc)

        print(f"  [章节拆分] {file_path.name}: {len(result)} 个 chunk")
        return result

    @staticmethod
    def _get_pdf_elements(file_path: str) -> list:
        """使用 unstructured.partition.auto.partition 解析 PDF，返回 Element 列表"""
        from unstructured.partition.auto import partition

        elements = partition(
            filename=file_path,
            strategy="fast", # auto fast hi_res
            include_page_breaks=False,
        )
        return list(elements) if elements else []

    @staticmethod
    def _remove_headers_footers(elements: list) -> list:
        """过滤掉页眉（Header）和页脚（Footer）元素"""
        from unstructured.documents.elements import Header, Footer

        return [e for e in elements if not isinstance(e, (Header, Footer))]

    @staticmethod
    def _is_heading(text: str) -> bool:
        """检查文本是否匹配中/英文章节标题模式"""
        text = text.strip()
        if not text:
            return False
        for pattern in _ZH_HEADING_PATTERNS:
            if pattern.match(text):
                return True
        return False

    @classmethod
    def _correct_elements(cls, elements: list) -> list:
        """校正元素类型：中文标题提升为 Title，误判的长文本降级为 NarrativeText"""
        from unstructured.documents.elements import Title, NarrativeText

        corrected: list = []
        for elem in elements:
            elem_text = str(elem.text).strip()
            if not elem_text:
                continue

            if cls._is_heading(elem_text):
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

    def _fallback_sentence_split(self, document: Document) -> List[Document]:
        """章节过大时回退到句子级拆分"""
        splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        return splitter.get_nodes_from_documents([document])
