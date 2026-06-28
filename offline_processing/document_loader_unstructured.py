"""
RAG 知识库 - 文档加载器模块
加载 PDF 并解析为 Markdown 格式
"""

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# Monkey-patch: unstructured 0.20+ 需要 IsExtracted 枚举，
# 但 unstructured-inference 0.7.x 版本中尚未提供该枚举
import unstructured_inference.constants as _uic  # noqa: E402

if not hasattr(_uic, "IsExtracted"):
    class IsExtracted(Enum):
        TRUE = True
        FALSE = False
        PARTIAL = "partial"

    _uic.IsExtracted = IsExtracted


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


@dataclass
class ParsedDocument:
    """单个 PDF 文件的解析结果"""
    markdown_text: str
    """PDF 解析后的 Markdown 文本"""
    file_name: str
    """文件名"""
    file_path: str
    """文件绝对路径"""


class DocumentLoader:
    """文档加载器，加载单个 PDF 并解析为 Markdown 格式"""

    SUPPORTED_EXTENSIONS = {
        ".pdf": "application/pdf",
    }

    def __init__(
        self,
        file_path: str | None = None,
        start_page: int | None = None,
        end_page: int | None = None,
    ):
        """
        Args:
            file_path: PDF 文件路径
            start_page: 起始页码（1-indexed，含），None 表示从第一页开始
            end_page: 结束页码（1-indexed，含），None 表示到最后一页
        """
        self.file_path = Path(file_path) if file_path else None
        self.start_page = start_page
        self.end_page = end_page

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def load(self, file_path: str | None = None) -> ParsedDocument:
        """加载指定 PDF 文件，返回 ParsedDocument"""
        path = Path(file_path or self.file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")
        markdown_text = self._load_single_file(path)
        return ParsedDocument(
            markdown_text=markdown_text,
            file_name=path.name,
            file_path=str(path.resolve()),
        )

    # ------------------------------------------------------------------
    # 文档解析核心
    # ------------------------------------------------------------------

    def _load_single_file(self, file_path: Path) -> str:
        """加载单个 PDF 文件，返回解析后的 Markdown 文本"""
        suffix = file_path.suffix.lower()
        if suffix not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"不支持的文件格式: {suffix}")

        elements = self._get_pdf_elements(str(file_path))
        elements = self._filter_by_page_range(elements)
        elements = self._remove_headers_footers(elements)
        elements = self._correct_elements(elements)
        markdown_text = self._elements_to_markdown(elements)

        return markdown_text

    @staticmethod
    def _get_pdf_elements(file_path: str) -> list:
        from unstructured.partition.pdf import partition_pdf

        elements = partition_pdf(
            filename=file_path,
            strategy="hi_res",
            include_page_breaks=False,
            languages=['chi_sim', 'eng'],
            infer_table_structure=True,
            encoding='utf-8',
        )
        return list(elements) if elements else []

    def _filter_by_page_range(self, elements: list) -> list:
        """按页码范围过滤元素"""
        if self.start_page is None and self.end_page is None:
            return elements
        filtered = []
        for e in elements:
            page = getattr(e.metadata, "page_number", None)
            if page is None:
                # 元素无页码信息时保留
                filtered.append(e)
            elif self.start_page is not None and page < self.start_page:
                continue
            elif self.end_page is not None and page > self.end_page:
                continue
            else:
                filtered.append(e)
        return filtered

    @staticmethod
    def _remove_headers_footers(elements: list) -> list:
        """过滤掉页眉（Header）和页脚（Footer）元素"""
        from unstructured.documents.elements import Header, Footer

        return [e for e in elements if not isinstance(e, (Header, Footer))]

    @staticmethod
    def _elements_to_markdown(elements: list) -> str:
        """将结构化元素列表转换为 Markdown 文本"""
        from unstructured.staging.base import elements_to_md

        return elements_to_md(elements)

    # ------------------------------------------------------------------
    # 元素类型校正
    # ------------------------------------------------------------------

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
        """校正元素类型：误判为 Title 的长文本降级为 NarrativeText"""
        from unstructured.documents.elements import Title, NarrativeText

        corrected: list = []
        for elem in elements:
            elem_text = str(elem.text).strip()

            if isinstance(elem, Title) and (
                len(elem_text) > _TITLE_MAX_LENGTH
                or not cls._is_heading(elem_text)
            ):
                corrected.append(NarrativeText(elem_text))
            else:
                corrected.append(elem)

        return corrected
