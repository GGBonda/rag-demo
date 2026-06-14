"""
RAG 知识库 - 文档加载器模块
加载 PDF 并解析为结构化元素
"""

import os
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List

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
    elements: list
    """PDF 解析后的结构化元素列表"""
    file_name: str
    """文件名"""
    file_path: str
    """文件绝对路径"""


class DocumentLoader:
    """文档加载器，加载 PDF 并解析为结构化元素"""

    SUPPORTED_EXTENSIONS = {
        ".pdf": "application/pdf",
    }

    def __init__(self, input_dir: str = "./documents"):
        self.input_dir = Path(input_dir)
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

    def load_all(self) -> List[ParsedDocument]:
        """加载目录下所有支持的文档，返回 ParsedDocument 列表"""
        results: List[ParsedDocument] = []
        files = self._get_supported_files()

        if not files:
            print(f"警告: 目录 '{self.input_dir}' 下未找到支持的文档文件")
            print(f"支持的格式: {', '.join(self.SUPPORTED_EXTENSIONS.keys())}")
            return results

        print(f"找到 {len(files)} 个文件待加载...")

        for file_path in files:
            try:
                elements = self._load_single_file(file_path)
                parsed = ParsedDocument(
                    elements=elements,
                    file_name=file_path.name,
                    file_path=str(file_path.resolve()),
                )
                results.append(parsed)
                print(f"  ✓ 已加载: {file_path.name} ({len(elements)} 个元素)")
            except Exception as e:
                print(f"  ✗ 加载失败: {file_path.name} - {e}")

        print(f"\n总共加载 {len(results)} 个文件")
        return results

    def load_file(self, file_path: str) -> ParsedDocument:
        """加载单个指定文件，返回 ParsedDocument"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        elements = self._load_single_file(path)
        return ParsedDocument(
            elements=elements,
            file_name=path.name,
            file_path=str(path.resolve()),
        )

    # ------------------------------------------------------------------
    # 文档解析核心
    # ------------------------------------------------------------------

    def _load_single_file(self, file_path: Path) -> list:
        """加载单个 PDF 文件，返回解析并校正后的元素列表"""
        suffix = file_path.suffix.lower()
        if suffix not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"不支持的文件格式: {suffix}")

        elements = self._get_pdf_elements(str(file_path))
        elements = self._remove_headers_footers(elements)
        elements = self._correct_elements(elements)

        return elements

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

            if isinstance(elem, Title) and (len(elem_text) > _TITLE_MAX_LENGTH or not cls._is_heading(elem_text)):
                corrected.append(NarrativeText(elem_text))
            else:
                corrected.append(elem)

            """
            if cls._is_heading(elem_text):
                if not isinstance(elem, Title):
                    corrected.append(Title(elem_text))
                else:
                    corrected.append(elem)
                continue
            """



        return corrected

