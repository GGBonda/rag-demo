"""
RAG 知识库 - 文档加载器模块
支持 PDF 格式的文件解析
"""

import os
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

from llama_index.core import Document
from llama_index.readers.file import UnstructuredReader


class DocumentLoader:
    """文档加载器，支持 PDF 文件解析"""

    SUPPORTED_EXTENSIONS = {
        ".pdf": "application/pdf",
    }

    def __init__(self, input_dir: str = "./documents"):
        self.input_dir = Path(input_dir)
        os.makedirs(self.input_dir, exist_ok=True)
        self._reader = UnstructuredReader()

    def _get_supported_files(self) -> List[Path]:
        """获取目录下所有支持的文件"""
        files = []
        for ext in self.SUPPORTED_EXTENSIONS:
            files.extend(self.input_dir.glob(f"**/*{ext}"))
        return sorted(files)

    def _load_single_file(self, file_path: Path) -> List[Document]:
        """加载单个文件，返回 Document 列表"""
        suffix = file_path.suffix.lower()
        if suffix not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"不支持的文件格式: {suffix}")

        documents = self._reader.load_data(
            file=file_path,
            split_documents=True,
            unstructured_kwargs={
                "strategy": "hi_res",  # auto 自动选择策略，文字型 PDF 用 fast，扫描件用 hi_res
                # 不使用任何需要 AI 模型的功能
            },
            extra_info={
                "file_name": file_path.name,
                "file_path": str(file_path.resolve()),
                "file_type": suffix.lstrip("."),
            },
        )

        if not documents:
            return []

        # UnstructuredReader 已将所有元素合并为一个 Document
        doc = documents[0]
        doc.metadata.update({
            "file_name": file_path.name,
            "file_path": str(file_path.resolve()),
            "file_type": suffix.lstrip("."),
        })
        return [doc]

    def load_all(self) -> List[Document]:
        """加载目录下所有支持的文档"""
        all_documents = []
        files = self._get_supported_files()

        if not files:
            print(f"警告: 目录 '{self.input_dir}' 下未找到支持的文档文件")
            print(f"支持的格式: {', '.join(self.SUPPORTED_EXTENSIONS.keys())}")
            return all_documents

        print(f"找到 {len(files)} 个文件待加载...")

        for file_path in files:
            try:
                docs = self._load_single_file(file_path)
                all_documents.extend(docs)
                print(f"  ✓ 已加载: {file_path.name} ({len(docs)} 个文档片段)")
            except Exception as e:
                print(f"  ✗ 加载失败: {file_path.name} - {e}")

        print(f"\n总共加载 {len(all_documents)} 个文档片段")
        return all_documents

    def load_file(self, file_path: str) -> List[Document]:
        """加载单个指定文件"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        return self._load_single_file(path)
