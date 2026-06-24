"""
RAG 知识库 - 文档加载器管理器
统一管理 Unstructured 和 MinerU 两种文档加载后端，
通过传入参数自由选择使用哪种加载器解析 PDF 文件。
"""

from pathlib import Path
from typing import List

from .document_loader_unstructured import DocumentLoader, ParsedDocument
from .document_loader_mineru import MinerUDocumentLoader


class DocumentLoaderManager:
    """文档加载器管理器，管理多种文档加载后端。

    用法示例::

        # 使用 Unstructured 后端
        manager = DocumentLoaderManager(backend="unstructured", input_dir="./documents")
        docs = manager.load_all()

        # 使用 MinerU 后端
        manager = DocumentLoaderManager(backend="mineru", input_dir="./documents")
        docs = manager.load_all()

        # MinerU 指定 OCR 模式
        manager = DocumentLoaderManager(
            backend="mineru",
            input_dir="./documents",
            method="ocr",
        )
    """

    SUPPORTED_BACKENDS = {
        "unstructured": DocumentLoader,
        "mineru": MinerUDocumentLoader,
    }

    def __init__(
        self,
        backend: str = "unstructured",
        input_dir: str = "./documents",
        **backend_kwargs,
    ):
        """
        Args:
            backend: 文档加载后端 - "unstructured" 或 "mineru"
            input_dir: 文档目录路径
            **backend_kwargs: 传递给具体加载后端的额外参数。
                对于 MinerU，支持 ``method``（"auto"/"txt"/"ocr"）。
        """
        backend_lower = backend.lower()
        if backend_lower not in self.SUPPORTED_BACKENDS:
            raise ValueError(
                f"不支持的文档加载后端: {backend}，"
                f"可选值: {', '.join(self.SUPPORTED_BACKENDS.keys())}"
            )

        self.backend = backend_lower
        self.input_dir = Path(input_dir)

        loader_cls = self.SUPPORTED_BACKENDS[backend_lower]
        self._loader = loader_cls(input_dir=str(input_dir), **backend_kwargs)

    @property
    def loader(self):
        """获取当前使用的加载器实例"""
        return self._loader

    @property
    def SUPPORTED_EXTENSIONS(self) -> dict:
        """代理到当前加载器的支持格式"""
        return self._loader.SUPPORTED_EXTENSIONS

    def load_all(self) -> List[ParsedDocument]:
        """加载目录下所有支持的文档"""
        return self._loader.load_all()

    def load_file(self, file_path: str) -> ParsedDocument:
        """加载单个指定文件"""
        return self._loader.load_file(file_path)
