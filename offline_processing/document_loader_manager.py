"""
RAG 知识库 - 文档加载器管理器
统一管理 Unstructured 和 MinerU 两种文档加载后端，
通过传入参数自由选择使用哪种加载器解析 PDF 文件。
"""

from .document_loader_unstructured import DocumentLoader, ParsedDocument
from .document_loader_mineru import MinerUDocumentLoader


class DocumentLoaderManager:
    """文档加载器管理器，管理多种文档加载后端。

    用法示例::

        # 使用 Unstructured 后端
        manager = DocumentLoaderManager(backend="unstructured", file_path="./doc.pdf")
        doc = manager.load()

        # 使用 MinerU 后端
        manager = DocumentLoaderManager(backend="mineru", file_path="./doc.pdf")
        doc = manager.load()

        # MinerU 指定 OCR 模式
        manager = DocumentLoaderManager(
            backend="mineru",
            file_path="./doc.pdf",
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
        file_path: str | None = None,
        **backend_kwargs,
    ):
        """
        Args:
            backend: 文档加载后端 - "unstructured" 或 "mineru"
            file_path: PDF 文件路径
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
        self.file_path = file_path

        loader_cls = self.SUPPORTED_BACKENDS[backend_lower]
        self._loader = loader_cls(file_path=file_path, **backend_kwargs)

    @property
    def loader(self):
        """获取当前使用的加载器实例"""
        return self._loader

    @property
    def SUPPORTED_EXTENSIONS(self) -> dict:
        """代理到当前加载器的支持格式"""
        return self._loader.SUPPORTED_EXTENSIONS

    def load(self) -> ParsedDocument:
        """加载指定的 PDF 文件"""
        return self._loader.load()
