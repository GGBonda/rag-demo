"""
离线处理模块 - 负责文档解析、按章分片、向量化、存入向量数据库
"""

from .document_loader import DocumentLoader, ParsedDocument
from .chunker import Chunker
from .embedding_engine import EmbeddingEngine
from .vector_store import VectorStoreManager
from .pipeline import OfflinePipeline

__all__ = [
    "DocumentLoader",
    "ParsedDocument",
    "Chunker",
    "EmbeddingEngine",
    "VectorStoreManager",
    "OfflinePipeline",
]
