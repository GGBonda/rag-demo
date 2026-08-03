"""Qdrant 客户端与集合管理。"""

from __future__ import annotations

import os
from collections.abc import Mapping

from qdrant_client import QdrantClient, models

from config import config


MARKDOWN_DOCUMENT_COLLECTION = "markdown_document"
RETRIEVE_CHUNK_COLLECTION = "retrieve_chunk"
PARAGRAPH_CHUNK_COLLECTION = "paragraph_chunk"
PARAGRAPH_DENSE_VECTOR = "dense"
PARAGRAPH_TEXT_SPARSE_VECTOR = "sparse"


class QdrantStore:
    """Qdrant 连接、集合初始化及各 collection 数据操作入口。"""

    _COLLECTION_NAMES = {
        MARKDOWN_DOCUMENT_COLLECTION,
        RETRIEVE_CHUNK_COLLECTION,
        PARAGRAPH_CHUNK_COLLECTION,
    }
    def __init__(
        self
    ) -> None:
        self.vector_size = config.embedding.openai_dimension
        if self.vector_size <= 0:
            raise ValueError("vector_size 必须为正整数")

        os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
        self.client = QdrantClient(
            host=config.qdrant.host,
            port=config.qdrant.port,
        )

    def initialize_collections(self) -> None:
        """初始化三个集合；重复调用不会重建已存在的集合。"""
        self._create_collection_if_missing(
            MARKDOWN_DOCUMENT_COLLECTION,
            vectors_config={},
        )
        self._create_collection_if_missing(
            RETRIEVE_CHUNK_COLLECTION,
            vectors_config={},
        )
        self._create_collection_if_missing(
            PARAGRAPH_CHUNK_COLLECTION,
            vectors_config={
                PARAGRAPH_DENSE_VECTOR: models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                PARAGRAPH_TEXT_SPARSE_VECTOR: models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                )
            },
        )

    def clear_collection(self, collection_name: str) -> None:
        """清空指定集合的数据，保留集合配置。"""
        self._validate_collection_name(collection_name)
        self.client.delete(
            collection_name=collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(must=[]),
            ),
            wait=True,
        )

    def _create_collection_if_missing(
        self,
        collection_name: str,
        vectors_config: models.VectorParams | dict,
        sparse_vectors_config: Mapping[
            str, models.SparseVectorParams
        ] | None = None,
    ) -> None:
        if self.client.collection_exists(collection_name):
            return
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config,
        )

    def _validate_collection_name(self, collection_name: str) -> None:
        if collection_name not in self._COLLECTION_NAMES:
            raise ValueError(f"不支持的 collection: {collection_name}")


__all__ = [
    "MARKDOWN_DOCUMENT_COLLECTION",
    "PARAGRAPH_CHUNK_COLLECTION",
    "PARAGRAPH_DENSE_VECTOR",
    "PARAGRAPH_TEXT_SPARSE_VECTOR",
    "QdrantStore",
    "RETRIEVE_CHUNK_COLLECTION",
]
