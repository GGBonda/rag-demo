"""paragraph_chunk 集合的数据操作。"""
from collections.abc import Sequence

from qdrant_client import models

from .qdrant_store import (
    PARAGRAPH_CHUNK_COLLECTION,
    PARAGRAPH_DENSE_VECTOR,
    PARAGRAPH_TEXT_SPARSE_VECTOR,
)
from .qdrant_store import QdrantStore

from offline_processing.chunker import ParagraphChunk


class ParagraphChunkStore:
    """ParagraphChunk 的批量写入、清空和向量查询。"""

    def __init__(self, store: QdrantStore) -> None:
        self._store = store

    def batch_insert(self, chunks: Sequence[ParagraphChunk]) -> int:
        """批量写入已生成向量的 ParagraphChunk。"""
        if not chunks:
            return 0

        points = []
        for chunk in chunks:
            if chunk.embedding_vector is None:
                raise ValueError(f"ParagraphChunk(id={chunk.id}) 缺少向量")
            points.append(
                models.PointStruct(
                    id=chunk.id,
                    vector={
                        PARAGRAPH_DENSE_VECTOR: list(chunk.embedding_vector),
                        PARAGRAPH_TEXT_SPARSE_VECTOR: self._bm25_document(chunk.ai_desc_text if chunk.ai_desc_text.strip() else chunk.text)
                    },
                    payload={
                        "retrieve_id": chunk.retrieve_id,
                        "text": chunk.text,
                        "type": chunk.type,
                        "ai_desc_text": chunk.ai_desc_text,
                    },
                )
            )

        self._store.client.upsert(
            collection_name=PARAGRAPH_CHUNK_COLLECTION,
            points=points,
            wait=True,
        )
        return len(points)

    def clear(self) -> None:
        """清空数据，保留集合及向量配置。"""
        self._store.clear_collection(PARAGRAPH_CHUNK_COLLECTION)

    def query(
        self,
        query_vector: Sequence[float],
        limit: int = 5,
        score_threshold: float | None = None,
    ) -> list[models.ScoredPoint]:
        """使用余弦相似度查询 ParagraphChunk。"""
        if len(query_vector) != self._store.vector_size:
            raise ValueError(f"查询向量维度必须为 {self._store.vector_size}，实际为 {len(query_vector)}")
        if limit <= 0:
            raise ValueError("limit 必须为正整数")

        response = self._store.client.query_points(
            collection_name=PARAGRAPH_CHUNK_COLLECTION,
            query=list(query_vector),
            using=PARAGRAPH_DENSE_VECTOR,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
        )
        return response.points

    def query_text_sparse(
        self,
        query_text: str,
        limit: int = 5,
        score_threshold: float | None = None,
    ) -> list[models.ScoredPoint]:
        """使用 BM25 稀疏向量检索 text 字段。"""
        return self._query_sparse(
            query_text,
            PARAGRAPH_TEXT_SPARSE_VECTOR,
            limit,
            score_threshold,
        )

    def _query_sparse(
        self,
        query_text: str,
        vector_name: str,
        limit: int,
        score_threshold: float | None,
    ) -> list[models.ScoredPoint]:
        if not query_text.strip():
            raise ValueError("query_text 不能为空")
        if limit <= 0:
            raise ValueError("limit 必须为正整数")

        response = self._store.client.query_points(
            collection_name=PARAGRAPH_CHUNK_COLLECTION,
            query=self._bm25_document(query_text),
            using=vector_name,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
        )
        return response.points

    @classmethod
    def _bm25_document(cls, text: str) -> models.Document:
        return models.Document(
            text=text,
            model="qdrant/bm25",
            options=models.Bm25Config(language="chinese"),
        )
