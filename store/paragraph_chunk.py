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
                        PARAGRAPH_DENSE_VECTOR: chunk.embedding_vector,
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
        query_vector: list[float],
        query_text: str,
        limit: int = 5,
        score_threshold: float | None = None,
    ) -> list[ParagraphChunk]:
        """使用稠密向量和 BM25 稀疏向量混合查询 ParagraphChunk。"""
        if len(query_vector) != self._store.vector_size:
            raise ValueError(f"查询向量维度必须为 {self._store.vector_size}，实际为 {len(query_vector)}")
        if not query_text.strip():
            raise ValueError("query_text 不能为空")
        if limit <= 0:
            raise ValueError("limit 必须为正整数")

        response = self._store.client.query_points(
            collection_name=PARAGRAPH_CHUNK_COLLECTION,
            prefetch=[
                models.Prefetch(
                    query=query_vector,
                    using=PARAGRAPH_DENSE_VECTOR,
                    limit=limit,
                    score_threshold=score_threshold,
                ),
                models.Prefetch(
                    query=self._bm25_document(query_text),
                    using=PARAGRAPH_TEXT_SPARSE_VECTOR,
                    limit=limit,
                ),
            ],
            query=models.RrfQuery(
                rrf=models.Rrf(weights=[0.8, 0.2]),
            ),
            limit=limit,
            with_payload=True,
        )
        print(response.model_dump_json(indent=2))
        return [
            ParagraphChunk(
                id=point.id,
                retrieve_id=(point.payload or {}).get("retrieve_id", 0),
                text=(point.payload or {}).get("text", ""),
                type=(point.payload or {}).get("type", "text"),
                ai_desc_text=(point.payload or {}).get("ai_desc_text", ""),
                score=point.score,
            )
            for point in response.points
        ]

    @classmethod
    def _bm25_document(cls, text: str) -> models.Document:
        return models.Document(
            text=text,
            model="qdrant/bm25",
            options=models.Bm25Config(language="chinese"),
        )
