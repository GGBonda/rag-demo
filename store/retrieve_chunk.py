"""retrieve_chunk 集合的数据操作。"""
from collections.abc import Mapping, Sequence

from qdrant_client import models

from .qdrant_store import RETRIEVE_CHUNK_COLLECTION
from .qdrant_store import QdrantStore

from offline_processing.chunker import RetrieveChunk



class RetrieveChunkStore:
    """RetrieveChunk 的批量写入、清空和精确查询。"""

    def __init__(self, store: QdrantStore) -> None:
        self._store = store

    def batch_insert(self, chunks: Sequence[RetrieveChunk]) -> int:
        """批量写入 RetrieveChunk，并返回写入数量。"""
        if not chunks:
            return 0

        points = [
            models.PointStruct(
                id=chunk.id,
                vector={},
                payload={
                    "doc_id": chunk.doc_id,
                    "title_path": chunk.title_path,
                    "text": chunk.text,
                },
            )
            for chunk in chunks
        ]
        self._store.client.upsert(
            collection_name=RETRIEVE_CHUNK_COLLECTION,
            points=points,
            wait=True,
        )
        return len(points)

    def clear(self) -> None:
        """清空数据，保留集合及索引。"""
        self._store.clear_collection(RETRIEVE_CHUNK_COLLECTION)

    def query(
        self,
        fields: Mapping[str, str | int | bool],
        limit: int = 10,
    ) -> list[models.Record]:
        """按 payload 字段做 AND 精确匹配。"""
        if not fields:
            raise ValueError("fields 不能为空")
        if limit <= 0:
            raise ValueError("limit 必须为正整数")

        records, _ = self._store.client.scroll(
            collection_name=RETRIEVE_CHUNK_COLLECTION,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key=field_name,
                        match=models.MatchValue(value=value),
                    )
                    for field_name, value in fields.items()
                ]
            ),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return records
