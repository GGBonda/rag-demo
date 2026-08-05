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

    def get_by_ids(self, ids: Sequence[int]) -> list[RetrieveChunk]:
        """根据 ID 列表批量查询 RetrieveChunk，并按输入 ID 顺序返回。"""
        if not ids:
            return []

        records = self._store.client.retrieve(
            collection_name=RETRIEVE_CHUNK_COLLECTION,
            ids=list(ids),
            with_payload=True,
            with_vectors=False,
        )
        records_by_id = {record.id: record for record in records}

        return [
            RetrieveChunk(
                id=record.id,
                doc_id=(record.payload or {}).get("doc_id", 0),
                title_path=(record.payload or {}).get("title_path", ""),
                text=(record.payload or {}).get("text", ""),
            )
            for chunk_id in ids
            if (record := records_by_id.get(chunk_id)) is not None
        ]
