"""markdown_document 集合的数据操作。"""
from collections.abc import Mapping, Sequence

from qdrant_client import models

from .qdrant_store import MARKDOWN_DOCUMENT_COLLECTION
from .qdrant_store import QdrantStore

from offline_processing.document_loader_mineru import MarkdownDocument


class MarkdownDocumentStore:
    """MarkdownDocument 的批量写入、清空和精确查询。"""

    def __init__(self, store: QdrantStore) -> None:
        self._store = store

    def batch_insert(self, documents: Sequence[MarkdownDocument]) -> int:
        """批量写入 MarkdownDocument，并返回写入数量。"""
        if not documents:
            return 0

        points = [
            models.PointStruct(
                id=document.id,
                vector={},
                payload={
                    "file_name": document.file_name,
                    "author": document.author,
                    "original_file_url": document.original_file_url,
                    "created_at": (
                        document.created_at.isoformat()
                        if document.created_at is not None
                        else None
                    ),
                    "business_team_id": document.business_team_id,
                    "markdown_text": document.markdown_text,
                    "md5": document.md5,
                },
            )
            for document in documents
        ]
        self._store.client.upsert(
            collection_name=MARKDOWN_DOCUMENT_COLLECTION,
            points=points,
            wait=True,
        )
        return len(points)

    def clear(self) -> None:
        """清空数据，保留集合及索引。"""
        self._store.clear_collection(MARKDOWN_DOCUMENT_COLLECTION)

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
            collection_name=MARKDOWN_DOCUMENT_COLLECTION,
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
