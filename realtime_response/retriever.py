"""
实时响应模块 - 检索器
从 Qdrant 向量数据库中检索与用户问题最相关的 IndexChunk
"""

from typing import Optional

from model import RetrieveChunk
from offline_processing.embedding_engine import EmbeddingEngine
from realtime_response.cross_encoder_reranker import CrossEncoderReranker
from store import (
    IndexChunkStore,
    QdrantStore,
    RetrieveChunkStore,
)


class Retriever:
    """检索器，负责从 Qdrant 向量数据库中检索与用户问题最相关的文档片段"""

    def __init__(self):
        """初始化检索器。"""
        self.embedding_engine = EmbeddingEngine()
        qdrant_store = QdrantStore()
        self.index_chunk_store = IndexChunkStore(qdrant_store)
        self.retrieve_chunk_store = RetrieveChunkStore(qdrant_store)

    def search(self, query: str, top_k: int = 5) -> list[RetrieveChunk]:
        query_vector = self.embedding_engine.embed_query(query)

        results = self.index_chunk_store.query(
            query_vector=query_vector,
            query_text=query,
            limit=20,
            candidate_limit=50,
            dense_score_threshold=0.5,
        )

        retrieve_ids = list(dict.fromkeys(hit.retrieve_id for hit in results))
        chunks = self.retrieve_chunk_store.get_by_ids(retrieve_ids)
        return CrossEncoderReranker().rerank(
            query=query,
            chunks=chunks,
            top_k=top_k,
        )
