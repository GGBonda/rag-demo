"""
实时响应模块 - 检索器
从 Qdrant 向量数据库中检索与用户问题最相关的 IndexChunk
"""

from typing import List, Dict, Any, Optional

from offline_processing.embedding_engine import EmbeddingEngine
from store import (
    IndexChunkStore,
    QdrantStore,
)


class Retriever:
    """检索器，负责从 Qdrant 向量数据库中检索与用户问题最相关的文档片段"""

    def __init__(self):
        """初始化检索器。"""
        self.embedding_engine = EmbeddingEngine()
        self.index_chunk_store = IndexChunkStore(QdrantStore())

    def search(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        检索与查询最相关的文档片段

        Args:
            query: 查询文本
            top_k: 返回最相似的前 K 个结果
            similarity_threshold: 相似度阈值（0~1）

        Returns:
            结果列表，每项包含: text, type, section_id, ai_desc_text
        """
        query_vector = self.embedding_engine.embed_query(query)
        results = self.index_chunk_store.query(
            query_vector=query_vector,
            query_text=query,
            limit=top_k,
            score_threshold=similarity_threshold,
        )

        return [
            {
                "text": hit.text,
                "type": hit.type,
                "section_id": hit.retrieve_id,
                "ai_desc_text": hit.ai_desc_text,
            }
            for hit in results
        ]
