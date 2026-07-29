"""
实时响应模块 - 检索器
从 Qdrant 向量数据库中检索与用户问题最相关的 ParagraphChunk
"""

from typing import List, Dict, Any, Optional

from offline_processing.embedding_engine import EmbeddingEngine
from offline_processing.vector_store import VectorStoreManager


class Retriever:
    """检索器，负责从 Qdrant 向量数据库中检索与用户问题最相关的文档片段"""

    def __init__(
        self,
        collection_name: str | None = None,
    ):
        """
        初始化检索器

        Args:
            collection_name: Qdrant 集合名称
        """
        self.embedding_engine = EmbeddingEngine()
        self.vector_store = VectorStoreManager(
            embedding_engine=self.embedding_engine,
            collection_name=collection_name,
        )

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
            结果列表，每项包含: text, type, score, section_id, ai_desc_text
        """
        return self.vector_store.search(
            query_text=query,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )

    def get_stats(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        return self.vector_store.get_stats()
