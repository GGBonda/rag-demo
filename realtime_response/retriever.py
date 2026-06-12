"""
实时响应模块 - 检索器
从向量数据库中加载索引，执行相似度检索，返回相关文档 chunk
"""

from typing import List, Dict, Any, Optional

from offline_processing.embedding_engine import EmbeddingEngine
from offline_processing.vector_store import VectorStoreManager


class Retriever:
    """检索器，负责从向量数据库中检索与用户问题最相关的文档片段"""

    def __init__(
        self,
        embedding_backend: str | None = None,
        table_name: str | None = None,
    ):
        """
        初始化检索器

        Args:
            embedding_backend: embedding 后端
            table_name: 数据库表名
        """
        self.embedding_engine = EmbeddingEngine(backend=embedding_backend)
        self.vector_store = VectorStoreManager(
            embedding_engine=self.embedding_engine,
            table_name=table_name,
        )

    def load_index(self) -> bool:
        """
        从数据库加载已有的向量索引

        Returns:
            是否成功加载
        """
        index = self.vector_store.load_index()
        return index is not None

    def search(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        检索与查询最相关的文档 chunk

        Args:
            query: 查询文本
            top_k: 返回最相似的前 K 个结果
            similarity_threshold: 相似度阈值（0~1）

        Returns:
            结果列表，每项包含: text, score, metadata
        """
        return self.vector_store.query(
            query_text=query,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )

    def get_stats(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        return self.vector_store.get_stats()
