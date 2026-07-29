"""
实时响应模块 - 回答生成器
接收用户提问，检索相关文档，生成自然语言回答
"""

from typing import List, Dict, Any

from .retriever import Retriever


class Responder:
    """回答生成器，负责处理用户提问并返回回答"""

    def __init__(
        self,
        collection_name: str | None = None,
    ):
        """
        初始化回答生成器

        Args:
            collection_name: Qdrant 集合名称
        """
        self.retriever = Retriever(
            collection_name=collection_name,
        )

    def ask(
        self,
        question: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        检索与问题最相关的文档片段

        Args:
            question: 用户提问
            top_k: 检索前 K 个相关 chunk

        Returns:
            相关文档片段列表
        """
        return self.retriever.search(query=question, top_k=top_k)

    def show_stats(self) -> None:
        """显示知识库统计信息"""
        stats = self.retriever.get_stats()
        print("\n知识库统计:")
        for key, value in stats.items():
            print(f"  - {key}: {value}")
