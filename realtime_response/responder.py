"""
实时响应模块 - 回答生成器
接收用户提问，检索相关文档，生成自然语言回答
"""

from typing import List, Dict, Any, Optional

from .retriever import Retriever


class Responder:
    """回答生成器，负责处理用户提问并返回回答"""

    def __init__(
        self,
        embedding_backend: str | None = None,
        table_name: str | None = None,
    ):
        """
        初始化回答生成器

        Args:
            embedding_backend: embedding 后端
            table_name: 数据库表名
        """
        self.retriever = Retriever(
            embedding_backend=embedding_backend,
            table_name=table_name,
        )

    def load_knowledge_base(self) -> bool:
        """加载知识库，返回是否成功"""
        return self.retriever.load_index()

    def ask(
        self,
        question: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        回答用户问题（纯检索模式，不生成自然语言回答）

        Args:
            question: 用户提问
            top_k: 检索前 K 个相关 chunk

        Returns:
            相关文档片段列表
        """
        return self.retriever.search(query=question, top_k=top_k)

    def ask_with_llm(
        self,
        question: str,
        top_k: int = 5,
        llm=None,
    ):
        """
        使用 LLM 进行检索增强生成（RAG）

        Args:
            question: 用户提问
            top_k: 检索前 K 个相关 chunk
            llm: LLM 实例（如 OpenAI），不传则降级为纯检索

        Returns:
            LLM 生成的回答，或检索结果列表
        """
        if llm is not None:
            query_engine = self.retriever.vector_store.as_query_engine(
                top_k=top_k, llm=llm
            )
            return query_engine.query(question)
        else:
            return self.ask(question, top_k=top_k)

    def show_stats(self) -> None:
        """显示知识库统计信息"""
        stats = self.retriever.get_stats()
        print("\n知识库统计:")
        for key, value in stats.items():
            print(f"  - {key}: {value}")
