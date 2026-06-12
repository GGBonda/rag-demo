"""
RAG 知识库 - PGvector 向量存储模块
负责向量数据库的连接、建表、写入、检索
"""

from typing import List, Optional, Dict, Any

import psycopg2
from psycopg2.extras import execute_values
from sqlalchemy import make_url

from llama_index.core import Document
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core import VectorStoreIndex, StorageContext

from config import config
from .embedding_engine import EmbeddingEngine


class VectorStoreManager:
    """PGvector 向量存储管理器"""

    def __init__(
        self,
        embedding_engine: Optional[EmbeddingEngine] = None,
        table_name: Optional[str] = None,
    ):
        """
        初始化向量存储管理器

        Args:
            embedding_engine: Embedding 引擎实例
            table_name: PGvector 表名
        """
        self.embedding_engine = embedding_engine or EmbeddingEngine()
        self.table_name = table_name or config.database.table_name
        self.db_cfg = config.database

        # 创建 LlamaIndex PGVectorStore 实例
        self._vector_store = PGVectorStore.from_params(
            host=self.db_cfg.host,
            port=self.db_cfg.port,
            database=self.db_cfg.database,
            user=self.db_cfg.user,
            password=self.db_cfg.password,
            table_name=self.table_name,
            embed_dim=self.embedding_engine.embedding_dimension,
            # 使用余弦相似度
            hnsw_kwargs={
                "hnsw_m": 16,
                "hnsw_ef_construction": 64,
                "hnsw_ef_search": 40,
                "hnsw_dist_method": "vector_cosine_ops",
            },
        )

        self._storage_context: Optional[StorageContext] = None
        self._index: Optional[VectorStoreIndex] = None

    def initialize_database(self) -> None:
        """
        初始化数据库：创建 PGvector 扩展和表结构
        这一步需要在 PostgreSQL 中启用 pgvector 扩展
        """

        conn = psycopg2.connect(**self.db_cfg.connection_params)
        conn.autocommit = True

        try:
            with conn.cursor() as cur:
                # 创建 pgvector 扩展
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                print("✓ PGvector 扩展已启用")
        finally:
            conn.close()

    def _ensure_database_exists(self) -> None:
        """确保目标数据库存在，不存在则创建"""
        # 连接到默认的 postgres 数据库来创建目标数据库
        params = self.db_cfg.connection_params.copy()
        params["database"] = "postgres"

        conn = psycopg2.connect(**params)
        conn.autocommit = True

        try:
            with conn.cursor() as cur:
                # 检查数据库是否存在
                cur.execute(
                    "SELECT 1 FROM pg_database WHERE datname = %s",
                    (self.db_cfg.database,),
                )
                if cur.fetchone() is None:
                    cur.execute(
                        f'CREATE DATABASE "{self.db_cfg.database}";'
                    )
                    print(f"✓ 数据库 '{self.db_cfg.database}' 已创建")
                else:
                    print(f"✓ 数据库 '{self.db_cfg.database}' 已存在")
        finally:
            conn.close()

    def setup(self) -> None:
        """完整的数据库初始化流程"""
        self._ensure_database_exists()
        self.initialize_database()
        print(f"✓ 向量存储表 '{self.table_name}' 已就绪")

    def build_index(self, nodes: List[Document]) -> VectorStoreIndex:
        """
        从节点列表构建向量索引（写入向量数据库）

        Args:
            nodes: 分片后的 Document/Node 列表

        Returns:
            VectorStoreIndex 实例
        """
        if not nodes:
            raise ValueError("节点列表为空，无法构建索引")

        print(f"正在为 {len(nodes)} 个 chunk 生成向量并写入数据库...")

        self._storage_context = StorageContext.from_defaults(
            vector_store=self._vector_store
        )

        # 构建索引：自动调用 embedding 模型生成向量，写入 PGvector
        self._index = VectorStoreIndex(
            nodes,
            storage_context=self._storage_context,
            embed_model=self.embedding_engine.get_model(),
            show_progress=True,
        )

        print(f"✓ 向量索引构建完成，已写入表 '{self.table_name}'")
        return self._index

    def load_index(self) -> Optional[VectorStoreIndex]:
        """
        从已有的向量数据库中加载索引（用于查询）

        Returns:
            VectorStoreIndex 实例，如果表为空则返回 None
        """
        self._storage_context = StorageContext.from_defaults(
            vector_store=self._vector_store
        )

        try:
            self._index = VectorStoreIndex.from_vector_store(
                self._vector_store,
                embed_model=self.embedding_engine.get_model(),
            )
            print(f"✓ 已从表 '{self.table_name}' 加载向量索引")
            return self._index
        except Exception as e:
            print(f"加载索引失败: {e}")
            return None

    def get_index(self) -> Optional[VectorStoreIndex]:
        """获取当前索引实例"""
        return self._index

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        similarity_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        查询最相似的文档 chunk

        Args:
            query_text: 查询文本
            top_k: 返回最相似的前 K 个结果
            similarity_threshold: 相似度阈值（0~1），低于此值的结果会被过滤

        Returns:
            结果列表，每项包含: text, score, metadata
        """
        if self._index is None:
            raise RuntimeError("索引未初始化，请先调用 build_index() 或 load_index()")

        retriever = self._index.as_retriever(
            similarity_top_k=top_k,
            # similarity_cutoff=similarity_threshold,  # 需要时启用
        )

        nodes = retriever.retrieve(query_text)

        results = []
        for node in nodes:
            results.append({
                "text": node.text,
                "score": node.score if hasattr(node, "score") else None,
                "metadata": node.metadata if hasattr(node, "metadata") else {},
            })

        return results

    def as_query_engine(
        self,
        top_k: int = 5,
        llm=None,
    ):
        """
        将索引转换为查询引擎（支持 LLM 生成回答）

        Args:
            top_k: 检索 top-K 个相关 chunk
            llm: 可选的 LLM 实例（用于生成自然语言回答）

        Returns:
            QueryEngine 实例
        """
        if self._index is None:
            raise RuntimeError("索引未初始化，请先调用 build_index() 或 load_index()")

        retriever_kwargs = {"similarity_top_k": top_k}

        return self._index.as_query_engine(
            retriever_kwargs=retriever_kwargs,
            llm=llm,
        )

    def get_stats(self) -> Dict[str, Any]:
        """获取向量存储的统计信息"""
        conn = psycopg2.connect(**self.db_cfg.connection_params)
        try:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {self.table_name};")
                count = cur.fetchone()[0]
                return {
                    "table_name": self.table_name,
                    "document_count": count,
                    "embedding_dimension": self.embedding_engine.embedding_dimension,
                    "database": self.db_cfg.database,
                }
        finally:
            conn.close()

    def clear(self, confirm: bool = False) -> None:
        """
        清空向量表数据

        Args:
            confirm: 必须显式设为 True 才会执行清空操作
        """
        if not confirm:
            print("警告: 请设置 confirm=True 来确认清空操作")
            return

        conn = psycopg2.connect(**self.db_cfg.connection_params)
        conn.autocommit = True
        try:
            with conn.cursor() as cur:
                cur.execute(f"TRUNCATE TABLE {self.table_name};")
                print(f"✓ 表 '{self.table_name}' 已清空")
        finally:
            conn.close()
