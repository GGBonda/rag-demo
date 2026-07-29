"""
RAG 知识库 - Qdrant 向量存储模块
基于 Qdrant 实现对 ParagraphChunk 数据的向量存储与检索
"""

from typing import List, Dict, Any
import os

from qdrant_client import QdrantClient
from qdrant_client import models

from config import config
from .chunker import ParagraphChunk
from .embedding_engine import EmbeddingEngine


class VectorStoreManager:
    """Qdrant 向量存储管理器，管理 ParagraphChunk 的存储与检索"""

    def __init__(
        self,
        embedding_engine: EmbeddingEngine | None = None,
        collection_name: str | None = None,
    ):
        """
        初始化向量存储管理器

        Args:
            embedding_engine: Embedding 引擎实例
            collection_name: Qdrant 集合名称
        """
        self.embedding_engine = embedding_engine or EmbeddingEngine()
        self.collection_name = collection_name or config.qdrant.collection_name
        self.qdrant_cfg = config.qdrant

        # httpx 通过 urllib.request.getproxies() 读取 macOS 系统代理，
        # 但不读取系统代理的例外列表（localhost/127.0.0.1），导致
        # localhost 请求也被发往代理服务器。此处补上 NO_PROXY 兜底。
        os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")

        self._client = QdrantClient(
            host=self.qdrant_cfg.host,
            port=self.qdrant_cfg.port,
        )

    def setup(self) -> None:
        """初始化 Qdrant 集合，不存在则创建"""
        if self._client.collection_exists(self.collection_name):
            print(f"✓ 集合 '{self.collection_name}' 已存在")
            return

        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.embedding_engine.embedding_dimension,
                distance=models.Distance.COSINE,
            ),
        )
        print(f"✓ 集合 '{self.collection_name}' 已创建 "
              f"(维度={self.embedding_engine.embedding_dimension}, 距离=余弦相似度)")

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: List[ParagraphChunk]) -> None:
        """
        将已嵌入的 ParagraphChunk 列表写入 Qdrant。
        调用前需确保已通过 EmbeddingEngine.embed_chunks() 生成向量。

        Args:
            chunks: 已嵌入的 ParagraphChunk 列表
        """
        if not chunks:
            raise ValueError("chunks 列表为空")

        # 确保集合存在
        if not self._client.collection_exists(self.collection_name):
            self.setup()

        points = []
        for i, chunk in enumerate(chunks):
            if chunk.embedding_vector is None:
                raise ValueError(
                    f"ParagraphChunk (text='{chunk.text[:50]}...') 尚未嵌入，"
                    f"请先调用 EmbeddingEngine.embed_chunks()"
                )

            # 以 chunk.id 作为 point id，若未分配则使用枚举序号
            point_id = chunk.id if chunk.id else i

            points.append(models.PointStruct(
                id=point_id,
                vector=chunk.embedding_vector,
                payload={
                    "text": chunk.text,
                    "type": chunk.type,
                    "section_id": chunk.section_id,
                    "ai_desc_text": chunk.ai_desc_text,
                },
            ))

        self._client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        print(f"✓ 已写入 {len(points)} 个 ParagraphChunk 到集合 '{self.collection_name}'")

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------

    def search(
        self,
        query_text: str,
        top_k: int = 5,
        similarity_threshold: float | None = None,
    ) -> List[Dict[str, Any]]:
        """
        检索与查询文本最相似的 ParagraphChunk

        Args:
            query_text: 查询文本
            top_k: 返回最相似的前 K 个结果
            similarity_threshold: 相似度阈值（0~1），低于此值的结果会被过滤

        Returns:
            结果列表，每项包含: text, type, score, section_id, ai_desc_text
        """
        query_vector = self.embedding_engine.embed_query(query_text)

        score_threshold = similarity_threshold if similarity_threshold is not None else 0.0

        results = self._client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
        )

        return [
            {
                "text": hit.payload.get("text", ""),
                "type": hit.payload.get("type", "text"),
                "score": hit.score,
                "section_id": hit.payload.get("section_id", 0),
                "ai_desc_text": hit.payload.get("ai_desc_text", ""),
            }
            for hit in results
        ]

    def search_by_vector(
        self,
        query_vector: List[float],
        top_k: int = 5,
        similarity_threshold: float | None = None,
    ) -> List[Dict[str, Any]]:
        """
        使用预计算的向量进行检索

        Args:
            query_vector: 查询向量
            top_k: 返回最相似的前 K 个结果
            similarity_threshold: 相似度阈值

        Returns:
            结果列表
        """
        results = self._client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=similarity_threshold or 0.0,
        )

        return [
            {
                "text": hit.payload.get("text", ""),
                "type": hit.payload.get("type", "text"),
                "score": hit.score,
                "section_id": hit.payload.get("section_id", 0),
                "ai_desc_text": hit.payload.get("ai_desc_text", ""),
            }
            for hit in results
        ]

    # ------------------------------------------------------------------
    # 管理
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """获取向量存储的统计信息"""
        if not self._client.collection_exists(self.collection_name):
            return {
                "collection_name": self.collection_name,
                "document_count": 0,
                "embedding_dimension": self.embedding_engine.embedding_dimension,
                "status": "集合不存在",
            }

        count_result = self._client.count(self.collection_name)
        return {
            "collection_name": self.collection_name,
            "document_count": count_result.count,
            "embedding_dimension": self.embedding_engine.embedding_dimension,
        }

    def clear(self, confirm: bool = False) -> None:
        """
        清空集合数据

        Args:
            confirm: 必须显式设为 True 才会执行清空操作
        """
        if not confirm:
            print("警告: 请设置 confirm=True 来确认清空操作")
            return

        if self._client.collection_exists(self.collection_name):
            self._client.delete_collection(self.collection_name)
            print(f"✓ 集合 '{self.collection_name}' 已删除")
        else:
            print(f"集合 '{self.collection_name}' 不存在，无需清空")
