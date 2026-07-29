"""
RAG 知识库 - Embedding 引擎模块
基于 BGE-M3 模型对 ParagraphChunk 进行向量化
"""

from typing import List

from config import config
from .chunker import ParagraphChunk


class EmbeddingEngine:
    """Embedding 引擎，基于 BGE-M3 模型"""

    def __init__(self):
        from FlagEmbedding import BGEM3FlagModel

        print("初始化 BGE-M3 Embedding 模型...")
        self._model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, devices=["cpu"])

    # ------------------------------------------------------------------
    # chunk embedding
    # ------------------------------------------------------------------

    def embed_chunks(self, chunks: List[ParagraphChunk]) -> None:
        """
        对 ParagraphChunk 列表进行 embedding。

        - text 类型: 对 text 字段进行 embedding
        - table / image 类型: 对 ai_desc_text 字段进行 embedding
        - 生成的向量赋值到 embedding_vector 字段

        Args:
            chunks: ParagraphChunk 列表
        """
        texts = [
            chunk.ai_desc_text
            if chunk.type in ("table", "image") and chunk.ai_desc_text
            else chunk.text
            for chunk in chunks
        ]

        embeddings = self._model.encode(
            texts,
            batch_size=12,
            max_length=8192,
        )["dense_vecs"]

        for chunk, vector in zip(chunks, embeddings):
            chunk.embedding_vector = vector.tolist()

    # ------------------------------------------------------------------
    # query embedding
    # ------------------------------------------------------------------

    def embed_query(self, query: str) -> List[float]:
        """将查询文本转换为向量"""
        result = self._model.encode([query], max_length=8192)
        return result["dense_vecs"][0].tolist()

    # ------------------------------------------------------------------
    # compat
    # ------------------------------------------------------------------

    def get_model(self):
        """获取底层 BGE-M3 模型实例"""
        return self._model

    @property
    def embedding_dimension(self) -> int:
        """BGE-M3 向量维度"""
        return 1024
