"""
RAG 知识库 - Embedding 引擎模块
通过 OpenAI 兼容的云平台 API 对 IndexChunk 进行向量化
"""

from typing import List

from config import config
from data_class import IndexChunk


class EmbeddingEngine:
    """Embedding 引擎，使用 OpenAI 兼容的云平台向量 API。"""

    BATCH_SIZE = 12

    def __init__(self):
        embedding_config = config.embedding
        self._model_name = embedding_config.openai_model.strip()
        if not self._model_name:
            raise ValueError("未配置 OPENAI_EMBEDDING_MODEL 环境变量")

        api_key = embedding_config.openai_api_key.strip()
        if not api_key:
            raise ValueError("未配置 OPENAI_API_KEY 环境变量")

        self._embedding_dimension = embedding_config.openai_dimension
        if self._embedding_dimension <= 0:
            raise ValueError("OPENAI_EMBEDDING_DIMENSION 必须为正整数")

        from openai import OpenAI

        client_options = {"api_key": api_key}
        base_url = embedding_config.openai_base_url.strip()
        if base_url:
            client_options["base_url"] = base_url
        self._client = OpenAI(**client_options)

        print(f"初始化云平台 Embedding 客户端: {self._model_name}")

    # ------------------------------------------------------------------
    # chunk embedding
    # ------------------------------------------------------------------

    def embed_chunks(self, chunks: List[IndexChunk]) -> None:
        """
        对 IndexChunk 列表进行 embedding。

        - 对 text 字段进行 embedding
        - 生成的向量赋值到 embedding_vector 字段

        Args:
            chunks: IndexChunk 列表
        """
        texts = [chunk.text for chunk in chunks]

        embeddings = self._embed_texts(texts)

        for chunk, vector in zip(chunks, embeddings):
            chunk.embedding_vector = vector

    # ------------------------------------------------------------------
    # query embedding
    # ------------------------------------------------------------------

    def embed_query(self, query: str) -> list[float]:
        """将查询文本转换为向量"""
        return self._embed_texts([query])[0]

    def _embed_texts(self, texts: List[str]) -> list[list[float]]:
        """分批调用云平台 API，并按输入顺序返回向量。"""
        embeddings: list[list[float]] = []

        for start in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[start:start + self.BATCH_SIZE]
            response = self._client.embeddings.create(
                model=self._model_name,
                input=batch,
            )
            response_data = sorted(response.data, key=lambda item: item.index)
            if len(response_data) != len(batch):
                raise RuntimeError(
                    f"Embedding API 返回了 {len(response_data)} 个向量，"
                    f"但请求中包含 {len(batch)} 条文本"
                )

            for item in response_data:
                vector = list(item.embedding)
                if len(vector) != self._embedding_dimension:
                    raise RuntimeError(
                        f"Embedding API 返回的向量维度为 {len(vector)}，"
                        f"与 OPENAI_EMBEDDING_DIMENSION="
                        f"{self._embedding_dimension} 不一致"
                    )
                embeddings.append(vector)

        return embeddings
