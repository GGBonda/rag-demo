"""
RAG 知识库 - Embedding 引擎模块
对接多种 embedding 大模型后端，生成文本向量

注意: 各后端采用延迟导入 (lazy import)，仅在需要时才加载对应库，
      避免未使用的后端因依赖兼容性问题导致整个模块加载失败。
"""

from typing import List

from llama_index.core.base.embeddings.base import BaseEmbedding

from config import config


class EmbeddingEngine:
    """Embedding 引擎，统一不同后端模型的调用接口"""

    BACKENDS = ["openai", "huggingface", "ollama"]

    def __init__(self, backend: str | None = None):
        """
        初始化 Embedding 引擎

        Args:
            backend: 模型后端 (openai / huggingface / ollama)
                     不传则使用 config 中的配置
        """
        self.backend = backend or config.embedding.backend

        if self.backend not in self.BACKENDS:
            raise ValueError(
                f"不支持的 embedding 后端: {self.backend}，可选: {self.BACKENDS}"
            )

        self._model = self._create_model()

    def _create_model(self) -> BaseEmbedding:
        """根据配置创建对应的 embedding 模型实例（延迟导入）"""
        emb_cfg = config.embedding

        if self.backend == "openai":
            from llama_index.embeddings.openai import OpenAIEmbedding

            print(f"初始化 OpenAI Embedding: 模型={emb_cfg.openai_model}")
            return OpenAIEmbedding(
                model=emb_cfg.openai_model,
                api_key=emb_cfg.openai_api_key,
                api_base=emb_cfg.openai_base_url,
            )

        elif self.backend == "huggingface":
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding

            print(f"初始化 HuggingFace Embedding: 模型={emb_cfg.hf_model}")
            return HuggingFaceEmbedding(
                model_name=emb_cfg.hf_model,
            )

        elif self.backend == "ollama":
            from llama_index.embeddings.ollama import OllamaEmbedding

            print(f"初始化 Ollama Embedding: 模型={emb_cfg.ollama_model}, "
                  f"地址={emb_cfg.ollama_base_url}")
            return OllamaEmbedding(
                model_name=emb_cfg.ollama_model,
                base_url=emb_cfg.ollama_base_url,
            )

    def get_model(self) -> BaseEmbedding:
        """获取底层 embedding 模型实例"""
        return self._model

    def embed_text(self, text: str) -> List[float]:
        """
        将单段文本转换为向量

        Args:
            text: 输入文本

        Returns:
            向量 (浮点数列表)
        """
        return self._model.get_text_embedding(text)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        批量将文本转换为向量

        Args:
            texts: 文本列表

        Returns:
            向量列表
        """
        return self._model.get_text_embedding_batch(texts)

    def embed_query(self, query: str) -> List[float]:
        """
        将查询文本转换为向量（某些模型对查询有特殊处理）

        Args:
            query: 查询文本

        Returns:
            向量
        """
        return self._model.get_query_embedding(query)

    @property
    def embedding_dimension(self) -> int:
        """获取当前模型的向量维度"""
        emb_cfg = config.embedding
        model_name = getattr(self._model, "model_name", "")

        dims = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
            "BAAI/bge-small-zh-v1.5": 512,
            "BAAI/bge-base-zh-v1.5": 768,
            "BAAI/bge-large-zh-v1.5": 1024,
            "sentence-transformers/all-MiniLM-L6-v2": 384,
            "sentence-transformers/all-mpnet-base-v2": 768,
            "nomic-embed-text": 768,
        }
        return dims.get(model_name, 1536)
