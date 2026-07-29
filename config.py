"""
RAG 知识库 - 全局配置模块
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class EmbeddingConfig:
    """Embedding 模型配置"""
    backend: str = os.getenv("EMBEDDING_BACKEND", "openai")
    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    # HuggingFace
    hf_model: str = os.getenv("HF_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    # Ollama
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")


@dataclass
class LLMConfig:
    """大语言模型配置"""
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "")
    openai_flash_model: str = os.getenv("OPENAI_FLASH_MODEL", "")
    openai_pro_model: str = os.getenv("OPENAI_PRO_MODEL", "")


@dataclass
class MinerUConfig:
    """MinerU 云 API 配置"""
    api_token: str = os.getenv("MINERU_API_TOKEN", "")


@dataclass
class QdrantConfig:
    """Qdrant 向量数据库配置"""
    host: str = os.getenv("QDRANT_HOST", "localhost")
    port: int = int(os.getenv("QDRANT_PORT", "6333"))
    collection_name: str = os.getenv("QDRANT_COLLECTION_NAME", "paragraph_chunks")


@dataclass
class ChunkConfig:
    """文档分片配置"""
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "512"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))


@dataclass
class Config:
    """全局配置"""
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    mineru: MinerUConfig = field(default_factory=MinerUConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)


# 全局单例
config = Config()
