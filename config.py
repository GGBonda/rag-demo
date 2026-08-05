"""
RAG 知识库 - 全局配置模块
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class EmbeddingConfig:
    """OpenAI 兼容的云平台 Embedding API 配置"""
    openai_api_key: str = os.getenv("OPENAI_EMBEDDING_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "")
    openai_base_url: str = os.getenv("OPENAI_EMBEDDING_BASE_URL", "")
    openai_dimension: int = int(os.getenv("OPENAI_EMBEDDING_DIMENSION", "1024"))


@dataclass
class LLMConfig:
    """大语言模型配置"""
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "")
    openai_flash_model: str = os.getenv("OPENAI_FLASH_MODEL", "")
    openai_pro_model: str = os.getenv("OPENAI_PRO_MODEL", "")


@dataclass
class VisionLLMConfig:
    """识图大模型配置"""
    openai_api_key: str = os.getenv("OPENAI_VISUAL_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_VISUAL_BASE_URL", "")
    openai_model: str = os.getenv("OPENAI_VISUAL_MODEL", "")


@dataclass
class MinerUConfig:
    """MinerU 云 API 配置"""
    api_token: str = os.getenv("MINERU_API_TOKEN", "")


@dataclass
class QdrantConfig:
    """Qdrant 向量数据库配置"""
    host: str = os.getenv("QDRANT_HOST", "localhost")
    port: int = int(os.getenv("QDRANT_PORT", "6333"))


@dataclass
class ChunkConfig:
    """文档分片配置"""
    retrieve_chunk_min_size: int = int(os.getenv("RETRIEVE_CHUNK_MIN_SIZE", "800"))
    retrieve_chunk_max_size: int = int(os.getenv("RETRIEVE_CHUNK_MAX_SIZE", "1500"))
    index_chunk_min_size: int = int(os.getenv("INDEX_CHUNK_MIN_SIZE", "200"))
    index_chunk_max_size: int = int(os.getenv("INDEX_CHUNK_MAX_SIZE", "500"))


@dataclass
class Config:
    """全局配置"""
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    vision_llm: VisionLLMConfig = field(default_factory=VisionLLMConfig)
    mineru: MinerUConfig = field(default_factory=MinerUConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)


# 全局单例
config = Config()
