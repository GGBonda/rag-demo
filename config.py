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
class DatabaseConfig:
    """PGvector 数据库配置"""
    host: str = os.getenv("PG_HOST", "localhost")
    port: int = int(os.getenv("PG_PORT", "5432"))
    database: str = os.getenv("PG_DATABASE", "rag_knowledge_base")
    user: str = os.getenv("PG_USER", "postgres")
    password: str = os.getenv("PG_PASSWORD", "postgres")
    table_name: str = os.getenv("PG_TABLE_NAME", "documents")

    @property
    def connection_string(self) -> str:
        return (
            f"postgresql+psycopg2://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )

    @property
    def connection_params(self) -> dict:
        """返回用于 psycopg2 连接的参数字典"""
        return {
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "user": self.user,
            "password": self.password,
        }


@dataclass
class ChunkConfig:
    """文档分片配置"""
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "512"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))


@dataclass
class Config:
    """全局配置"""
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)


# 全局单例
config = Config()
