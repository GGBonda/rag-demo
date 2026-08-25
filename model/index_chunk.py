"""索引分片数据模型。"""

from dataclasses import dataclass


@dataclass
class IndexChunk:
    """索引分片数据类"""

    """主键id"""
    id: int = 0
    """所属章节id"""
    retrieve_id: int = 0
    """索引文本"""
    text: str = ""
    """向量"""
    embedding_vector: list[float] | None = None
