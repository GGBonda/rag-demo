"""
实时响应模块 - 负责处理用户提问、检索相关文档、返回回答
"""

from .retriever import Retriever
from .responder import Responder

__all__ = ["Retriever", "Responder"]
