"""Markdown 文档数据模型。"""

import time
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class MarkdownDocument:
    """PDF 解析后的 Markdown 文档"""
    """主键 ID 直接用时间戳作为id，demo项目，不考虑太复杂的唯一性标识方案"""
    id: int = field(default_factory=lambda: int(time.time()))
    """文件名"""
    file_name: str = ""
    """作者"""
    author: str = ""
    """原文件 URL"""
    original_file_url: str = ""
    """文档创建时间"""
    created_at: datetime | None = None
    """Markdown 文本"""
    markdown_text: str = ""
    """原文件内容的 MD5 摘要"""
    md5: str = ""
