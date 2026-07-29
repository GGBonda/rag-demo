"""
RAG 知识库 - 文档分片模块
对 Markdown 文本按最小章节（任意级别标题）分片
"""

import re
from dataclasses import dataclass
from typing import List, Optional

from langchain_text_splitters import MarkdownHeaderTextSplitter

from config import config
from .document_loader_mineru import MarkdownDocument

"""章节分片数据类"""
@dataclass
class RetrieveChunk:
    """主键id"""
    id: int = 0
    """所属文档id"""
    doc_id: int = 0
    """章节标题路径"""
    title_path: str = ""
    """章节index"""
    section_index: int = 0
    """章节文本"""
    text: str = ""

"""段落分片数据类"""
@dataclass
class ParagraphChunk:
    """主键id"""
    id: int = 0
    """所属章节id"""
    section_id: int = 0
    """段落文本"""
    text: str = ""
    """文本类型，可选枚举text、table、image、code"""
    type: str = "text"
    """AI 描述文本，table、image、code 类型分片需要由 AI 总结后生成相应描述"""
    ai_desc_text: str = ""
    """向量"""
    embedding_vector: List[float] = None


def _merge_small_chunks(
    chunks: List[ParagraphChunk] | List[RetrieveChunk],
    min_size: int,
    max_size: int
) -> None:
    """原地将过小分块向后合并，且不拆分已经超过 max_size 的分块。"""
    if min_size < 0 or max_size < min_size:
        raise ValueError("需要满足 0 <= min_size <= max_size")

    index = 0

    while index < len(chunks):
        chunk = chunks[index]

        while (
            (not hasattr(chunk, 'type') or chunk.type == "text")
            and len(chunk.text) < min_size
            and index + 1 < len(chunks)
        ):
            next_chunk = chunks[index + 1]
            if hasattr(next_chunk, 'type') and next_chunk.type != "text":
                break

            merged_text = f"{chunk.text}\n{next_chunk.text}"
            if len(merged_text) > max_size:
                break

            next_chunk.text = merged_text
            del chunks[index]
            chunk = next_chunk

        index += 1


class Chunker:
    """文档分片器，基于 langchain-text-splitters 按最小章节拆分"""

    def __init__(
        self
    ):
        # 按标题层级切分 Markdown
        self._heading_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
                ("####", "h4"),
                ("#####", "h5"),
                ("######", "h6"),
            ],
            strip_headers=False,
        )

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def chunk_markdown(
        self,
        document: MarkdownDocument,
    ) -> List[RetrieveChunk]:
        """将 Markdown 文档按最小章节拆分为 RetrieveChunk 列表"""
        if not document.markdown_text.strip():
            return []

        # 第一步：按标题层级切分
        docs = self._heading_splitter.split_text(document.markdown_text)

        result: List[RetrieveChunk] = []
        for doc in docs:
            sub_texts = [doc.page_content]

            for sub_text in sub_texts:
                if not sub_text.strip():
                    continue

                # 从 metadata 拼接标题路径
                title_path = self._build_title_path(doc.metadata)

                chunk = RetrieveChunk(
                    doc_id=document.id or 0,
                    section_index=len(result),
                    title_path=title_path,
                    text=sub_text,
                )
                result.append(chunk)

        _merge_small_chunks(result, 800, 1500)
        print(f"  [Markdown 分片] {document.file_name}: {len(result)} 个 chunk")
        return result

    # 用于识别 base64 图片的正则
    _BASE64_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(data:image/[^;]+;base64,")

    # 用于识别 Markdown 表格分隔行的正则（如 |---|:---:|---|）
    _TABLE_SEPARATOR_RE = re.compile(r"^\|?[\s:]*-{2,}[\s:]*(?:\|[\s:]*-{2,}[\s:]*)*\|?$")

    # 用于识别完整的 HTML table 标签，兼容属性、大小写和多行内容
    _HTML_TABLE_RE = re.compile(r"<table\b[^>]*>.*?</table\s*>", re.IGNORECASE | re.DOTALL)

    # 用于剔除文本开头的 Markdown 标题行
    _HEADING_LINE_RE = re.compile(r"^#{1,6}\s+[^\n]*\n?")

    # 用于识别三个或更多反引号组成的代码块边界
    _CODE_FENCE_RE = re.compile(r"^\s*(`{3,})(.*)$")

    def chunk_section_chunks(
        self,
        section_chunks: List[RetrieveChunk],
    ) -> List[ParagraphChunk]:
        """将 RetrieveChunk 列表按段落拆分为 ParagraphChunk 列表"""
        result: List[ParagraphChunk] = []

        for section in section_chunks:
            # 剔除开头的标题行；普通文本按换行分片，代码块整体保留
            #text_without_heading = self._HEADING_LINE_RE.sub("", section.text).strip()
            paragraphs: List[str] = []
            code_lines: List[str] = []
            code_fence_length = 0

            for line in section.text.splitlines():
                fence_match = self._CODE_FENCE_RE.match(line)

                if code_lines:
                    code_lines.append(line)
                    if (
                        fence_match
                        and len(fence_match.group(1)) >= code_fence_length
                        and not fence_match.group(2).strip()
                    ):
                        paragraphs.append("\n".join(code_lines))
                        code_lines = []
                        code_fence_length = 0
                    continue

                if fence_match:
                    code_lines = [line]
                    code_fence_length = len(fence_match.group(1))
                elif line.strip():
                    paragraphs.append(line)

            # 未闭合的代码块保留至章节末尾，避免丢失内容
            if code_lines:
                paragraphs.append("\n".join(code_lines))

            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue

                chunk_type = self._detect_paragraph_type(para)
                chunk = ParagraphChunk(
                    section_id=section.id,
                    text=para,
                    type=chunk_type,
                )
                result.append(chunk)

        _merge_small_chunks(result, 200, 500)
        return result

    def _detect_paragraph_type(self, text: str) -> str:
        """检测段落类型：code / image / table / text"""
        first_line = text.lstrip().splitlines()[0] if text.strip() else ""
        if self._CODE_FENCE_RE.match(first_line):
            return "code"
        if self._BASE64_IMAGE_RE.search(text):
            return "image"
        if self._is_markdown_table(text):
            return "table"
        return "text"

    @staticmethod
    def _is_markdown_table(text: str) -> bool:
        """判断文本是否为 Markdown 或 HTML 表格"""
        stripped_text = text.strip()
        if Chunker._HTML_TABLE_RE.search(stripped_text):
            return True

        lines = stripped_text.split("\n")
        if len(lines) < 2:
            return False
        # 表格至少有一行包含分隔行（如 |---|---|）
        for line in lines:
            if Chunker._TABLE_SEPARATOR_RE.match(line.strip()):
                return True
        return False

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    @staticmethod
    def _build_title_path(metadata: dict) -> str:
        """从 header metadata 构建标题路径，如 'Title > Subsection'"""
        parts = [v for v in metadata.values() if v]
        return " > ".join(parts) if parts else ""
