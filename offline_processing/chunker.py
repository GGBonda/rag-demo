"""
RAG 知识库 - 文档分片模块
对 Markdown 文本按最小章节（任意级别标题）分片
"""

import re
from typing import List

from langchain_text_splitters import MarkdownHeaderTextSplitter

from config import config
from data_class import IndexChunk, MarkdownDocument
from data_class.retrieve_chunk import DATA_IMAGE_RE, RetrieveChunk

_TABLE_SEPARATOR_RE = re.compile(r"^\|?[\s:]*-{2,}[\s:]*(?:\|[\s:]*-{2,}[\s:]*)*\|?$")
_HTML_TABLE_RE = re.compile(r"<table\b[^>]*>.*?</table\s*>", re.IGNORECASE | re.DOTALL)

"""判断文本是否为 Markdown 或 HTML 表格"""
def _is_markdown_table(text: str) -> bool:
    stripped_text = text.strip()
    if _HTML_TABLE_RE.search(stripped_text):
        return True

    lines = stripped_text.split("\n")
    if len(lines) < 2:
        return False
    # 表格至少有一行包含分隔行（如 |---|---|）
    for line in lines:
        if _TABLE_SEPARATOR_RE.match(line.strip()):
            return True
    return False

# 用于识别三个或更多反引号组成的代码块边界
CODE_FENCE_RE = re.compile(r"^\s*(`{3,})(.*)$")

def _normal_text_length(text: str) -> int:
    """计算普通文本长度，忽略图片、表格和 fenced code block。"""
    lines_without_code: List[str] = []
    code_fence_length = 0

    for line in text.splitlines(keepends=True):
        fence_match = CODE_FENCE_RE.match(line)
        if code_fence_length:
            if (
                fence_match
                and len(fence_match.group(1)) >= code_fence_length
                and not fence_match.group(2).strip()
            ):
                code_fence_length = 0
            continue

        if fence_match:
            code_fence_length = len(fence_match.group(1))
            continue

        lines_without_code.append(line)

    normal_text = "".join(lines_without_code)
    normal_text = _HTML_TABLE_RE.sub("", normal_text)
    normal_text = DATA_IMAGE_RE.sub("", normal_text)

    lines = normal_text.splitlines(keepends=True)
    table_line_indexes: set[int] = set()
    for index, line in enumerate(lines):
        if not _TABLE_SEPARATOR_RE.match(line.strip()):
            continue
        if index == 0 or "|" not in lines[index - 1]:
            continue

        table_line_indexes.update((index - 1, index))
        next_index = index + 1
        while next_index < len(lines) and "|" in lines[next_index]:
            table_line_indexes.add(next_index)
            next_index += 1

    return sum(
        len(line)
        for index, line in enumerate(lines)
        if index not in table_line_indexes
    )

"""检测文本类型：code / image / table / text。"""
def detect_text_type(text: str) -> str:
    first_line = text.lstrip().splitlines()[0] if text.strip() else ""
    if CODE_FENCE_RE.match(first_line):
        return "code"
    if DATA_IMAGE_RE.search(text):
        return "image"
    if _is_markdown_table(text):
        return "table"
    return "text"


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
            if not doc.page_content.strip():
                continue

            # 从 metadata 拼接标题路径
            title_path = self._build_title_path(doc.metadata)

            chunk = RetrieveChunk(
                doc_id=document.id or 0,
                title_path=title_path,
                text=doc.page_content,
            )
            result.append(chunk)

        self._merge_small_retrieve_chunks(result)

        for retrieve_index, chunk in enumerate(result, start=1):
            chunk.id = int(f"{document.id}{retrieve_index:0{3}d}")

        print(f"  [Markdown 分片] {document.file_name}: {len(result)} 个 chunk")
        return result

    @staticmethod
    def _merge_small_retrieve_chunks(chunks: List[RetrieveChunk]) -> None:
        """原地将普通文本部分过小的 RetrieveChunk 向后合并。"""
        min_size = config.chunk.retrieve_chunk_min_size
        max_size = config.chunk.retrieve_chunk_max_size
        if min_size < 0 or max_size < min_size:
            raise ValueError("需要满足 0 <= min_size <= max_size")

        index = 0

        while index < len(chunks):
            chunk = chunks[index]

            while (
                _normal_text_length(chunk.text) < min_size
                and index + 1 < len(chunks)
            ):
                next_chunk = chunks[index + 1]
                merged_text = f"{chunk.text}\n{next_chunk.text}"
                if _normal_text_length(merged_text) > max_size:
                    break

                next_chunk.text = merged_text
                del chunks[index]
                chunk = next_chunk

            index += 1

    @staticmethod
    def _merge_small_index_chunks(chunks: List[IndexChunk]) -> None:
        """原地合并过小的纯文本 IndexChunk。"""
        min_size = config.chunk.index_chunk_min_size
        max_size = config.chunk.index_chunk_max_size

        if min_size < 0 or max_size < min_size:
            raise ValueError("需要满足 0 <= min_size <= max_size")

        index = 0

        while index < len(chunks):
            chunk = chunks[index]

            while (
                detect_text_type(chunk.text) == "text"
                and len(chunk.text) < min_size
                and index + 1 < len(chunks)
            ):
                next_chunk = chunks[index + 1]
                if detect_text_type(next_chunk.text) != "text":
                    break

                merged_text = f"{chunk.text}\n{next_chunk.text}"
                if len(merged_text) > max_size:
                    break

                next_chunk.text = merged_text
                del chunks[index]
                chunk = next_chunk

            index += 1

    def chunk_section_chunks(
        self,
        section_chunks: List[RetrieveChunk],
    ) -> List[IndexChunk]:
        """将 RetrieveChunk 列表按段落拆分为 IndexChunk 列表"""
        result: List[IndexChunk] = []

        for section in section_chunks:
            section_result: List[IndexChunk] = []

            index_texts: List[str] = []
            code_lines: List[str] = []
            code_fence_length = 0

            for line in section.text.splitlines():
                fence_match = CODE_FENCE_RE.match(line)

                if code_lines:
                    code_lines.append(line)
                    if (
                        fence_match
                        and len(fence_match.group(1)) >= code_fence_length
                        and not fence_match.group(2).strip()
                    ):
                        index_texts.append("\n".join(code_lines))
                        code_lines = []
                        code_fence_length = 0
                    continue

                if fence_match:
                    code_lines = [line]
                    code_fence_length = len(fence_match.group(1))
                elif line.strip():
                    index_texts.append(line)

            # 未闭合的代码块保留至章节末尾，避免丢失内容
            if code_lines:
                index_texts.append("\n".join(code_lines))

            for index_text in index_texts:
                index_text = index_text.strip()
                if not index_text:
                    continue

                chunk = IndexChunk(
                    retrieve_id=section.id,
                    text=index_text,
                )
                section_result.append(chunk)

            self._merge_small_index_chunks(section_result)

            for index_chunk_index, chunk in enumerate(section_result, start=1):
                chunk.id = int(f"{section.id}{index_chunk_index:0{3}d}")

            result.extend(section_result)

        return result

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    @staticmethod
    def _build_title_path(metadata: dict) -> str:
        """从 header metadata 构建标题路径，如 'Title > Subsection'"""
        parts = [v for v in metadata.values() if v]
        return " > ".join(parts) if parts else ""
