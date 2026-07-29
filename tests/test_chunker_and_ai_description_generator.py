import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from offline_processing.ai_description_generator import (
    ParagraphChunkDescriptionGenerator,
)
from offline_processing.chunker import Chunker
from offline_processing.document_loader_mineru import MarkdownDocument


AI_DESCRIPTION_TYPES = {"table", "code", "image"}


def process_markdown_document(markdown_file: str | Path) -> None:
    """二阶段分片指定的 Markdown 文档，并为非纯文本分片生成 AI 描述。"""
    markdown_path = Path(markdown_file).expanduser().resolve()
    if not markdown_path.is_file():
        raise FileNotFoundError(f"Markdown 文件不存在: {markdown_path}")
    if markdown_path.suffix.lower() != ".md":
        raise ValueError(f"文件不是 Markdown 格式: {markdown_path}")

    chunker = Chunker()
    description_generator = ParagraphChunkDescriptionGenerator()

    print(f"正在处理: {markdown_path.name}")
    document = MarkdownDocument(
        id=1,
        file_name=markdown_path.name,
        markdown_text=markdown_path.read_text(encoding="utf-8"),
    )

    section_chunks = chunker.chunk_markdown(document)
    for section_id, section_chunk in enumerate(section_chunks, start=1):
        section_chunk.id = section_id

    paragraph_chunks = chunker.chunk_section_chunks(section_chunks)
    for paragraph_id, paragraph_chunk in enumerate(paragraph_chunks, start=1):
        paragraph_chunk.id = paragraph_id

    ai_chunks = [
        chunk
        for chunk in paragraph_chunks
        if chunk.type in AI_DESCRIPTION_TYPES
    ]
    description_generator.generate_descriptions(ai_chunks)

    missing_descriptions = [
        chunk.id for chunk in ai_chunks if not chunk.ai_desc_text
    ]
    if missing_descriptions:
        raise AssertionError(
            f"以下分片未生成 AI 描述: {missing_descriptions}"
        )

    type_counts = Counter(chunk.type for chunk in paragraph_chunks)
    print(
        f"分片完成: 一级章节 {len(section_chunks)} 个，"
        f"二级段落 {len(paragraph_chunks)} 个，类型统计 {dict(type_counts)}"
    )
    for chunk in ai_chunks:
        print(
            f"  ParagraphChunk(id={chunk.id}, type={chunk.type}, "
            f"section_id={chunk.section_id})"
        )
        print(f"  AI 描述: {chunk.ai_desc_text}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="测试指定 Markdown 文档的二阶段分片与 AI 描述生成"
    )
    parser.add_argument(
        "--file",
        required=True,
        help="需要测试的 Markdown 文件路径",
    )
    args = parser.parse_args()
    process_markdown_document(args.file)


if __name__ == "__main__":
    main()
