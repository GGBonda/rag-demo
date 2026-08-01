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
MARKDOWN_DIR = PROJECT_ROOT / "doc_markdown"


def process_markdown_document(file_name: str) -> None:
    """二阶段分片指定的 Markdown 文档，并为非纯文本分片生成 AI 描述。"""
    markdown_path = MARKDOWN_DIR / file_name

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

    description_generator.generate_descriptions(paragraph_chunks)

    for chunk in paragraph_chunks:
        if chunk.type in AI_DESCRIPTION_TYPES:
            print(f"========================================AI 生成描述{chunk.type}=================================================")
            print(chunk.ai_desc_text)
        else:
            print("================================================================================================================")
            print(chunk.text)


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
