import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from offline_processing.ai_description_generator import (
    AIDescriptionGenerator,
)
from offline_processing.chunker import Chunker
from model import MarkdownDocument
from offline_processing.embedding_engine import EmbeddingEngine
from store import (
    MarkdownDocumentStore,
    IndexChunkStore,
    QdrantStore,
    RetrieveChunkStore,
)


MARKDOWN_DIR = PROJECT_ROOT / "doc_markdown"
MARKDOWN_EXTENSIONS = {".md", ".markdown"}


def ingest_all_markdown_documents() -> None:
    """解析、切分并描述 doc_markdown 下的文档，然后批量写入 Qdrant。"""
    markdown_paths = sorted(
        path
        for path in MARKDOWN_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in MARKDOWN_EXTENSIONS
    )
    if not markdown_paths:
        print(f"未在 {MARKDOWN_DIR} 中找到 Markdown 文档")
        return

    chunker = Chunker()
    description_generator = AIDescriptionGenerator()
    embedding_engine = EmbeddingEngine()
    qdrant_store = QdrantStore()
    markdown_document_store = MarkdownDocumentStore(qdrant_store)
    retrieve_chunk_store = RetrieveChunkStore(qdrant_store)
    index_chunk_store = IndexChunkStore(qdrant_store)

    markdown_document_store.clear()
    retrieve_chunk_store.clear()
    index_chunk_store.clear()

    for document_id, markdown_path in enumerate(markdown_paths, start=1):
        print(f"正在处理: {markdown_path.name}")
        document = MarkdownDocument(
            file_name=markdown_path.name,
            markdown_text=markdown_path.read_text(encoding="utf-8"),
        )
        retrieve_chunks = chunker.chunk_markdown(document)
        index_chunks = chunker.chunk_section_chunks(
            retrieve_chunks
        )

        description_generator.generate_image_code_table_desc(index_chunks)

        retrieve_id_index_chunk_count_dict = dict(Counter(chunk.retrieve_id for chunk in index_chunks))
        for chunk in retrieve_chunks:
            index_chunks.extend(description_generator.generate_retrieve_desc([chunk], retrieve_id_index_chunk_count_dict[chunk.id]))

        embedding_engine.embed_chunks(index_chunks)

        document_count = markdown_document_store.batch_insert([document])
        retrieve_count = retrieve_chunk_store.batch_insert(retrieve_chunks)
        index_count = index_chunk_store.batch_insert(index_chunks)

        print("入库完成:")
        print(f"  MarkdownDocument: {document_count}")
        print(f"  RetrieveChunk: {retrieve_count}")
        print(f"  IndexChunk: {index_count}")


if __name__ == "__main__":
    ingest_all_markdown_documents()
