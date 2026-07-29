from pathlib import Path

from offline_processing.document_loader_mineru import MarkdownDocument
from offline_processing.chunker import Chunker
from offline_processing.embedding_engine import EmbeddingEngine
from offline_processing.vector_store import VectorStoreManager

if __name__ == '__main__':

    """
    loader = MinerUDocumentLoader(file_path="/Users/shengjunhui/Downloads/rag_test_pdf/阿里开发规范.pdf", start_page=3)

    doc = loader.load() 
    """

    markdown_path = (
        Path(__file__).resolve().parent.parent / "documents"/ "pdf_2_markdown.md"
    )
    doc = MarkdownDocument(
        markdown_text=markdown_path.read_text(encoding="utf-8"),
        file_name=markdown_path.name
    )

    chunker = Chunker()

    sectionChunks = chunker.chunk_markdown(doc)

    paragraphChunks = chunker.chunk_section_chunks(sectionChunks)

    for i, chunk in enumerate(paragraphChunks):
        print(f"=========================================================={i}, type：{chunk.type}, node length: {len(chunk.text)}")
        print(chunk.text)

"""
    embeddingEngine = EmbeddingEngine()

    embeddingEngine.embed_chunks(paragraphChunks)

    vectorStoreManager = VectorStoreManager(embeddingEngine, "paragraph_chunk")

    vectorStoreManager.add_chunks(paragraphChunks)


"""
