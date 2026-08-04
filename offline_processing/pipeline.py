"""
离线处理模块 - 入库流水线
编排完整的离线处理流程: 加载 → 按章节分片 → 按段落分片 → 向量化 → 存储
"""

from .document_loader_mineru import MinerUDocumentLoader
from .chunker import Chunker
from .embedding_engine import EmbeddingEngine


class OfflinePipeline:
    """离线入库流水线，负责将文档解析、按章分片、向量化后写入 Qdrant 向量数据库"""

    def __init__(
        self,
        file_path: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        **loader_kwargs,
    ):
        from store import (
            PARAGRAPH_CHUNK_COLLECTION,
            MarkdownDocumentStore,
            ParagraphChunkStore,
            QdrantStore,
            RetrieveChunkStore,
        )

        self.file_path = file_path

        self.loader = MinerUDocumentLoader(
            file_path=file_path,
            **loader_kwargs,
        )
        self.chunker = Chunker()
        self.embedding_engine = EmbeddingEngine()
        self.qdrant_store = QdrantStore()
        self.markdown_document_store = MarkdownDocumentStore(self.qdrant_store)
        self.retrieve_chunk_store = RetrieveChunkStore(self.qdrant_store)
        self.paragraph_chunk_store = ParagraphChunkStore(self.qdrant_store)
        self.paragraph_collection_name = PARAGRAPH_CHUNK_COLLECTION

    def ingest(self, rebuild: bool = False) -> None:
        """
        执行文档入库：加载 → 章节分片 → 段落分片 → 向量化 → 存入 Qdrant

        Args:
            rebuild: 是否重建索引（清空旧数据）
        """
        print("=" * 60)
        print("RAG 知识库 - 离线处理: 文档入库流程")
        print("=" * 60)

        # Step 1: QdrantStore 单例构造时已初始化集合
        print("\n[1/5] Qdrant 集合已就绪...")

        if rebuild:
            self.markdown_document_store.clear()
            self.retrieve_chunk_store.clear()
            self.paragraph_chunk_store.clear()

        # Step 2: 加载文档
        print("\n[2/5] 加载文档...")
        doc = self.loader.load()

        if doc is None:
            print("文档加载失败，入库流程终止")
            return

        # Step 3: 按章节分片
        print("\n[3/5] 按章节分片...")
        section_chunks = self.chunker.chunk_markdown(doc)

        if not section_chunks:
            print("分片后没有有效内容，入库流程终止")
            return

        print(f"  生成 {len(section_chunks)} 个 RetrieveChunk")

        # Step 4: 按段落分片
        print("\n[4/5] 按段落分片 + 向量化...")
        paragraph_chunks = self.chunker.chunk_section_chunks(section_chunks)

        if not paragraph_chunks:
            print("段落分片后没有有效内容，入库流程终止")
            return

        print(f"  生成 {len(paragraph_chunks)} 个 ParagraphChunk")

        # 向量化
        self.embedding_engine.embed_chunks(paragraph_chunks)

        # Step 5: 写入 Qdrant
        print(f"\n[5/5] 写入 Qdrant...")
        self.markdown_document_store.batch_insert([doc])
        self.retrieve_chunk_store.batch_insert(section_chunks)
        self.paragraph_chunk_store.batch_insert(paragraph_chunks)

        # 输出统计
        paragraph_count = self.qdrant_store.client.count(
            self.paragraph_collection_name
        ).count
        print("\n" + "=" * 60)
        print("入库完成! 统计信息:")
        print(f"  - 集合: {self.paragraph_collection_name}")
        print(f"  - 文档 chunk 总数: {paragraph_count}")
        print(f"  - 向量维度: {self.qdrant_store.vector_size}")
        print("=" * 60)
