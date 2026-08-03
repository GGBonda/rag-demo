"""
离线处理模块 - 入库流水线
编排完整的离线处理流程: 加载 → 按章节分片 → 按段落分片 → 向量化 → 存储
"""

from .document_loader_mineru import MinerUDocumentLoader
from .chunker import Chunker
from .embedding_engine import EmbeddingEngine
from .vector_store import VectorStoreManager


class OfflinePipeline:
    """离线入库流水线，负责将文档解析、按章分片、向量化后写入 Qdrant 向量数据库"""

    def __init__(
        self,
        file_path: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        collection_name: str | None = None,
        **loader_kwargs,
    ):
        self.file_path = file_path

        self.loader = MinerUDocumentLoader(
            file_path=file_path,
            **loader_kwargs,
        )
        self.chunker = Chunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.embedding_engine = EmbeddingEngine()
        self.vector_store = VectorStoreManager(
            embedding_engine=self.embedding_engine,
            collection_name=collection_name,
        )

    def ingest(self, rebuild: bool = False) -> None:
        """
        执行文档入库：加载 → 章节分片 → 段落分片 → 向量化 → 存入 Qdrant

        Args:
            rebuild: 是否重建索引（清空旧数据）
        """
        print("=" * 60)
        print("RAG 知识库 - 离线处理: 文档入库流程")
        print("=" * 60)

        if rebuild:
            self.vector_store.clear(confirm=True)

        # Step 1: 初始化 Qdrant 集合
        print("\n[1/5] 初始化 Qdrant 集合...")
        self.vector_store.setup()

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
        self.vector_store.add_chunks(paragraph_chunks)

        # 输出统计
        stats = self.vector_store.get_stats()
        print("\n" + "=" * 60)
        print("入库完成! 统计信息:")
        print(f"  - 集合: {stats['collection_name']}")
        print(f"  - 文档 chunk 总数: {stats['document_count']}")
        print(f"  - 向量维度: {stats['embedding_dimension']}")
        print("=" * 60)
