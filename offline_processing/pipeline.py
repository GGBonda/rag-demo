"""
离线处理模块 - 入库流水线
编排完整的离线处理流程: 加载（含按章分片）→ 向量化 → 存储
"""

from .document_loader import DocumentLoader
from .embedding_engine import EmbeddingEngine
from .vector_store import VectorStoreManager


class OfflinePipeline:
    """离线入库流水线，负责将文档解析、按章分片、向量化后写入向量数据库"""

    def __init__(
        self,
        documents_dir: str = "./documents",
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        embedding_backend: str | None = None,
        table_name: str | None = None,
    ):
        self.documents_dir = documents_dir

        self.loader = DocumentLoader(
            input_dir=documents_dir,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.embedding_engine = EmbeddingEngine(backend=embedding_backend)
        self.vector_store = VectorStoreManager(
            embedding_engine=self.embedding_engine,
            table_name=table_name,
        )

    def ingest(self, rebuild: bool = False) -> None:
        """
        执行文档入库：加载（含按章分片）→ 向量化 → 存入 PGvector

        Args:
            rebuild: 是否重建索引（清空旧数据）
        """
        print("=" * 60)
        print("RAG 知识库 - 离线处理: 文档入库流程")
        print("=" * 60)

        if rebuild:
            self.vector_store.clear(confirm=True)

        # Step 1: 初始化数据库
        print("\n[1/3] 初始化数据库...")
        self.vector_store.setup()

        # Step 2: 加载文档并按章节分片
        print("\n[2/3] 加载文档并按章节分片...")
        nodes = self.loader.load_all()

        if not nodes:
            print("没有文档需要处理，入库流程终止")
            return

        # Step 3: 向量化 + 写入数据库
        print(f"\n[3/3] 向量化并写入数据库...")
        self.vector_store.build_index(nodes)

        # 输出统计
        stats = self.vector_store.get_stats()
        print("\n" + "=" * 60)
        print("入库完成! 统计信息:")
        print(f"  - 数据库: {stats['database']}")
        print(f"  - 表名: {stats['table_name']}")
        print(f"  - 文档 chunk 总数: {stats['document_count']}")
        print(f"  - 向量维度: {stats['embedding_dimension']}")
        print("=" * 60)
