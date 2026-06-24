#!/usr/bin/env python3
"""
RAG 知识库 - 离线处理 CLI 入口

模块划分:
  - offline_processing: 离线处理（文档解析、按章分片、向量化、入库）→ CLI
  - realtime_response:  实时响应（用户提问、检索、返回回答）→ HTTP API (api_server.py)
"""

import argparse
import sys

from offline_processing import OfflinePipeline
from config import config


def cmd_ingest(args):
    """子命令: 文档入库（离线处理）"""
    loader_kwargs = {}
    if hasattr(args, "mineru_method") and args.mineru_method:
        loader_kwargs["method"] = args.mineru_method

    pipeline = OfflinePipeline(
        documents_dir=args.documents_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_backend=args.embedding_backend,
        table_name=args.table_name,
        loader_backend=args.loader_backend,
        **loader_kwargs,
    )
    pipeline.ingest(rebuild=args.rebuild)


def main():
    parser = argparse.ArgumentParser(
        description="RAG 知识库 - 离线处理 (基于 LlamaIndex + PGvector)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 文档入库
  python main.py ingest --documents_dir ./my_docs

  # 启动实时响应 API 服务
  uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
  或
  python api_server.py
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # ---- ingest 子命令 ----
    ingest_parser = subparsers.add_parser("ingest", help="文档入库（离线处理）")
    ingest_parser.add_argument(
        "--documents_dir", default="./documents", help="文档目录"
    )
    ingest_parser.add_argument(
        "--chunk_size", type=int, default=None,
        help=f"分片大小 (默认: {config.chunk.chunk_size})"
    )
    ingest_parser.add_argument(
        "--chunk_overlap", type=int, default=None,
        help=f"分片重叠量 (默认: {config.chunk.chunk_overlap})"
    )
    ingest_parser.add_argument(
        "--embedding_backend", default=None,
        choices=["openai", "huggingface", "ollama"],
        help=f"Embedding 后端 (默认: {config.embedding.backend})"
    )
    ingest_parser.add_argument(
        "--table_name", default=None,
        help=f"数据库表名 (默认: {config.database.table_name})"
    )
    ingest_parser.add_argument(
        "--rebuild", action="store_true",
        help="重建索引（清空旧数据）"
    )
    ingest_parser.add_argument(
        "--loader_backend", default="unstructured",
        choices=["unstructured", "mineru"],
        help="文档加载后端 (默认: unstructured)"
    )
    ingest_parser.add_argument(
        "--mineru_method", default=None,
        choices=["auto", "txt", "ocr"],
        help="MinerU 解析方式，仅 --loader_backend=mineru 时生效 (默认: auto)"
    )
    ingest_parser.set_defaults(func=cmd_ingest)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
