#!/usr/bin/env python3
"""
RAG 知识库 - 实时响应 HTTP API 服务

启动方式:
    uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
    python api_server.py

端点:
    POST /api/search   - 语义检索
    GET  /api/stats    - 知识库统计
    GET  /api/health   - 健康检查
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from data_class import RetrieveChunk
from realtime_response import Retriever

app = FastAPI(
    title="RAG Knowledge Base API",
    description="实时响应模块 - 语义检索与知识库统计",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchResponse(BaseModel):
    query: str
    results: list[RetrieveChunk]


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/search", response_model=SearchResponse)
def search(req: SearchRequest):
    retriever = Retriever()

    results = retriever.search(query=req.query, top_k=req.top_k)

    if not results:
        raise HTTPException(
            status_code=404,
            detail="未检索到相关内容，请先运行 ingest 命令入库文档",
        )

    return SearchResponse(query=req.query, results=results)


@app.get("/api/stats")
def stats():
    retriever = Retriever()
    return retriever.get_stats()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
