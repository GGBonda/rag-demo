"""调用远程 Cross-Encoder 模型对召回分片进行精排。"""

import requests

from config import config
from offline_processing.chunker import RetrieveChunk


class CrossEncoderReranker:
    """通过远程 rerank API 对 RetrieveChunk 按相关性重新排序。"""

    def __init__(self) -> None:
        self._base_url = config.cross_encoder.base_url.strip().rstrip("/")
        if not self._base_url:
            raise ValueError("未配置 CROSS_ENCODER_BASE_URL 环境变量")

        self._api_key = config.cross_encoder.api_key.strip()
        if not self._api_key:
            raise ValueError("未配置 CROSS_ENCODER_API_KEY 环境变量")

        self._model = config.cross_encoder.model.strip()
        if not self._model:
            raise ValueError("未配置 CROSS_ENCODER_MODEL 环境变量")

    def rerank(
        self,
        query: str,
        chunks: list[RetrieveChunk],
        top_k: int,
    ) -> list[RetrieveChunk]:
        """按 query 与分片正文的相关性降序返回最多 top_k 个分片。"""
        if top_k <= 0:
            raise ValueError("top_k 必须为正整数")
        if not chunks:
            return []

        top_n = min(top_k, len(chunks))
        response = requests.post(
            f"{self._base_url}/rerank",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._model,
                "query": query,
                "documents": [chunk.text for chunk in chunks],
                "top_n": top_n,
                "return_documents": False,
            },
            timeout=30,
        )
        response.raise_for_status()

        results = response.json().get("results")
        if not isinstance(results, list):
            raise RuntimeError("Cross-Encoder API 响应中缺少 results 列表")

        try:
            ranked_results = sorted(
                results,
                key=lambda item: float(item["relevance_score"]),
                reverse=True,
            )
            ranked_chunks = [
                chunks[int(item["index"])]
                for item in ranked_results[:top_n]
            ]
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise RuntimeError(
                "Cross-Encoder API 返回了无效的精排结果"
            ) from exc

        return ranked_chunks
