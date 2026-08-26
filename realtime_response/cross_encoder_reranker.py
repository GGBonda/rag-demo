"""调用远程 Cross-Encoder 模型对召回分片进行精排。"""

import time

import requests

from config import config
from data_class import RetrieveChunk


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
        documents: list[dict[str, str]] = []
        document_chunk_indexes: list[int] = []
        for chunk_index, chunk in enumerate(chunks):
            chunk_documents = chunk.analyse_rerank()
            documents.extend(chunk_documents)
            document_chunk_indexes.extend(
                [chunk_index] * len(chunk_documents)
            )

        print(
            f"[Cross-Encoder 精排] 请求开始: data_class={self._model}, "
            f"chunks={len(chunks)}, documents={len(documents)}, "
            f"top_n={top_n}"
        )
        start_time = time.perf_counter()
        response = requests.post(
            self._base_url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "data_class": self._model,
                "input": {
                    "query": {"text": query},
                    "documents": documents,
                },
                "parameters": {
                    "top_n": len(documents),
                    "return_documents": False,
                },
            },
            timeout=30,
        )
        print(
            f"[Cross-Encoder 精排] 请求结束: "
            f"response={response.text}，耗时={time.perf_counter() - start_time:.2f}秒"
        )
        response.raise_for_status()

        output = response.json().get("output")
        results = output.get("results") if isinstance(output, dict) else None
        if not isinstance(results, list):
            raise RuntimeError(
                "Cross-Encoder API 响应中缺少 output.results 列表"
            )

        try:
            chunk_scores: dict[int, float] = {}
            for item in results:
                document_index = int(item["index"])
                chunk_index = document_chunk_indexes[document_index]
                score = float(item["relevance_score"])
                chunk_scores[chunk_index] = max(
                    score,
                    chunk_scores.get(chunk_index, float("-inf")),
                )

            ranked_chunk_indexes = sorted(
                chunk_scores,
                key=chunk_scores.__getitem__,
                reverse=True,
            )
            ranked_chunks = [
                chunks[chunk_index]
                for chunk_index in ranked_chunk_indexes[:top_n]
            ]
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise RuntimeError(
                "Cross-Encoder API 返回了无效的精排结果"
            ) from exc

        return ranked_chunks
