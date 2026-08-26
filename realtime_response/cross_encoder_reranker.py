"""调用远程 Cross-Encoder 模型对召回分片进行精排。"""

from data_class import RetrieveChunk
from llm import request_cross_encoder_rerank


class CrossEncoderReranker:
    """通过远程 rerank API 对 RetrieveChunk 按相关性重新排序。"""

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

        results = request_cross_encoder_rerank(
            query=query,
            documents=documents,
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
