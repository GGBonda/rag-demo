from .http_request import post_json
from config import config

def request_cross_encoder_rerank(
    query: str,
    documents: list[dict[str, str]],
) -> list[dict]:
    """调用远程 Cross-Encoder rerank API。"""
    url = config.cross_encoder.base_url
    api_key = config.cross_encoder.api_key
    model = config.cross_encoder.model

    print(f"[Cross-Encoder 精排] 开始处理 {len(documents)} 个文档")
    response_data = post_json(
        url=url,
        api_key=api_key,
        payload={
            "data_class": model,
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
        api_name="Cross-Encoder 精排",
    )

    output = response_data.get("output")
    results = output.get("results") if isinstance(output, dict) else None
    if not isinstance(results, list):
        raise RuntimeError(
            "Cross-Encoder API 响应中缺少 output.results 列表"
        )
    return results