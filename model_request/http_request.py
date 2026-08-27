import time
import requests

def post_json(
    url: str,
    api_key: str,
    payload: dict[str, object],
    timeout: int,
    api_name: str,
) -> dict:
    print(f"[{api_name}] 请求开始")

    start_time = time.perf_counter()
    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    print(f"[{api_name}] 请求结束, 耗时={time.perf_counter() - start_time:.2f}秒")

    response.raise_for_status()

    try:
        response_data = response.json()
    except ValueError as exc:
        raise RuntimeError(f"{api_name} 返回的内容不是有效 JSON") from exc
    if not isinstance(response_data, dict):
        raise RuntimeError(f"{api_name} 返回的内容必须是 JSON 对象")
    return response_data