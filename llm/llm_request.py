"""统一封装项目中的大模型 HTTP 请求。"""

import time
from pathlib import Path

import requests

from config import config


CHAT_COMPLETION_TIMEOUT_SECONDS = 120
CROSS_ENCODER_TIMEOUT_SECONDS = 30
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
SYSTEM_PROMPT_MARKER = "[system]"
USER_PROMPT_MARKER = "[user]"

_TABLE_SYSTEM_PROMPT = "ai_desc_for_table.txt"
_CODE_SYSTEM_PROMPT = "ai_desc_for_code.txt"
_IMAGE_SYSTEM_PROMPT = "ai_desc_for_image.txt"
_RETRIEVE_SYSTEM_PROMPT = "ai_desc_question_for_retrieve_chunk.txt"


def request_table_description(
    user_content: str | list[dict],
) -> tuple[str | None, str | None]:
    """请求生成表格描述。"""
    return _request_chat_completion(
        user_content=user_content,
        system_prompt_file=_TABLE_SYSTEM_PROMPT,
        use_vision_model=False,
        task_name="表格描述",
    )


def request_code_description(
    user_content: str | list[dict],
) -> tuple[str | None, str | None]:
    """请求生成代码描述。"""
    return _request_chat_completion(
        user_content=user_content,
        system_prompt_file=_CODE_SYSTEM_PROMPT,
        use_vision_model=False,
        task_name="代码描述",
    )


def request_image_description(
    user_content: str | list[dict],
) -> tuple[str | None, str | None]:
    """请求生成图片描述。"""
    return _request_chat_completion(
        user_content=user_content,
        system_prompt_file=_IMAGE_SYSTEM_PROMPT,
        use_vision_model=True,
        task_name="图片描述",
    )


def request_retrieve_description(
    user_content: str | list[dict],
) -> tuple[str | None, str | None]:
    """请求生成召回分片的索引简介和问题。"""
    return _request_chat_completion(
        user_content=user_content,
        system_prompt_file=_RETRIEVE_SYSTEM_PROMPT,
        use_vision_model=isinstance(user_content, list),
        task_name="召回文档简介",
        response_format={"type": "json_object"},
    )


def _request_chat_completion(
    user_content: str | list[dict],
    system_prompt_file: str,
    use_vision_model: bool,
    task_name: str,
    response_format: dict | None = None,
) -> tuple[str | None, str | None]:
    api_key, base_url, model = _resolve_chat_config(use_vision_model)
    payload: dict[str, object] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": _load_system_prompt(system_prompt_file),
            },
            {"role": "user", "content": user_content},
        ],
    }
    if response_format is not None:
        payload["response_format"] = response_format

    print(f"[大模型请求] 请求开始: task={task_name}, model={model}")
    start_time = time.perf_counter()
    response_data = _post_json(
        url=f"{base_url.rstrip('/')}/chat/completions",
        api_key=api_key,
        payload=payload,
        timeout=CHAT_COMPLETION_TIMEOUT_SECONDS,
        api_name="大模型 API",
    )
    print(
        f"[大模型请求] 请求结束: task={task_name}, model={model}, "
        f"耗时={time.perf_counter() - start_time:.2f}秒"
    )

    try:
        choice = response_data["choices"][0]
        content = choice["message"]["content"]
        finish_reason = choice.get("finish_reason")
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            "大模型 API 响应中缺少 choices[0].message.content"
        ) from exc

    if content is not None and not isinstance(content, str):
        raise RuntimeError(
            "大模型 API 响应中的 choices[0].message.content 必须为字符串"
        )
    if not isinstance(finish_reason, str):
        finish_reason = None
    return content, finish_reason


def request_cross_encoder_rerank(
    query: str,
    documents: list[dict[str, str]],
) -> list[dict]:
    """调用远程 Cross-Encoder rerank API。"""
    url = _require_config(
        config.cross_encoder.base_url,
        "CROSS_ENCODER_BASE_URL",
    ).rstrip("/")
    api_key = _require_config(
        config.cross_encoder.api_key,
        "CROSS_ENCODER_API_KEY",
    )
    model = _require_config(
        config.cross_encoder.model,
        "CROSS_ENCODER_MODEL",
    )
    print(
        f"[Cross-Encoder 精排] 请求开始: model={model}, "
        f"documents={len(documents)}"
    )
    start_time = time.perf_counter()
    response_data = _post_json(
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
        timeout=CROSS_ENCODER_TIMEOUT_SECONDS,
        api_name="Cross-Encoder API",
    )
    print(
        f"[Cross-Encoder 精排] 请求结束: model={model}, "
        f"耗时={time.perf_counter() - start_time:.2f}秒"
    )

    output = response_data.get("output")
    results = output.get("results") if isinstance(output, dict) else None
    if not isinstance(results, list):
        raise RuntimeError(
            "Cross-Encoder API 响应中缺少 output.results 列表"
        )
    return results


def _resolve_chat_config(use_vision_model: bool) -> tuple[str, str, str]:
    if use_vision_model:
        api_key = _require_config(
            config.vision_llm.openai_api_key,
            "OPENAI_VISUAL_API_KEY",
        )
        base_url = config.vision_llm.openai_base_url.strip()
        model = _require_config(
            config.vision_llm.openai_model,
            "OPENAI_VISUAL_MODEL",
        )
    else:
        api_key = _require_config(
            config.llm.openai_api_key,
            "OPENAI_API_KEY",
        )
        base_url = config.llm.openai_base_url.strip()
        model = _require_config(
            config.llm.openai_pro_model,
            "OPENAI_PRO_MODEL",
        )

    return api_key, (base_url or DEFAULT_OPENAI_BASE_URL).rstrip("/"), model


def _require_config(value: str, environment_name: str) -> str:
    resolved_value = value.strip()
    if not resolved_value:
        raise ValueError(f"未配置 {environment_name} 环境变量")
    return resolved_value


def _load_system_prompt(file_name: str) -> str:
    content = (PROMPTS_DIR / file_name).read_text(encoding="utf-8").strip()
    system_marker, separator, _ = content.partition(USER_PROMPT_MARKER)
    if not separator or not system_marker.strip().startswith(SYSTEM_PROMPT_MARKER):
        raise ValueError(
            f"提示词文件 {file_name} 必须包含 [system] 和 [user] 标记"
        )

    system_content = system_marker.strip()[len(SYSTEM_PROMPT_MARKER):].strip()
    if not system_content:
        raise ValueError(f"提示词文件 {file_name} 的 system 内容不能为空")
    return system_content


def _post_json(
    url: str,
    api_key: str,
    payload: dict[str, object],
    timeout: int,
    api_name: str,
) -> dict:
    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()

    try:
        response_data = response.json()
    except ValueError as exc:
        raise RuntimeError(f"{api_name} 返回的内容不是有效 JSON") from exc
    if not isinstance(response_data, dict):
        raise RuntimeError(f"{api_name} 返回的内容必须是 JSON 对象")
    return response_data
