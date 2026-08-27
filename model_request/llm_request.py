from .http_request import post_json

from config import config
from prompts import load_system_prompt

def request_table_description(
    user_content: str | list[dict],
) -> tuple[str | None, str | None]:
    """请求生成表格描述。"""
    return _request_chat_completion(
        user_content=user_content,
        system_prompt_file="ai_desc_for_table.txt",
        use_vision_model=False,
        task_name="表格描述",
    )


def request_code_description(
    user_content: str | list[dict],
) -> tuple[str | None, str | None]:
    """请求生成代码描述。"""
    return _request_chat_completion(
        user_content=user_content,
        system_prompt_file="ai_desc_for_code.txt",
        use_vision_model=False,
        task_name="代码描述",
    )


def request_image_description(
    user_content: str | list[dict],
) -> tuple[str | None, str | None]:
    """请求生成图片描述。"""
    return _request_chat_completion(
        user_content=user_content,
        system_prompt_file="ai_desc_for_image.txt",
        use_vision_model=True,
        task_name="图片描述",
    )


def request_retrieve_description(
    user_content: str | list[dict],
) -> tuple[str | None, str | None]:
    """请求生成召回分片的索引简介和问题。"""
    return _request_chat_completion(
        user_content=user_content,
        system_prompt_file="ai_desc_question_for_retrieve_chunk.txt",
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
                "content": load_system_prompt(system_prompt_file),
            },
            {"role": "user", "content": user_content},
        ],
    }
    if response_format is not None:
        payload["response_format"] = response_format

    response_data = post_json(
        url=f"{base_url.rstrip('/')}/chat/completions",
        api_key=api_key,
        payload=payload,
        timeout=120,
        opt_desc=task_name,
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

def _resolve_chat_config(use_vision_model: bool) -> tuple[str, str, str]:
    if use_vision_model:
        api_key = config.vision_llm.openai_api_key.strip()
        base_url = config.vision_llm.openai_base_url.strip()
        model = config.vision_llm.openai_model.strip()
    else:
        api_key = config.llm.openai_api_key.strip()
        base_url = config.llm.openai_base_url.strip()
        model = config.llm.openai_pro_model.strip()

    return api_key, base_url, model



