"""Prompt 文件加载与解析。"""

from pathlib import Path
from string import Template


PROMPTS_DIR = Path(__file__).resolve().parent
SYSTEM_PROMPT_MARKER = "[system]"
USER_PROMPT_MARKER = "[user]"


def _load_prompt_content(file_name: str) -> tuple[str, str]:
    content = (PROMPTS_DIR / file_name).read_text(encoding="utf-8").strip()
    system_marker, separator, user_content = content.partition(USER_PROMPT_MARKER)
    if not separator or not system_marker.strip().startswith(SYSTEM_PROMPT_MARKER):
        raise ValueError(
            f"提示词文件 {file_name} 必须包含 [system] 和 [user] 标记"
        )

    system_content = system_marker.strip()[len(SYSTEM_PROMPT_MARKER):].strip()
    user_content = user_content.strip()
    return system_content, user_content


def load_prompt(file_name: str) -> tuple[str, Template]:
    system_content, user_content = _load_prompt_content(file_name)
    if not system_content or not user_content:
        raise ValueError(f"提示词文件 {file_name} 的 system 和 user 内容不能为空")
    return system_content, Template(user_content)


def load_system_prompt(file_name: str) -> str:
    system_content, _ = _load_prompt_content(file_name)
    if not system_content:
        raise ValueError(f"提示词文件 {file_name} 的 system 内容不能为空")
    return system_content
