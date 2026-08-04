"""Story length configuration — constants for long-form novel writing."""

import os

# 长篇默认配置（短篇/中篇已移除，只保留长篇）
DEFAULT_CHAPTER_WORDS = 3000
DEFAULT_MAX_TOKENS = 4096
# reasoning 模型（如 step-3.7-flash）的推理深度；低档压 reasoning 预算，
# 防止其 max_tokens 里的推理 token 挤空正文 content（实测偶发只出 123 字）
REASONING_EFFORT = "low"
NARRATIVE_PACING = (
    "渐进展开，intro充分铺垫(前5-10%)，development多线并进，"
    "climax在70-80%处，伏笔长线回收"
)
TYPICAL_CHAPTERS = "100章以上"


def env_bool(name: str, default: bool = False) -> bool:
    """将环境变量解析为布尔。接受 1/true/yes/on（大小写不敏感）。"""
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")
