"""Shared JSON response parser — used by all agents to extract structured output.

Strategy: direct parse → markdown code block → regex {.*} → defaults.

All dicts returned here are sanitized through ``strip_none`` so that ``null``
fields produced by the LLM never reach downstream consumers. This is the single
boundary where unstructured LLM output is normalized.

Parse statistics are tracked via ``ParseStats`` so fallback is never silent.
"""

import json
import re
from collections import Counter


def strip_none(obj):
    """递归移除 None 值（dict 值 + list 元素）。

    LLM 生成的结构化 JSON 常把可选嵌套字段写成 ``null``。而
    ``dict.get(key, default)`` 只在键**不存在**时返回 default，键存在但值为
    None 时返回 None，导致下游 ``None.get(...)`` 崩溃。清理后这些字段回归到
    "缺省"路径，``.get(key, default)`` 才真正生效。
    """
    if isinstance(obj, dict):
        return {k: strip_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [strip_none(v) for v in obj if v is not None]
    return obj


class ParseStats:
    """Module-level parse strategy counter — fallback is never silent.

    Tracks how often each parse strategy fires across all calls:
    - ``direct``:   json.loads succeeded on first try
    - ``markdown``: extracted from ```json code block
    - ``repaired``: _repair_json fixed syntax errors (trailing comma, missing quote)
    - ``fallback``: all strategies failed, returned defaults
    """

    _counts: Counter = Counter()

    @classmethod
    def record(cls, method: str) -> None:
        cls._counts[method] += 1

    @classmethod
    def snapshot(cls) -> dict:
        return dict(cls._counts)

    @classmethod
    def reset(cls) -> None:
        cls._counts.clear()


def _repair_json(text: str) -> str | None:
    """Iteratively repair common LLM JSON syntax errors.

    LLM 输出偶尔带语法错误，例如字符串值漏掉左引号（``"数量": 10枚"``）或
    末尾多余逗号。这些错误会让 ``json.loads`` 整体失败，导致整份结构化结果
    被丢弃。这里逐类修复，每轮后重试解析，最多 6 轮。

    返回 ``json.loads`` 可接受的字符串；修复不了返回 None。
    """
    t = text.strip()
    for _ in range(6):
        try:
            json.loads(t)
            return t
        except json.JSONDecodeError:
            pass
        fixed = re.sub(r",\s*([}\]])", r"\1", t)  # 尾逗号 ,} 或 ,]
        fixed = re.sub(
            r':\s*([^"{}\[\],\s][^"{}]*?)"(?=\s*[,}\]])', r': "\1"', fixed
        )  # 漏左引号的字符串值 10枚" -> "10枚"
        if fixed == t:
            return None
        t = fixed
    try:
        json.loads(t)
        return t
    except json.JSONDecodeError:
        return None


def _try_load(text: str) -> tuple[object | None, str]:
    """``json.loads``，失败时先走 ``_repair_json`` 修复再试。

    Returns ``(data, method)`` where method is ``"direct"`` or ``"repaired"``.
    """
    try:
        return json.loads(text), "direct"
    except json.JSONDecodeError:
        pass
    repaired = _repair_json(text)
    if repaired is not None:
        try:
            return json.loads(repaired), "repaired"
        except json.JSONDecodeError:
            pass
    return None, ""


def parse_json_response(text: str, defaults: dict | None = None) -> dict:
    """Extract JSON from LLM text output with fallback defaults.

    Args:
        text: Raw LLM response text (may contain markdown wrapping).
        defaults: Fallback dict if all parsing fails.

    Returns:
        Parsed dict (or defaults if unparseable). Never contains nested None
        values — ``strip_none`` is applied to every successful parse and to the
        fallback defaults alike.

    When fallback to defaults occurs, the result includes ``raw_output`` and
    ``parse_method`` = ``"fallback"`` so downstream code can distinguish a
    real parse from a default substitution.
    """
    if defaults is None:
        defaults = {}

    if not text or not text.strip():
        ParseStats.record("fallback")
        result = strip_none(dict(defaults))
        result["raw_output"] = text or ""
        result["parse_method"] = "fallback"
        return result

    # 候选文本：原始 → markdown 代码块 → 最外层 {...}
    candidates = [("direct", text)]
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m:
        candidates.append(("markdown", m.group(1)))
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        candidates.append(("regex", m.group(0)))

    for label, cand in candidates:
        data, method = _try_load(cand)
        if isinstance(data, dict):
            # method = "direct" or "repaired" from _try_load;
            # label = "direct" / "markdown" / "regex" for the candidate source.
            # For stats: prefer the repair method when repair was needed,
            # otherwise use the candidate label (markdown for code block).
            stat = method if method == "repaired" else label
            if stat == "regex":
                stat = "direct"
            ParseStats.record(stat)
            return strip_none(data)

    # 全部失败：返回 defaults 并保留原始输出
    ParseStats.record("fallback")
    result = strip_none(dict(defaults))
    result["raw_output"] = text
    result["parse_method"] = "fallback"
    return result
