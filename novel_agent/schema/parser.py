"""Shared JSON response parser — used by all agents to extract structured output.

Strategy: direct parse → markdown code block → regex {.*} → defaults.
"""

import json
import re


def parse_json_response(text: str, defaults: dict | None = None) -> dict:
    """Extract JSON from LLM text output with fallback defaults.

    Args:
        text: Raw LLM response text (may contain markdown wrapping).
        defaults: Fallback dict if all parsing fails.

    Returns:
        Parsed dict (or defaults if unparseable).
    """
    if defaults is None:
        defaults = {}

    if not text or not text.strip():
        return dict(defaults)

    # 1. Direct JSON parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Extract from markdown code block
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 3. Find JSON object in text
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    # 4. Return defaults with raw text preserved
    result = dict(defaults)
    result["raw_output"] = text
    return result
