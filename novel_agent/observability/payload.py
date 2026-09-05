"""Deterministic trace payload: hashes, truncation, scores, events.

No I/O, no SDK. Default contract: do not upload chapter text or full prompts.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

PROMPT_TRUNCATE_CHARS = 2000
ISSUE_TRUNCATE_CHARS = 200
ISSUE_LIMIT = 5


def _text_units(text: str) -> int:
    cjk = sum(1 for ch in text if "\u3400" <= ch <= "\u9fff")
    if text and cjk >= len(text) * 0.2:
        return cjk
    return len(text.split())


def content_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def truncate(text: str, limit: int = PROMPT_TRUNCATE_CHARS) -> str:
    value = text or ""
    if len(value) <= limit:
        return value
    return value[:limit] + "…"


def chapter_input(state: Mapping[str, Any]) -> dict[str, Any]:
    outline = str(state.get("chapter_outline") or "")
    return {
        "chapter": state.get("chapter_number"),
        "outline_chars": len(outline),
        "target_words": state.get("target_chapter_words"),
        "run_id": state.get("writing_run_id") or "",
    }


def compact_meta(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "run_id": state.get("writing_run_id") or "",
        "chapter": state.get("chapter_number"),
        "loop_retries": state.get("loop_retries", 0),
    }


def chapter_tags(state: Mapping[str, Any], *, source: str = "api") -> list[str]:
    tags = [source]
    return tags


def redact_tool_args(args: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = dict(args or {})
    query = payload.get("query")
    if isinstance(query, str):
        payload["query"] = truncate(query, 200)
    return payload


def _editor_issue_summaries(report: Mapping[str, Any]) -> list[str]:
    issues = report.get("issues") or report.get("problems") or []
    if not isinstance(issues, list):
        return []
    out: list[str] = []
    for item in issues[:ISSUE_LIMIT]:
        if isinstance(item, str):
            out.append(truncate(item, ISSUE_TRUNCATE_CHARS))
        elif isinstance(item, dict):
            desc = item.get("description") or item.get("message") or ""
            if desc:
                out.append(truncate(str(desc), ISSUE_TRUNCATE_CHARS))
    return out


def outcome_output(values: Mapping[str, Any], *, interrupted: bool = False) -> dict[str, Any]:
    draft = str(values.get("draft_content") or "")
    gate = values.get("quality_gate_report") or {}
    wb = values.get("worldbuilding_report") or {}
    entities = wb.get("new_entities") if isinstance(wb, dict) else None
    return {
        "interrupted": interrupted,
        "approved": bool(values.get("human_approved")),
        "content_chars": len(draft),
        "content_hash": content_hash(draft) if draft else "",
        "gate_passed": bool(gate.get("passed")) if isinstance(gate, dict) else None,
        "loop_retries": values.get("loop_retries", 0),
        "wb_entity_count": len(entities) if isinstance(entities, list) else 0,
    }


def outcome_scores(values: Mapping[str, Any]) -> list[dict[str, Any]]:
    scores: list[dict[str, Any]] = []
    gate = values.get("quality_gate_report") or {}
    if isinstance(gate, dict) and "passed" in gate:
        scores.append({"name": "quality_gate", "value": 1.0 if gate.get("passed") else 0.0})
    style = values.get("style_report") or {}
    if isinstance(style, dict) and "paragraph_structure_score" in style:
        scores.append(
            {"name": "style_structure", "value": float(style["paragraph_structure_score"])}
        )
    editor = values.get("editor_report") or {}
    if isinstance(editor, dict) and editor.get("overall_score") is not None:
        scores.append({"name": "editor", "value": float(editor["overall_score"])})
    continuity = values.get("continuity_report") or {}
    if isinstance(continuity, dict) and continuity.get("overall_score") is not None:
        scores.append({"name": "continuity", "value": float(continuity["overall_score"])})
    draft = str(values.get("draft_content") or "")
    if draft:
        scores.append({"name": "content_units", "value": float(_text_units(draft))})
    return scores


def outcome_events(values: Mapping[str, Any], *, interrupted: bool = False) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    gate = values.get("quality_gate_report") or {}
    if isinstance(gate, dict) and gate.get("passed") is False:
        events.append(
            {
                "name": "quality_gate.failed",
                "metadata": {"violations": list(gate.get("violations") or [])},
            }
        )
    editor = values.get("editor_report") or {}
    if isinstance(editor, dict):
        summaries = _editor_issue_summaries(editor)
        if summaries:
            events.append({"name": "editor.issues", "metadata": {"items": summaries}})
    if interrupted:
        events.append({"name": "human.interrupt", "metadata": {}})
    elif values.get("human_approved"):
        events.append({"name": "human.approve", "metadata": {}})
    elif values.get("human_feedback"):
        action = ""
        feedback = values.get("human_feedback")
        if isinstance(feedback, dict):
            action = str(feedback.get("action") or "")
        if action == "reject":
            events.append({"name": "human.reject", "metadata": {}})
    return events


def outcome_tags(values: Mapping[str, Any]) -> list[str]:
    tags: list[str] = []
    gate = values.get("quality_gate_report") or {}
    if isinstance(gate, dict) and gate.get("passed") is False:
        tags.append("gate-failed")
    return tags
