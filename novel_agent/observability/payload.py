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
        "v0_gate": state.get("evolution_v0_gate_score"),
        "scene_first": bool(state.get("scene_first")),
        "gate_first": bool(state.get("deterministic_gate_first")),
        "skip_reviews": bool(state.get("skip_reviews")),
    }


def chapter_tags(state: Mapping[str, Any], *, source: str = "api") -> list[str]:
    tags = [source]
    if state.get("deterministic_gate_first"):
        tags.append("gate-first")
    rounds = state.get("evolution_max_rounds")
    if rounds is not None:
        tags.append(f"evo-r{rounds}")
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


def _reviews_skipped(values: Mapping[str, Any]) -> str | None:
    gate = values.get("quality_gate_report") or {}
    if values.get("deterministic_gate_first") and isinstance(gate, dict) and gate.get("passed"):
        editor = values.get("editor_report") or {}
        if not editor or (isinstance(editor, dict) and editor.get("unavailable")):
            return "gate_first"
    if values.get("skip_reviews"):
        return "skip_reviews"
    return None


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
        "termination": values.get("evolution_termination") or "",
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
    skipped = _reviews_skipped(values)
    if not skipped:
        from novel_agent.services.evolution import composite_score, extract_scores

        extracted = extract_scores(values)
        editor = extracted.get("editor_overall") or 0
        continuity = extracted.get("continuity_overall") or 0
        if editor:
            scores.append({"name": "editor", "value": float(editor)})
        if continuity:
            scores.append({"name": "continuity", "value": float(continuity)})
        if editor or continuity:
            scores.append({"name": "composite", "value": float(composite_score(extracted))})
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
    skipped = _reviews_skipped(values)
    if skipped:
        events.append({"name": "reviews.skipped", "metadata": {"reason": skipped}})
    termination = values.get("evolution_termination") or ""
    if termination == "v0_gate":
        from novel_agent.services.evolution import composite_score, extract_scores

        extracted = extract_scores(values)
        events.append(
            {
                "name": "evolution.v0_gate",
                "metadata": {
                    "composite": composite_score(extracted),
                    "threshold": values.get("evolution_v0_gate_score"),
                },
            }
        )
    elif termination:
        events.append(
            {
                "name": "evolution.rewrite",
                "metadata": {
                    "termination": termination,
                    "version": values.get("evolution_version"),
                },
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
    skipped = _reviews_skipped(values)
    if skipped:
        tags.append("skipped-reviews")
    gate = values.get("quality_gate_report") or {}
    if isinstance(gate, dict) and gate.get("passed") is False:
        tags.append("gate-failed")
    termination = values.get("evolution_termination") or ""
    if termination == "v0_gate":
        tags.append("evo-skip")
    elif termination:
        tags.append("evo-rewrite")
    return tags
