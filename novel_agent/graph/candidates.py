"""Serializable Evolution candidate records."""

from typing import Any, TypedDict


class EvolutionCandidate(TypedDict, total=False):
    version: int
    draft_content: str
    editor_report: dict[str, Any]
    continuity_report: dict[str, Any]
    worldbuilding_report: dict[str, Any]
    quality_guard_report: dict[str, Any]
    quality_gate_report: dict[str, Any]
    outline_coverage: float | None
    required_facts_missing: int
    scores: dict[str, Any]
    composite_score: float
    content_length: int


def candidate_from_state(
    state: dict,
    version: int,
    scores: dict[str, Any],
    quality_guard_report: dict | None = None,
) -> EvolutionCandidate:
    return {
        "version": version,
        "draft_content": state.get("draft_content", ""),
        "editor_report": state.get("editor_report", {}) or {},
        "continuity_report": state.get("continuity_report", {}) or {},
        "worldbuilding_report": state.get("worldbuilding_report", {}) or {},
        "quality_guard_report": quality_guard_report or {},
        "quality_gate_report": state.get("quality_gate_report", {}) or {},
        "outline_coverage": state.get("outline_coverage"),
        "required_facts_missing": state.get("required_facts_missing", 0),
        "scores": scores,
        "composite_score": scores.get("composite", 0),
        "content_length": len(state.get("draft_content", "")),
    }


def candidate_to_state(candidate: EvolutionCandidate) -> dict[str, Any]:
    return {
        "draft_content": candidate.get("draft_content", ""),
        "editor_report": candidate.get("editor_report", {}) or {},
        "continuity_report": candidate.get("continuity_report", {}) or {},
        "worldbuilding_report": candidate.get("worldbuilding_report", {}) or {},
        "outline_coverage": candidate.get("outline_coverage"),
        "required_facts_missing": candidate.get("required_facts_missing", 0),
        "quality_gate_report": candidate.get("quality_gate_report", {}) or {},
    }
