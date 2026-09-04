"""Deterministic Langfuse payloads: no chapter text, skip editor when gated."""

from novel_agent.observability.payload import (
    chapter_input,
    outcome_events,
    outcome_scores,
)


def test_chapter_input_omits_draft_text():
    payload = chapter_input(
        {
            "chapter_number": 3,
            "chapter_outline": "大纲" * 20,
            "target_chapter_words": 3000,
            "draft_content": "不应上传的正文",
            "writing_run_id": "run-1",
        }
    )
    assert payload["chapter"] == 3
    assert payload["target_words"] == 3000
    assert "不应上传" not in str(payload)
    assert "draft" not in payload


def test_gate_first_skip_omits_editor_and_composite():
    scores = outcome_scores(
        {
            "deterministic_gate_first": True,
            "quality_gate_report": {"passed": True},
            "editor_report": {},
            "continuity_report": {"overall_score": 90},
            "style_report": {"paragraph_structure_score": 80.0},
            "draft_content": "中文正文" * 40,
        }
    )
    names = [item["name"] for item in scores]
    assert "quality_gate" in names
    assert "style_structure" in names
    assert "content_units" in names
    assert "editor" not in names
    assert "continuity" not in names
    assert "composite" not in names


def test_v0_gate_emits_event():
    events = outcome_events(
        {
            "evolution_termination": "v0_gate",
            "evolution_v0_gate_score": 78,
            "editor_report": {"overall_score": 80},
            "continuity_report": {"overall_score": 80},
            "style_report": {"paragraph_structure_score": 80.0},
        }
    )
    v0 = next(item for item in events if item["name"] == "evolution.v0_gate")
    assert v0["metadata"]["threshold"] == 78
    assert "composite" in v0["metadata"]
