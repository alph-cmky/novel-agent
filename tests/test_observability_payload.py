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


def test_scores_include_editor_and_continuity():
    scores = outcome_scores(
        {
            "quality_gate_report": {"passed": True},
            "editor_report": {"overall_score": 80},
            "continuity_report": {"overall_score": 90},
            "style_report": {"paragraph_structure_score": 80.0},
            "draft_content": "中文正文" * 40,
        }
    )
    names = [item["name"] for item in scores]
    assert "quality_gate" in names
    assert "style_structure" in names
    assert "content_units" in names
    assert "editor" in names
    assert "continuity" in names


def test_events_include_editor_issues():
    events = outcome_events(
        {
            "editor_report": {
                "overall_score": 80,
                "issues": [{"description": "对话不够自然"}],
            },
        }
    )
    issue_events = [item for item in events if item["name"] == "editor.issues"]
    assert len(issue_events) == 1
    assert "对话不够自然" in issue_events[0]["metadata"]["items"][0]
