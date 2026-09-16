"""Unit tests for S1 conditional Hard Gate → Editor/Continuity orchestration."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from novel_agent.graph.agent_loop import EmptyDraftError, run_agent_loop


def _long_draft() -> str:
    """CJK-heavy draft long enough for typical hard-gate thresholds."""
    return "正文内容" * 500


def _make_patches(*, draft: str, gate_passed: bool, editor_verdict: str = "accept"):
    orch = MagicMock()
    orch.analyze = AsyncMock(return_value={"narrative_stage": "rising", "context_needed": {}})
    for name in ("input_tokens", "output_tokens", "cached_tokens", "reasoning_tokens", "model_calls"):
        setattr(orch, name, 1)

    writer = MagicMock()
    writer.tool_call_counts = {}
    for name in ("input_tokens", "output_tokens", "cached_tokens", "reasoning_tokens", "model_calls"):
        setattr(writer, name, 1)
    writer.write_with_loop = AsyncMock(return_value=(draft, MagicMock()))

    editor = MagicMock()
    editor.review = AsyncMock(
        return_value=(
            {
                "overall_score": 70,
                "verdict": editor_verdict,
                "issues": [{"dimension": "pacing", "description": "慢"}],
            },
            MagicMock(),
        )
    )
    for name in ("input_tokens", "output_tokens", "cached_tokens", "reasoning_tokens", "model_calls"):
        setattr(editor, name, 2)

    continuity = MagicMock()
    continuity.audit = AsyncMock(return_value=({"overall_score": 80, "issues": []}, MagicMock()))
    for name in ("input_tokens", "output_tokens", "cached_tokens", "reasoning_tokens", "model_calls"):
        setattr(continuity, name, 3)

    wb = MagicMock()
    wb.extract = AsyncMock(
        return_value=(
            {
                "new_entities": [{"name": "甲"}],
                "updated_entities": [],
                "conflicts": [],
                "foreshadowings": [],
            },
            MagicMock(),
        )
    )
    for name in ("input_tokens", "output_tokens", "cached_tokens", "reasoning_tokens", "model_calls"):
        setattr(wb, name, 4)

    style = MagicMock()
    style.analyze.return_value = MagicMock(model_dump=lambda: {})

    gate_result = {
        "passed": gate_passed,
        "violations": [] if gate_passed else ["minimum_length"],
        "content_units": 1000,
        "minimum_units": 600,
        "target_units": 600,
    }

    return {
        "orch": orch,
        "writer": writer,
        "editor": editor,
        "continuity": continuity,
        "wb": wb,
        "OrchestratorAgent": MagicMock(return_value=orch),
        "WriterAgent": MagicMock(return_value=writer),
        "EditorAgent": MagicMock(return_value=editor),
        "ContinuityAgent": MagicMock(return_value=continuity),
        "WorldbuildingAgent": MagicMock(return_value=wb),
        "StyleAnalyzer": MagicMock(return_value=style),
        "gate": MagicMock(return_value=gate_result),
        "_config_for": MagicMock(return_value=MagicMock(max_tokens=4096)),
        "_get_chapter_store": MagicMock(return_value=None),
    }


def _run(state: dict[str, Any], p: dict) -> dict:
    with (
        patch("novel_agent.graph.agent_loop.OrchestratorAgent", p["OrchestratorAgent"]),
        patch("novel_agent.graph.agent_loop.WriterAgent", p["WriterAgent"]),
        patch("novel_agent.graph.agent_loop.EditorAgent", p["EditorAgent"]),
        patch("novel_agent.graph.agent_loop.ContinuityAgent", p["ContinuityAgent"]),
        patch("novel_agent.graph.agent_loop.WorldbuildingAgent", p["WorldbuildingAgent"]),
        patch("novel_agent.graph.agent_loop.StyleAnalyzer", p["StyleAnalyzer"]),
        patch("novel_agent.graph.agent_loop.QualityService.check_draft_hard_gates", p["gate"]),
        patch("novel_agent.graph.agent_loop._config_for", p["_config_for"]),
        patch("novel_agent.graph.agent_loop._get_chapter_store", p["_get_chapter_store"]),
        patch("novel_agent.graph.agent_loop.StyleCheckTool", MagicMock()),
        patch("novel_agent.graph.agent_loop.HumanizeTool", MagicMock()),
    ):
        return asyncio.run(run_agent_loop(state, max_rounds=2, max_retries=2))


def test_hard_gate_pass_skips_editor_and_continuity():
    p = _make_patches(draft=_long_draft(), gate_passed=True)
    result = _run(
        {
            "chapter_number": 1,
            "chapter_outline": "大纲",
            "target_chapter_words": 600,
            "persist_dir": "/tmp/na-s1",
        },
        p,
    )

    assert result["quality_gate_passed"] is True
    assert result["editor_skipped"] is True
    assert result["continuity_skipped"] is True
    assert result["editor_report"].get("skipped") is True
    assert result["continuity_report"].get("skipped") is True
    assert p["EditorAgent"].call_count == 0
    assert p["ContinuityAgent"].call_count == 0
    assert p["writer"].write_with_loop.await_count == 1
    assert p["editor"].review.await_count == 0
    assert p["continuity"].audit.await_count == 0


def test_hard_gate_fail_invokes_editor_and_continuity():
    p = _make_patches(draft="短稿", gate_passed=False, editor_verdict="accept")
    result = _run(
        {
            "chapter_number": 2,
            "chapter_outline": "大纲",
            "target_chapter_words": 3000,
            "persist_dir": "/tmp/na-s1",
        },
        p,
    )

    assert result["quality_gate_passed"] is False
    assert result["editor_skipped"] is False
    assert result["continuity_skipped"] is False
    assert p["EditorAgent"].call_count == 1
    assert p["ContinuityAgent"].call_count == 1
    assert p["editor"].review.await_count == 1
    assert p["continuity"].audit.await_count == 1
    assert result["editor_report"]["verdict"] == "accept"
    assert result["continuity_report"]["overall_score"] == 80


def test_empty_writer_output_raises():
    p = _make_patches(draft="", gate_passed=False)
    with pytest.raises(EmptyDraftError, match="empty draft"):
        _run(
            {
                "chapter_number": 3,
                "chapter_outline": "大纲",
                "target_chapter_words": 600,
                "persist_dir": "/tmp/na-s1",
            },
            p,
        )
    # Must fail before Editor is constructed
    assert p["EditorAgent"].call_count == 0


def test_revision_feedback_passed_to_writer():
    p = _make_patches(draft=_long_draft(), gate_passed=True)
    result = _run(
        {
            "chapter_number": 4,
            "chapter_outline": "大纲",
            "target_chapter_words": 600,
            "persist_dir": "/tmp/na-s1",
            "revision_feedback": "请加强冲突",
        },
        p,
    )

    assert result["revision_feedback_consumed"] is True
    assert p["writer"].write_with_loop.await_args.kwargs.get("revision_feedback") == "请加强冲突"


def test_worldbuilding_uses_get_all_world_entities():
    p = _make_patches(draft=_long_draft(), gate_passed=True)
    mgr = MagicMock()
    mgr.get_recent_chapters.return_value = []
    mgr.count_chapters.return_value = 0
    mgr.get_relevant_foreshadowings.return_value = []
    mgr.get_all_world_entities.return_value = [
        {"name": "旧实体", "entity_type": "character"}
    ]

    with (
        patch("novel_agent.graph.agent_loop.OrchestratorAgent", p["OrchestratorAgent"]),
        patch("novel_agent.graph.agent_loop.WriterAgent", p["WriterAgent"]),
        patch("novel_agent.graph.agent_loop.EditorAgent", p["EditorAgent"]),
        patch("novel_agent.graph.agent_loop.ContinuityAgent", p["ContinuityAgent"]),
        patch("novel_agent.graph.agent_loop.WorldbuildingAgent", p["WorldbuildingAgent"]),
        patch("novel_agent.graph.agent_loop.StyleAnalyzer", p["StyleAnalyzer"]),
        patch("novel_agent.graph.agent_loop.QualityService.check_draft_hard_gates", p["gate"]),
        patch("novel_agent.graph.agent_loop._config_for", p["_config_for"]),
        patch("novel_agent.graph.agent_loop._get_chapter_store", p["_get_chapter_store"]),
        patch("novel_agent.graph.agent_loop.StyleCheckTool", MagicMock()),
        patch("novel_agent.graph.agent_loop.HumanizeTool", MagicMock()),
        patch("novel_agent.storage.manager.ProjectManager", return_value=mgr),
    ):
        result = asyncio.run(
            run_agent_loop(
                {
                    "chapter_number": 5,
                    "chapter_outline": "大纲",
                    "target_chapter_words": 600,
                    "persist_dir": "/tmp/na-s1",
                    "project_id": "proj-1",
                },
                max_rounds=2,
            )
        )

    mgr.get_all_world_entities.assert_called_once_with("proj-1")
    assert p["WorldbuildingAgent"].call_args.kwargs["existing_entities"] == [
        {"name": "旧实体", "entity_type": "character"}
    ]
    assert result["worldbuilding_report"]["new_entities"][0]["name"] == "甲"


def test_worldbuilding_zero_entities_surfaces_warning():
    p = _make_patches(draft=_long_draft(), gate_passed=True)
    p["wb"].extract = AsyncMock(
        return_value=(
            {
                "new_entities": [],
                "updated_entities": [],
                "conflicts": [],
                "foreshadowings": [],
            },
            MagicMock(),
        )
    )
    result = _run(
        {
            "chapter_number": 6,
            "chapter_outline": "大纲",
            "target_chapter_words": 600,
            "persist_dir": "/tmp/na-s1",
        },
        p,
    )

    assert result["worldbuilding_warnings"]
    assert "0 entities" in result["worldbuilding_warnings"][0]
    assert result["worldbuilding_report"].get("zero_entities_warning")
