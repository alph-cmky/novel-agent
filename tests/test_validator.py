"""Tests for OutputValidator 3-layer validation."""

from novel_agent.schema.models import (
    ContinuityReport,
    EditorReport,
    OrchestratorReport,
    WorldbuildingReport,
)
from novel_agent.schema.validator import OutputValidator, parse_validated


class TestValidInput:
    def test_valid_orchestrator(self):
        raw = {
            "narrative_stage": "development",
            "stage_analysis": "middle arc",
            "chapter_strategy": {
                "pacing": "fast",
                "key_scenes": ["fight", "reveal"],
                "ending_type": "cliffhanger",
                "foreshadowings_to_address": [],
            },
            "context_needed": {
                "characters": ["林风"],
                "world_elements": [],
                "recent_reference": "",
            },
        }
        result = OutputValidator.validate("orchestrator", raw)
        assert result.valid is True
        assert isinstance(result.data, OrchestratorReport)
        assert result.data.narrative_stage == "development"

    def test_valid_editor(self):
        raw = {
            "overall_score": 85,
            "verdict": "pass",
            "issues": [
                {
                    "severity": "minor",
                    "category": "dialogue",
                    "description": "too verbose",
                    "suggestion": "shorten",
                    "phrase": "",
                    "location": "",
                }
            ],
            "ai_flavor": {
                "overall_score": 90,
                "banned_phrases": [],
                "cliches": [],
                "sentence_pattern_issues": [],
                "structural_issues": [],
            },
        }
        result = OutputValidator.validate("editor", raw)
        assert result.valid is True
        assert isinstance(result.data, EditorReport)
        assert result.data.overall_score == 85

    def test_valid_continuity(self):
        raw = {
            "overall_score": 90,
            "verdict": "pass",
            "inconsistencies": [],
        }
        result = OutputValidator.validate("continuity", raw)
        assert result.valid is True
        assert isinstance(result.data, ContinuityReport)

    def test_valid_worldbuilding(self):
        raw = {
            "new_entities": [
                {
                    "entity_type": "character",
                    "name": "林风",
                    "properties": {"age": 25},
                    "first_appearance_chapter": 1,
                    "relationships": [],
                }
            ],
            "conflicts": [],
            "chapter_events": [],
            "updated_entities": [],
        }
        result = OutputValidator.validate("worldbuilding", raw)
        assert result.valid is True
        assert isinstance(result.data, WorldbuildingReport)
        assert len(result.data.new_entities) == 1


class TestCoercion:
    def test_list_fields_coerced(self):
        raw = {
            "overall_score": 70,
            "verdict": "pass",
            "inconsistencies": {
                "severity": "minor",
                "description": "test",
            },  # single dict, should be coerced to list
        }
        result = OutputValidator.validate("continuity", raw)
        assert result.valid is True
        assert result.warnings != []

    def test_score_fields_coerced_to_int(self):
        raw = {
            "overall_score": "85",  # string, should be coerced
            "verdict": "pass",
            "inconsistencies": [],
        }
        result = OutputValidator.validate("continuity", raw)
        assert result.valid is True
        assert result.data.overall_score == 85

    def test_invalid_score_becomes_zero(self):
        raw = {
            "overall_score": "not_a_number",
            "verdict": "pass",
            "inconsistencies": [],
        }
        result = OutputValidator.validate("continuity", raw)
        assert result.data.overall_score == 0


class TestInvalidInput:
    def test_none_input(self):
        result = OutputValidator.validate("editor", None)
        assert result.valid is False
        assert "None" in result.errors[0]

    def test_unknown_agent_type(self):
        result = OutputValidator.validate("unknown_agent", {})
        assert result.valid is False
        assert "Unknown agent_type" in result.errors[0]

    def test_non_dict_input(self):
        result = OutputValidator.validate("editor", ["not", "a", "dict"])
        assert result.valid is False

    def test_empty_dict_gets_defaults(self):
        result = OutputValidator.validate("editor", {})
        assert isinstance(result.data, EditorReport)
        assert result.data.overall_score == 0


class TestParseValidated:
    def test_editor_coerces_score_string_to_int(self):
        text = (
            '{"overall_score": "85", "dimensions": {"rhythm": "70"},'
            ' "issues": [], "verdict": "pass"}'
        )
        report = parse_validated("editor", text)
        assert report["overall_score"] == 85
        assert report["dimensions"]["rhythm"] == 70

    def test_editor_coerces_single_issue_to_list(self):
        text = '{"overall_score": 80, "issues": {"description": "x"}, "verdict": "pass"}'
        report = parse_validated("editor", text)
        assert isinstance(report["issues"], list)
        assert len(report["issues"]) == 1

    def test_orchestrator_drops_none_fields(self):
        text = (
            '{"narrative_stage": "development", "chapter_strategy":'
            ' {"pacing": "fast", "tension_profile": null}}'
        )
        result = parse_validated("orchestrator", text)
        cs = result["chapter_strategy"]
        assert "tension_profile" not in cs
        assert cs["pacing"] == "fast"

    def test_orchestrator_coerces_nested_list(self):
        text = '{"chapter_strategy": {"key_scenes": "开篇冲突"}}'
        result = parse_validated("orchestrator", text)
        assert result["chapter_strategy"]["key_scenes"] == ["开篇冲突"]

    def test_continuity_keeps_aligned_fields(self):
        text = (
            '{"overall_score": 90, "inconsistencies": [{"category": "character",'
            ' "severity": "major", "current": "黑发", "previous": "金发",'
            ' "description": "发色矛盾", "suggestion": "统一发色"}]}'
        )
        report = parse_validated("continuity", text)
        inc = report["inconsistencies"][0]
        assert inc["category"] == "character"
        assert inc["current"] == "黑发"
        assert inc["previous"] == "金发"
        assert inc["suggestion"] == "统一发色"

    def test_worldbuilding_keeps_resolved_foreshadowings(self):
        text = '{"resolved_foreshadowings": [{"description": "伏笔X", "resolved_chapter": 3}]}'
        report = parse_validated("worldbuilding", text)
        assert report["resolved_foreshadowings"][0]["description"] == "伏笔X"

    def test_garbage_falls_back_to_defaults(self):
        report = parse_validated("worldbuilding", "not json", defaults={"new_entities": []})
        assert report["new_entities"] == []
