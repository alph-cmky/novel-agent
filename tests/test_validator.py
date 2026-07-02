"""Tests for OutputValidator 3-layer validation."""

from novel_agent.schema.models import (
    ContinuityReport,
    EditorReport,
    OrchestratorReport,
    WorldbuildingReport,
)
from novel_agent.schema.validator import OutputValidator


class TestValidInput:
    def test_valid_orchestrator(self):
        raw = {
            "narrative_stage": "development",
            "stage_analysis": "middle arc",
            "chapter_strategy": {
                "primary_storyline": "main plot",
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
            "rhythm_score": 80,
            "dialogue_score": 85,
            "logic_score": 90,
            "writing_quality_score": 82,
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
