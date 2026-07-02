"""Output validation with 3-layer strategy: parse → validate → fallback.

Layer 1: Parse raw dict/JSON into the target Pydantic model.
Layer 2: Validate required fields and types, collect errors.
Layer 3: On failure, return sensible defaults + error metadata (never crash).
"""

from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from novel_agent.schema.models import (
    ContinuityReport,
    EditorReport,
    OrchestratorReport,
    WorldbuildingReport,
)


@dataclass
class ValidationResult:
    """Result of validating an agent output."""

    valid: bool
    data: OrchestratorReport | EditorReport | ContinuityReport | WorldbuildingReport
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        if isinstance(self.data, OrchestratorReport):
            return self.data.model_dump()
        if isinstance(self.data, EditorReport):
            return self.data.model_dump()
        if isinstance(self.data, ContinuityReport):
            return self.data.model_dump()
        return self.data.model_dump()


class OutputValidator:
    """Validates raw agent outputs against their Pydantic schemas.

    Three-layer strategy:
    1. Try direct Pydantic construction
    2. Coerce common type mismatches and retry
    3. Return default instance with captured errors
    """

    MODEL_MAP = {
        "orchestrator": OrchestratorReport,
        "editor": EditorReport,
        "continuity": ContinuityReport,
        "worldbuilding": WorldbuildingReport,
    }

    @staticmethod
    def validate(
        agent_type: str,
        raw: dict[str, Any] | None,
    ) -> ValidationResult:
        """Validate raw dict against the target agent schema.

        Args:
            agent_type: One of orchestrator, editor, continuity, worldbuilding.
            raw: The raw dict from agent output (may be None or malformed).

        Returns:
            ValidationResult with valid flag, parsed data, and any errors.
        """
        model_cls = OutputValidator.MODEL_MAP.get(agent_type)
        if model_cls is None:
            return ValidationResult(
                valid=False,
                data=OrchestratorReport(),
                errors=[f"Unknown agent_type: {agent_type}"],
            )

        if raw is None:
            return ValidationResult(
                valid=False,
                data=model_cls(),
                errors=["Agent returned None output"],
            )

        if not isinstance(raw, dict):
            return ValidationResult(
                valid=False,
                data=model_cls(),
                errors=[f"Expected dict, got {type(raw).__name__}"],
            )

        # Layer 1: direct parse
        try:
            instance = model_cls(**raw)
            return ValidationResult(valid=True, data=instance)
        except ValidationError as e:
            layer1_errors = [f"{err['loc']}: {err['msg']}" for err in e.errors()]

        # Layer 2: coerce common issues and retry
        coerced = OutputValidator._coerce(raw, agent_type)
        try:
            instance = model_cls(**coerced)
            return ValidationResult(
                valid=True,
                data=instance,
                warnings=[f"Coerced fields: {layer1_errors}"],
            )
        except ValidationError:
            pass

        # Layer 3: return defaults with errors
        return ValidationResult(
            valid=False,
            data=model_cls(),
            errors=layer1_errors,
        )

    @staticmethod
    def _coerce(raw: dict, agent_type: str) -> dict:
        """Fix common LLM output mistakes before retrying validation."""
        fixed = dict(raw)

        # Ensure list fields are actually lists
        list_fields = {
            "orchestrator": [
                "key_scenes", "foreshadowings_to_address",
                "characters", "world_elements",
            ],
            "editor": [
                "issues", "banned_phrases", "cliches",
                "sentence_pattern_issues", "structural_issues",
            ],
            "continuity": ["inconsistencies"],
            "worldbuilding": [
                "new_entities", "conflicts",
                "chapter_events", "updated_entities",
            ],
        }

        for fname in list_fields.get(agent_type, []):
            if fname in fixed and not isinstance(fixed[fname], list):
                fixed[fname] = [fixed[fname]] if fixed[fname] else []

        # Coerce score fields to int
        score_fields = [
            "overall_score", "rhythm_score", "dialogue_score",
            "logic_score", "writing_quality_score",
        ]
        for fname in score_fields:
            if fname in fixed and not isinstance(fixed[fname], int):
                try:
                    fixed[fname] = int(fixed[fname])
                except (ValueError, TypeError):
                    fixed[fname] = 0

        # Coerce nested objects
        if agent_type == "editor" and "ai_flavor" in fixed:
            if not isinstance(fixed["ai_flavor"], dict):
                fixed["ai_flavor"] = {}
        if agent_type == "orchestrator":
            if "chapter_strategy" in fixed and not isinstance(fixed["chapter_strategy"], dict):
                fixed["chapter_strategy"] = {}
            if "context_needed" in fixed and not isinstance(fixed["context_needed"], dict):
                fixed["context_needed"] = {}

        return fixed
