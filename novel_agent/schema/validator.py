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
from novel_agent.schema.parser import parse_json_response


@dataclass
class ValidationResult:
    """Result of validating an agent output."""

    valid: bool
    data: OrchestratorReport | EditorReport | ContinuityReport | WorldbuildingReport
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        # exclude_none keeps output consistent with strip_none: a field the LLM
        # left null (or omitted) must not reappear as an explicit None, otherwise
        # downstream ``dict.get(key, {}).get(...)`` would crash again.
        return self.data.model_dump(exclude_none=True)


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
        """Fix common LLM output mistakes before retrying validation.

        Uses dotted paths so nested fields (e.g. ``chapter_strategy.key_scenes``,
        ``dimensions.rhythm``) are corrected in place, not just top-level fields.
        """
        fixed = dict(raw)

        list_paths = {
            "orchestrator": [
                "chapter_strategy.key_scenes",
                "chapter_strategy.foreshadowings_to_address",
                "chapter_strategy.storylines",
                "chapter_strategy.character_arcs",
                "chapter_strategy.foreshadowing_management",
                "context_needed.characters",
                "context_needed.world_elements",
                "context_needed.cross_timeline_references",
            ],
            "editor": [
                "issues",
                "banned_phrases",
                "cliches",
                "sentence_pattern_issues",
                "structural_issues",
                "highlights",
            ],
            "continuity": ["inconsistencies"],
            "worldbuilding": [
                "new_entities",
                "conflicts",
                "chapter_events",
                "updated_entities",
                "foreshadowings",
                "resolved_foreshadowings",
            ],
        }

        int_paths = [
            "overall_score",
            "dimensions.consistency",
            "dimensions.writing",
            "dimensions.ai_flavor",
            "dimensions.dialogue",
            "dimensions.plot",
            "dimensions.instruction",
            "dimensions.creativity",
            "dimensions.controllability",
            "chapter_strategy.suggested_chapter_words",
        ]

        def _set(d: dict, path: str, fn) -> None:
            keys = path.split(".")
            for k in keys[:-1]:
                nxt = d.get(k)
                if not isinstance(nxt, dict):
                    return  # intermediate structure missing — Pydantic will default it
                d = nxt
            leaf = keys[-1]
            if leaf in d and d[leaf] is not None:
                d[leaf] = fn(d[leaf])

        for path in list_paths.get(agent_type, []):
            _set(
                fixed,
                path,
                lambda v: ([v] if v else []) if not isinstance(v, list) else v,
            )

        for path in int_paths:
            _set(fixed, path, OutputValidator._coerce_int)

        # Coerce nested objects to dict
        if agent_type == "editor" and "ai_flavor" in fixed:
            if not isinstance(fixed["ai_flavor"], dict):
                fixed["ai_flavor"] = {}
        if agent_type == "orchestrator":
            if "chapter_strategy" in fixed and not isinstance(fixed["chapter_strategy"], dict):
                fixed["chapter_strategy"] = {}
            if "context_needed" in fixed and not isinstance(fixed["context_needed"], dict):
                fixed["context_needed"] = {}

        return fixed

    @staticmethod
    def _coerce_int(v: Any) -> int:
        if isinstance(v, int) and not isinstance(v, bool):
            return v
        try:
            return int(v)
        except (ValueError, TypeError):
            return 0


def parse_validated(
    agent_type: str,
    text: str,
    defaults: dict | None = None,
) -> dict:
    """Parse raw LLM output, then coerce field types via OutputValidator.

    Combines the two normalization boundaries:
    1. ``parse_json_response`` strips nested ``null`` values.
    2. ``OutputValidator.validate`` coerces types (scores → int, items → list)
       and fills Pydantic defaults for the target agent schema.

    Returns a plain dict (None fields dropped) safe to consume downstream.
    """
    raw = parse_json_response(text, defaults)
    return OutputValidator.validate(agent_type, raw).to_dict()
