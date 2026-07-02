"""Pydantic models for all agent outputs — 3-layer validation boundary."""

from typing import Any

from pydantic import BaseModel, Field

# ── Orchestrator ──────────────────────────────────────


class ChapterStrategy(BaseModel):
    primary_storyline: str = ""
    pacing: str = "normal"
    key_scenes: list[str] = Field(default_factory=list)
    ending_type: str = "cliffhanger"
    foreshadowings_to_address: list[str] = Field(default_factory=list)
    suggested_chapter_words: int = 3000


class ContextNeeded(BaseModel):
    characters: list[str] = Field(default_factory=list)
    world_elements: list[str] = Field(default_factory=list)
    recent_reference: str = ""


class OrchestratorReport(BaseModel):
    narrative_stage: str = "development"
    stage_analysis: str = ""
    chapter_strategy: ChapterStrategy = Field(default_factory=ChapterStrategy)
    context_needed: ContextNeeded = Field(default_factory=ContextNeeded)


# ── Editor ────────────────────────────────────────────


class EditorIssue(BaseModel):
    severity: str = "minor"
    category: str = ""
    description: str = ""
    suggestion: str = ""
    phrase: str = ""
    location: str = ""


class AIFlavorReport(BaseModel):
    overall_score: int = 100
    banned_phrases: list[str] = Field(default_factory=list)
    cliches: list[str] = Field(default_factory=list)
    sentence_pattern_issues: list[str] = Field(default_factory=list)
    structural_issues: list[str] = Field(default_factory=list)


class EditorReport(BaseModel):
    overall_score: int = 0
    verdict: str = ""
    issues: list[EditorIssue] = Field(default_factory=list)
    ai_flavor: AIFlavorReport = Field(default_factory=AIFlavorReport)
    rhythm_score: int = 0
    dialogue_score: int = 0
    logic_score: int = 0
    writing_quality_score: int = 0


# ── Continuity ────────────────────────────────────────


class Inconsistency(BaseModel):
    severity: str = "minor"
    category: str = ""
    description: str = ""
    current: str = ""
    previous: str = ""
    suggestion: str = ""


class ContinuityReport(BaseModel):
    overall_score: int = 0
    verdict: str = ""
    inconsistencies: list[Inconsistency] = Field(default_factory=list)


# ── Worldbuilding ─────────────────────────────────────


class WorldEntity(BaseModel):
    entity_type: str = ""
    name: str = ""
    properties: dict[str, Any] = Field(default_factory=dict)
    first_appearance_chapter: int = 0
    relationships: list[dict[str, str]] = Field(default_factory=list)


class WorldbuildingConflict(BaseModel):
    existing_entity: str = ""
    conflict_type: str = ""
    description: str = ""
    new_info: str = ""
    existing_info: str = ""
    severity: str = "minor"


class WorldbuildingReport(BaseModel):
    new_entities: list[WorldEntity] = Field(default_factory=list)
    conflicts: list[WorldbuildingConflict] = Field(default_factory=list)
    chapter_events: list[str] = Field(default_factory=list)
    updated_entities: list[str] = Field(default_factory=list)
