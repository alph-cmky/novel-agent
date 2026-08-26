"""Pydantic models for all agent outputs — 3-layer validation boundary."""

from typing import Any, TypedDict

from pydantic import BaseModel, Field

from novel_agent.schema.enums import EvolutionAction

# ── Orchestrator ──────────────────────────────────────


class ChapterStrategy(BaseModel):
    """Narrative strategy for a single chapter — orchestrator output.

    Fields are classified per the design doc into:
    - GLOBAL: always injected into Writer prompt
    - CONDITIONAL: injected only when narrative_mode matches
    - AUXILIARY: injected as [参考] hints
    """

    # ── Original fields (GLOBAL) ──
    pacing: str = "normal"
    key_scenes: list[str] = Field(default_factory=list)
    ending_type: str = "cliffhanger"
    foreshadowings_to_address: list[str] = Field(default_factory=list)
    suggested_chapter_words: int = 3000

    # ── Phase 1: Structural layer ──
    # CONDITIONAL: climax stage only
    climax_sequence: dict | None = None
    # AUXILIARY: boundary detection
    stage_boundary: dict | None = None
    # CONDITIONAL: unit_arc / hybrid mode
    unit_arc: dict | None = None
    # CONDITIONAL: multi_perspective / ensemble mode
    pov_config: dict | None = None
    # CONDITIONAL: non-linear modes
    time_structure: dict | None = None
    # CONDITIONAL: near ending
    ending_tone: dict | None = None
    # GLOBAL: storylines for multi-line works
    storylines: list[dict] = Field(default_factory=list)
    storyline_intersection: dict | None = None

    # ── Phase 3: Character + Experience layer ──
    character_arcs: list[dict] = Field(default_factory=list)
    character_emotional_state: dict = Field(default_factory=dict)
    tension_profile: dict | None = None
    foreshadowing_management: list[dict] = Field(default_factory=list)
    scene_composition: dict | None = None


class ContextNeeded(BaseModel):
    characters: list[str] = Field(default_factory=list)
    world_elements: list[str] = Field(default_factory=list)
    recent_reference: str = ""
    # Phase 1 additions
    cross_timeline_references: list[str] = Field(default_factory=list)
    perspective_specific: str = ""


class OrchestratorReport(BaseModel):
    narrative_stage: str = "development"
    stage_analysis: str = ""
    chapter_strategy: ChapterStrategy = Field(default_factory=ChapterStrategy)
    context_needed: ContextNeeded = Field(default_factory=ContextNeeded)


# ── Editor ────────────────────────────────────────────


class EditorIssue(BaseModel):
    severity: str = "minor"
    category: str = ""
    dimension: str = ""  # 对应 system prompt 中的 dimension 字段
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


class EditorDimensions(BaseModel):
    """Per-dimension scores matching Editor system prompt output."""

    rhythm: int = 0
    ai_flavor: int = 0
    dialogue: int = 0
    logic: int = 0
    writing: int = 0


class EditorReport(BaseModel):
    overall_score: int = 0
    verdict: str = ""
    dimensions: EditorDimensions = Field(default_factory=EditorDimensions)
    issues: list[EditorIssue] = Field(default_factory=list)
    highlights: list[str] = Field(default_factory=list)
    ai_flavor: AIFlavorReport = Field(default_factory=AIFlavorReport)


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
    # Existing entities may be reported by name for compatibility, or with
    # properties that should be merged into the stored entity.
    updated_entities: list[dict[str, Any] | str] = Field(default_factory=list)
    # Foreshadowing lifecycle — extracted by WorldbuildingAgent
    foreshadowings: list[dict[str, Any]] = Field(default_factory=list)
    resolved_foreshadowings: list[dict[str, Any]] = Field(default_factory=list)


class EvolutionCandidate(TypedDict, total=False):
    """Serializable candidate data carried by the evolution state."""

    version: int
    draft_content: str
    editor_report: dict[str, Any]
    continuity_report: dict[str, Any]
    worldbuilding_report: dict[str, Any]
    quality_guard_report: dict[str, Any]
    quality_gate_report: dict[str, Any]
    style_report: dict[str, Any]
    outline_coverage: float | None
    required_facts_missing: int
    scores: dict[str, Any]
    composite_score: float
    content_length: int


class EvolutionDecision(BaseModel):
    """Deterministic evolution decision consumed by the Graph."""

    action: EvolutionAction
    reason: str
    details: dict[str, Any] = Field(default_factory=dict)
