"""Model router — classifies tasks and routes to budget or quality models.

Budget model (deepseek-chat): structural analysis, reviews, extraction.
Quality model (claude-sonnet-4): creative writing, dialogue, final prose.
"""

import os
from dataclasses import dataclass
from enum import Enum


class TaskClass(Enum):
    CREATIVE = "creative_writing"
    STRUCTURAL = "structural_analysis"
    REVIEW = "review"
    EXTRACTION = "extraction"


@dataclass
class RouteConfig:
    model: str
    temperature: float
    max_tokens: int = 4096


TASK_ROUTES: dict[TaskClass, tuple[str, float]] = {
    TaskClass.CREATIVE: ("QUALITY_MODEL", 0.85),
    TaskClass.STRUCTURAL: ("BUDGET_MODEL", 0.4),
    TaskClass.REVIEW: ("BUDGET_MODEL", 0.3),
    TaskClass.EXTRACTION: ("BUDGET_MODEL", 0.2),
}

FALLBACK_MODEL = "deepseek-chat"


class ModelRouter:
    """Routes tasks to the appropriate model based on task classification."""

    def __init__(self, budget_model: str | None = None, quality_model: str | None = None):
        self._budget_override = budget_model
        self._quality_override = quality_model

    def resolve(self, task: TaskClass) -> RouteConfig:
        env_key, temperature = TASK_ROUTES[task]
        budget = self._budget_override or os.getenv("BUDGET_MODEL", FALLBACK_MODEL)
        if "QUALITY" in env_key:
            model = self._quality_override or os.getenv("QUALITY_MODEL") or budget
        else:
            model = budget
        return RouteConfig(model=model, temperature=temperature)

    def route_for(self, agent_name: str) -> RouteConfig:
        mapping = {
            "writer": TaskClass.CREATIVE,
            "orchestrator": TaskClass.STRUCTURAL,
            "editor": TaskClass.REVIEW,
            "continuity": TaskClass.REVIEW,
            "worldbuilding": TaskClass.EXTRACTION,
        }
        return self.resolve(mapping.get(agent_name, TaskClass.STRUCTURAL))


router = ModelRouter()
