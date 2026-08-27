"""Story length configuration — constants for long-form novel writing."""

import os
from collections.abc import Mapping
from dataclasses import dataclass

# 长篇默认配置（短篇/中篇已移除，只保留长篇）
DEFAULT_CHAPTER_WORDS = 3000
DEFAULT_MAX_TOKENS = 4096
# reasoning 模型（如 step-3.7-flash）的推理深度；低档压 reasoning 预算，
# 防止其 max_tokens 里的推理 token 挤空正文 content（实测偶发只出 123 字）
REASONING_EFFORT = "low"
NARRATIVE_PACING = (
    "渐进展开，intro充分铺垫(前5-10%)，development多线并进，climax在70-80%处，伏笔长线回收"
)
TYPICAL_CHAPTERS = "100章以上"

# ── Style analysis thresholds ──────────────────────────
# Paragraph structure — configurable, not scattered as magic numbers.
SHORT_NARRATIVE_PARAGRAPH_THRESHOLD = 40
SINGLE_SENTENCE_RATIO_THRESHOLD = 0.30
SHORT_NARRATIVE_RATIO_THRESHOLD = 0.40
MAX_CONSECUTIVE_SHORT_PARAGRAPHS = 3
# Descriptive-evidence cutoff: narrative single sentences up to this length
# are reported as "kinetic beats" (动作节拍). Beats carry NO score penalty —
# classification into beat vs fragmented narration is semantic, so the count
# is evidence for Editor/humans, never an automatic deduction.
KINETIC_BEAT_CHAR_LIMIT = 12


def env_bool(name: str, default: bool = False) -> bool:
    """将环境变量解析为布尔。接受 1/true/yes/on（大小写不敏感）。"""
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class ExecutionProfile:
    """Resolved execution switches for one chapter run."""

    skip_reviews: bool = False
    review_interval: int = 1
    skip_worldbuilding: bool = False
    skip_evolution_enrichment: bool = False

    @classmethod
    def from_state(cls, state: Mapping[str, object]) -> "ExecutionProfile":
        return cls(
            skip_reviews=bool(state.get("skip_reviews", False)),
            review_interval=max(int(state.get("review_interval", 1) or 1), 1),
            skip_worldbuilding=bool(state.get("skip_worldbuilding", False)),
            skip_evolution_enrichment=bool(state.get("skip_evolution_enrichment", False)),
        )

    def should_review(self, chapter_number: int) -> bool:
        return not self.skip_reviews and chapter_number % self.review_interval == 0

    def should_worldbuild(self) -> bool:
        return not self.skip_worldbuilding

    def should_enrich_evolution(self) -> bool:
        return not self.skip_evolution_enrichment
