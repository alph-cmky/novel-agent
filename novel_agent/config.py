"""Story length configuration — enum and defaults for multi-length support.

短篇 short:  3-10 chapters, fast pacing, compact arcs
中篇 novella: 20-50 chapters, balanced arc
长篇 long:    100+ chapters, slow-burn development
"""

from dataclasses import dataclass
from enum import Enum


class StoryLength(Enum):
    SHORT = "short"
    NOVELLA = "novella"
    LONG = "long"


@dataclass
class LengthConfig:
    default_chapter_words: int
    max_tokens: int
    narrative_pacing: str  # injected into orchestrator prompt
    typical_chapters: str  # human-readable range


_LENGTH_DEFAULTS: dict[StoryLength, LengthConfig] = {
    StoryLength.SHORT: LengthConfig(
        default_chapter_words=1500,
        max_tokens=2048,
        narrative_pacing="快速推进，跳过intro直接进入development，3-5章内到达climax，章节少但每章有实质推进",
        typical_chapters="3-10章",
    ),
    StoryLength.NOVELLA: LengthConfig(
        default_chapter_words=3000,
        max_tokens=4096,
        narrative_pacing=(
            "平衡发展，intro 1-2章，development充分展开，"
            "climax在总章数60-70%处，resolution完整收尾"
        ),
        typical_chapters="20-50章",
    ),
    StoryLength.LONG: LengthConfig(
        default_chapter_words=3000,
        max_tokens=4096,
        narrative_pacing="渐进展开，intro充分铺垫(前5-10%)，development多线并进，climax在70-80%处，伏笔长线回收",
        typical_chapters="100章以上",
    ),
}


def get_length_config(story_length: str | StoryLength) -> LengthConfig:
    """Get defaults for a story length tier."""
    if isinstance(story_length, str):
        story_length = StoryLength(story_length)
    return _LENGTH_DEFAULTS[story_length]


def get_length_defaults(story_length: str | StoryLength) -> dict:
    """Get defaults as a plain dict (for JSON serialization / state passing)."""
    cfg = get_length_config(story_length)
    return {
        "story_length": (
            story_length.value
            if isinstance(story_length, StoryLength)
            else story_length
        ),
        "default_chapter_words": cfg.default_chapter_words,
        "max_tokens": cfg.max_tokens,
        "narrative_pacing": cfg.narrative_pacing,
        "typical_chapters": cfg.typical_chapters,
    }
