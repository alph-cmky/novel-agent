"""Scene planning and assembly helpers for V2 chapter generation."""

import re
from typing import Any


def build_scene_plan(
    chapter_outline: str,
    target_words: int,
    strategy: dict | None = None,
    max_scenes: int = 4,
) -> list[dict[str, Any]]:
    """Build a small deterministic scene plan from strategy or outline text."""
    raw = (strategy or {}).get("chapter_strategy", {}).get("key_scenes", [])
    outlines: list[str] = []
    for item in raw:
        if isinstance(item, str) and item.strip():
            outlines.append(item.strip())
        elif isinstance(item, dict):
            text = item.get("description") or item.get("summary") or item.get("goal")
            if text:
                outlines.append(str(text).strip())
    if not outlines:
        outlines = [
            part.strip()
            for part in re.split(r"(?<=[。！？.!?])\s*|\n+", chapter_outline)
            if part.strip()
        ]
    if not outlines:
        outlines = [chapter_outline.strip() or "完成本章核心情节"]
    outlines = outlines[:max_scenes]
    base = max(target_words // len(outlines), 1)
    remainder = max(target_words - base * len(outlines), 0)
    return [
        {
            "scene_index": index,
            "outline": outline,
            "target_words": base + (1 if index <= remainder else 0),
        }
        for index, outline in enumerate(outlines, start=1)
    ]


def assemble_scenes(scene_drafts: list[str]) -> str:
    """Join scene drafts while preserving a clean chapter boundary."""
    return "\n\n".join(draft.strip() for draft in scene_drafts if draft.strip()).strip()


def build_scene_outcome(scene_content: str, max_chars: int = 400) -> str:
    """Extract a compact, deterministic scene ending for the next scene's context.

    Instead of dumping 1200 chars of raw text, this takes the last 1–2
    paragraphs (the "ending beat") and caps at ``max_chars``. No LLM call.
    """
    paragraphs = [p.strip() for p in scene_content.split("\n") if p.strip()]
    if not paragraphs:
        return ""
    ending = paragraphs[-1] if len(paragraphs) == 1 else "\n".join(paragraphs[-2:])
    if len(ending) > max_chars:
        ending = ending[-max_chars:]
    return f"[前场结局]\n{ending}"
