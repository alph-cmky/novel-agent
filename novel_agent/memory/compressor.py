"""Context Compressor — intelligent compression when token budget exceeded."""

import os
import re
from dataclasses import dataclass, field

from langchain_openai import ChatOpenAI


@dataclass
class CompressionStrategy:
    trigger_threshold: int = 40000  # tokens
    target_tokens: int = 20000  # after compression
    preserve_patterns: list[str] = field(default_factory=lambda: [
        r"伏笔|预示|暗示|以后会|将来",
        r"第.*?次.*?出现|第一.*?见到|初.*?登场",
        r"规则|设定|体系|能力|功法|修炼",
        r"关系|认识|结识|结盟|背叛",
    ])


def estimate_tokens(text: str) -> int:
    """Rough token estimation: Chinese ~1.5 char/token, English ~4 char/token."""
    chinese_chars = len(re.findall(r"[一-鿿]", text))
    other_chars = len(text) - chinese_chars
    return int(chinese_chars / 1.5 + other_chars / 4)


def extract_critical_snippets(text: str, patterns: list[str]) -> list[str]:
    """Extract sentences matching preserve patterns."""
    sentences = re.split(r"[。！？!?\n]", text)
    snippets = []
    for pattern in patterns:
        for sent in sentences:
            if re.search(pattern, sent):
                snippets.append(sent.strip())
    return snippets[:10]  # Limit to avoid token bloat


class ContextCompressor:
    """Compresses old chapter content into summaries for Recent Memory."""

    def __init__(self, strategy: CompressionStrategy | None = None):
        self.strategy = strategy or CompressionStrategy()
        self._model = ChatOpenAI(
            model=os.getenv("BUDGET_MODEL", "deepseek-chat"),
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            max_tokens=600,
            temperature=0.3,
        )

    def should_compress(self, context: dict[str, str]) -> bool:
        """Check if total context exceeds threshold."""
        total = sum(estimate_tokens(v) for v in context.values())
        return total > self.strategy.trigger_threshold

    async def compress(
        self,
        chapters: list[dict],
        recent_count: int = 3,
    ) -> dict[str, str]:
        """Compress chapter list into Recent Memory format.

        Args:
            chapters: List of {chapter_number, draft_content, ...}
            recent_count: Number of most recent chapters to keep full

        Returns:
            {"recent_summary": compressed summary, "critical_snippets": [...]}
        """
        if len(chapters) <= recent_count:
            return {
                "recent_summary": _build_simple_summary(chapters),
                "critical_snippets": "",
            }

        # Keep recent chapters full, compress older ones
        recent = chapters[-recent_count:]
        older = chapters[:-recent_count]

        # Extract critical info from older chapters
        all_text = "\n".join(
            c.get("draft_content", "")[:2000] for c in older
        )
        critical = extract_critical_snippets(
            all_text, self.strategy.preserve_patterns
        )

        # Build compressed summary
        summary = await self._llm_compress(older)

        # Recent chapters stay full
        recent_text = _build_simple_summary(recent)

        return {
            "recent_summary": f"{summary}\n\n## 最近章节\n{recent_text}",
            "critical_snippets": "\n".join(critical) if critical else "",
        }

    async def _llm_compress(self, chapters: list[dict]) -> str:
        """Use LLM to generate a concise summary of older chapters."""
        chapter_texts = []
        for c in chapters:
            cn = c.get("chapter_number", "?")
            draft = c.get("draft_content", "")[:1500]
            chapter_texts.append(f"第{cn}章:\n{draft}\n")

        prompt = (
            "请将以下章节压缩为简洁的摘要（中文，300字以内）。"
            "保留：关键事件、角色行为变化、新设定、伏笔。"
            "忽略：环境描写、战斗细节、日常对话。\n\n"
            + "\n".join(chapter_texts)
        )

        try:
            response = await self._model.ainvoke(prompt)
            return response.content or ""
        except Exception:
            return _build_simple_summary(chapters)


def _build_simple_summary(chapters: list[dict]) -> str:
    """Fallback: simple concatenation of chapter excerpts."""
    parts = []
    for c in chapters:
        draft = c.get("draft_content", "")
        cn = c.get("chapter_number", "?")
        if draft:
            excerpt = draft[:300] + ("..." if len(draft) > 300 else "")
            parts.append(f"第{cn}章: {excerpt}")
    return "\n\n".join(parts) if parts else ""
