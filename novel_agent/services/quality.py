"""Deterministic quality checks that do not require an LLM call."""

import re


class QualityService:
    """Evaluate hard quality constraints without side effects."""

    @staticmethod
    def text_units(text: str) -> int:
        if not text:
            return 0
        cjk = len(re.findall(r"[\u3400-\u9fff]", text))
        if cjk >= len(text) * 0.2:
            return cjk
        return len(re.findall(r"\b[\w']+\b", text))

    @classmethod
    def check_draft_hard_gates(
        cls,
        content: str,
        *,
        target_words: int,
        chapter_outline: str = "",
    ) -> dict:
        """Check non-negotiable draft requirements before LLM scoring.

        The hard gate rejects only grossly short content (< 50% of target);
        the narrative-extension layer handles the 50%–100% gap.
        """
        units = cls.text_units(content.strip())
        minimum = max(600, int(target_words * 0.5)) if target_words else 1
        violations: list[str] = []
        if not content.strip():
            violations.append("empty_content")
        if units < minimum:
            violations.append("minimum_length")
        if not chapter_outline.strip():
            violations.append("missing_chapter_outline")
        return {
            "passed": not violations,
            "violations": violations,
            "content_units": units,
            "minimum_units": minimum,
            "target_units": target_words,
        }
