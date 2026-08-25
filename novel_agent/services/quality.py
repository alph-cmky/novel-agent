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

    @staticmethod
    def check_story_integrity(
        content: str,
        *,
        scene_plan: list[dict] | None = None,
        scene_drafts: list[str] | None = None,
        required_facts: list[str] | None = None,
        canon_conflicts: list[dict] | None = None,
    ) -> dict:
        """Run deterministic, evidence-producing story checks."""
        violations: list[str] = []
        findings: list[dict] = []
        plans = scene_plan or []
        drafts = scene_drafts or []
        if plans and len(plans) != len(drafts):
            violations.append("scene_count_mismatch")
            findings.append(
                {
                    "type": "scene_count_mismatch",
                    "planned": len(plans),
                    "generated": len(drafts),
                }
            )
        for index, draft in enumerate(drafts, start=1):
            if not draft.strip():
                violations.append(f"missing_scene:{index}")
                findings.append({"type": "missing_scene", "scene_index": index})

        for fact in required_facts or []:
            if fact and fact not in content:
                violations.append("required_fact_missing")
                findings.append({"type": "required_fact_missing", "fact": fact})

        for conflict in canon_conflicts or []:
            keywords = [str(item) for item in conflict.get("keywords", []) if item]
            if keywords and all(keyword in content for keyword in keywords):
                violations.append("canon_conflict_exposed")
                findings.append(
                    {
                        "type": "canon_conflict_exposed",
                        "severity": conflict.get("severity", "major"),
                        "keywords": keywords,
                    }
                )
        return {"passed": not violations, "violations": violations, "findings": findings}
