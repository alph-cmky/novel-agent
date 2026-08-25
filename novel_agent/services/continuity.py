"""Deterministic continuity checks over structured story events and plot threads."""


class ContinuityService:
    """Own deterministic continuity checks used by context compilation."""

    @staticmethod
    def check_timeline(
        events: list[dict],
        foreshadowings: list[dict],
        *,
        current_chapter: int,
        dormant_after: int = 3,
    ) -> dict:
        findings: list[dict] = []
        ordered = [event.get("chapter_number", 0) for event in events]
        if ordered != sorted(ordered):
            findings.append(
                {
                    "type": "event_order_violation",
                    "severity": "major",
                    "chapters": ordered,
                }
            )

        death_chapters: dict[str, int] = {}
        for event in events:
            subject = str(event.get("subject", "")).strip()
            action = str(event.get("action", "")).lower()
            chapter = int(event.get("chapter_number", 0) or 0)
            if subject and any(
                marker in action for marker in ("死亡", "死去", "战死", "died", "dies")
            ):
                death_chapters[subject] = min(death_chapters.get(subject, chapter), chapter)
            if subject and subject in death_chapters and chapter > death_chapters[subject]:
                findings.append(
                    {
                        "type": "dead_character_reappeared",
                        "severity": "critical",
                        "subject": subject,
                        "death_chapter": death_chapters[subject],
                        "appearance_chapter": chapter,
                    }
                )

        for item in foreshadowings:
            status = item.get("status", "")
            planted = int(item.get("planted_chapter", 0) or 0)
            expected = item.get("expected_resolve_chapter")
            if status in {"resolved", "abandoned"} or not planted:
                continue
            overdue = expected is not None and int(expected) < current_chapter
            dormant = expected is None and current_chapter - planted > dormant_after
            if overdue or dormant:
                findings.append(
                    {
                        "type": "overdue_foreshadowing" if overdue else "dormant_foreshadowing",
                        "severity": "major" if overdue else "warning",
                        "description": item.get("description", ""),
                        "planted_chapter": planted,
                        "expected_resolve_chapter": expected,
                    }
                )

        return {
            "passed": not any(f["severity"] == "critical" for f in findings),
            "findings": findings,
        }
