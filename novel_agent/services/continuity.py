"""Deterministic continuity checks over structured story events and plot threads."""

import re

_ZH_DEATH = re.compile(r"死亡|死去|战死|身死|毙命")
_EN_DEATH = re.compile(r"(?<![a-z])(?:died|dies)(?![a-z])")
_DEATH_PREFIXES = (
    "不",
    "未",
    "没有",
    "不曾",
    "不是",
    "讨论",
    "关于",
    "提及",
    "仿佛",
    "好像",
    "not",
)


def _negated(text: str, start: int) -> bool:
    prefix = text[max(0, start - 4) : start].rstrip()
    return any(prefix.endswith(item) for item in _DEATH_PREFIXES)


def is_death_action(action: str) -> bool:
    """True when the action states a death, not when it merely mentions the word."""
    text = str(action or "").lower()
    for match in _EN_DEATH.finditer(text):
        if _negated(text, match.start()):
            continue
        return True
    for match in _ZH_DEATH.finditer(text):
        if _negated(text, match.start()):
            continue
        return True
    return False


def _is_name_char(char: str) -> bool:
    return char.isalpha() or "\u3400" <= char <= "\u9fff"


def subject_mentioned(subject: str, text: str, known_names: set[str] | None = None) -> bool:
    """True when the name stands on its own, not inside a longer name or word.

    A one-character name only counts beside a boundary. A longer name does not
    count when a known longer name covers that same span, such as 「林」 inside 「林风」.
    """
    if not subject or not text or subject not in text:
        return False
    names = known_names or set()
    start = 0
    while True:
        index = text.find(subject, start)
        if index < 0:
            return False
        if _mention_stands_alone(subject, text, index, names):
            return True
        start = index + len(subject)
    return False


def _mention_stands_alone(subject: str, text: str, index: int, known_names: set[str]) -> bool:
    for name in known_names:
        if len(name) <= len(subject) or subject not in name:
            continue
        offset = 0
        while True:
            found = name.find(subject, offset)
            if found < 0:
                break
            begin = index - found
            if begin >= 0 and text.startswith(name, begin):
                return False
            offset = found + 1
    if len(subject) == 1:
        before = text[index - 1] if index else ""
        after = text[index + 1] if index + 1 < len(text) else ""
        if (before and _is_name_char(before)) or (after and _is_name_char(after)):
            return False
    return True


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
            if subject and is_death_action(action):
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

    @staticmethod
    def check_draft_against_timeline(
        events: list[dict],
        draft: str,
        *,
        current_chapter: int,
    ) -> dict:
        """Critical when this draft names a character who already died.

        Only confirmed deaths count. A mention such as 「讨论死亡」 does not.
        Findings already recorded on earlier chapters do not open the review arm.
        """
        deaths: dict[str, int] = {}
        for event in events:
            subject = str(event.get("subject", "")).strip()
            chapter = int(event.get("chapter_number", 0) or 0)
            if not subject or chapter >= current_chapter or chapter <= 0:
                continue
            if is_death_action(str(event.get("action", ""))):
                deaths[subject] = min(deaths.get(subject, chapter), chapter)

        findings: list[dict] = []
        text = draft or ""
        known_names = {str(event.get("subject") or "").strip() for event in events}
        known_names.discard("")
        for subject, death_chapter in deaths.items():
            if subject_mentioned(subject, text, known_names):
                findings.append(
                    {
                        "type": "dead_character_reappeared",
                        "severity": "critical",
                        "subject": subject,
                        "death_chapter": death_chapter,
                        "appearance_chapter": current_chapter,
                    }
                )
        return {
            "passed": not findings,
            "findings": findings,
        }
