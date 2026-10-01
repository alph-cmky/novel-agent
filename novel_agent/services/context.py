"""Build an auditable context packet for chapter agents."""

from dataclasses import asdict, dataclass
from typing import Any

from novel_agent.services.continuity import ContinuityService, is_death_action
from novel_agent.storage.manager import _property_text


@dataclass(frozen=True)
class ContextPacket:
    project_id: str
    chapter_number: int
    character_context: str
    world_context: str
    recent_summary: str
    unresolved_foreshadowings: list[str]
    timeline_events: list[dict[str, Any]]
    timeline_findings: list[dict[str, Any]]

    def to_state(self) -> dict[str, Any]:
        """Single context contract: the packet lives only under context_packet."""
        return {"context_packet": asdict(self)}


_RECENT_CHAPTERS = 3
_EXCERPT_CHARS = 400
_EVENT_WINDOW = 8
_PROPERTY_CHARS = 160
_FORESHADOW_LIMIT = 40
_CHARACTER_LIMIT = 16
_WORLD_ENTITY_LIMIT = 24


def _merge_context_lines(existing: str, lines: list[str]) -> str:
    """Keep compiled lines and add a named entry only when that name is absent."""
    kept = existing
    for line in lines:
        name = line.split(":", 1)[0].split("]")[-1].strip(" -")
        if name and name in kept:
            continue
        kept = f"{kept}\n{line}" if kept else line
    return kept


def _event_chapter(event: dict) -> int:
    try:
        return int(event.get("chapter_number") or 0)
    except (TypeError, ValueError):
        return 0


def _death_chapter(event: dict) -> int | None:
    subject = str(event.get("subject") or "").strip()
    if subject and is_death_action(str(event.get("action") or "")):
        return _event_chapter(event)
    return None


def _compress_events(events: list[dict], chapter_number: int) -> list[dict]:
    """Keep the recent window, real deaths, and events after those deaths.

    Metaphorical mentions such as 「讨论死亡」 are not deaths. Events from before
    a death stay out of the packet unless they fall inside the recent window.
    """
    prior = [event for event in events if _event_chapter(event) < chapter_number]
    died_at: dict[str, int] = {}
    for event in prior:
        chapter = _death_chapter(event)
        subject = str(event.get("subject") or "").strip()
        if chapter is None or not subject:
            continue
        died_at[subject] = min(died_at.get(subject, chapter), chapter)
    cutoff = chapter_number - _EVENT_WINDOW
    kept = []
    for event in prior:
        subject = str(event.get("subject") or "").strip()
        chapter = _event_chapter(event)
        after_death = subject in died_at and chapter >= died_at[subject]
        if chapter >= cutoff or after_death:
            kept.append(event)
    return kept


def closing_repeats(previous: str, draft: str) -> bool:
    """True when this chapter's closing window reuses the previous one."""
    previous_grams = _content_grams(_closing_window(previous))
    draft_grams = _content_grams(_closing_window(draft))
    if len(previous_grams) < 8 or not draft_grams:
        return False
    shared = previous_grams & draft_grams
    return len(shared) >= 8 and len(shared) / len(previous_grams) >= 0.25


def _closing_window(text: str, limit: int = 180) -> str:
    stripped = (text or "").strip()
    if len(stripped) <= limit:
        return stripped
    tail = stripped[-limit:]
    for index, char in enumerate(tail):
        if char not in "。！？\n":
            continue
        rest = tail[index + 1 :].strip()
        if len(rest) >= limit // 2:
            return rest
    return tail.strip()


def _content_grams(text: str, size: int = 4) -> set[str]:
    compact = "".join((text or "").split())
    grams: set[str] = set()
    for index in range(len(compact) - size + 1):
        gram = compact[index : index + size]
        if all("\u4e00" <= char <= "\u9fff" for char in gram):
            grams.add(gram)
    return grams


def format_timeline_event(event: dict | str) -> str:
    """One prompt line. Drop ids, evidence, and timestamps."""
    if not isinstance(event, dict):
        return f"- {event}"
    chapter = event.get("chapter_number", "?")
    subject = str(event.get("subject") or "")
    action = str(event.get("action") or event.get("description") or event.get("summary") or "")
    obj = str(event.get("object") or event.get("object_value") or "")
    location = str(event.get("location") or "")
    body = " ".join(part for part in (subject, action, obj) if part)
    if location:
        body = f"{body} @{location}".strip()
    return f"- [第{chapter}章] {body}".rstrip()


def format_timeline_events(events: list) -> str:
    return "\n".join(format_timeline_event(event) for event in events or [])


_SKIPPED_FINDING_TYPES = {"overdue_foreshadowing", "dormant_foreshadowing"}


def format_timeline_finding(finding: dict | str) -> str:
    """One warning line. Foreshadowing status is already in the open-thread list."""
    if not isinstance(finding, dict):
        return f"- {finding}"
    kind = str(finding.get("type") or "")
    if kind in _SKIPPED_FINDING_TYPES:
        return ""
    if kind == "dead_character_reappeared":
        subject = finding.get("subject") or "角色"
        death = finding.get("death_chapter", "?")
        return f"- {subject}已在第{death}章死亡，不要再让其出场"
    if kind == "event_order_violation":
        return "- 已记录事件的章节顺序不一致"
    subject = str(finding.get("subject") or finding.get("description") or "")
    if kind and subject:
        return f"- {kind}: {subject}"
    return f"- {subject or kind}" if subject or kind else ""


def format_timeline_findings(findings: list) -> str:
    lines = [format_timeline_finding(item) for item in findings or []]
    return "\n".join(line for line in lines if line)


def _planted_chapter(row: dict) -> int:
    try:
        return int(row.get("planted_chapter") or 0)
    except (TypeError, ValueError):
        return 0


def _event_text(events: list) -> str:
    parts: list[str] = []
    for event in events or []:
        if not isinstance(event, dict):
            parts.append(str(event))
            continue
        parts.append(str(event.get("subject") or ""))
        parts.append(str(event.get("action") or ""))
        parts.append(str(event.get("object") or event.get("object_value") or ""))
    return "\n".join(part for part in parts if part)


def _phrase_overlap(description: str, text: str, size: int = 4) -> bool:
    if len(description) < size or len(text) < size:
        return False
    for index in range(len(description) - size + 1):
        gram = description[index : index + size]
        if not any(ch.isalnum() or "\u4e00" <= ch <= "\u9fff" for ch in gram):
            continue
        if gram in text:
            return True
    return False


def _prefer_foreshadowings(rows: list[dict], limit: int | None, events: list) -> list[dict]:
    """Keep the earliest open threads, then ones that still touch recent events.

    A risk sort plus a newest-first limit drops the opening threads once later
    chapters plant enough new ones.
    """
    open_rows = [
        row
        for row in rows
        if str(row.get("status") or "open") in {"open", "planted", "hinted", "advanced"}
    ]
    ordered = sorted(open_rows, key=_planted_chapter)
    if limit is None or len(ordered) <= limit:
        return ordered
    chosen = ordered[: limit // 2]
    chosen_ids = {id(row) for row in chosen}
    text = _event_text(events)
    live = [
        row
        for row in ordered
        if id(row) not in chosen_ids and _phrase_overlap(str(row.get("description") or ""), text)
    ]
    live.sort(key=_planted_chapter, reverse=True)
    for row in live:
        if len(chosen) >= limit:
            break
        chosen.append(row)
        chosen_ids.add(id(row))
    for row in ordered:
        if len(chosen) >= limit:
            break
        if id(row) in chosen_ids:
            continue
        chosen.append(row)
    return chosen


def _mention_text(events: list[dict], foreshadowings: list[dict]) -> str:
    """Names the entity cap must keep: recent actions and open threads."""
    parts: list[str] = []
    for event in events:
        parts.append(str(event.get("subject") or ""))
        parts.append(str(event.get("action") or ""))
        parts.append(str(event.get("object") or ""))
    for item in foreshadowings:
        parts.append(str(item.get("description") or ""))
    return "\n".join(part for part in parts if part)


class ContextCompiler:
    """Compile structured memory plus a short recent-prose excerpt.

    Character and world rows are capped. Names in the recent excerpt stay.
    Earlier chapter drafts are not copied into the packet; only the last few
    chapters' endings are.
    """

    def __init__(self, manager):
        self.manager = manager

    def compile(
        self,
        project_id: str,
        chapter_number: int,
        snapshot_id: str | None = None,
        task: str = "full",
    ) -> ContextPacket:
        del task  # projections no longer shrink the packet by role
        snapshot = self.manager.get_canon_snapshot(snapshot_id) if snapshot_id else None
        if snapshot:
            payload = snapshot["payload"]
            events = _compress_events(payload.get("story_events", []), chapter_number)
            foreshadowings = _prefer_foreshadowings(
                payload.get("foreshadowings", []), _FORESHADOW_LIMIT, events
            )
            context = self.manager.build_context_from_snapshot(
                snapshot,
                chapter_number,
                max_recent_chapters=_RECENT_CHAPTERS,
                max_entities=_WORLD_ENTITY_LIMIT,
                max_characters=_CHARACTER_LIMIT,
                excerpt_chars=_EXCERPT_CHARS,
                property_chars=_PROPERTY_CHARS,
                mention_text=_mention_text(events, foreshadowings),
            )
        else:
            events = self.manager.get_context_story_events(
                project_id, chapter_number, window=_EVENT_WINDOW
            )
            foreshadowings = _prefer_foreshadowings(
                self.manager.get_relevant_foreshadowings(project_id, chapter_number, limit=None),
                _FORESHADOW_LIMIT,
                events,
            )
            context = self.manager.build_context(
                project_id,
                chapter_number,
                max_recent_chapters=_RECENT_CHAPTERS,
                max_entities=_WORLD_ENTITY_LIMIT,
                max_characters=_CHARACTER_LIMIT,
                excerpt_chars=_EXCERPT_CHARS,
                property_chars=_PROPERTY_CHARS,
                mention_text=_mention_text(events, foreshadowings),
            )
        timeline_findings = ContinuityService.check_timeline(
            events,
            foreshadowings,
            current_chapter=chapter_number,
        )["findings"]
        unresolved = []
        for item in foreshadowings:
            if item.get("status") not in {"open", "planted", "hinted", "advanced"}:
                continue
            text = (
                f"[第{item.get('planted_chapter', '?')}章] {item.get('description', '')}".strip()
            )
            if text:
                unresolved.append(text)
        return ContextPacket(
            project_id=project_id,
            chapter_number=chapter_number,
            character_context=context.get("character_context", ""),
            world_context=context.get("world_context", ""),
            recent_summary=context.get("recent_summary", ""),
            unresolved_foreshadowings=unresolved,
            timeline_events=list(events),
            timeline_findings=timeline_findings,
        )

    # ── Task projections ──────────────────────────────────
    # Copy the compiled packet. Roles do not get a separate quota.

    @staticmethod
    def project(packet: dict) -> dict:
        """Copy the compiled packet. Missing fields stay empty, present fields stay whole."""
        return {
            "character_context": packet.get("character_context") or "",
            "world_context": packet.get("world_context") or "",
            "recent_summary": packet.get("recent_summary") or "",
            "unresolved_foreshadowings": list(packet.get("unresolved_foreshadowings") or []),
            "timeline_events": list(packet.get("timeline_events") or []),
            "timeline_findings": list(packet.get("timeline_findings") or []),
        }

    @staticmethod
    def for_orchestrator(packet: dict) -> dict:
        return ContextCompiler.project(packet)

    @staticmethod
    def for_writer(packet: dict) -> dict:
        return ContextCompiler.project(packet)

    @staticmethod
    def for_editor(packet: dict) -> dict:
        return ContextCompiler.project(packet)

    @staticmethod
    def for_continuity(packet: dict) -> dict:
        return ContextCompiler.project(packet)

    def apply_context_needed(
        self,
        packet: dict,
        context_needed: dict,
        project_id: str,
        chapter_number: int,
    ) -> dict:
        """Fold the Orchestrator's context declaration into the packet via real retrieval.

        context_needed is the Orchestrator → ContextCompiler demand signal.
        Instead of appending text hints, this method queries Storage for the
        actual entities/events the chapter needs and replaces the packet's
        character_context / world_context / timeline_events with the results.

        perspective_specific and recent_reference remain as text annotations
        (POV metadata, not entity retrieval).
        """
        packet = dict(packet)

        # ── Character retrieval: query matching entities by name ──
        char_names = context_needed.get("characters", [])
        if char_names:
            chars = self.manager.get_entities_by_names(
                project_id, char_names, entity_type="character"
            )
            char_lines = [
                f"- {c['name']}: {_property_text(c.get('properties') or '', _PROPERTY_CHARS)}"
                for c in chars
                if c.get("name")
            ]
            if char_lines:
                packet["character_context"] = _merge_context_lines(
                    packet.get("character_context") or "", char_lines
                )

        # ── World element retrieval: query matching non-character entities ──
        world_names = context_needed.get("world_elements", [])
        if world_names:
            world_ents = self.manager.get_entities_by_names(project_id, world_names)
            # Exclude characters — they're handled above
            world_ents = [e for e in world_ents if e.get("entity_type") != "character"]
            world_lines = [
                f"- [{e['entity_type']}] {e['name']}: "
                f"{_property_text(e.get('properties') or '', _PROPERTY_CHARS)}"
                for e in world_ents
                if e.get("name")
            ]
            if world_lines:
                packet["world_context"] = _merge_context_lines(
                    packet.get("world_context") or "", world_lines
                )

        # ── Cross-timeline retrieval: query events by subject ──
        cross_timeline = context_needed.get("cross_timeline_references", [])
        if cross_timeline:
            cross_events = self.manager.get_story_events_by_subjects(project_id, cross_timeline)
            if cross_events:
                existing = list(packet.get("timeline_events") or [])
                # Merge deduplicated: avoid duplicating events already in packet
                existing_keys = {
                    e.get("id") or e.get("action", "") for e in existing if isinstance(e, dict)
                }
                added = []
                for ev in cross_events:
                    key = ev.get("id") or ev.get("action", "")
                    if key not in existing_keys:
                        added.append(ev)
                packet["timeline_events"] = existing + _compress_events(added, chapter_number)

        # ── POV metadata: text annotation (not entity retrieval) ──
        persp_specific = context_needed.get("perspective_specific", "")
        if persp_specific:
            char_ctx = packet.get("character_context", "")
            hint = f"[视角特定信息: {persp_specific}]"
            packet["character_context"] = f"{char_ctx}\n{hint}" if char_ctx else hint

        # ── Recent reference: text annotation (not entity retrieval) ──
        recent_ref = context_needed.get("recent_reference", "")
        if recent_ref:
            recent_sum = packet.get("recent_summary", "")
            hint = f"[本章需要回顾 — {recent_ref}]"
            packet["recent_summary"] = f"{recent_sum}\n{hint}" if recent_sum else hint

        return packet
