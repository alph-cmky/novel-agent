"""Build an auditable context packet for chapter agents."""

from dataclasses import asdict, dataclass
from typing import Any

from novel_agent.services.continuity import ContinuityService, is_death_action
from novel_agent.storage.manager import _excerpt


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


class ContextCompiler:
    """Compile structured memory plus a short recent-prose excerpt.

    Character and world text stay whole. Earlier chapter drafts are not copied
    into the packet; only the last few chapters' endings are.
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
            context = self.manager.build_context_from_snapshot(
                snapshot,
                chapter_number,
                max_recent_chapters=_RECENT_CHAPTERS,
                max_entities=None,
                excerpt_chars=_EXCERPT_CHARS,
                property_chars=_PROPERTY_CHARS,
            )
            payload = snapshot["payload"]
            foreshadowings = payload.get("foreshadowings", [])[:_FORESHADOW_LIMIT]
            events = _compress_events(payload.get("story_events", []), chapter_number)
        else:
            context = self.manager.build_context(
                project_id,
                chapter_number,
                max_recent_chapters=_RECENT_CHAPTERS,
                max_entities=None,
                excerpt_chars=_EXCERPT_CHARS,
                property_chars=_PROPERTY_CHARS,
            )
            foreshadowings = self.manager.get_relevant_foreshadowings(
                project_id, chapter_number, limit=_FORESHADOW_LIMIT
            )
            events = self.manager.get_context_story_events(
                project_id, chapter_number, window=_EVENT_WINDOW
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
            text = f"[第{item.get('planted_chapter', '?')}章] {item.get('description', '')}".strip()
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
                f"- {c['name']}: {_excerpt(str(c.get('properties') or ''), _PROPERTY_CHARS)}"
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
                f"{_excerpt(str(e.get('properties') or ''), _PROPERTY_CHARS)}"
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
