"""Build an auditable, bounded context packet for chapter agents."""

from dataclasses import asdict, dataclass
from typing import Any

from novel_agent.services.continuity import ContinuityService


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


class ContextCompiler:
    """Compile structured and recent project memory into one packet."""

    def __init__(
        self,
        manager,
        *,
        recent_chapters: int = 3,
        max_context_chars: int = 24000,
    ):
        self.manager = manager
        self.recent_chapters = max(recent_chapters, 0)
        self.max_context_chars = max(max_context_chars, 1)

    def compile(
        self,
        project_id: str,
        chapter_number: int,
        snapshot_id: str | None = None,
        task: str = "full",
    ) -> ContextPacket:
        # The run starts with an Orchestrator view. It only needs a bounded
        # canon sample; context_needed performs the precise retrieval later.
        entity_limit = 20 if task == "orchestrator" else None
        snapshot = self.manager.get_canon_snapshot(snapshot_id) if snapshot_id else None
        if snapshot:
            context = self.manager.build_context_from_snapshot(
                snapshot,
                chapter_number,
                max_recent_chapters=self.recent_chapters,
                max_entities=entity_limit,
            )
            payload = snapshot["payload"]
            foreshadowings = payload.get("foreshadowings", [])
            events = payload.get("story_events", [])
            if task == "orchestrator":
                foreshadowings = [
                    item
                    for item in foreshadowings
                    if item.get("status") in {"open", "planted", "hinted", "advanced"}
                ][:25]
                events = events[-90:]
        else:
            context = self.manager.build_context(
                project_id,
                chapter_number,
                max_recent_chapters=self.recent_chapters,
                max_entities=entity_limit,
            )
            # Task-aware retrieval: bounded, relevance-ranked reads —
            # no full foreshadowings/story_events table scan.
            foreshadowings = self.manager.get_relevant_foreshadowings(project_id, chapter_number)
            events = self.manager.get_relevant_story_events(project_id, chapter_number)
        timeline_findings = ContinuityService.check_timeline(
            events,
            foreshadowings,
            current_chapter=chapter_number,
        )["findings"]
        unresolved = [
            f"[第{item.get('planted_chapter', '?')}章] {item.get('description', '')}"
            for item in foreshadowings
            if item.get("status") in {"open", "planted", "hinted", "advanced"}
        ]
        section_budget = max(self.max_context_chars // 3, 1)
        character_context = ContextCompiler.bound(
            context.get("character_context", ""), section_budget
        )
        world_context = ContextCompiler.bound(context.get("world_context", ""), section_budget)
        recent_summary = ContextCompiler.bound(context.get("recent_summary", ""), section_budget)
        unresolved = [item for item in unresolved if item.strip()][:40]
        return ContextPacket(
            project_id=project_id,
            chapter_number=chapter_number,
            character_context=character_context,
            world_context=world_context,
            recent_summary=recent_summary,
            unresolved_foreshadowings=unresolved,
            timeline_events=events[-30:],
            timeline_findings=timeline_findings,
        )

    @staticmethod
    def bound(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        marker = "\n[context compacted]\n"
        available = max(limit - len(marker), 0)
        head = available // 2
        tail = available - head
        return text[:head] + marker + text[-tail:]

    # ── Task-aware projections ────────────────────────────

    @staticmethod
    def for_orchestrator(packet: dict) -> dict:
        """Minimal context for Orchestrator planning.

        Planning needs storyline inputs: recent summaries, active characters,
        open foreshadowings and recent timeline facts. Full worldbuilding is
        not required to decide strategy — a tight excerpt keeps the plan
        consistent with canon at a fraction of the size (≈2-4K tokens).
        """
        return {
            "character_context": ContextCompiler.bound(packet.get("character_context", ""), 2000),
            "world_context": ContextCompiler.bound(packet.get("world_context", ""), 1000),
            "recent_summary": ContextCompiler.bound(packet.get("recent_summary", ""), 2000),
            "unresolved_foreshadowings": (packet.get("unresolved_foreshadowings") or [])[:10],
            "timeline_events": (packet.get("timeline_events") or [])[-8:],
            "timeline_findings": (packet.get("timeline_findings") or [])[:5],
        }

    @staticmethod
    def for_writer(packet: dict) -> dict:
        """Minimal context for Writer: chars + summary + foreshadowings + events.

        world_context is explicitly empty — Writer does not need full worldbuilding.
        All keys are present so Writer can read them without nil-checks.
        """
        char_budget = 5000 // 3
        return {
            "character_context": ContextCompiler.bound(
                packet.get("character_context", ""), char_budget
            ),
            "world_context": "",
            "recent_summary": ContextCompiler.bound(packet.get("recent_summary", ""), char_budget),
            "unresolved_foreshadowings": (packet.get("unresolved_foreshadowings") or [])[:5],
            "timeline_events": (packet.get("timeline_events") or [])[-5:],
            "timeline_findings": (packet.get("timeline_findings") or [])[:3],
        }

    @staticmethod
    def for_extension(packet: dict) -> dict:
        """Minimal context for Narrative Extension: active chars + top foreshadowings only.

        No world_context, no recent_summary, no timeline — only what the
        continuation needs to stay on-plot.
        """
        return {
            "character_context": ContextCompiler.bound(packet.get("character_context", ""), 1500),
            "unresolved_foreshadowings": (packet.get("unresolved_foreshadowings") or [])[:3],
        }

    @staticmethod
    def for_editor(packet: dict, budget_chars: int = 5000) -> dict:
        """Minimal context for Editor: brief summary + active chars for consistency."""
        char_budget = budget_chars // 2
        return {
            "character_context": ContextCompiler.bound(
                packet.get("character_context", ""), char_budget
            ),
            "recent_summary": ContextCompiler.bound(packet.get("recent_summary", ""), char_budget),
            "unresolved_foreshadowings": (packet.get("unresolved_foreshadowings") or [])[:3],
        }

    @staticmethod
    def for_continuity(packet: dict) -> dict:
        """Minimal context for Continuity: structured events + findings + foreshadowings."""
        return {
            "timeline_events": (packet.get("timeline_events") or [])[-10:],
            "timeline_findings": (packet.get("timeline_findings") or [])[:5],
            "unresolved_foreshadowings": (packet.get("unresolved_foreshadowings") or [])[:8],
            "character_context": ContextCompiler.bound(
                packet.get("character_context", ""), 4000 // 3
            ),
        }

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
            char_lines = [f"- {c['name']}: {c['properties']}" for c in chars if c.get("name")]
            if char_lines:
                packet["character_context"] = "\n".join(char_lines)

        # ── World element retrieval: query matching non-character entities ──
        world_names = context_needed.get("world_elements", [])
        if world_names:
            world_ents = self.manager.get_entities_by_names(project_id, world_names)
            # Exclude characters — they're handled above
            world_ents = [e for e in world_ents if e.get("entity_type") != "character"]
            world_lines = [
                f"- [{e['entity_type']}] {e['name']}: {e['properties']}"
                for e in world_ents
                if e.get("name")
            ]
            if world_lines:
                packet["world_context"] = "\n".join(world_lines)

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
                for ev in cross_events:
                    key = ev.get("id") or ev.get("action", "")
                    if key not in existing_keys:
                        existing.append(ev)
                packet["timeline_events"] = existing[-30:]

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
