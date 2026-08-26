"""Build an auditable, bounded context packet for chapter agents."""

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

from novel_agent.services.continuity import ContinuityService


def estimate_tokens(text: str) -> int:
    """Rough token estimation: Chinese ~1.5 char/token, other ~4 char/token."""
    if not text:
        return 0
    cjk = len(re.findall(r"[\u3400-\u9fff\u3000-\u303f\uff00-\uffef]", text))
    other = len(text) - cjk
    return int(cjk / 1.5 + other / 4)


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
    ) -> ContextPacket:
        snapshot = self.manager.get_canon_snapshot(snapshot_id) if snapshot_id else None
        if snapshot:
            context = self.manager.build_context_from_snapshot(
                snapshot, chapter_number, max_recent_chapters=self.recent_chapters
            )
            payload = snapshot["payload"]
            foreshadowings = payload.get("foreshadowings", [])
            events = payload.get("story_events", [])
        else:
            context = self.manager.build_context(
                project_id,
                chapter_number,
                max_recent_chapters=self.recent_chapters,
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
        character_context = self._bound(context.get("character_context", ""), section_budget)
        world_context = self._bound(context.get("world_context", ""), section_budget)
        recent_summary = self._bound(context.get("recent_summary", ""), section_budget)
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

    def compile_for_run(self, run_id: str) -> ContextPacket:
        run = self.manager.get_writing_run(run_id)
        if not run:
            raise ValueError("Run not found")
        return self.compile(
            run["project_id"],
            run["chapter_number"],
            snapshot_id=run.get("input_snapshot_id"),
        )

    def _bound(self, text: str, limit: int) -> str:
        return ContextCompiler.bound(text, limit)

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
    def for_orchestrator(packet: dict, budget_chars: int = 4000) -> dict:
        """Minimal context for Orchestrator planning.

        Planning needs storyline inputs: recent summaries, active characters,
        open foreshadowings and recent timeline facts. Full worldbuilding is
        not required to decide strategy — a tight excerpt keeps the plan
        consistent with canon at a fraction of the size (≈2-4K tokens).
        """
        return {
            "character_context": ContextCompiler.bound(
                packet.get("character_context", ""), budget_chars // 2
            ),
            "world_context": ContextCompiler.bound(
                packet.get("world_context", ""), budget_chars // 4
            ),
            "recent_summary": ContextCompiler.bound(
                packet.get("recent_summary", ""), budget_chars // 2
            ),
            "unresolved_foreshadowings": (packet.get("unresolved_foreshadowings") or [])[:10],
            "timeline_events": (packet.get("timeline_events") or [])[-8:],
            "timeline_findings": (packet.get("timeline_findings") or [])[:5],
        }

    @staticmethod
    def for_writer(packet: dict, budget_chars: int = 5000) -> dict:
        """Minimal context for Writer: chars + summary + foreshadowings + events.

        world_context is explicitly empty — Writer does not need full worldbuilding.
        All keys are present so Writer can read them without nil-checks.
        """
        char_budget = budget_chars // 3
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
    def for_extension(packet: dict, budget_chars: int = 1500) -> dict:
        """Minimal context for Narrative Extension: active chars + top foreshadowings only.

        No world_context, no recent_summary, no timeline — only what the
        continuation needs to stay on-plot.
        """
        return {
            "character_context": ContextCompiler.bound(
                packet.get("character_context", ""), budget_chars
            ),
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
    def for_continuity(packet: dict, budget_chars: int = 4000) -> dict:
        """Minimal context for Continuity: structured events + findings + foreshadowings."""
        return {
            "timeline_events": (packet.get("timeline_events") or [])[-10:],
            "timeline_findings": (packet.get("timeline_findings") or [])[:5],
            "unresolved_foreshadowings": (packet.get("unresolved_foreshadowings") or [])[:8],
            "character_context": ContextCompiler.bound(
                packet.get("character_context", ""), budget_chars // 3
            ),
        }

    @staticmethod
    def for_evolution(
        current_scores: dict,
        previous_scores: dict | None = None,
        delta: dict | None = None,
        guard_report: dict | None = None,
        improvement_plan: dict | None = None,
        budget_chars: int = 3000,
    ) -> dict:
        """Minimal context for Evolution: metrics + deltas + violations + plan only."""
        return {
            "current_scores": current_scores,
            "previous_scores": previous_scores or {},
            "delta": delta or {},
            "guard_violations": (guard_report or {}).get("violations", []),
            "improvement_plan": improvement_plan or {},
        }

    @staticmethod
    def apply_context_needed(packet: dict, context_needed: dict) -> dict:
        """Fold the Orchestrator's context declaration into the packet.

        context_needed is the Orchestrator → ContextCompiler demand signal:
        characters and world elements this chapter touches, POV-scoped
        knowledge, cross-timeline references and recent plot points the
        Writer must see. ContextCompiler owns packet shaping — hints are
        merged here, once, and every downstream projection consumes the
        enriched packet.
        """
        packet = dict(packet)
        chars = ", ".join(context_needed.get("characters", []))
        world = ", ".join(context_needed.get("world_elements", []))

        char_ctx = packet.get("character_context", "")
        if chars:
            hint = f"[本章涉及角色: {chars}]"
            char_ctx = f"{char_ctx}\n{hint}" if char_ctx else hint

        world_ctx = packet.get("world_context", "")
        if world:
            hint = f"[本章涉及设定: {world}]"
            world_ctx = f"{world_ctx}\n{hint}" if world_ctx else hint

        persp_specific = context_needed.get("perspective_specific", "")
        if persp_specific:
            char_ctx = (
                f"{char_ctx}\n[视角特定信息: {persp_specific}]"
                if char_ctx
                else f"[视角特定信息: {persp_specific}]"
            )

        cross_timeline = context_needed.get("cross_timeline_references", [])
        if cross_timeline:
            hint = f"[跨时间线参考: {', '.join(cross_timeline)}]"
            world_ctx = f"{world_ctx}\n{hint}" if world_ctx else hint

        recent_ref = context_needed.get("recent_reference", "")
        recent_sum = packet.get("recent_summary", "")
        if recent_ref:
            hint = f"[主编提示：本章需要回顾 — {recent_ref}]"
            recent_sum = f"{recent_sum}\n{hint}" if recent_sum else hint

        # Write back only what a demand signal touched — an empty declaration
        # must not inject new keys (or placeholder noise) into the packet.
        if chars or persp_specific:
            packet["character_context"] = char_ctx
        if world or cross_timeline:
            packet["world_context"] = world_ctx
        if recent_ref:
            packet["recent_summary"] = recent_sum
        return packet

    @staticmethod
    def context_metrics(context: dict, budget_tokens: int = 3500) -> dict:
        """Code-level statistics for logging/debugging. No LLM call.

        All values are approximate — do not use for billing. ``budget_tokens``
        is the target ceiling for this task's context.
        """
        text = json.dumps(context, ensure_ascii=False, default=str)
        chars = len(text)
        tokens = estimate_tokens(text)
        return {
            "context_chars": chars,
            "estimated_tokens": tokens,
            "budget_tokens": budget_tokens,
            "utilization": round(tokens / max(budget_tokens, 1), 2),
        }
