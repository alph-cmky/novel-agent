"""Chainlit Web UI for novel-agent — interactive chapter writing with HITL.

Usage:
    chainlit run novel_agent/api/app.py
"""

from pathlib import Path

import chainlit as cl
from langgraph.errors import GraphInterrupt
from langgraph.types import Command

from novel_agent.graph.chapter import build_chapter_graph
from novel_agent.storage.manager import ProjectManager

DEFAULT_PROJECT_DIR = Path.cwd() / "novel-data"


def _get_manager() -> ProjectManager:
    return ProjectManager(DEFAULT_PROJECT_DIR)


def _build_initial_state(
    project_id: str,
    chapter_number: int,
    outline: str,
    ctx: dict,
    existing_entities: list[dict] | None = None,
    story_length: str = "long",
    target_chapter_words: int = 3000,
) -> dict:
    return {
        "project_id": project_id,
        "chapter_number": chapter_number,
        "chapter_outline": outline,
        "story_length": story_length,
        "target_chapter_words": target_chapter_words,
        "draft_content": "",
        "editor_report": {},
        "continuity_report": {},
        "worldbuilding_report": {},
        "retry_count": 0,
        "human_approved": False,
        "human_feedback": {},
        "rewrite_instructions": "",
        "orchestrator_strategy": {},
        "existing_world_entities": existing_entities or [],
        "character_context": ctx["character_context"],
        "world_context": ctx["world_context"],
        "recent_summary": ctx["recent_summary"],
        "unresolved_foreshadowings": [],
        "trace_id": "",
        "persist_dir": str(DEFAULT_PROJECT_DIR.absolute()),
    }


@cl.on_chat_start
async def start():
    """Show project selection on first load."""
    mgr = _get_manager()
    projects = mgr.list_projects()

    elements = []
    if projects:
        for p in projects:
            chapters = mgr.get_all_chapters(p["id"])
            elements.append(
                cl.Text(
                    name=f"Project: {p['title']}",
                    content=f"ID: {p['id']}\nGenre: {p['genre']}\nChapters: {len(chapters)}",
                    display="side",
                )
            )

    await cl.Message(
        content=(
            "# Novel Agent\n"
            "Welcome to the multi-Agent novel writing framework.\n\n"
            "**Commands:**\n"
            "- `write <chapter> <outline> [--words N]` — generate a chapter\n"
            "- `list` — show all projects\n"
            "- `new <name> [title] [genre] [--length short|novella|long] [--words N]`\n"
            "  create a new project\n"
            "- `select <project_id>` — switch project\n"
            "- `chapters` — list written chapters\n"
        ),
        elements=elements,
    ).send()

    cl.user_session.set("project_id", "")
    cl.user_session.set("mgr", mgr)


@cl.on_message
async def on_message(message: cl.Message):
    """Handle user commands."""
    mgr = cl.user_session.get("mgr") or _get_manager()
    project_id = cl.user_session.get("project_id") or ""
    text = message.content.strip()

    # ── new <name> [title] [genre] [--length short|novella|long] [--words N] ──
    if text.startswith("new ") or text.startswith("create "):
        length = "long"
        chapter_words = 3000
        if "--length" in text:
            import re
            m = re.search(r"--length\s+(\w+)", text)
            if m and m.group(1) in ("short", "novella", "long"):
                length = m.group(1)
        if "--words" in text:
            m = re.search(r"--words\s+(\d+)", text)
            if m:
                chapter_words = int(m.group(1))
        clean = text
        for flag in ("--length", "--words"):
            clean = re.sub(rf"{flag}\s+\S+", "", clean).strip()
        parts = clean.split(maxsplit=3)
        name = parts[1] if len(parts) > 1 else "untitled"
        title = parts[2] if len(parts) > 2 else name
        genre = parts[3] if len(parts) > 3 else ""
        pid = mgr.init_project(
            name=name, title=title, genre=genre,
            story_length=length, target_chapter_words=chapter_words,
        )
        cl.user_session.set("project_id", pid)
        await cl.Message(
            content=f"Created project **{title}** (id=`{pid}`)\n"
            f"Length: {length}, Chapter words: {chapter_words}"
        ).send()

    # ── list ──
    elif text == "list":
        projects = mgr.list_projects()
        if not projects:
            await cl.Message(content="No projects yet. Use `new <name>` to create one.").send()
            return
        lines = ["## Projects\n"]
        for p in projects:
            chs = mgr.get_all_chapters(p["id"])
            lines.append(
                f"- `{p['id']}` — **{p['title']}** ({p['genre']}) — "
                f"{len(chs)} chapters"
            )
        await cl.Message(content="\n".join(lines)).send()

    # ── select <project_id> ──
    elif text.startswith("select "):
        pid = text.split()[1].strip()
        proj = mgr.get_project(pid)
        if not proj:
            await cl.Message(content=f"Project `{pid}` not found.").send()
            return
        cl.user_session.set("project_id", pid)
        chapters = mgr.get_all_chapters(pid)
        await cl.Message(
            content=f"Selected **{proj['title']}** (id=`{pid}`)\n"
            f"{len(chapters)} chapters written."
        ).send()

    # ── chapters ──
    elif text == "chapters":
        if not project_id:
            await cl.Message(content="Select a project first: `select <project_id>`").send()
            return
        chapters = mgr.get_all_chapters(project_id)
        if not chapters:
            await cl.Message(content="No chapters written yet.").send()
            return
        lines = ["## Chapters\n"]
        for c in chapters:
            lines.append(
                f"- Ch.{c['chapter_number']}: {c.get('outline', '')[:50]} "
                f"({c.get('status', 'draft')})"
            )
        await cl.Message(content="\n".join(lines)).send()

    # ── write <chapter> <outline> [--words N] ──
    elif text.startswith("write "):
        if not project_id:
            await cl.Message(content="Select a project first: `select <project_id>`").send()
            return

        chapter_words_override = None
        import re
        wm = re.search(r"--words\s+(\d+)", text)
        if wm:
            chapter_words_override = int(wm.group(1))

        clean = re.sub(r"--words\s+\S+", "", text).strip()
        parts = clean.split(maxsplit=2)
        if len(parts) < 3:
            await cl.Message(content="Usage: `write <chapter_number> <outline>`").send()
            return

        try:
            chapter_number = int(parts[1])
        except ValueError:
            await cl.Message(content="Chapter number must be an integer.").send()
            return

        outline = parts[2]

        await _run_pipeline(
            mgr, project_id, chapter_number, outline,
            chapter_words_override=chapter_words_override,
        )

    # ── help ──
    else:
        await cl.Message(
            content=(
                "**Available commands:**\n"
                "- `new <name>` — create a project\n"
                "- `list` — show projects\n"
                "- `select <id>` — switch project\n"
                "- `chapters` — list chapters\n"
                "- `write <n> <outline>` — generate chapter\n"
            )
        ).send()


async def _human_review_chainlit(interrupt_data: dict) -> dict:
    """Present draft to user in Chainlit and wait for approve/reject decision.

    Shows draft preview, scores, and issues. User clicks Approve or Reject.
    On reject, a follow-up message collects comments.
    """
    chapter = interrupt_data.get("chapter_number", "?")
    editor_score = interrupt_data.get("editor_score", "?")
    continuity_score = interrupt_data.get("continuity_score", "?")
    retries = interrupt_data.get("retry_count", 0)
    draft = interrupt_data.get("draft_preview", "")

    # Build the review display
    review_lines = [
        f"## Human Review — Chapter {chapter}",
        "",
        "| Metric | Score |",
        "|--------|-------|",
        f"| Editor | {editor_score}/100 |",
        f"| Continuity | {continuity_score}/100 |",
        f"| Retries | {retries} |",
        "",
    ]

    editor_issues = interrupt_data.get("editor_issues", [])[:5]
    if editor_issues:
        review_lines.append("### Editor Issues")
        for i in editor_issues:
            review_lines.append(
                f"- [{i.get('severity', '?')}] {i.get('description', '')[:120]}"
            )
        review_lines.append("")

    continuity_issues = interrupt_data.get("continuity_issues", [])[:5]
    if continuity_issues:
        review_lines.append("### Continuity Issues")
        for i in continuity_issues:
            review_lines.append(
                f"- [{i.get('severity', '?')}] {i.get('description', '')[:120]}"
            )
        review_lines.append("")

    if draft:
        review_lines.append("### Draft Preview")
        review_lines.append(draft[:600] + "...")
        review_lines.append("")

    # Send the review message
    await cl.Message(content="\n".join(review_lines)).send()

    # Show draft as side element
    draft_full = interrupt_data.get("draft_full", "")
    if draft_full:
        await cl.Message(
            content="Full draft available in side panel",
            elements=[
                cl.Text(
                    name=f"Chapter {chapter} Draft",
                    content=draft_full[:3000] + ("..." if len(draft_full) > 3000 else ""),
                    display="side",
                )
            ],
        ).send()

    # Ask for decision using Chainlit action buttons
    res = await cl.AskActionMessage(
        content="Do you approve this chapter?",
        actions=[
            cl.Action(name="approve", label="Approve", value="approve",
                      description="Accept this chapter as-is"),
            cl.Action(name="reject", label="Reject", value="reject",
                      description="Request rewrite with feedback"),
        ],
    ).send()

    action = res.get("value", "approve")
    comments = ""

    if action == "reject":
        comments_msg = await cl.AskUserMessage(
            content="What needs to change? (optional)",
            timeout=300,
        ).send()
        if comments_msg:
            comments = comments_msg.get("output", "")

    return {"action": action, "comments": comments}


async def _run_pipeline(
    mgr: ProjectManager,
    project_id: str,
    chapter_number: int,
    outline: str,
    chapter_words_override: int | None = None,
):
    """Run the full Agent pipeline with Human-in-the-loop support.

    Catches GraphInterrupt when the graph reaches human_review_node,
    presents draft to user via Chainlit buttons, and resumes with feedback.
    """
    proj = mgr.get_project(project_id)
    project_title = proj["title"] if proj else "Unknown"

    story_length = proj.get("story_length", "long") if proj else "long"
    target_words = chapter_words_override or (
        proj.get("target_chapter_words", 3000) if proj else 3000
    )

    ctx = mgr.build_context(project_id, chapter_number)
    existing_entities = mgr.get_all_world_entities(project_id)
    initial_state = _build_initial_state(
        project_id, chapter_number, outline, ctx,
        existing_entities, story_length, target_words,
    )

    graph = build_chapter_graph()
    config = {"configurable": {"thread_id": f"project-{project_id}"}}

    await cl.Message(content=f"## Chapter {chapter_number}: {outline}\n").send()

    # Run pipeline with HITL loop
    async with cl.Step(name="Pipeline", type="tool") as step:
        step.input = (
            f"Orchestrator → Writer → Editor → Continuity → Worldbuilding → Human Review\n"
            f"Chapter {chapter_number}: {outline[:100]}"
        )

        state = initial_state
        while True:
            try:
                result = await graph.ainvoke(state, config)
                break
            except GraphInterrupt as e:
                interrupt_data = e.args[0] if e.args else {}
                feedback = await _human_review_chainlit(interrupt_data)
                state = Command(resume=feedback)

        step.output = "Pipeline complete"

    draft = result.get("draft_content", "")
    editor = result.get("editor_report", {})
    continuity = result.get("continuity_report", {})
    worldbuilding = result.get("worldbuilding_report", {})

    # Save result
    mgr.save_chapter(project_id, chapter_number, outline=outline, draft_content=draft)
    mgr.save_world_entities(project_id, worldbuilding_report=worldbuilding)

    # Display results
    editor_score = editor.get("overall_score", "?")
    continuity_score = continuity.get("overall_score", "?")

    elements = [
        cl.Text(
            name=f"Chapter {chapter_number} — Draft",
            content=draft[:3000] + ("..." if len(draft) > 3000 else ""),
            display="side",
        )
    ]

    new_entities = worldbuilding.get("new_entities", [])
    entity_text = "\n".join(
        f"- [{e.get('entity_type', '?')}] {e.get('name', '?')}"
        for e in new_entities[:10]
    ) or "None"

    await cl.Message(
        content=(
            f"## Chapter {chapter_number} Complete\n\n"
            f"**Project:** {project_title} (`{project_id}`)\n\n"
            f"### Scores\n"
            f"| Metric | Score |\n"
            f"|--------|-------|\n"
            f"| Editor | {editor_score}/100 |\n"
            f"| Continuity | {continuity_score}/100 |\n\n"
            f"### New Entities\n{entity_text}\n\n"
            f"### Preview\n{draft[:200]}..."
        ),
        elements=elements,
    ).send()

    # Show issues if any
    issues = editor.get("issues", [])
    if issues:
        issue_lines = "\n".join(
            f"- [{i.get('severity', '?')}] {i.get('description', '')}"
            for i in issues[:5]
        )
        await cl.Message(content=f"### Editor Issues\n{issue_lines}").send()

    inconsistencies = continuity.get("inconsistencies", [])
    if inconsistencies:
        inc_lines = "\n".join(
            f"- [{i.get('severity', '?')}] {i.get('description', '')}"
            for i in inconsistencies[:5]
        )
        await cl.Message(content=f"### Continuity Issues\n{inc_lines}").send()
