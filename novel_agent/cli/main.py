"""CLI entry point for novel-agent."""

import asyncio
import os
from pathlib import Path

import click
from langgraph.errors import GraphInterrupt
from langgraph.types import Command

from novel_agent import __version__
from novel_agent.agents.base import AgentConfig
from novel_agent.agents.writer import WriterAgent
from novel_agent.graph.chapter import build_chapter_graph
from novel_agent.storage.manager import ProjectManager
from novel_agent.trace.viewer import list_traces, show_trace

DEFAULT_PROJECT_DIR = Path.cwd() / "novel-data"


def _get_manager(project_dir: str | None) -> ProjectManager:
    path = Path(project_dir) if project_dir else DEFAULT_PROJECT_DIR
    return ProjectManager(path)


def _run_async(coro):
    return asyncio.run(coro)


def _human_review_cli(interrupt_data: dict) -> dict:
    """Interactive human review in CLI mode.

    Presents draft + scores to the user and prompts for approve/reject.
    Returns the feedback dict to resume the graph with.
    """
    chapter = interrupt_data.get("chapter_number", "?")
    editor_score = interrupt_data.get("editor_score", "?")
    continuity_score = interrupt_data.get("continuity_score", "?")
    retries = interrupt_data.get("retry_count", 0)

    click.echo(f"\n{'=' * 60}")
    click.echo(f"  HUMAN REVIEW — Chapter {chapter}")
    click.echo(f"{'=' * 60}")
    click.echo(f"  Editor: {editor_score}/100  |  Continuity: {continuity_score}/100")
    click.echo(f"  Retries: {retries}")
    click.echo()

    # Show draft preview
    draft = interrupt_data.get("draft_preview", "")
    if draft:
        click.echo("  ── Draft Preview ──")
        click.echo(draft[:600])
        click.echo("  ...\n")

    # Show key issues
    editor_issues = interrupt_data.get("editor_issues", [])
    if editor_issues:
        click.echo(f"  Editor Issues ({len(editor_issues)}):")
        for i in editor_issues[:5]:
            click.echo(f"    [{i.get('severity', '?')}] {i.get('description', '')[:100]}")

    continuity_issues = interrupt_data.get("continuity_issues", [])
    if continuity_issues:
        click.echo(f"\n  Continuity Issues ({len(continuity_issues)}):")
        for i in continuity_issues[:5]:
            click.echo(f"    [{i.get('severity', '?')}] {i.get('description', '')[:100]}")

    wb_entities = interrupt_data.get("wb_new_entities", 0)
    wb_conflicts = interrupt_data.get("wb_conflicts", 0)
    if wb_entities or wb_conflicts:
        click.echo(f"\n  Worldbuilding: {wb_entities} entities, {wb_conflicts} conflicts")

    click.echo()

    # Prompt for decision
    decision = click.prompt(
        "  Approve or reject?",
        type=click.Choice(["approve", "reject", "a", "r"]),
        default="a",
        show_choices=True,
    )

    action = "approve" if decision in ("approve", "a") else "reject"
    comments = ""
    if action == "reject":
        comments = click.prompt(
            "  Comments (what needs to change)",
            type=str,
            default="",
        )

    click.echo()
    return {"action": action, "comments": comments}


async def _run_graph_with_hitl(
    graph, initial_state: dict, config: dict,
) -> dict:
    """Run the graph, handling human-in-the-loop interrupts.

    When the graph hits human_review_node, LangGraph raises GraphInterrupt.
    We catch it, present the draft to the user via CLI, and resume with
    their feedback. This loop continues until the graph reaches END.
    """
    state = initial_state
    while True:
        try:
            result = await graph.ainvoke(state, config)
            return result
        except GraphInterrupt as e:
            # The graph paused at human_review_node — present to user
            interrupt_data = e.args[0] if e.args else {}
            feedback = _human_review_cli(interrupt_data)
            # Resume with human feedback
            state = Command(resume=feedback)


@click.group()
@click.version_option(version=__version__)
def cli():
    """novel-agent — Open-source multi-Agent novel writing framework."""


@cli.command()
@click.option("--name", "-n", required=True, help="Project name")
@click.option("--title", "-t", default="", help="Novel title")
@click.option("--genre", "-g", default="", help="Genre (e.g. 都市, 悬疑)")
@click.option(
    "--length", "-l", default="long",
    type=click.Choice(["short", "novella", "long"]),
    help="Story length",
)
@click.option(
    "--chapter-words", "-w", default=3000, type=int,
    help="Target words per chapter",
)
@click.option("--dir", "-d", default=None, help="Project directory")
def init(
    name: str, title: str, genre: str,
    length: str, chapter_words: int, dir: str | None,
):
    """Initialize a new novel project."""
    mgr = _get_manager(dir)
    pid = mgr.init_project(
        name=name, title=title or name, genre=genre,
        story_length=length, target_chapter_words=chapter_words,
    )
    project_dir = Path(dir) if dir else DEFAULT_PROJECT_DIR
    click.echo(f"Project '{name}' created (id={pid})")
    click.echo(f"  Length: {length}, Chapter words: {chapter_words}")
    click.echo(f"  Data directory: {project_dir.absolute()}")


@cli.command()
@click.option("--chapter", "-c", type=int, required=True, help="Chapter number")
@click.option("--outline", "-o", required=True, help="Chapter outline")
@click.option("--project", "-p", default=None, help="Project ID")
@click.option("--dir", "-d", default=None, help="Project directory")
@click.option("--model", "-m", default=None, help="Model to use for Writer")
@click.option(
    "--chapter-words", "-w", default=None, type=int,
    help="Override target words per chapter",
)
def write(
    chapter: int, outline: str, project: str | None,
    dir: str | None, model: str | None, chapter_words: int | None,
):
    """Generate a chapter using the full multi-Agent pipeline.

    Writer → Editor → Continuity → Human Review.
    """
    mgr = _get_manager(dir)
    persist_dir = str((Path(dir) if dir else DEFAULT_PROJECT_DIR).absolute())

    # Get or create project
    if project:
        proj = mgr.get_project(project)
        if not proj:
            click.echo(f"Error: Project '{project}' not found. Run 'novel-agent init' first.")
            return
        project_id = project
    else:
        project_id = mgr.get_or_create_default_project()
        proj = mgr.get_project(project_id)

    # Read length config from project, CLI flag overrides
    story_length = proj.get("story_length", "long") if proj else "long"
    target_words = chapter_words or (proj.get("target_chapter_words", 3000) if proj else 3000)

    click.echo(f"Project: {proj['title']} (id={project_id})")
    click.echo(f"  Length: {story_length}, Target: {target_words} words/chapter")
    click.echo(f"Chapter {chapter} — {outline[:80]}...")
    click.echo()

    # Build context from previous chapters
    ctx = mgr.build_context(project_id, chapter)
    if ctx["recent_summary"]:
        click.echo(f"Context: {len(ctx['recent_summary'])} chars from previous chapters\n")

    # Load existing world entities for conflict detection
    existing_entities = mgr.get_all_world_entities(project_id)

    # Build initial state
    initial_state = {
        "project_id": project_id,
        "chapter_number": chapter,
        "chapter_outline": outline,
        "story_length": story_length,
        "target_chapter_words": target_words,
        "draft_content": "",
        "editor_report": {},
        "continuity_report": {},
        "worldbuilding_report": {},
        "orchestrator_strategy": {},
        "existing_world_entities": existing_entities,
        "retry_count": 0,
        "human_approved": False,
        "character_context": ctx["character_context"],
        "world_context": ctx["world_context"],
        "recent_summary": ctx["recent_summary"],
        "unresolved_foreshadowings": [],
        "trace_id": "",
        "persist_dir": persist_dir,
    }

    # Build the graph and run
    graph = build_chapter_graph()
    config = {"configurable": {"thread_id": f"project-{project_id}"}}

    # Model override via CLI flag
    if model:
        os.environ["QUALITY_MODEL"] = model

    click.echo("Running pipeline: Writer → Editor → Continuity → Review\n")

    async def _run():
        return await _run_graph_with_hitl(graph, initial_state, config)

    result = _run_async(_run())

    # Show results
    draft = result.get("draft_content", "")
    editor = result.get("editor_report", {})
    continuity = result.get("continuity_report", {})
    worldbuilding = result.get("worldbuilding_report", {})

    # Save chapter
    mgr.save_chapter(project_id, chapter, outline=outline, draft_content=draft)
    # Persist extracted world entities
    saved_entities = mgr.save_world_entities(project_id, worldbuilding_report=worldbuilding)
    if saved_entities:
        click.echo(f"  World Entities: {saved_entities} saved")

    click.echo(f"\n{'─' * 60}")
    click.echo(draft[:500] + ("..." if len(draft) > 500 else ""))
    click.echo(f"{'─' * 60}")

    # Summary
    editor_score = editor.get("overall_score", "?")
    continuity_score = continuity.get("overall_score", "?")
    click.echo("\nResults:")
    click.echo(f"  Editor:       {editor_score}/100 — {editor.get('verdict', '?')}")
    click.echo(f"  Continuity:   {continuity_score}/100")

    new_entities = worldbuilding.get("new_entities", [])
    conflicts = worldbuilding.get("conflicts", [])
    if new_entities:
        click.echo(f"  New Entities: {len(new_entities)}")
        for e in new_entities[:3]:
            click.echo(f"    - {e.get('entity_type', '?')}: {e.get('name', '?')}")
    if conflicts:
        click.echo(f"  WB Conflicts: {len(conflicts)}")

    editor_issues = editor.get("issues", [])
    if editor_issues:
        click.echo(f"\n  Editor Issues ({len(editor_issues)}):")
        for issue in editor_issues[:5]:
            desc = issue.get('description', issue.get('phrase', ''))
            click.echo(f"    [{issue.get('severity', '?')}] {desc}")

    inconsistencies = continuity.get("inconsistencies", [])
    if inconsistencies:
        click.echo(f"\n  Continuity Issues ({len(inconsistencies)}):")
        for inc in inconsistencies[:5]:
            click.echo(f"    [{inc.get('severity', '?')}] {inc.get('description', '')[:80]}")

    click.echo(f"\nChapter {chapter} saved to {persist_dir}")


@cli.command()
@click.option("--chapter", "-c", type=int, required=True, help="Chapter number")
@click.option("--outline", "-o", required=True, help="Chapter outline")
@click.option("--project", "-p", default=None, help="Project ID")
@click.option("--dir", "-d", default=None, help="Project directory")
@click.option("--model", "-m", default=None, help="Model to use")
@click.option(
    "--chapter-words", "-w", default=None, type=int,
    help="Target words per chapter",
)
def quick(
    chapter: int, outline: str, project: str | None,
    dir: str | None, model: str | None, chapter_words: int | None,
):
    """Quick generate with Writer only (skip Editor + Continuity)."""
    mgr = _get_manager(dir)

    if project:
        proj = mgr.get_project(project)
        if not proj:
            click.echo(f"Error: Project '{project}' not found.")
            return
        project_id = project
    else:
        project_id = mgr.get_or_create_default_project()
        proj = mgr.get_project(project_id)

    target_words = chapter_words or (proj.get("target_chapter_words", 3000) if proj else 3000)
    click.echo(f"Quick write Chapter {chapter} ({target_words} words)...")

    ctx = mgr.build_context(project_id, chapter)

    config = AgentConfig()
    if model:
        config.model = model
    else:
        config.model = os.getenv("QUALITY_MODEL", os.getenv("BUDGET_MODEL", "deepseek-chat"))

    writer = WriterAgent(
        config=config,
        chapter_store=mgr.chapter_store,
        project_id=project_id,
        target_chapter_words=target_words,
    )

    async def _write():
        content, trace = await writer.write(
            chapter_number=chapter,
            outline=outline,
            character_context=ctx["character_context"],
            recent_summary=ctx["recent_summary"],
            target_chapter_words=target_words,
        )
        return content, trace

    content, trace = _run_async(_write())

    mgr.save_chapter(project_id, chapter, outline=outline, draft_content=content)

    click.echo(f"\n{'─' * 60}")
    click.echo(content[:500] + ("..." if len(content) > 500 else ""))
    click.echo(f"{'─' * 60}")
    click.echo(f"\nSaved. Tokens: {trace.input_tokens}/{trace.output_tokens}")


@cli.command()
@click.option("--dir", "-d", default=None, help="Project directory")
def list(dir: str | None):
    """List all projects."""
    mgr = _get_manager(dir)
    projects = mgr.list_projects()
    if not projects:
        click.echo("No projects found. Run 'novel-agent init' to create one.")
        return
    for p in projects:
        chapters = mgr.get_all_chapters(p["id"])
        click.echo(f"  [{p['id']}] {p['title']} ({len(chapters)} chapters) — {p['genre']}")


@cli.group()
def trace():
    """View and replay pipeline traces."""


@trace.command("show")
@click.argument("filepath", type=click.Path(exists=True))
def trace_show(filepath: str):
    """Display a trace file with formatted output."""
    show_trace(filepath)


@trace.command("ls")
@click.option("--dir", "-d", default="./traces", help="Trace directory")
def trace_list(dir: str):
    """List all trace files."""
    list_traces(dir)


if __name__ == "__main__":
    cli()
