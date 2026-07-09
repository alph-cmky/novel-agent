"""CLI entry point for novel-agent — service + operations."""

from pathlib import Path

import click

from novel_agent import __version__
from novel_agent.api.routes import _build_export_content
from novel_agent.storage.manager import ProjectManager
from novel_agent.trace.viewer import list_traces, show_trace

DEFAULT_PROJECT_DIR = Path.cwd() / "novel-data"


def _get_manager(project_dir: str | None) -> ProjectManager:
    path = Path(project_dir) if project_dir else DEFAULT_PROJECT_DIR
    return ProjectManager(path)


@click.group()
@click.version_option(version=__version__)
def cli():
    """novel-agent — Open-source multi-Agent novel writing framework."""


# ── serve ──────────────────────────────────────────────


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, type=int, help="Port to listen on")
@click.option("--reload", is_flag=True, help="Enable auto-reload (dev mode)")
def serve(host: str, port: int, reload: bool):
    """Start the Web server."""
    import uvicorn

    click.echo(f"novel-agent server starting at http://{host}:{port}")
    uvicorn.run(
        "novel_agent.api.app:app",
        host=host,
        port=port,
        reload=reload,
    )


# ── export ─────────────────────────────────────────────


@cli.command("export")
@click.option("--project", "-p", default=None, help="Project ID (default: first project)")
@click.option(
    "--format", "-f", "fmt",
    default="md",
    type=click.Choice(["md", "txt"]),
    help="Export format",
)
@click.option("--output", "-o", default=None, help="Output file path (default: stdout)")
@click.option("--dir", "-d", default=None, help="Project data directory")
def export_novel(
    project: str | None,
    fmt: str,
    output: str | None,
    dir: str | None,
):
    """Export novel as Markdown or plain text.

    \b
    Examples:
      novel-agent export                        # stdout
      novel-agent export -o novel.md            # save to file
      novel-agent export -f txt -o novel.txt    # plain text
    """
    mgr = _get_manager(dir)

    if project:
        proj = mgr.get_project(project)
        if not proj:
            click.echo(f"Error: Project '{project}' not found.", err=True)
            raise SystemExit(1)
    else:
        projects = mgr.list_projects()
        if not projects:
            click.echo("Error: No projects found.", err=True)
            raise SystemExit(1)
        proj = projects[0]
        click.echo(f"Using project: {proj['title']} ({proj['id']})")

    chapters = mgr.get_all_chapters(proj["id"])
    outlines = mgr.get_outline(proj["id"])
    content = _build_export_content(proj, chapters, outlines, fmt)

    if output:
        Path(output).write_text(content, encoding="utf-8")
        click.echo(f"Exported to {output}")
    else:
        click.echo(content)


# ── trace ──────────────────────────────────────────────


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
