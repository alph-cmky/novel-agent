"""Rich CLI trace viewer — replays agent pipeline traces.

Usage:
    novel-agent trace show traces/trace-xxx.json
    novel-agent trace list traces/
"""

import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def list_traces(trace_dir: str = "./traces"):
    """List all trace files in the directory."""
    path = Path(trace_dir)
    if not path.exists():
        Console().print("[red]No traces directory found.[/red]")
        return

    files = sorted(path.glob("trace-*.json"), reverse=True)
    if not files:
        Console().print("[yellow]No trace files found.[/yellow]")
        return

    table = Table(title="Pipeline Traces")
    table.add_column("File", style="cyan")
    table.add_column("Project", style="green")
    table.add_column("Chapter", justify="right")
    table.add_column("Steps", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Elapsed", justify="right")

    for f in files[:20]:
        try:
            data = json.loads(f.read_text())
            meta = data.get("meta", {})
            table.add_row(
                f.name,
                meta.get("project_id", "?")[:12],
                str(meta.get("chapter_number", "?")),
                str(data.get("step_count", 0)),
                f"{data.get('total_tokens_input', 0) + data.get('total_tokens_output', 0)}",
                f"{data.get('meta', {}).get('pipeline_elapsed_s', 0):.1f}s",
            )
        except (json.JSONDecodeError, KeyError):
            table.add_row(f.name, "?", "?", "?", "?", "?")

    Console().print(table)


def show_trace(filepath: str):
    """Display a single trace file with Rich formatting."""
    path = Path(filepath)
    if not path.exists():
        Console().print(f"[red]File not found: {filepath}[/red]")
        return

    console = Console()
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        console.print("[red]Invalid trace file.[/red]")
        return

    meta = data.get("meta", {})

    # Header
    header = Text()
    header.append("Pipeline Trace: ", style="bold")
    header.append(f"Chapter {meta.get('chapter_number', '?')}")
    header.append(f"\nProject: {meta.get('project_id', '?')}")
    header.append(f"\nElapsed: {meta.get('pipeline_elapsed_s', 0):.1f}s")
    total_tok = data.get("total_tokens_input", 0) + data.get("total_tokens_output", 0)
    header.append(f"\nTotal tokens: {total_tok}")
    header.append(f"\nTool calls: {data.get('total_tool_calls', 0)}")
    console.print(Panel(header, title="Summary"))

    # Step table
    table = Table(title="Agent Steps")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Agent", style="cyan")
    table.add_column("Action", style="green")
    table.add_column("Input Tokens", justify="right")
    table.add_column("Output Tokens", justify="right")
    table.add_column("Duration", justify="right")
    table.add_column("Tools", justify="right")

    for i, step in enumerate(data.get("steps", []), 1):
        tools = ", ".join(tc["tool"] for tc in step.get("tool_calls", [])) or "-"
        table.add_row(
            str(i),
            step.get("agent", "?"),
            step.get("action", "?")[:30],
            str(step.get("input_tokens", 0)),
            str(step.get("output_tokens", 0)),
            f"{step.get('duration_ms', 0)}ms",
            tools[:25],
        )

    if data["steps"]:
        console.print(table)
    else:
        console.print("[yellow]No steps recorded.[/yellow]")
