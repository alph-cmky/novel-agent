# Novel-Agent — Claude Code Instructions

Chinese long-form fiction generation system.

Pre-production: no production users or legacy data to preserve.

## Environment

```bash
uv sync                                   # install deps
uv run novel-agent serve [--reload]       # backend http://0.0.0.0:8000
uv run ruff format --check .             # format check
uv run ruff check .                      # lint
uv run pytest -q                         # tests
```

Python 3.12+. Use `uv` for all commands.

## Architecture: Agent Loop

```
Orchestrator → Writer agent loop → deterministic QualityGate → output
```

Writer drives a single tool-calling loop. It generates, then calls tools to
self-check and revise, then outputs the final chapter. No fixed graph nodes,
no evolution subgraph. The loop is bounded by `max_rounds` and deterministic
guardrails.

```
Writer loop:
  generate → analyze_style → humanize_passage (if AI flavor) → editor_review
  → check_continuity → worldbuilding_extract → revise → finalize
```

### Loop Driver

- Writer is the **only** loop driver. It decides what tools to call and when.
- Other components (Editor, StyleAnalyzer, WorldbuildingAgent) are **tools**,
  not agents. They do not drive loops.
- Do not build nested agent loops (an agent that calls an agent that loops).
  Keep the loop flat: one driver, many tools.

### Deterministic Guardrails

The LLM drives the loop; deterministic code bounds it.

- `max_rounds` caps total tool-calling rounds (default 12).
- `QualityService.check_draft_hard_gates` runs after the loop — empty or
  grossly short content is rejected regardless of loop outcome.
- `strip_writer_preamble` removes meta-commentary from the final output.
- If the loop exhausts rounds or returns empty, the runner raises — no silent
  fallback to a partial draft.

### Tool Design

- **Deterministic tools** (`analyze_style`, `check_continuity`) measure and
  retrieve — 0 LLM calls. They return evidence, not verdicts.
- **LLM tools** (`humanize_passage`, `editor_review`,
  `worldbuilding_extract`) do one semantic task each. They do not call other
  tools or drive loops.
- Each tool has a clear input schema, returns `ToolResult`, and writes its
  result to a shared context dict for the runner to extract.
- Do not add a tool that duplicates another tool's judgment. If two tools
  measure the same thing, one is redundant.

## Critical Rules

- Prefer deletion and simplification over adding abstractions.
- Do not add a new tool when an existing tool or deterministic function is
  sufficient.
- Do not use an LLM for work that can be done deterministically (measuring,
  retrieving, filtering, formatting).
- Do not duplicate the same information across State, Context, tool results,
  and Storage.
- Never silently reintroduce the LangGraph DAG, EvolutionService,
  ContextCompressor, or deprecated prompt paths.
- Treat the current code contract as authoritative; remove historical
  compatibility when verified unused.

## Architecture Boundaries

State is workflow state. Storage is durable truth. Context is a derived
task view.

```
State → ContextCompiler → task-specific context → Writer loop
```

- Do not pass raw `NovelState` into the Writer or any tool.
- Do not reconstruct Context manually inside the loop.
- Do not add LLM summarization for ordinary context compression.

## LLM Usage

Before adding an LLM-backed tool ask:

1. Can deterministic code solve it? (measurement, retrieval, filtering)
2. Can an existing tool solve it?
3. Is the additional semantic judgment actually necessary?

Normal chapter generation should remain inexpensive. The loop's LLM cost is
bounded by `max_rounds` — each round is at most one generation + one tool call.
Track per-tool token usage; a loop that burns rounds without improving is a
bug, not a feature.

## Novel Generation

Chapter target is a minimum completion target, not a request to pad text.

- `target_words = 3000`, preferred range `3000–3450`
- Do not forcibly truncate natural text above 3450.

When below target, use Narrative Extension — extend the story forward
(consequences, reactions, discoveries, decisions, new beats). Never pad by
repeating environment, emotion, dialogue, or explanation.

Natural prose is more important than satisfying a stylistic heuristic.

## Tool Layers

- `analyze_style` — deterministic text measurement (0 LLM). Evidence only.
- `humanize_passage` — LLM rewrite to remove AI writing patterns. Preserves
  plot and characters; only rewrites prose.
- `editor_review` — LLM literary judgment (8 dimensions). Scores and flags
  issues; does not rewrite.
- `check_continuity` — deterministic retrieval + LLM audit. Cross-chapter
  consistency.
- `worldbuilding_extract` — LLM extraction. Entities, conflicts, foreshadowings.
- `QualityService` — deterministic hard gates. Runs after the loop, not
  inside it.

Do not duplicate the same judgment across tools. `analyze_style` measures;
`editor_review` judges; `humanize_passage` rewrites. Each tool has one job.

## De-AI Flavor

`humanize_passage` is the only tool that rewrites prose to remove AI writing
patterns. Its prompt is distilled from humanizer-zh and focused on patterns
that matter for Chinese web fiction.

- `analyze_style` identifies the patterns (deterministic).
- `humanize_passage` rewrites them (LLM).
- `editor_review` judges the result (LLM).

Do not add AI-flavor rules to `editor_review` that duplicate `analyze_style`'s
deterministic checks. Do not add AI-flavor rewriting to the Writer prompt
that duplicates `humanize_passage`.
