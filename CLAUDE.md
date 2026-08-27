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

## Critical Rules

- Prefer deletion and simplification over adding abstractions.
- Do not add a new Agent when an existing Agent, Service, or deterministic function is sufficient.
- Do not use an LLM for work that can be done deterministically (routing, scoring, formatting, filtering, statistics).
- Do not duplicate the same information across State, Context, Agent parameters, and Storage.
- Never silently reintroduce Writer V1, ContextCompressor, legacy context parameters, or deprecated prompt paths.
- Treat the current code contract as authoritative; remove historical compatibility when verified unused.

## Architecture Boundaries

State is workflow state. Storage is durable truth. Context is a derived task view.

```
State → ContextCompiler → task-specific context → Agent
```

- Do not pass raw `NovelState` into an Agent.
- Do not reconstruct Context manually inside Agents.
- Do not add LLM summarization for ordinary context compression.

Core flow: Orchestrator → Writer → deterministic Quality/Style → optional Editor → optional Continuity → Worldbuilding → Evolution → SelectBest → Human Review.

Directory boundaries are enforced by code structure; do not move business logic across them unless a concrete bug requires it.

## LLM Usage

Before adding an LLM call ask:

1. Can deterministic code solve it?
2. Can an existing Agent solve it?
3. Is the additional semantic judgment actually necessary?

Normal chapter generation should remain inexpensive.

## Novel Generation

Chapter target is a minimum completion target, not a request to pad text.

- `target_words = 3000`, preferred range `3000–3450`
- Do not forcibly truncate natural text above 3450.

When below target, use Narrative Extension — extend the story forward (consequences, reactions, discoveries, decisions, new beats). Never pad by repeating environment, emotion, dialogue, or explanation.

Natural prose is more important than satisfying a stylistic heuristic.

## Editing Layers

- `Editor` — semantic literary judgment (the only LLM literary reviewer).
- `QualityService` — deterministic hard constraints.
- `StyleAnalyzer` — deterministic text measurement.
- `EvolutionService` — deterministic version comparison and termination.

Do not duplicate the same judgment in multiple layers. Prefer conditional review: style-only change does not rerun unrelated heavy review.
