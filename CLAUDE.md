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

## Architecture: S1

```
Orchestrator
  → Writer loop [generate → analyze_style → humanize_passage if needed]
  → deterministic Hard Gate
       PASS → deterministic timeline check on this draft
              critical finding → Editor arm
              clean → skip Editor and Continuity
       FAIL → Editor → Writer rewrite ≤ 2 → Continuity
  → Worldbuilding (always)
  → output
```

Writer is the only loop driver. Editor, Continuity, and Worldbuilding sit
outside the loop. They are not Writer tools. Do not put `editor_review`,
`check_continuity`, or `worldbuilding_extract` back inside the Writer loop.

No LangGraph DAG, no evolution subgraph, no scene-first path. The loop is
bounded by `max_rounds` (runner default 8) and the hard gate.

### Loop Driver

- Writer decides when to call `analyze_style`, `humanize_passage`, and
  `search_context`.
- Do not build nested agent loops. One driver, tools only inside that loop.
- Editor and Continuity each do one review call. They do not drive a loop.
  An Editor `rewrite` verdict re-enters the Writer loop at most twice.

### Deterministic Guardrails

The LLM writes; deterministic code decides whether review runs.

- `QualityService.check_draft_hard_gates` runs after the Writer loop. It
  rejects only an empty draft, length under half the target (1500 when the
  target is 3000), or a missing outline. Do not raise this threshold to force
  the Editor arm.
- On PASS, `ContinuityService.check_timeline` still audits this draft for
  event order and dead-character reappearance. A critical finding enters the
  Editor arm. A clean PASS skips both Editor and the Continuity LLM.
- On FAIL, Editor runs, then Writer may rewrite, then the Continuity LLM
  audits. If a rewrite later passes the hard gate and the timeline check,
  skip the Continuity LLM.
- `strip_writer_preamble` removes meta-commentary from the final output.
- Empty Writer output raises. Do not store an empty draft as success.
- Worldbuilding always runs after the review decision.

### What each component does

- `analyze_style` — deterministic measurement inside the Writer loop. Evidence
  only, 0 LLM calls.
- `humanize_passage` — the only prose rewrite for AI flavor. Preserves plot
  and characters.
- `search_context` — retrieval inside the Writer loop. It does not judge.
- Editor — literary judgment on the FAIL arm. Scores and flags issues. It
  does not rewrite, and it does not repeat `analyze_style`'s checks.
- Continuity LLM — cross-chapter audit on the FAIL arm only.
- `ContinuityService` — deterministic timeline audit. Shared by context
  compilation and the PASS-path gate. Death means the action states a death.
  Mentions such as 「讨论死亡」 are not deaths.
- Worldbuilding — extraction after the review decision. Entities, conflicts,
  foreshadowings. It does not drive the loop.
- `QualityService` — hard gate after the Writer loop, not inside it.

Do not duplicate a judgment. `analyze_style` measures, `humanize_passage`
rewrites, Editor judges, `ContinuityService` checks time and death.

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

## Context

State is workflow state. Storage is durable truth. Context is one derived
packet.

```
Storage → ContextCompiler.compile → context_packet → Orchestrator and Writer
```

`compile` is the only place that builds the packet. The loop reads
`context_packet`. It does not assemble a second packet from raw tables.

- Do not pass raw `NovelState` into the Writer or any review component.
- Do not reconstruct context inside the loop.
- Do not add LLM summarization for ordinary context compression.
- Role projections (`for_writer`, `for_editor`, `for_continuity`,
  `for_orchestrator`) copy the packet through. Do not give a role a second
  character budget, and do not clear `world_context`.
- `compile(..., task=)` does not change the packet. Do not bring task-based
  shrinking back.
- Orchestrator `context_needed` may add named characters, world elements, and
  cross-timeline events. Those additions use the same excerpt, death
  retention, and foreshadowing cap as `compile`. They must not replace a
  section with unbounded text or drop a confirmed death that fell outside a
  raw tail slice.

Compression keeps the fields. It does not blank them.

- Recent prose: last 3 chapters, each a sentence-bounded tail of about 400
  characters.
- Events: the 8 chapters before the current one, plus confirmed deaths and
  that subject's events from the death chapter up to the window. Drop the
  current chapter and anything after it.
- Entity properties: tail excerpt of about 160 characters. Names stay.
- Unresolved foreshadowings: at most 40.
- Worldbuilding and the Orchestrator use these same caps. Do not load every
  entity or every open foreshadowing into a prompt.

## LLM Usage

Before adding an LLM-backed step ask:

1. Can deterministic code solve it? (measurement, retrieval, filtering)
2. Can an existing tool or service solve it?
3. Is the additional semantic judgment actually necessary?

Normal chapter generation should stay inexpensive. A PASS chapter pays for
the Orchestrator, the Writer loop, and Worldbuilding. Editor and the
Continuity LLM are the FAIL arm, not the default. Each Writer round is at
most one generation plus one tool call. A loop that burns rounds without
improving the draft is a bug.

## Novel Generation

Chapter target is a minimum completion target, not a request to pad text.

- `target_words = 3000`, preferred range `3000–3450`
- Do not forcibly truncate natural text above 3450.

When below target, use Narrative Extension — extend the story forward
(consequences, reactions, discoveries, decisions, new beats). Never pad by
repeating environment, emotion, dialogue, or explanation.

Natural prose is more important than satisfying a stylistic heuristic.

## De-AI Flavor

`humanize_passage` is the only rewrite that removes AI writing patterns. Its
prompt is distilled from humanizer-zh and focused on patterns that matter
for Chinese web fiction.

- `analyze_style` identifies the patterns (deterministic).
- `humanize_passage` rewrites them (LLM).

Do not add AI-flavor rules to Editor that duplicate `analyze_style`. Do not
add AI-flavor rewriting to the Writer prompt that duplicates
`humanize_passage`.
