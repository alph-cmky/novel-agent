# Novel-Agent Refactor Baseline

Baseline recorded before refactoring on 2026-08-25, branch `master`, commit
`e6e7e85`.

## 1. Test Count

`247` tests passed and `1` test was skipped (`248` collected).

## 2. Test Result

`uv run pytest -q`: **PASS**

```text
247 passed, 1 skipped in 2.68s
```

## 3. Lint Result

- `uv run ruff check .`: **PASS**
- `uv run ruff format --check .`: **FAIL**; 57 files would be reformatted.

The formatting failure is pre-existing and is not being fixed as part of the
baseline commit.

## 4. Type Check Result

No mypy/pyright configuration is present in the repository. Running
`uv run mypy .` nevertheless produced **349 errors in 39 files**, primarily
because the invocation uses an incompatible interpreter/stub environment and
cannot resolve several installed dependencies. This is recorded as a baseline
failure, not a refactor regression.

## 5. Current Package Tree

```text
novel_agent/
├── agents/ (base, continuity, editor, evolution_orchestrator, orchestrator, worldbuilding, writer)
├── api/ (app, chainlit_app, graph_data, outline, routes, run_service, sse)
├── cli/ (main)
├── context/ (compiler)
├── graph/ (candidates, chapter, evolution, quality_gates, scenes, state, timeline_checker)
├── memory/ (compressor, embeddings)
├── observability/ (langfuse)
├── routing/ (__init__)
├── schema/ (enums, models, parser, validator)
├── storage/ (manager, models, outbox_worker)
├── style/ (ai_flavor)
├── tools/ (base, continuity, search, style)
├── __init__.py
└── config.py
```

## 6. References To Candidate Deletion Targets

Counts below are repository text-search matches before any deletion:

- `chainlit`: 16 matches across source, dependency metadata, docs, and tests/config.
- `novel_agent.routing` / routing imports: 4 source/test import matches, plus 1 README/CLAUDE reference.
- `timeline_checker`: 3 source/test import or module references.
- `graph.candidates`: 3 source/test import references.
- `graph.scenes`: 3 source/test import references.
- `tools.style`: 2 source/test import references.
- `tools.continuity`: 2 source/test import references.
- `outbox_worker`: 2 source imports plus its dedicated test.

The counts include direct references only; dynamic imports were also checked in
the repository search and none were found for these module names.

## 7. LangGraph State Fields

`NovelState` currently contains:

```text
project_id, writing_run_id, chapter_number, chapter_outline, story_length,
target_chapter_words, narrative_mode, narrative_perspective, draft_content,
editor_report, continuity_report, worldbuilding_report, orchestrator_strategy,
skip_orchestrator, skip_reviews, skip_worldbuilding, review_interval,
skip_evolution_enrichment, character_context, world_context, recent_summary,
unresolved_foreshadowings, context_packet_hash, context_packet, timeline_events,
timeline_findings, scene_first, scene_plan, scene_drafts,
existing_world_entities, human_approved, human_feedback,
evolution_human_rejects, human_review_exhausted, evolution_max_rounds,
evolution_convergence_threshold, evolution_round, evolution_version,
evolution_history, evolution_candidates, evolution_improvement_plan,
evolution_termination, evolution_best_candidate_version, quality_guard_report,
quality_gate_report, deterministic_gate_first, writer_prompt_profile,
persist_dir, trace_id
```

No state field names or checkpoint serialization behavior are changed by this
baseline.

## 8. API Endpoints

```text
GET/POST /projects
GET/PATCH/DELETE /projects/{project_id}
GET/PUT /projects/{project_id}/outline
POST /projects/{project_id}/outline/generate
GET /projects/{project_id}/graph
GET /projects/{project_id}/chapters
GET/POST /projects/{project_id}/chapters/{chapter_number}
GET/POST /projects/{project_id}/chapters/{chapter_number}/scenes
POST /projects/{project_id}/chapters/{chapter_number}/write
POST /projects/{project_id}/chapters/{chapter_number}/approve
POST /projects/{project_id}/chapters/{chapter_number}/reject
PUT/DELETE /projects/{project_id}/chapters/{chapter_number}/draft
GET /projects/{project_id}/canon
GET /projects/{project_id}/outbox
GET /projects/{project_id}/events
GET /projects/{project_id}/chapters/{chapter_number}/versions
GET /projects/{project_id}/export
GET /runs/{run_id}
POST /runs/{run_id}/versions
POST /runs/{run_id}/commit
GET /runs/{run_id}/proposals
POST /runs/{run_id}/scenes/{scene_index}/rewrite
POST /runs/{run_id}/cancel
POST /runs/{run_id}/retry
GET /versions/{version_id}/diff
POST /versions/{version_id}/scenes/{scene_index}/review
POST /versions/{version_id}/restore
POST /outbox/{event_id}/process
POST /outbox/{event_id}/retry
POST /proposals/{proposal_id}/accept
POST /proposals/{proposal_id}/reject
```

The endpoint contract and SSE event contract are unchanged.

## 9. CLI Commands

```text
novel-agent --version
novel-agent serve [--host HOST] [--port PORT] [--reload]
novel-agent export [-p PROJECT] [-f md|txt] [-o OUTPUT] [--dir DIR]
```

## 10. SQLite Schema Status

SQLite schema is defined in `novel_agent/storage/models.py` and initialized by
`init_db`. All columns are declared in the `CREATE TABLE` statements directly —
no migration or backfill logic. Tables currently created include `projects`,
`chapters`, `writing_runs`, `canon_snapshots`, `chapter_versions`,
`canon_proposals`, `outbox_events`, `story_events`, `world_entities`,
`world_relations`, `foreshadowings`, and `outlines`. No SQLite schema or
checkpoint format changes are part of this baseline.

## BASELINE FAILURES

- Ruff format check: 57 files unformatted.
- Mypy: 349 errors in 39 files under the available invocation/environment.
