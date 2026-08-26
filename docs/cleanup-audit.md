# Cleanup Audit — Pre-production Architecture Reset

Baseline: `aef24209d8ff6a445359520cd89e2a79b8c1aabb`

## Phase 0: Audit Summary

| Symbol | File | Callers | Test Callers | Prod Use | Legacy Reason | Action |
|---|---|---|---|---|---|---|
| `ContextCompressor` | `memory/compressor.py` | orchestrator.py, chapter.py | test_compressor.py | Orchestrator node | Replaced by ContextCompiler | DELETE |
| `CompressionStrategy` | `memory/compressor.py` | compressor.py | test_compressor.py | — | Part of old context system | DELETE |
| `_compressor` param | `agents/orchestrator.py` | chapter.py:232-239 | — | Orchestrator node | ContextCompressor dependency | DELETE |
| `_story_arc` | `agents/orchestrator.py` | analyze():315, get_arc_summary():420 | — | get_arc_summary() has 0 callers | Write-only, never read | DELETE |
| `WRITER_SYSTEM_PROMPT` (V1) | `agents/writer.py` | system_prompt property | test_chapter.py | Only when prompt_profile!="v2" | V2 is default and only path | DELETE |
| `writer_prompt_profile` | `graph/state.py`, `chapter.py:353` | chapter.py writer_node | test_chapter.py | Always "v2" | V1/V2 branch | DELETE |
| `prompt_profile` param | `agents/writer.py` | WriterAgent constructor | test_chapter.py | — | V1/V2 selection | DELETE |
| `primary_storyline` | `schema/models.py`, `orchestrator.py` | orchestrator defaults:284, prompt:63 | test_validator.py | Never read by code | Replaced by `storylines` | DELETE |
| Editor legacy flat scores | `schema/models.py:109-112` | validator.py coercion:157-160 | test_validator.py | Never read in production | Replaced by `dimensions` | DELETE |
| `_migrate()` | `storage/models.py` | `init_db()` | test_manager.py | Schema evolution | Pre-prod: CREATE TABLE has all cols | DELETE (Phase D) |
| `rewrite_instructions` legacy path | `graph/chapter.py:371-377` | writer_node | — | Old feedback loop | Replaced by evolution_improvement_plan | DELETE (Phase B) |
| `retry_count` in state | `graph/state.py` comment, `chapter.py:910`, `sse.py:94` | human_review_node, sse | — | Only displayed, not used for logic | Old linear retry | DELETE from state usage |
| `style/ai_flavor.py` | — | — | — | Already deleted (only .pyc remains) | Old style tool | CONFIRM DELETE |
| `tools/detect_ai_flavor.py` | — | — | — | Already deleted (only .pyc remains) | Old style tool | CONFIRM DELETE |
| `tools/style.py` | — | — | — | Already deleted (only .pyc remains) | Old style tool | CONFIRM DELETE |
| `context/compiler.py` | — | — | — | Already deleted (only .pyc remains) | Old context compiler | CONFIRM DELETE |
| `routing/` dir | — | — | — | Empty (only __pycache__) | Unused | DELETE |
| Writer individual context params | `agents/writer.py` write/write_stream | chapter.py writer_node | — | Overridden by context_packet | ContextCompiler projections exist | CONSOLIDATE |
| Orchestrator context params | `agents/orchestrator.py` analyze | chapter.py orchestrator_node | — | Overridden by context_packet | ContextCompiler exists | CONSOLIDATE |
| `context_packet` in State | `graph/state.py:45` | Multiple nodes | — | Stored in state, rebuildable | Derived from ContextCompiler | CONSOLIDATE (evaluate removal) |
| `character_context` etc. in State | `graph/state.py:40-43` | orchestrator_node, writer_node, routes | — | Derived from context_packet | Derived state | CONSOLIDATE |
| `scene_drafts` in State | `graph/state.py:50` | writer_node, quality_gate | — | Only current node | Temporary | KEEP (needed for quality gate) |
| `style_report` in State | `graph/state.py:32` | editor_node, evolution | — | Recomputed each time | Derived | KEEP (needed by evolution) |
| `backfill_world_relations` | `storage/manager.py` | graph_data.py | test_relations.py | Legacy backfill | Pre-prod: no old data to backfill | DELETE (Phase D) |
| `get_chapter_reports` | `storage/manager.py:1270` | — | — | Unused | No callers | DELETE (Phase D) |
| `organization` alias in graph_data | `api/graph_data.py:13,25` | — | — | Entity type display | Not legacy API | KEEP |
| `ContextCompiler` projections | `services/context.py` | chapter.py, routes.py | test_context_service.py | Current context system | Current contract | KEEP |
| `StyleAnalyzer` | `style/analyzer.py` | editor_node | test_ai_flavor.py | Current style system | Current contract | KEEP |
| `Parser` | `schema/parser.py` | All agents | test_parser.py | Current parser | Current contract | KEEP |
| `OutputValidator` | `schema/validator.py` | All agents | test_validator.py | Current validator | Current contract | KEEP (trim legacy coercion) |
