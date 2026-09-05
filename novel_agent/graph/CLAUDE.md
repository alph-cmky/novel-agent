# Graph Rules

The agent loop runner (`graph/agent_loop.py`) replaces the LangGraph DAG. The
LangGraph code (`graph/chapter.py`) is retained as a fallback but is not the
default path.

Before adding a state field, verify:

1. the value does not already exist in State or Storage;
2. it cannot be recomputed;
3. it must survive the loop result.

Do not store large durable objects (full chapter versions, world snapshots,
derived context) in State when they already exist in Storage — store
identifiers and compact decision metadata instead.

The loop runner assembles the final state dict from: Writer output (draft
content), shared tool context (editor report, worldbuilding report), and
deterministic QualityGate. Token accounting is per-role, compatible with the
eval adapter's expectations.
