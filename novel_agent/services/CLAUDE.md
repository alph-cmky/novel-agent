# Services Rules

`ContextCompiler` is the single source of task-specific Agent context. Task
projections: orchestrator, writer, extension, editor, continuity.

Retrieval pattern: task need → relevant query → minimal data → minimal
context. Use narrow queries (`ORDER BY ... LIMIT ...`) for recent data.

`QualityService` runs after the Writer loop as a deterministic hard gate. It
does not run inside the loop. Empty or grossly short content is rejected
regardless of loop outcome.

`EvolutionService` is removed in the agent loop architecture. The Writer
self-revises via tool calls; `max_rounds` bounds the loop. Do not reintroduce
evolution termination logic, version comparison, or improvement-plan
generation as a separate service.
