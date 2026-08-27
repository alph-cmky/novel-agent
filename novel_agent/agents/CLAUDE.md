# Agents Rules

Writer is the only component responsible for prose generation.

Writer prompts should remain short and high-value. Do not add checklist-style writing rules unless a real evaluation problem justifies them. Avoid forcing fixed dialogue ratios, paragraph sizes, mandatory cliffhangers, or mechanical sentence-length alternation.

Orchestrator decides what the chapter needs — it is not a complete novel-analysis engine. Only emit mode-specific planning fields when the execution mode needs them.

Editor and Continuity receive only the context necessary for their task. Do not re-fetch facts already present in the projected context.

EvolutionOrchestratorAgent is fallback, not default. Prefer deterministic `EvolutionService` rules. When revising a draft, preserve existing facts and unaffected narrative unless the improvement plan explicitly requires otherwise.
