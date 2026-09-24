# Style Rules

`analyze_style` is a deterministic tool (0 LLM) exposed to the Writer loop.
It measures prose; it does not prescribe prose mechanically. Style findings
are evidence for the Writer, not verdicts.

Distinguish narrative / dialogue / mixed paragraphs. Support common Chinese
quotation marks (`""`, `「」`, `『』`, `""`). Pure dialogue must not be
treated as fragmented narration. Short sentences in action scenes are not
automatically bad.

Isolated words (突然, 忽然, 下一秒, 就在这时) are not proof of AI writing.
Do not reward keyword-based cliffhangers or use fixed global dialogue ratios
as quality requirements.

Before adding a new style metric, show: the real failure pattern, why
existing metrics miss it, expected effect, and false-positive risk. Prefer
fewer meaningful metrics over many correlated scores.

AI-flavor rewriting is `humanize_passage`'s job (LLM tool), not
`analyze_style`'s. Do not add rewriting logic to the deterministic style
tool.
