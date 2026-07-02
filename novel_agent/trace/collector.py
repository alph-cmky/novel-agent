"""TraceCollector — records agent traces during pipeline execution.

Collects all TraceStep objects, enriches with pipeline metadata,
and writes to JSON for later replay and analysis.
"""

import json
import time
from pathlib import Path

from novel_agent.agents.base import TraceStep


class TraceCollector:
    """Collects and persists agent traces as structured JSON."""

    def __init__(self, output_dir: str = "./traces"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.steps: list[dict] = []
        self.meta: dict = {
            "pipeline_start": time.time(),
            "project_id": "",
            "chapter_number": 0,
        }

    def start(self, project_id: str, chapter_number: int):
        self.meta["project_id"] = project_id
        self.meta["chapter_number"] = chapter_number
        self.meta["pipeline_start"] = time.time()
        self.steps.clear()

    def record(self, step: TraceStep):
        entry = step.to_dict()
        entry["timestamp"] = time.time()
        self.steps.append(entry)

    def finish(self) -> Path:
        elapsed = time.time() - self.meta["pipeline_start"]
        record = {
            "meta": {
                **self.meta,
                "pipeline_elapsed_s": round(elapsed, 2),
            },
            "total_tokens_input": sum(s["input_tokens"] for s in self.steps),
            "total_tokens_output": sum(s["output_tokens"] for s in self.steps),
            "total_tool_calls": sum(len(s["tool_calls"]) for s in self.steps),
            "step_count": len(self.steps),
            "steps": self.steps,
        }
        filename = (
            f"trace-{self.meta['project_id']}-ch{self.meta['chapter_number']}"
            f"-{int(self.meta['pipeline_start'])}.json"
        )
        path = self.output_dir / filename
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
        return path
