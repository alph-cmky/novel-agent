"""单章 token 实测：按生产路径（ContextCompiler + chapter graph）跑一章，
从 NovelState 读出各节点真实 token 消耗。只写 checkpoint，不落章节库。

用法：.venv/bin/python scripts/measure_chapter_tokens.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from novel_agent.graph.chapter import aclose_checkpointers, build_chapter_graph_async
from novel_agent.services.context import ContextCompiler
from novel_agent.storage.manager import ProjectManager

PERSIST = "novel-data/eval_long_novel"
PROJECT = "01b58117"
THREAD = f"{PROJECT}:ch1-tokmeas"

CH1_OUTLINE = """第一章 断刃初鸣
- 开篇：林绝在落霞宗外门杂役院劈柴，练气三层停滞两年，被同门讥为废柴。
- 冲突起因：外门执事刘恒克扣杂役弟子本月灵米，林绝出面理论反遭刁难。
- 关键场景：争执中林绝掌心被柴刃划破，血溅祖传断刃「青冥断片」，断刃泛起青芒，剑魂低语，林绝首次察觉断刃藏有秘密。
- 收束：林绝隐下异象，忍辱领罚；夜里断刃在枕下微鸣，暗示修行停滞与断刃封印有关，埋下与赵乾长老一脉对抗的伏笔。"""


async def main() -> None:
    mgr = ProjectManager(PERSIST)
    ctx = ContextCompiler(mgr).compile(PROJECT, 1).to_state()

    graph = await build_chapter_graph_async(persist_dir=PERSIST)
    initial_state = {
        "project_id": PROJECT,
        "chapter_number": 1,
        "chapter_outline": CH1_OUTLINE,
        "story_length": "long",
        "target_chapter_words": 2000,
        "narrative_mode": None,
        "narrative_perspective": "",
        "context_packet": ctx.get("context_packet", {}),
        "scene_first": False,
        "persist_dir": PERSIST,
        "evolution_max_rounds": 1,
    }
    config = {"configurable": {"thread_id": THREAD}}

    try:
        async for ev in graph.astream(initial_state, config=config, stream_mode="updates"):
            for node, delta in ev.items():
                keys = sorted(delta.keys()) if isinstance(delta, dict) else []
                print(f"[node] {node}: {keys}")
    finally:
        await aclose_checkpointers()

    snap = await build_chapter_graph_async(persist_dir=PERSIST)
    final = await snap.aget_state(config)
    await aclose_checkpointers()
    v = final.values

    def tok(prefix: str) -> str:
        i, o = v.get(f"{prefix}_input_tokens", 0), v.get(f"{prefix}_output_tokens", 0)
        c, r = v.get(f"{prefix}_cached_tokens", 0), v.get(f"{prefix}_reasoning_tokens", 0)
        return f"in={i} out={o} cached={c} reasoning={r}"

    print("\n===== TOKEN 报告（provider usage_metadata 实测）=====")
    print(f"orchestrator: {tok('orchestrator')}")
    print(f"writer:       {tok('writer')}")
    print(f"editor:       {tok('editor')}")
    total_in = sum(v.get(f"{p}_input_tokens", 0) for p in ("orchestrator", "writer", "editor"))
    total_out = sum(v.get(f"{p}_output_tokens", 0) for p in ("orchestrator", "writer", "editor"))
    print(f"三章节点合计: in={total_in} out={total_out}")
    print("\n===== 调用统计 =====")
    print(
        f"writer: model={v.get('writer_model_calls', 0)} "
        f"tool={v.get('writer_tool_calls', 0)} search={v.get('writer_search_calls', 0)}"
    )
    print(
        f"evolution: rule_plans={v.get('evolution_rule_plan_calls', 0)} "
        f"llm_enrichments={v.get('evolution_llm_enrichment_calls', 0)} "
        f"rounds={len(v.get('evolution_history', []))} "
        f"termination={v.get('evolution_termination', '') or '(interrupted at review)'}"
    )
    draft = v.get("draft_content", "")
    print(f"\ndraft: {len(draft)} chars")
    print("editor score:", (v.get("editor_report") or {}).get("overall_score"))
    print("continuity score:", (v.get("continuity_report") or {}).get("overall_score"))


if __name__ == "__main__":
    asyncio.run(main())
