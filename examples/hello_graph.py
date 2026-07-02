"""
Minimal LangGraph demo: a mock chapter-writing pipeline.

This is a checkpoint to verify the stack works before building real agents.
Run: uv run python examples/hello_graph.py
"""
from typing import Literal, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph

# ── State ──────────────────────────────────────────────

class NovelState(TypedDict):
    chapter_number: int
    outline: str
    draft: str
    editor_score: int
    approved: bool


# ── Nodes ──────────────────────────────────────────────

def writer_node(state: NovelState) -> dict:
    """Mock: generate chapter content from outline."""
    print(f"  [Writer] Generating chapter {state['chapter_number']}...")
    draft = (
        f"Chapter {state['chapter_number']}\n\n"
        f"基于大纲「{state['outline'][:40]}...」展开的正文内容。\n"
        f"这里是主角的冒险故事..."
    )
    return {"draft": draft}


def editor_node(state: NovelState) -> dict:
    """Mock: review draft and assign a score."""
    print(f"  [Editor] Reviewing draft ({len(state['draft'])} chars)...")
    # Simulated review: count certain AI-flavor keywords
    ai_markers = ["值得注意的是", "总而言之", "他的眼中闪过"]
    score = 90
    for marker in ai_markers:
        if marker in state["draft"]:
            score -= 10
    return {"editor_score": score}


def human_review(state: NovelState) -> dict:
    """Interactive: ask user to approve or reject."""
    print(f"\n  ┌─ Chapter {state['chapter_number']} Preview ──────────────")
    print(f"  │ {state['draft'][:100]}...")
    print(f"  │ Editor Score: {state['editor_score']}/100")
    print("  └────────────────────────────────────────────")

    choice = input("  [Human] Approve? (y/n): ").strip().lower()
    return {"approved": choice == "y"}


# ── Router ─────────────────────────────────────────────

def route_after_editor(state: NovelState) -> Literal["human_review", "writer"]:
    if state["editor_score"] >= 70:
        return "human_review"
    else:
        print("  → Score too low, sending back to Writer...")
        return "writer"


def route_after_human(state: NovelState) -> Literal["__end__", "writer"]:
    if state["approved"]:
        return "__end__"
    else:
        print("  → Rejected, sending back to Writer...")
        return "writer"


# ── Build Graph ────────────────────────────────────────

def build_graph():
    workflow = StateGraph(NovelState)

    workflow.add_node("writer", writer_node)
    workflow.add_node("editor", editor_node)
    workflow.add_node("human_review", human_review)

    workflow.set_entry_point("writer")
    workflow.add_edge("writer", "editor")
    workflow.add_conditional_edges("editor", route_after_editor)
    workflow.add_conditional_edges("human_review", route_after_human)

    return workflow.compile(checkpointer=MemorySaver())


# ── Run ────────────────────────────────────────────────

if __name__ == "__main__":
    graph = build_graph()
    config = {"configurable": {"thread_id": "demo-1"}}

    print("=" * 50)
    print("LangGraph Chapter Writing Demo")
    print("=" * 50)

    initial_state = {
        "chapter_number": 1,
        "outline": "主角在废弃工厂发现了一个神秘装置，触碰后被传送到异世界。",
        "draft": "",
        "editor_score": 0,
        "approved": False,
    }

    result = graph.invoke(initial_state, config)

    print(f"\n{'=' * 50}")
    print("Final State:")
    print(f"  draft length: {len(result['draft'])} chars")
    print(f"  editor_score: {result['editor_score']}")
    print(f"  approved: {result['approved']}")
    print(f"{'=' * 50}")
