"""AI-powered outline generation for novel projects."""

from novel_agent.agents.base import AgentConfig
from novel_agent.agents.orchestrator import OrchestratorAgent
from novel_agent.model_router import ModelRouter, TaskClass
from novel_agent.schema.enums import OutlineStatus
from novel_agent.schema.parser import parse_json_response
from novel_agent.storage.manager import ProjectManager

router = ModelRouter()

OUTLINE_SYSTEM_PROMPT = """你是一个小说主编，负责为一本小说规划全书章节大纲。

根据小说的梗概、篇幅和类型，规划每一章的标题和简短概要。
章节数量取决于篇幅：短篇3-10章，中篇20-50章，长篇50-100+章。

## 输出格式

```json
{
  "chapters": [
    {
      "chapter_number": 1,
      "title": "章节标题",
      "summary": "本章概要，50-200字"
    }
  ]
}
```

只输出JSON。
"""

# A full-book outline (50-100+ chapters × 50-200字 summary) needs far more
# than the 4096 default; otherwise the model hits its output limit and returns
# truncated/empty content (``finish_reason="length"``), yielding no chapters.
# 16K is still occasionally exceeded by verbose 100-chapter outlines, so give
# headroom up to 32K — the model stops early (finish_reason="stop") when done.
OUTLINE_MAX_TOKENS = 32768


async def generate_outline(mgr: ProjectManager, project_id: str) -> list[dict]:
    """Generate a chapter outline for a project using AI.

    Returns a list of {chapter_number, title, summary} dicts.
    """
    project = mgr.get_project(project_id)
    if not project:
        raise ValueError(f"Project {project_id} not found")

    title = project.get("title") or project.get("name", "")
    genre = project.get("genre", "")

    length_label = "长篇"
    chapter_range = "50-100章"

    route = router.resolve(TaskClass.STRUCTURAL)
    agent_config = AgentConfig(
        model=route.model,
        api_key=route.api_key,
        base_url=route.base_url,
        temperature=0.7,
        max_tokens=OUTLINE_MAX_TOKENS,
        is_reasoning=route.is_reasoning,
    )

    agent = OrchestratorAgent(config=agent_config)

    messages = [
        {"role": "system", "content": OUTLINE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"请为以下小说规划全书大纲：\n\n"
                f"- 书名：{title}\n"
                f"- 类型：{genre or '未指定'}\n"
                f"- 篇幅：{length_label}（约{chapter_range}）\n"
                f"- 梗概：{project.get('outline', '') or '请根据书名和类型自由发挥'}\n\n"
                f"只输出JSON。"
            ),
        },
    ]

    content, _ = await agent.run_with_tools(
        messages,
        max_rounds=1,
        action="generate_outline",
    )

    result = parse_json_response(content, defaults={"chapters": []})
    chapters = result.get("chapters", [])

    if not chapters:
        raise ValueError(
            "模型未返回有效章节（输出可能超长被截断），请重试"
        )

    # Ensure required fields
    for i, ch in enumerate(chapters):
        ch.setdefault("chapter_number", i + 1)
        ch.setdefault("title", f"第{ch['chapter_number']}章")
        ch.setdefault("summary", "")
        ch["status"] = OutlineStatus.PENDING.value
        ch["sort_order"] = ch["chapter_number"]

    # Save to database
    if chapters:
        mgr.save_outline(project_id, chapters)

    return chapters
