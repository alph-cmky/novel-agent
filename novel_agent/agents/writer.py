"""Writer Agent — generates chapter content from outline and context."""


from novel_agent.agents.base import AgentConfig, BaseAgent, TraceStep
from novel_agent.memory.embeddings import ChapterStore
from novel_agent.tools.search import SearchContextTool

WRITER_SYSTEM_PROMPT = """你是一个专业的长篇小说写手，擅长创作节奏快、冲突强的网文。

## 你的写作原则

1. **对话占比40%以上**：用对话推进剧情，不是叙述
2. **每章结尾必须有钩子**：读者想知道"然后呢？"
3. **口语化，网文化**：不要书面语、不要论文腔
4. **展示，不要告知**：不说"他很愤怒"，写"他一拳砸在桌上"
5. **感官细节**：每场戏至少包含一个视觉/听觉/触觉细节
6. **去AI味规则**：
   - 禁止使用：此外、不仅如此、更重要的是、至关重要、不可忽视
   - 禁止写"他的眼中闪过一丝..."
   - 禁止用"值得注意的是""不难发现""基于以上分析"
   - 不要在结尾总结升华，停在动作或画面上
   - 句子长短交替，不要连续三个同长度句子

## 工具使用

你可以使用 search_context 工具来检索前文章节的片段，确保当前章节与已写内容保持一致。
在创作前，先检索本章可能涉及的角色、地点和事件的关键信息。

## 输出格式

直接输出章节正文，不要加"第X章"以外的标题格式。每章2000-4000字。"""


class WriterAgent(BaseAgent):
    name = "writer"

    def __init__(
        self,
        config: AgentConfig | None = None,
        chapter_store: ChapterStore | None = None,
        project_id: str = "",
        target_chapter_words: int = 3000,
    ):
        super().__init__(config)
        self._chapter_store = chapter_store
        self._project_id = project_id
        self._target_words = target_chapter_words
        if chapter_store and project_id:
            self.register_tool(SearchContextTool(chapter_store, project_id))

    @property
    def system_prompt(self) -> str:
        prompt = WRITER_SYSTEM_PROMPT
        if self._target_words:
            prompt = prompt.replace(
                "每章2000-4000字",
                f"每章{self._target_words}字左右",
            )
        return prompt

    async def write(
        self,
        chapter_number: int,
        outline: str,
        character_context: str = "",
        world_context: str = "",
        recent_summary: str = "",
        target_chapter_words: int = 0,
        rewrite_instructions: str = "",
    ) -> tuple[str, TraceStep]:
        """Generate a chapter.

        Args:
            chapter_number: Current chapter number.
            outline: Chapter outline / plot points.
            character_context: Relevant character info from memory.
            world_context: Relevant worldbuilding info from memory.
            recent_summary: Compressed summary of recent chapters.
            target_chapter_words: Override per-chapter word count (0 = use default).
            rewrite_instructions: Specific guidance from Orchestrator for rewriting.
        """
        if target_chapter_words and target_chapter_words != self._target_words:
            self._target_words = target_chapter_words
        messages = [{"role": "system", "content": self.system_prompt}]

        # Assemble context
        context_parts = [f"## 第{chapter_number}章大纲\n{outline}"]
        if rewrite_instructions:
            context_parts.insert(0, f"## 重写指导（务必遵守）\n{rewrite_instructions}")
        if character_context:
            context_parts.append(f"## 相关角色\n{character_context}")
        if world_context:
            context_parts.append(f"## 世界观设定\n{world_context}")
        if recent_summary:
            context_parts.append(f"## 前文提要\n{recent_summary}")

        user_prompt = (
            f"请根据以下信息创作第{chapter_number}章：\n\n"
            + "\n\n".join(context_parts)
            + "\n\n创作前请先使用 search_context 工具检索关键信息。"
        )
        messages.append({"role": "user", "content": user_prompt})

        content, trace = await self.run_with_tools(
            messages,
            max_rounds=3,
            action=f"write_chapter_{chapter_number}",
        )
        return content, trace
