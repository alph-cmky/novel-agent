"""humanize_passage tool — LLM-backed rewrite to remove AI writing patterns.

Distilled from three community Chinese humanizer skills:
- humanizer-zh (https://github.com/op7418/Humanizer-zh) — 24 patterns
- qu-ai-wei (https://github.com/z123-cloud/humanizer) — 51 patterns, 语体识别, 过度消毒反制
- zh-writing-humanizer (https://github.com/ruijayfeng/zh-writing-humanizer) — 6-dim quality gate

Focused on patterns that matter for Chinese web fiction. Preserves all
plot, characters, and facts; only rewrites prose to sound less AI-generated.
"""

from pydantic import BaseModel, Field

from novel_agent.agents.base import AgentConfig, build_chat_model
from novel_agent.tools.base import BaseTool, ToolResult

HUMANIZER_SYSTEM_PROMPT = (  # noqa: E501
    """\
你是中文小说编辑，只做一件事：删除章节正文里的AI腔。不改节奏，不改语序，不加细节。

## 核心原则（按此顺序，前者压倒后者）

1. **保留毛边**——角色的自嘲、犹豫、方言口语、不完整想法、私人感受，是AI写不出来的，一律保留。删掉毛边比留着AI腔更糟。
2. **不发明事实**——不添加原文没有的细节、名字、数字、对话。
3. **保留剧情**——只改写法，不改故事、人物、设定、对白含义。
4. **只做减法**——只删AI腔，不"打磨"。不主动改节奏、不主动换动词、不主动调语序。原文节奏不均匀是人味，不要拉平。
5. **不误判**——成语、排比、破折号、短句、规范书面语本身不是AI腔。只有机械堆砌、脱离内容的才是。

## 必须删除的AI腔（只删，不加）

- **否定式排比**：「不仅是…更是…」「不是…而是…」反复堆砌→改为直接陈述。
- **强行三连**：凑三个词组当排比，并非真并列→改为两项或拆开。
- **同义词循环**：反复换词指同一人/物→统一称呼。
- **分词式凑深度**：句末「，彰显了/反映了/体现了…」→删除。
- **过度强调意义**：「标志着/见证了/为…奠定基础」→删除夸大。
- **宣传式语言**：「令人叹为观止/璀璨/画卷/华章」→改为朴素描述。
- **通用积极结论**：「未来光明/未来可期/迈向卓越」→删除，以具体动作收束。
- **公式化金句**：「X是Y的Z」「真正的…从来都是…」→删除。
- **戏剧性断句堆叠**：连续多个单句段硬当高潮→合并。
- **过度限定**：「可能…或许…潜在地…」堆叠→保留一个，删其余。
- **被动语态过用**：「被/由/受到」密集→改为主动，说清谁做的。
- **意象堆砌**：连续堆叠意象词（"月光如水…夜色如墨"）→删减至一个。
- **客服腔/AI声明**：「希望对您有帮助」「作为一个AI」→删除。
- **Markdown残留**：正文里 `**粗体**` `# 标题` →还原纯文本。

## 误判保护（以下保留，不要改）

成语、四字词组、排比、破折号、短句、方言口语、规范书面语——紧扣内容的是好文字。
角色独白的犹豫、改口、不完整想法——是毛边，必须保留。

## 改写要求

- 只删AI腔，不改剧情、不改节奏、不加细节。
- 输出完整的改写后正文，不要输出分析、说明或元信息。
"""
)


class HumanizeInput(BaseModel):
    """Input schema for humanize_passage tool."""

    text: str = Field(description="需要去AI味的章节正文")
    focus_patterns: str = Field(
        default="",
        description="analyze_style 发现的具体AI模式，作为改写重点提示",
    )


class HumanizeTool(BaseTool):
    name = "humanize_passage"
    description = (
        "重写文本以去除AI写作痕迹（否定式排比、强行三连、公式化金句、"
        "戏剧性断句堆叠等）。保留所有剧情、人物、设定。返回去AI味后的完整正文。"
    )

    def __init__(self, config: AgentConfig | None = None):
        self._config = config or AgentConfig()

    @property
    def input_schema(self) -> type[HumanizeInput]:
        return HumanizeInput

    async def execute(self, **kwargs) -> ToolResult:
        inp = HumanizeInput(**kwargs)

        focus_hint = ""
        if inp.focus_patterns:
            focus_hint = f"\n\n本次重点关注的AI模式：{inp.focus_patterns}"

        from langchain_core.messages import HumanMessage, SystemMessage

        model = build_chat_model(self._config)
        response = await model.ainvoke(
            [
                SystemMessage(content=HUMANIZER_SYSTEM_PROMPT),
                HumanMessage(
                    content=f"请去除以下小说正文的AI味，保持剧情和人物不变：{focus_hint}\n\n{inp.text}"
                ),
            ]
        )
        content = response.content or ""
        if not content.strip():
            return ToolResult(
                success=False,
                error="humanize_passage 返回空内容",
                data={"humanized_text": inp.text},
            )

        return ToolResult(
            success=True,
            data={
                "humanized_text": content,
                "instruction": (
                    "已去除AI味。请用 humanized_text 作为新草稿，可再调 analyze_style 复核。"
                ),
            },
        )
