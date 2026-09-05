"""humanize_passage tool — LLM-backed rewrite to remove AI writing patterns.

Distilled from humanizer-zh (https://github.com/op7418/Humanizer-zh),
focused on patterns that matter for Chinese web fiction. Preserves all
plot, characters, and facts; only rewrites prose to sound less AI-generated.
"""

from pydantic import BaseModel, Field

from novel_agent.agents.base import AgentConfig, build_chat_model
from novel_agent.tools.base import BaseTool, ToolResult

HUMANIZER_SYSTEM_PROMPT = """\
你是文字编辑，专门去除中文小说文本中的AI生成痕迹，使文字更自然、更有人味。

## 核心原则

1. **保留所有剧情、人物、设定、对白含义**——只改写法，不改故事。
2. **不要新增事实**——不添加原文没有的细节、名字、数字。
3. **维持语调**——保持原文的叙事视角和情感基调。
4. **注入灵魂**——去除AI模式后，让文字有真实人的节奏和个性。

## 必须消除的AI写作模式

### 句式套路
- **否定式排比**：「不仅是…更是…」「不是…而是…」当修辞套路反复堆砌→改为直接陈述。
- **强行三连**：凑三个词组当排比，三者并非真并列→改为两项或拆开。
- **同义词循环**：反复换词指同一人/物（"主人公…主要角色…中心人物"）→用统一称呼。
- **系动词回避**：用「作为/代表/充当/标志着」替代简单的「是」→改回简单动词。
- **分词式肤浅分析**：句末加「，彰显了/反映了/体现了…」凑深度→删除或改为具体动作。
- **虚假范围**：「从X到Y」当宏大叙事，X、Y并非真在一条尺度上→改为具体列举。

### 内容夸饰
- **过度强调意义**：「标志着/见证了/是…的体现/为…奠定基础/不可磨灭的印记」→删除夸大，只留事实。
- **宣传式语言**：「令人叹为观止/迷人的/充满活力/坐落于」→改为朴素描述。
- **通用积极结论**：「未来光明/激动人心的时代/迈向卓越」→删除空泛正能量，以具体动作收束。
- **模糊归因**：「专家认为/观察者指出/行业报告显示」→删除无来源的权威背书。

### 节奏与金句
- **戏剧性断句堆叠**：连续多个单句段硬当高潮（"他来了。他没有回头。他消失了。"）
  →合并或改为自然节奏。
- **公式化金句**：「X是Y的Z」「真正的…从来都是…」当结语→信任读者，删除金句。
- **重复句首**：连续多句用同一主语起头→变化句首，打破单调。
- **过度限定**：「可能…或许…潜在地…」限定词堆叠→保留必要的一个，删除其余。
- **假坦白/谄媚**：「说实话」「老实讲」假开场→直接陈述。

## 改写要求

- 保持原文的剧情推进、场景结构、对白内容不变。
- 可以调整句子顺序、合并段落、变化句长，但不要删减剧情信息。
- 输出完整的改写后正文，不要输出分析、说明或元信息。
"""


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
