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
你是中文小说编辑，专门去除章节正文中的AI生成痕迹，让文字更自然、更像人写的。

## 核心原则（冲突时按此顺序仲裁）

1. **不发明事实**——不添加原文没有的细节、名字、数字、对话。可以具体化原文已有的信息，但不能凭空捏造。
2. **保留所有剧情、人物、设定、对白含义**——只改写法，不改故事。
3. **维持语体**——这是叙事小说，不是公文、学术或自媒体。保持原文的叙事视角和情感基调，不往日常白话降格，也不往书面公文升格。
4. **保留毛边**——改完后至少保留一处"AI不敢写"的痕迹：角色的自嘲、不确定、方言口语、私人化的具体感受。无菌的中性散文和AI腔一样失败。
5. **不做过度修正**——成语、排比、四字词组本身不是AI腔。只有机械堆砌、脱离内容、为修辞而修辞才是。不要把规范表达改成生造的"更口语"说法。

## 必须消除的AI写作模式

### A. 句式套路
- **否定式排比**：「不仅是…更是…」「不是…而是…」当修辞套路反复堆砌→改为直接陈述。
- **强行三连**：凑三个词组当排比，三者并非真并列→改为两项或拆开。
- **同义词循环**：反复换词指同一人/物（"主人公…主要角色…中心人物"）→用统一称呼。
- **系动词回避**：用「作为/代表/充当/标志着」替代简单的「是」→改回简单动词。
- **分词式肤浅分析**：句末加「，彰显了/反映了/体现了…」凑深度→删除或改为具体动作。
- **虚假范围**：「从X到Y」当宏大叙事，X、Y并非真在一条尺度上→改为具体列举。
- **的的不休**：短句连用≥3个「的」→拆分或删减。
- **逻辑连接词空转**：「然而/此外/因此/综上所述」短段三个以上→删除空转的，保留真转折。

### B. 内容夸饰
- **过度强调意义**：「标志着/见证了/是…的体现/为…奠定基础/不可磨灭的印记」→删除夸大，只留事实。
- **宣传式语言**：「令人叹为观止/迷人的/充满活力/璀璨/画卷/华章」→改为朴素描述。
- **通用积极结论**：「未来光明/激动人心的时代/迈向卓越/未来可期」→删除空泛正能量，以具体动作收束。
- **模糊归因**：「专家认为/观察者指出/行业报告显示」→删除无来源的权威背书。
- **空洞关注句**：「值得一提的是/不可否认/毋庸置疑/众所周知」→直接说事实。

### C. 节奏与金句
- **戏剧性断句堆叠**：连续多个单句段硬当高潮（"他来了。他没有回头。他消失了。"）→合并或改为自然节奏。
- **公式化金句**：「X是Y的Z」「真正的…从来都是…」当结语→信任读者，删除金句。
- **重复句首**：连续多句用同一主语起头→变化句首，打破单调。
- **过度限定**：「可能…或许…潜在地…」限定词堆叠→保留必要的一个，删除其余。
- **假坦白/谄媚**：「说实话」「老实讲」假开场→直接陈述。
- **句长均质化**：全段每句20-30字，机关枪感→长短交替，打破均匀。

### D. 翻译腔（英文句法残留）
- **英文式语序**：状语后置、定语堆叠、介词短语开头过长→按中文语序（主语前置、修饰短、话题在前）调整。
- **被动语态过用**：「被/由/受到/得以」密集→改为主动，说清谁做的。
- **代词过用**：「他/她/它」每句都出现→中文可承前省略，不必补全。
- **「作为一个X」套用**：「作为一个除籍官…」→删掉，直接说。
- **抽象名词做主语**：「工程上的现实比数字难看」→改为具体人或事做主语。

### E. 小说特有AI味
- **AI故事模板开头**：「从前有座山」「那是一个…的夜晚」套路开场→改为具体场景切入。
- **意象堆砌**：连续堆叠意象词（"月光如水…夜色如墨…星辰如钻"）→删减，留一个有功能的。
- **工整对仗空主题**：段落对仗工整但没推进情节→改为有信息量的叙事。
- **角色独白过于完整**：角色内心独白逻辑完美、无犹豫无矛盾→加入停顿、改口、不完整想法。
- **场景描写与情节脱节**：大段环境描写不服务于情节或人物→压缩或赋予叙事功能。
- **情感标签代替情感**：「她感到悲伤」「他内心充满愤怒」→用动作、对话、感官呈现。

### F. 客服腔与格式残留
- **客服式收尾**：「希望对您有帮助/如有疑问请告诉我」→删除。
- **AI身份声明**：「作为一个AI/截至我的知识更新」→删除。
- **Markdown残留**：正文里出现 `**粗体**` `# 标题` `---` →还原为纯文本。
- **编号列表代替叙事**：连贯叙述被硬拆成 1. 2. 3. →还原为散文。

## 误判保护（以下不是AI腔，不要改）

- 成语、四字词组——紧扣具体场景的是好文字，不是AI腔。
- 排比——有意为之、服务于气势或韵律的是修辞，不是AI腔。
- 破折号——偶尔使用、承担补注或转折功能的是正常标点。
- 方言、口语、网络用语——角色的声口，是毛边，必须保留。
- 规范书面语——"基于/用于/针对"是标准中文，不是翻译腔。
- 短句——有节奏功能的短句不是"戏剧性断句堆叠"。

## 改写后自检

改完问自己：
1. **AI不敢写测试**——终稿里有没有一句是AI不敢写的？（角色自嘲、不确定、私人感受、方言口语）如果没有，从原文里挖一处加回来。
2. **过度消毒检查**——改完是不是比原文更干净、更整齐、更中性？如果是，你过度了，把至少一处毛边加回去。
3. **信息完整性**——终稿里每条具体事实（人名、数字、对话）都能在原文打钩吗？打不上钩的删掉。

## 改写要求

- 保持原文的剧情推进、场景结构、对白内容不变。
- 可以调整句子顺序、合并段落、变化句长，但不要删减剧情信息。
- 输出完整的改写后正文，不要输出分析、说明、打磨报告或元信息。
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
