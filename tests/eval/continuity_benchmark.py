"""Continuity Agent benchmark — injects known bugs and scores detection.

Each test case is a chapter pair with injected inconsistencies.
The ContinuityAgent audits the second chapter and we measure:
- precision: detected issues that match ground truth / total detected
- recall: detected issues that match ground truth / total ground truth
- f1: harmonic mean of precision and recall
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class InjectedBug:
    """A known inconsistency injected into a test chapter."""

    category: str  # character, timeline, worldbuilding
    severity: str  # critical, major, minor
    description: str
    location_hint: str  # where in the chapter the bug was placed
    keywords: list[str] = field(default_factory=list)  # 可检测特征词，供评分匹配


@dataclass
class BenchmarkCase:
    """A single benchmark test case."""

    name: str
    description: str
    chapter_number: int
    chapter_outline: str
    draft_content: str
    injected_bugs: list[InjectedBug] = field(default_factory=list)
    previous_context: str = ""  # context from earlier chapters


@dataclass
class BenchmarkResult:
    """Results for a single benchmark case."""

    case_name: str
    total_injected: int
    total_detected: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    details: list[dict] = field(default_factory=list)

    def __post_init__(self):
        if self.total_detected > 0:
            self.precision = self.true_positives / self.total_detected
        if self.total_injected > 0:
            self.recall = self.true_positives / self.total_injected
        if self.precision + self.recall > 0:
            self.f1 = 2 * self.precision * self.recall / (self.precision + self.recall)


class ContinuityBenchmark:
    """Runs continuity detection benchmarks against injected bugs."""

    def __init__(self):
        self.cases: list[BenchmarkCase] = []
        self.results: list[BenchmarkResult] = []

    def add_case(self, case: BenchmarkCase):
        self.cases.append(case)

    def load_cases(self, cases):
        """Load benchmark cases from a list, JSON string, or file path."""
        if isinstance(cases, list):
            data = cases
        elif os.path.isfile(cases):
            data = json.loads(open(cases).read())
        else:
            data = json.loads(cases)

        for entry in data:
            bugs = [
                InjectedBug(
                    category=b.get("category", ""),
                    severity=b.get("severity", "minor"),
                    description=b.get("description", ""),
                    location_hint=b.get("location_hint", ""),
                    keywords=b.get("keywords", []),
                )
                for b in entry.get("injected_bugs", [])
            ]
            self.add_case(BenchmarkCase(
                name=entry.get("name", "unnamed"),
                description=entry.get("description", ""),
                chapter_number=entry.get("chapter_number", 1),
                chapter_outline=entry.get("chapter_outline", ""),
                draft_content=entry.get("draft_content", ""),
                injected_bugs=bugs,
                previous_context=entry.get("previous_context", ""),
            ))

    def score_detection(self, case: BenchmarkCase, audit_report: dict) -> BenchmarkResult:
        """Compare audit results against ground truth injected bugs."""
        reported = audit_report.get("inconsistencies", [])
        total_injected = len(case.injected_bugs)
        total_detected = len(reported)

        # Simple keyword-overlap matching between reported issues and injected bugs
        matched_bug_indices: set[int] = set()
        details = []

        for issue in reported:
            desc = issue.get("description", "")
            best_match = -1
            best_overlap = 0

            for i, bug in enumerate(case.injected_bugs):
                if i in matched_bug_indices:
                    continue
                overlap = _keyword_match(desc, bug)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = i

            is_match = best_overlap >= 0.5 and best_match >= 0
            if is_match and best_match >= 0:
                matched_bug_indices.add(best_match)

            details.append({
                "reported": desc[:100],
                "matched": is_match,
                "best_ground_truth": case.injected_bugs[best_match].description[:100]
                if best_match >= 0 else "",
                "overlap_score": round(best_overlap, 2),
            })

        true_positives = len(matched_bug_indices)
        false_positives = total_detected - true_positives
        false_negatives = total_injected - true_positives

        return BenchmarkResult(
            case_name=case.name,
            total_injected=total_injected,
            total_detected=total_detected,
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            details=details,
        )

    async def run(self, persist_dir: str = "./novel-data/chroma_data") -> list[BenchmarkResult]:
        """Run all benchmark cases against the ContinuityAgent."""
        from novel_agent.agents.base import AgentConfig
        from novel_agent.agents.continuity import ContinuityAgent
        from novel_agent.memory.embeddings import ChapterStore

        self.results = []

        for case in self.cases:
            config = AgentConfig(
                model=os.getenv("BUDGET_MODEL", "deepseek-chat"),
                temperature=0.1,
            )
            store = ChapterStore(persist_dir)
            # 每个用例独立 project_id，避免跨用例上下文污染
            project_id = f"benchmark_{case.name}"
            # 把前文上下文写入向量库，供 check_continuity 工具检索比对
            if case.previous_context:
                store.index_chapter(
                    project_id=project_id,
                    chapter_number=case.chapter_number - 1,
                    content=case.previous_context,
                )
            agent = ContinuityAgent(config=config, chapter_store=store, project_id=project_id)

            report, _ = await agent.audit(
                chapter_number=case.chapter_number,
                draft_content=case.draft_content,
            )

            result = self.score_detection(case, report)
            self.results.append(result)

        return self.results

    def summary(self) -> dict[str, Any]:
        """Aggregate results across all cases."""
        if not self.results:
            return {"error": "No results"}

        total_injected = sum(r.total_injected for r in self.results)
        total_detected = sum(r.total_detected for r in self.results)
        total_tp = sum(r.true_positives for r in self.results)

        macro_precision = sum(r.precision for r in self.results) / len(self.results)
        macro_recall = sum(r.recall for r in self.results) / len(self.results)
        macro_f1 = sum(r.f1 for r in self.results) / len(self.results)

        return {
            "cases": len(self.results),
            "total_bugs_injected": total_injected,
            "total_issues_reported": total_detected,
            "true_positives": total_tp,
            "macro_precision": round(macro_precision, 3),
            "macro_recall": round(macro_recall, 3),
            "macro_f1": round(macro_f1, 3),
            "per_case": [
                {
                    "name": r.case_name,
                    "precision": round(r.precision, 3),
                    "recall": round(r.recall, 3),
                    "f1": round(r.f1, 3),
                    "tp": r.true_positives,
                    "fp": r.false_positives,
                    "fn": r.false_negatives,
                }
                for r in self.results
            ],
        }


def _keyword_overlap(text_a: str, text_b: str) -> float:
    """Simple Jaccard-like keyword overlap between two strings."""
    def tokenize(s: str) -> set[str]:
        keywords = set()
        for phrase in s.replace("，", ",").replace("。", ",").split(","):
            phrase = phrase.strip().lower()
            if len(phrase) >= 2:
                keywords.add(phrase)
        return keywords

    a_tokens = tokenize(text_a)
    b_tokens = tokenize(text_b)

    if not a_tokens and not b_tokens:
        return 0.0
    if not a_tokens or not b_tokens:
        return 0.0

    intersection = a_tokens & b_tokens
    union = a_tokens | b_tokens
    return len(intersection) / len(union)


def _keyword_match(reported_desc: str, bug: InjectedBug) -> float:
    """返回 bug 特征词在报告描述中的命中率；无特征词时退回短语 Jaccard。"""
    if bug.keywords:
        desc = reported_desc.lower()
        hits = sum(1 for kw in bug.keywords if kw.lower() in desc)
        return hits / len(bug.keywords)
    return _keyword_overlap(reported_desc, bug.description)


# ── Built-in benchmark cases ──────────────────────────

# stress case 的正文填充：把 bug 推到 draft_content[:4000] 截断位置之后。
# 填充句刻意不含任何 bug 特征词，保证各 bug 特征词首次出现都在 >4000 字符处。
_STRESS_PAD = (
    "林风负手立于城头，远眺群山如黛，暮色四合，归鸦点点掠过天际。"
    "他心中反复盘算着此行的安排，前路凶险难测，却势在必行。"
    "城下市声渐起，贩夫走卒络绎不绝，一派升平景象。"
    "他收回目光，转身走下城楼，穿过长街，回到下榻的客栈，掩门休息。"
)

BUILTIN_CASES = [
    {
        "name": "character_name_swap",
        "description": "主角名字从'林风'突然变成'林峰'",
        "chapter_number": 3,
        "chapter_outline": "主角在城中遭遇伏击，展现出新的能力",
        "draft_content": (
            "林峰走进城主府的大门，守卫拦住了他。\n"
            '"我是来找城主的，"林峰平静地说。\n'
            "守卫看了看他，摇头道：城主今日不见客。\n"
            "林峰微微一笑，从怀中取出一枚令牌。守卫脸色大变，连忙让开道路。\n"
            "穿过庭院，林峰看到了坐在大厅中的城主。\n"
            '"你终于来了，"城主抬起头，"我等你很久了，林风。"\n'
            "林峰纠正道：我是林峰。\n"
            "城主愣了一下，随即笑道：是我记错了。"
        ),
        "injected_bugs": [
            {
                "category": "character",
                "severity": "critical",
                "description": "主角名字从'林风'变成'林峰'，前后不一致",
                "location_hint": "整章使用'林峰'而前文章节使用'林风'",
                "keywords": ["林风", "林峰"],
            },
            {
                "category": "character",
                "severity": "critical",
                "description": "城主称呼主角为'林风'，主角却自称'林峰'，存在身份混淆",
                "location_hint": "对话中",
                "keywords": ["城主", "身份混淆"],
            },
        ],
        "previous_context": "前两章中，主角的名字是'林风'，来自青云镇的年轻剑客。",
    },
    {
        "name": "timeline_contradiction",
        "description": "时间线矛盾：前文说3天前发生的事，本章说1周前",
        "chapter_number": 5,
        "chapter_outline": "主角在修炼中突破瓶颈",
        "draft_content": (
            "距离那场大战已经过去了一周，但林风的伤还未痊愈。\n"
            "他盘坐在密室中，感受着体内灵力的流动。\n"
            "一周前的那场战斗，让他险些丧命。\n"
            "不过现在，他感觉自己的修为瓶颈终于松动了。"
        ),
        "injected_bugs": [
            {
                "category": "timeline",
                "severity": "major",
                "description": "大战时间从3天前变成1周前，时间线矛盾",
                "location_hint": "章节开头的时间描述",
                "keywords": ["3天前", "一周前"],
            },
        ],
        "previous_context": "第4章结尾：大战发生在3天前，林风受伤后一直在养伤。",
    },
    {
        "name": "worldbuilding_rule_violation",
        "description": "世界观规则违反：前文设定灵力只能通过修炼获得，本章出现了丹药增灵",
        "chapter_number": 4,
        "chapter_outline": "主角获得了一枚神奇的丹药",
        "draft_content": (
            "老者从袖中取出一枚丹药，递到林风面前。\n"
            '"服下此丹，你的灵力可瞬间提升一个大境界，"老者说。\n'
            "林风接过丹药，感受着其中澎湃的灵力波动。\n"
            "他毫不犹豫地吞了下去，体内的灵力果然暴涨。"
        ),
        "injected_bugs": [
            {
                "category": "worldbuilding",
                "severity": "critical",
                "description": (
                    "前文明确设定灵力只能通过自身修炼获得，不能借助外力。"
                    "本章出现了可提升灵力的丹药，违反世界观规则。"
                ),
                "location_hint": "丹药增灵的情节",
                "keywords": ["灵力", "丹药", "修炼"],
            },
        ],
        "previous_context": "这个世界中，灵力只能通过自身的艰苦修炼获得，没有任何捷径。这是铁律。",
    },
    {
        "name": "character_skill_swap",
        "description": "角色招式串换：萧烈使出了林风的标志性剑招'碎星剑'",
        "chapter_number": 6,
        "chapter_outline": "萧烈在比武场上展露实力",
        "draft_content": (
            "比武场上，萧烈深吸一口气，缓缓拔剑。\n"
            '"看好了，"萧烈低喝一声，使出了成名绝技碎星剑。\n'
            "剑光如星辰坠落，场边众人纷纷惊呼。\n"
            "这一招正是江湖中传说的碎星剑，据说只有林风会使。"
        ),
        "injected_bugs": [
            {
                "category": "character",
                "severity": "major",
                "description": "萧烈使出了林风的标志性剑招'碎星剑'，招式归属错误",
                "location_hint": "比武场面",
                "keywords": ["萧烈", "碎星剑"],
            },
        ],
        "previous_context": "林风的标志性剑招是'碎星剑'，好友萧烈只会'烈焰掌'。",
    },
    {
        "name": "character_title_mismatch",
        "description": "次要角色称呼错位：城主府守门老者由'赵伯'误称为'王伯'",
        "chapter_number": 7,
        "chapter_outline": "主角再次拜访城主府",
        "draft_content": (
            "再次来到城主府时，门房换了张生面孔。\n"
            '"王伯今日不在，"年轻的守卫歉意地说。\n'
            "林风点点头，他记得此前接待他的那位老者姓赵，人人都叫他赵伯。\n"
            "不过这只是小事，他也没再多问。"
        ),
        "injected_bugs": [
            {
                "category": "character",
                "severity": "minor",
                "description": "城主府守门老者由'赵伯'误写为'王伯'，次要角色称呼不一致",
                "location_hint": "门房对话",
                "keywords": ["王伯", "赵伯"],
            },
        ],
        "previous_context": "城主府大门由一位姓赵的老者看管，人人都称他'赵伯'。",
    },
    {
        "name": "timeline_death_contradiction",
        "description": "时间线重大矛盾：上一章末主角与魔尊同归于尽，本章却若无其事出现在集市",
        "chapter_number": 9,
        "chapter_outline": "林风在市集挑选药材",
        "draft_content": (
            "清晨的市集热闹非凡，林风挤在人群里挑选药材。\n"
            '"这株灵芝怎么卖？"他随口问道。\n'
            "摊主抬头看他，愣了片刻，随即笑道：客官好眼力，三枚灵石。\n"
            "林风付了钱，把灵芝揣进怀里，转身继续逛。\n"
            "街角卖糖人的阿婆认出了他，招呼道：林少侠，您回来了？"
        ),
        "injected_bugs": [
            {
                "category": "timeline",
                "severity": "critical",
                "description": "上一章林风与魔尊同归于尽，本章却毫发无损出现在市集",
                "location_hint": "市集场景",
                "keywords": ["复活", "市集"],
            },
        ],
        "previous_context": "第8章末林风与魔尊同归于尽，坠崖身亡，全城已为他设灵堂出殡。",
    },
    {
        "name": "timeline_event_order",
        "description": "时间线小矛盾：城主收徒大典时间由'昨日'误写为'三天前'",
        "chapter_number": 10,
        "chapter_outline": "众人议论城主收徒之事",
        "draft_content": (
            "酒馆里，几个汉子聊起城主的收徒大典。\n"
            '"你们听说没有，三天前城主收了林家的女儿做关门弟子，"一个络腮胡压低声音说。\n'
            "旁边有人附和：难怪这两天城主府灯火通明。\n"
            "林风恰好路过，不禁想起昨日大典上的盛况。"
        ),
        "injected_bugs": [
            {
                "category": "timeline",
                "severity": "minor",
                "description": "城主收徒大典时间由'昨日'误写为'三天前'，时间细节不符",
                "location_hint": "酒馆对话",
                "keywords": ["三天前", "昨日"],
            },
        ],
        "previous_context": "前文交代：城主昨日刚为林家女儿举办收徒大典，仪式盛大。",
    },
    {
        "name": "worldbuilding_teleport_beyond_range",
        "description": "世界观规则违反：传音符从青云城直达千里之外的京城，超出'只能同城传音'的限制",
        "chapter_number": 11,
        "chapter_outline": "主角收到千里之外的紧急传讯",
        "draft_content": (
            "深夜，一道流光从窗外疾射而入，落在林风掌心化为一张符纸。\n"
            "符纸上只有四个字：速来京城。\n"
            "林风认得这笔迹，是远在千里之外的师父所写。\n"
            "他不由心惊：师父竟能从京城直接传音到青云城，这远超传音符本来的极限。"
        ),
        "injected_bugs": [
            {
                "category": "worldbuilding",
                "severity": "major",
                "description": "传音符从京城直达青云城，违反'传音符只能同城传音'的世界观规则",
                "location_hint": "收到传音符的场景",
                "keywords": ["传音符", "京城"],
            },
        ],
        "previous_context": "此界传音符有铁则：只能经传送阵同城传音，跨城传音不可能。",
    },
    {
        "name": "worldbuilding_currency_mismatch",
        "description": "世界观小细节矛盾：修士市集中用'银两'交易，而此界修士只认灵石",
        "chapter_number": 12,
        "chapter_outline": "主角在修士市集采买",
        "draft_content": (
            "修士市集依山而建，摊位密布。\n"
            "林风在一个卖丹药的摊位前停下，摊主报价：两枚灵石。\n"
            "他正要掏灵石，却见隔壁摊位有人摸出十两银子拍在桌上。\n"
            "那摊主竟也欣然收下，把一炉丹药推了过去。\n"
            "林风看得皱眉，心想这市集的规矩竟如此随意。"
        ),
        "injected_bugs": [
            {
                "category": "worldbuilding",
                "severity": "minor",
                "description": "修士市集中用银两购买丹药，而此界修士交易只用灵石",
                "location_hint": "市集交易细节",
                "keywords": ["银两", "灵石"],
            },
        ],
        "previous_context": "此界修士之间交易只认灵石，金银凡物在修士市集中并无价值。",
    },
    {
        "name": "quantity_3_bugs",
        "description": "单章注入3个不同类型的不一致，检验多 bug 场景的查全率",
        "chapter_number": 13,
        "chapter_outline": "宗门大比开幕，多线并进",
        "draft_content": (
            "宗门大比的锣声响起，各峰弟子齐聚演武场。\n"
            "大师姐苏雪款款走上高台，主持开幕仪式。\n"
            "长老们坐在主席台上，交头接耳。\n"
            '"这次大比，提前到上周就开始报名了，"二长老捻着胡须说。\n'
            "身旁的三长老摇头：我分明记得公告写的是明日才截止。\n"
            "台下，几名外门弟子压低声音议论：听说后山禁地这两日能随便进出，不少人都去偷看了稀罕物。\n"
            "林风站在人群中，把这些话都听在耳里。"
        ),
        "injected_bugs": [
            {
                "category": "character",
                "severity": "critical",
                "description": "大师姐的名字由'苏若雪'误写为'苏雪'，前后不一致",
                "location_hint": "开幕仪式上的名字",
                "keywords": ["苏雪", "苏若雪"],
            },
            {
                "category": "timeline",
                "severity": "major",
                "description": "大比报名时间由'明日截止'写成'上周开始'，时间线矛盾",
                "location_hint": "长老对话",
                "keywords": ["上周", "明日"],
            },
            {
                "category": "worldbuilding",
                "severity": "critical",
                "description": "前文设定后山禁地禁止擅闯，本章弟子却能随意进出禁地",
                "location_hint": "外门弟子议论",
                "keywords": ["禁地", "擅闯"],
            },
        ],
        "previous_context": "前文明确：大师姐叫苏若雪；大比报名明日截止；禁地严禁弟子擅入。",
    },
    {
        "name": "quantity_5_bugs",
        "description": "单章注入5个不同类型的不一致，检验高密度多 bug 场景",
        "chapter_number": 14,
        "chapter_outline": "主角设宴招待多年未见的旧友",
        "draft_content": (
            "林风在青云城最大的酒楼醉仙居设宴，招待多年未见的旧友萧牧。\n"
            "萧牧推开雅间的门，笑道：多年不见，你可还记得咱们三年前闯黑风寨的事？\n"
            "林风摇头：那是五年前的事，你记岔了。\n"
            "两人坐下对饮，萧牧的随从阿贵在一旁殷勤伺候，替二人斟酒。\n"
            "席间萧牧提起，他如今在京城经营商号，生意做得颇大，来往都靠银两结算。\n"
            "林风听得皱眉，心想修士之间不是向来只用灵石么。\n"
            "掌柜又进来添了一坛好酒，说是昨日刚从江南运来的。\n"
            "萧牧接口道：掌柜说笑了，这坛酒分明是前天才从江南启程。"
        ),
        "injected_bugs": [
            {
                "category": "character",
                "severity": "critical",
                "description": "旧友的名字由'萧烈'误写为'萧牧'，主要配角姓名前后不一致",
                "location_hint": "酒楼雅间",
                "keywords": ["萧牧", "萧烈"],
            },
            {
                "category": "timeline",
                "severity": "major",
                "description": "黑风寨往事时间由'三年前'写成'五年前'，时间线矛盾",
                "location_hint": "叙旧对话",
                "keywords": ["三年前", "五年前"],
            },
            {
                "category": "character",
                "severity": "minor",
                "description": "随从的名字由'阿福'误写为'阿贵'，次要角色称呼不一致",
                "location_hint": "伺候斟酒",
                "keywords": ["阿贵", "阿福"],
            },
            {
                "category": "worldbuilding",
                "severity": "minor",
                "description": "修士商号用银两结算，而此界修士交易只用灵石",
                "location_hint": "京城商号对话",
                "keywords": ["银两", "灵石"],
            },
            {
                "category": "timeline",
                "severity": "minor",
                "description": "美酒到货时间由'昨日'写成'前天'，时间细节不符",
                "location_hint": "掌柜添酒",
                "keywords": ["前天", "昨日"],
            },
        ],
        "previous_context": "前文明确：主角旧友名萧烈，随从阿福；三年前闯黑风寨；交易只用灵石。",
    },
    {
        "name": "stress_tail_truncation",
        "description": "超长章节：bug 全部放在 >4000 字符处，量化截断对 recall 的影响",
        "chapter_number": 15,
        "chapter_outline": "主角在客栈歇脚，听闻坊间诸多传闻",
        "draft_content": _STRESS_PAD * 50 + (
            "第二日，林风在客栈听到掌柜絮叨。\n"
            "掌柜说：昨儿个我去族老那儿对账，发现大师姐的名字被写成了苏雪，可族谱上明明写着苏若雪。\n"
            "掌柜又说：还有人私下传言，说有人用传音符从京城直接传到青云城，全程没有中转，这在过去是绝无可能的。\n"
            "林风越听越不对劲，他分明记得收徒大典就办在昨日，可账房那本账上，大典却记成了上周。"
        ),
        "injected_bugs": [
            {
                "category": "character",
                "severity": "critical",
                "description": "大师姐的名字由'苏若雪'误写为'苏雪'，前后不一致",
                "location_hint": "掌柜对账处（>4000 字符位置）",
                "keywords": ["苏雪", "苏若雪"],
            },
            {
                "category": "worldbuilding",
                "severity": "critical",
                "description": "传音符从京城直达青云城，违反'只能同城传音'的世界观规则",
                "location_hint": "掌柜传言（>4000 字符位置）",
                "keywords": ["传音符", "京城"],
            },
            {
                "category": "timeline",
                "severity": "major",
                "description": "收徒大典时间由'昨日'写成'上周'，时间线矛盾",
                "location_hint": "账房记录（>4000 字符位置）",
                "keywords": ["上周", "昨日"],
            },
        ],
        "previous_context": "前文设定：大师姐叫苏若雪；传音符只能同城传音；城主昨日办收徒大典。",
    },
]
