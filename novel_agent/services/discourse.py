"""Discourse contract for one chapter.

The book policy is fixed. The orchestrator fills a small contract and a beat
list; deterministic checks compare the draft to that contract. Reviewers do
not rewrite. The revision director folds their findings into one brief.
"""

from __future__ import annotations

import re

BOOK_DISCOURSE_POLICY = (
    "全书话语规范：主题由事件承担，旁白不解释主题；"
    "因果可以留下未收的线；章末用动作、打断或未决收束，不以主角想通了作结；"
    "时间以顺叙为主，闪回必须有交代；人物选择可以含糊，不必当场裁判。"
)

DEFAULT_CONTRACT = {
    "theme": "implicit",
    "causal": "loose_end",
    "ending": "action",
    "time": "linear",
    "moral": "ambiguous",
}

_REALIZATION = re.compile(
    r"终于明白|这一刻.{0,8}懂|他意识到|她意识到|真正的意义|这告诉我们|主题是"
)
_THEME_TOLD = re.compile(r"真正的意义|这告诉我们|主题是|说到底")


def ensure_discourse(strategy: dict | None) -> dict:
    """Attach the book policy and fill a contract when the planner omitted one."""
    strategy = dict(strategy or {})
    raw = strategy.get("discourse_contract")
    contract = dict(DEFAULT_CONTRACT)
    if isinstance(raw, dict):
        for key in DEFAULT_CONTRACT:
            value = raw.get(key)
            if isinstance(value, str) and value.strip():
                contract[key] = value.strip()
    strategy["discourse_contract"] = contract
    strategy["book_policy"] = BOOK_DISCOURSE_POLICY
    beats = strategy.get("beats")
    if isinstance(beats, list):
        strategy["beats"] = [beat for beat in beats if isinstance(beat, dict)]
    else:
        strategy["beats"] = []
    return strategy


def check_discourse_contract(draft: str, contract: dict | None) -> dict:
    """Flag clustered structural misses. A single soft echo is not a miss."""
    contract = contract or DEFAULT_CONTRACT
    tail = (draft or "")[-400:]
    findings: list[dict] = []

    ending = contract.get("ending") or "action"
    if ending not in {"acceptance", "realization"} and len(_REALIZATION.findall(tail)) >= 1:
        if _REALIZATION.search(tail):
            findings.append(
                {
                    "type": "ending_by_realization",
                    "severity": "major",
                    "description": (
                        "章末落在主角想通了或点明意义上，合同要求用动作、打断或未决收束。"
                    ),
                }
            )

    theme = contract.get("theme") or "implicit"
    told = _THEME_TOLD.findall(draft or "")
    if theme == "implicit" and len(told) >= 2:
        findings.append(
            {
                "type": "theme_explained",
                "severity": "major",
                "description": "旁白反复把主题说破。主题留在事件里，不要解释。",
            }
        )

    return {"passed": not findings, "findings": findings, "contract": dict(contract)}


def direct_revision(
    *,
    gate_report: dict | None = None,
    timeline_report: dict | None = None,
    contract_report: dict | None = None,
    editor_report: dict | None = None,
    continuity_report: dict | None = None,
) -> str:
    """One brief. Canon and contract outrank prose notes."""
    lines: list[str] = []

    if gate_report and not gate_report.get("passed"):
        for item in gate_report.get("violations") or []:
            lines.append(f"- 硬门禁：{item}")

    for item in (timeline_report or {}).get("findings") or []:
        if item.get("type") != "dead_character_reappeared":
            continue
        lines.append(
            f"- 正典：角色「{item.get('subject', '')}」"
            f"已在第{item.get('death_chapter', '?')}章死亡，"
            "本章不得让其活着出场。删掉这个人的在场。"
        )

    for item in (contract_report or {}).get("findings") or []:
        lines.append(f"- 话语合同：{item.get('description', '')}")

    for item in (continuity_report or {}).get("issues") or []:
        if item.get("severity") not in {"critical", "major"}:
            continue
        text = item.get("description") or item.get("detail") or ""
        if text:
            lines.append(f"- 连续性：{text}")

    editor = editor_report or {}
    if editor.get("verdict") == "rewrite":
        for item in (editor.get("issues") or [])[:5]:
            text = item.get("description") or ""
            if text:
                dim = item.get("dimension") or "文学"
                lines.append(f"- 文学（{dim}）：{text}")

    if not lines:
        return ""
    return "按下面这一份意见重写整章，先改正典和合同，再改文笔：\n" + "\n".join(lines)


def needs_rewrite(
    *,
    gate_passed: bool,
    timeline_report: dict | None,
    contract_report: dict | None,
    editor_report: dict | None,
    continuity_report: dict | None,
) -> bool:
    if not gate_passed:
        return True
    if timeline_report and not timeline_report.get("passed", True):
        return True
    if contract_report and not contract_report.get("passed", True):
        return True
    if (editor_report or {}).get("verdict") == "rewrite":
        return True
    for item in (continuity_report or {}).get("issues") or []:
        if item.get("severity") in {"critical", "major"}:
            return True
    return False
