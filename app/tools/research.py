"""research —— 有界的研究函数：给定对象与维度，回一份结构化事实，**网页正文不进主环**。

**为什么不是让模型自己多调几次 web_search。** 三条实测/机制上的理由：

1. **重复搜同一件事**。真实会话统计（``output/`` 336 个旧格式会话）里 web_search 共 51 次调用，
   同一会话换措辞重搜 2~3 次是主要形态（露营厨具 3 次、Sony vs Bose 3 次），自由发挥没带来
   信息增量，只带来往返。
2. **上下文口子**。``web_search`` 每条 content 是整页正文（截断后仍达单条 1500 / 整批 15000
   字符），全部原样进主环 messages。本函数一次 fast 模型归纳成 schema，主环只见结论与出处。
3. **可预扣配额**。函数内部**没有反馈环**（模板展开 → 并行搜 → 一次归纳，固定三步），调用
   次数在入参确定的那一刻就是已知上界 ``len(targets)``，所以能预先扣配额、能并行、延迟不随
   target 数线性增长。自由 web_search 三样都做不到。

**这不是子 Agent。** 内部那次 fast 模型调用没有 loop、没有工具面、不判断何时终止，也不占主环
的 ``MAIN_MAX_ITERS``；与 ``shopping_summary`` / ``chat_fallback`` / ``image_understand`` 内部
调模型同构。要加「结果不好就再搜一轮」请让主 agent 再调一次本函数（次数记在主环账上、受配额
约束），不要在这里装 loop。

**与 web_search 的分工**：本函数吃「已有明确对象」的三类——单品口碑、多品对比、品类选购维度；
没有对象的两类（把新说法翻译成品类词、召回全空时查背景）仍走裸 ``web_search``。两者并存。

**归纳是有损压缩，所以有两道护栏**：① 每条 claim 必须带 url，且 url 必须出现在本次搜索结果里，
否则整条丢弃（防归纳模型编造出处）；② 原始结果落 ``session_dir/research_<n>.json``——不进上下文
但可回取。后者治的是「漏抽维度」「把众说纷纭压成确定 claim」这两种**静默失败**，没有盘上原文就
无从归因。``aspects`` 同时是查询词与**归纳的抽取目标**：主 agent 指定抽什么，归纳模型不自由发挥。
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from app.agent.invoke import call_structured
from app.agent.llm import get_fast_llm
from app.api import monitor
from app.api.context import get_session_dir
from app.tools._args import StrListArg
from app.tools._shell import tool
from app.tools.web_search import WebResult, search_web
from app.utils.env import env_int

logger = logging.getLogger(__name__)

#: 单次 research 最多研究几个对象。超出的 target 直接截断（不报错——主 agent 想比 5 个手机时，
#: 回 3 个 + 一句说明，比整条失败有用）。配额本身（会话级）在 C3 的 retrieval_budget 里。
RESEARCH_MAX_TARGETS = env_int("RESEARCH_MAX_TARGETS", 3)

#: 每个 target 发一条模板查询、取几条结果。aspects 合进同一条查询，不额外发搜索。
RESEARCH_RESULTS_PER_TARGET = env_int("RESEARCH_RESULTS_PER_TARGET", 5)


class ResearchClaim(BaseModel):
    """一条带出处的事实。``url`` 必须来自本次搜索结果，否则这条会被丢弃。"""

    text: str = Field(description="一句话事实，不要写成推荐语")
    url: str = Field(default="", description="出处 URL，必须是本次搜索结果里出现过的")
    aspect: str = Field(default="", description="这条对应哪个 aspect")


class ResearchFinding(BaseModel):
    """一个研究对象的归纳结果。"""

    target: str
    pros: list[str] = Field(default_factory=list, description="优点，每条尽量带数字或具体场景")
    cons: list[str] = Field(default_factory=list, description="缺点/争议点，没查到就留空别编")
    price_range: str = Field(default="", description="公网价格区间原文，如 $249~$299；没查到留空")
    claims: list[ResearchClaim] = Field(default_factory=list)
    data_note: str = Field(
        default="",
        description="资料薄/结论有分歧/没查到时如实写，供主 agent 决定要不要在文案里标注",
    )


class _ResearchDraft(BaseModel):
    """归纳模型的产出（只有 findings，其余字段由函数侧填，不让模型编）。"""

    findings: list[ResearchFinding] = Field(default_factory=list)


class ResearchOutput(BaseModel):
    """research 的结构化返回。主环看到的就是它，**不含网页正文**。"""

    targets: list[str]
    aspects: list[str]
    findings: list[ResearchFinding] = Field(default_factory=list)
    searched: int = 0  # 实际发出的搜索条数（配额记账用）
    raw_path: str = ""  # 原始结果落盘路径（不进上下文，供事后归因）
    note: str = ""  # 降级 / 截断 / 全空时的说明


def _build_query(target: str, aspects: list[str]) -> str:
    """模板展开：一个 target 一条查询，aspects 合进同一条（C3 口径：不额外发搜索）。

    刻意加 ``review`` 与年份——真实会话里模型自己写的查询，带这两样的结果质量明显高
    （``best X 2024 review`` 一类），把它固化进模板就不必再靠 prompt 教模型怎么写查询。
    """
    parts = [target.strip()]
    parts.extend(a.strip() for a in aspects if a.strip())
    parts.extend(["review", str(datetime.now(tz=UTC).year)])
    return " ".join(p for p in parts if p)


def _dump_raw(payload: dict) -> str:
    """把原始搜索结果落 ``session_dir/research_<n>.json``，返回相对路径（落不下就回空串）。

    **不进上下文但可回取**：归纳漏抽维度、把分歧压成确定 claim 都是静默失败，盘上留原文才有
    归因依据。落盘失败不影响主流程——研究结果已经拿到了，为了存档把整轮打挂是本末倒置。
    """
    sd = get_session_dir()
    if sd is None:  # 无 session 作用域（单测 / 离线直调）
        return ""
    try:
        n = len(list(sd.glob("research_*.json"))) + 1
        path = sd / f"research_{n}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path.name
    except Exception:
        logger.warning("research 原始结果落盘失败", exc_info=True)
        return ""


def _drop_ungrounded(findings: list[ResearchFinding], allowed: set[str]) -> int:
    """丢弃 url 不在本次搜索结果里的 claim，返回丢了几条。

    归纳模型编造出处是这条链上最可能的幻觉形态（它读的是真正文，但 url 是从记忆里凑的）。
    这里**只删 claim 不删 finding**——pros/cons 是归纳出来的判断、本就不逐条带 url，claim 才是
    「我有出处」的那一档，出处对不上就不配留在这一档。
    """
    dropped = 0
    for f in findings:
        kept = [c for c in f.claims if c.url and c.url in allowed]
        dropped += len(f.claims) - len(kept)
        f.claims = kept
    return dropped


_SUMMARY_SYSTEM = """你在为购物助手归纳公网资料。只做归纳，不做推荐——「哪个更适合这位用户」\
由主流程判断，你不知道用户的预算和偏好。

规则：
1. 只用给定资料里的信息，资料没提到的维度留空，**不要补充常识**。
2. 每条 claim 必须带 url，且 url 只能从资料里逐字复制，不确定就不写这条。
3. 资料里有分歧（不同来源说法相反）时，写进 data_note，不要挑一个当定论。
4. pros / cons 每条尽量带数字或具体场景，避免「性价比高」这类空话。
5. 用户给了 aspects 时，优先抽这些维度；资料里没有该维度就在 data_note 说明。"""


def _summary_input(
    targets: list[str], aspects: list[str], packs: list[tuple[str, list[WebResult]]]
) -> str:
    """把搜索结果拼成归纳输入。正文在这里进的是**归纳模型**的上下文，不是主环的。"""
    lines: list[str] = []
    if aspects:
        lines.append(f"需要重点抽取的维度（aspects）：{', '.join(aspects)}")
    for target, results in packs:
        lines.append(f"\n## 研究对象：{target}")
        if not results:
            lines.append("（没搜到资料）")
            continue
        for r in results:
            lines.append(f"- 标题：{r.title}\n  url：{r.url}\n  正文：{r.content}")
    return "\n".join(lines)


@tool
async def research(targets: StrListArg, aspects: StrListArg | None = None) -> ResearchOutput:
    """研究几个**已知对象**（商品/品牌/品类）的公网口碑并归纳成结构化事实；不产候选、不下推荐结论。

    何时调用：用户点名比较（「A 和 B 哪个好」）、问某款值不值、问某品类怎么选。
    参数：targets 最多 3 个研究对象（商品名/品牌/品类）；aspects 要重点了解的维度
    （如 ["续航", "降噪"]），会同时用于检索与归纳抽取，留空则按品类常见维度。
    不要用它找商品——它不返回可下单的候选，找商品用 item_search。
    把新说法翻译成品类词、或召回全空时查背景，用 web_search。
    """
    targets = [t.strip() for t in targets if t and t.strip()][:RESEARCH_MAX_TARGETS]
    aspects = [a.strip() for a in (aspects or []) if a and a.strip()]
    await monitor.report_tool_start("research", targets=targets, aspects=aspects)
    if not targets:
        out = ResearchOutput(targets=[], aspects=aspects, note="没有给出研究对象（targets 为空）。")
        await monitor.report_tool_end("research", results=0, degraded=True, result=out.note)
        return out

    # 并行发：每 target 一条模板查询，aspects 合进同一条。上界 = len(targets)，事前可知。
    packs_raw = await asyncio.gather(
        *(
            search_web(_build_query(t, aspects), RESEARCH_RESULTS_PER_TARGET)
            for t in targets
        )
    )
    packs = [(t, ws.results) for t, (ws, _) in zip(targets, packs_raw, strict=True)]
    degraded_notes = [ws.note for ws, deg in packs_raw if deg and ws.note]
    allowed_urls = {r.url for _, results in packs for r in results if r.url}

    raw_path = _dump_raw(
        {
            "targets": targets,
            "aspects": aspects,
            "packs": [
                {"target": t, "results": [r.model_dump() for r in results]} for t, results in packs
            ],
        }
    )

    if not allowed_urls:  # 全降级 / 全空：不调归纳模型，白花一次解码
        note = degraded_notes[0] if degraded_notes else "公网没搜到这些对象的资料，请如实说明。"
        out = ResearchOutput(
            targets=targets, aspects=aspects, searched=len(targets), raw_path=raw_path, note=note
        )
        await monitor.report_tool_end("research", results=0, degraded=True, result=note)
        return out

    notes: list[str] = list(degraded_notes)
    try:
        draft = await call_structured(
            get_fast_llm(),
            [("system", _SUMMARY_SYSTEM), ("user", _summary_input(targets, aspects, packs))],
            _ResearchDraft,
        )
        findings = draft.findings
    except Exception:
        # 归纳失败不抛：主 agent 至少该知道「查到了但没归纳出来」，而不是收到一个异常。
        logger.warning("research 归纳失败", exc_info=True)
        findings = []
        notes.append("资料已查到但归纳失败，可据 raw_path 的原始结果人工核，或如实说明不确定。")

    dropped = _drop_ungrounded(findings, allowed_urls)
    if dropped:
        notes.append(f"已丢弃 {dropped} 条出处对不上本次搜索结果的说法。")

    out = ResearchOutput(
        targets=targets,
        aspects=aspects,
        findings=findings,
        searched=len(targets),
        raw_path=raw_path,
        note="；".join(notes),
    )
    summary = " / ".join(f"{f.target}: {len(f.claims)} 条" for f in findings) or out.note
    await monitor.report_tool_end(
        "research", results=len(findings), degraded=bool(degraded_notes), result=summary
    )
    return out
