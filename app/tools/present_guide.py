"""present_guide —— 选购标准类答案的终结出口：分节标准 + 明说的假设 + 来源列表。

**为什么不继续用 ``chat_fallback``。** 「这个品类怎么挑」这类答案的主体是**判断依据**，形状是固定
的（几条标准，每条几个要点，末尾一份出处），此前它和「你好」「谢谢」共用一个非购物兜底出口：
答案整段塞进 ``message``，前端只能按 Markdown 正文排，来源混在正文末尾、分节靠模型自己打标题。
把结构变成字段之后，前端那张卡才对得齐（一节一块、来源单独一栏可点），而模型少一件要操心的事。

**与 ``shopping_summary`` 同一条口径：文案由主模型在入参里给，工具只排版。** 不在工具里再调一次
fast 模型「归纳一下」——2026-09-16 那次教训（1500+ 字符的电动牙刷选购指南被归纳成一句客套话）
对本工具同样成立，而且更致命：这里根本没有商品卡兜底，答案被换掉就什么都不剩。

**C7：本工具不带选项。** 「你最看重哪条」的可点选项由同一轮的 ``ask_user(closes_turn=true)`` 发
（D2 已落地的那条路），不在这里再开第二套 chips —— 两个职责重叠的出口，模型会乱选。

**来源只校验形状**（http(s) 开头、去重、封顶），不校验「是不是本会话搜过的 url」：模型的来源
本来就是从 ``research`` 的 claim 上抄的，而那条链已经在 ``research._drop_ungrounded`` 做过
grounding（url 必须出现在本次搜索结果里）。在这里再拦一道，拦掉的只会是 research 已经放过的。
"""

from __future__ import annotations

import json
import logging
from typing import Annotated

from pydantic import BaseModel, BeforeValidator, Field

from app.api import monitor
from app.tools._args import StrListArg, coerce_stringified_list
from app.tools._shell import tool
from app.utils.env import env_int

logger = logging.getLogger("shoppingx.tools.present_guide")

#: 一次最多讲几节标准。超出截断——再多用户就不读了，真要细讲让他挑一条继续问。
GUIDE_MAX_SECTIONS = env_int("GUIDE_MAX_SECTIONS", 6)
#: 一节最多几个要点。
GUIDE_MAX_POINTS = env_int("GUIDE_MAX_POINTS", 5)
#: 来源最多列几条。
GUIDE_MAX_SOURCES = env_int("GUIDE_MAX_SOURCES", 8)


class GuideSection(BaseModel):
    """一节选购标准：一个标题 + 几条要点。"""

    title: str
    points: StrListArg = Field(default_factory=list, description="这条标准怎么看，每条一句")


class GuideSource(BaseModel):
    """一条出处。``title`` 缺省时排版退回域名。"""

    title: str = ""
    url: str = ""


def _coerce_sections(v: object) -> object:
    """模型侧容错：JSON 字符串 → list；``{标题: [要点]}`` 映射 → 条目列表。"""
    v = coerce_stringified_list(v)
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except ValueError:
            return v
    if isinstance(v, dict):
        return [{"title": k, "points": p} for k, p in v.items()]
    return v


def _coerce_sources(v: object) -> object:
    """模型侧容错：JSON 字符串 → list；裸 url 字符串列表 → 条目列表。"""
    v = coerce_stringified_list(v)
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except ValueError:
            return v
    if isinstance(v, list):
        return [{"url": x} if isinstance(x, str) else x for x in v]
    return v


SectionListArg = Annotated[list[GuideSection], BeforeValidator(_coerce_sections)]
SourceListArg = Annotated[list[GuideSource], BeforeValidator(_coerce_sources)]


class GuideOutput(BaseModel):
    """present_guide 的结构化返回：前端那张卡与主环看到的是同一份。

    ``markdown`` 是排好版的最终答案，由 ``adapter._merge_terminal_body`` 并回 ``final_text``
    （终结工具的产出就是最终答案），历史回看与落盘产物读的都是它。
    """

    topic: str = ""
    sections: list[GuideSection] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    sources: list[GuideSource] = Field(default_factory=list)
    closing: str = ""
    markdown: str = ""
    note: str = ""


def _clean_sections(raw: list[GuideSection]) -> list[GuideSection]:
    """丢掉空节、逐节截断要点，整体封顶。标题空但有要点的节保留（标题不是必需的信息）。"""
    out: list[GuideSection] = []
    for s in raw:
        points = [p.strip() for p in s.points if p and p.strip()][:GUIDE_MAX_POINTS]
        title = (s.title or "").strip()
        if not title and not points:
            continue
        out.append(GuideSection(title=title, points=points))
    return out[:GUIDE_MAX_SECTIONS]


def _clean_sources(raw: list[GuideSource]) -> tuple[list[GuideSource], int]:
    """只留 http(s) 链接、按 url 去重、封顶；返回 (留下的, 丢掉几条)。

    丢掉的条数要如实回给模型：它以为自己列了出处，实际被吞掉，那答案里「据某评测」那句就成了
    无出处的断言。
    """
    out: list[GuideSource] = []
    seen: set[str] = set()
    dropped = 0
    for s in raw:
        url = (s.url or "").strip()
        if not url.startswith(("http://", "https://")):
            dropped += 1
            continue
        if url in seen:
            continue
        seen.add(url)
        out.append(GuideSource(title=(s.title or "").strip(), url=url))
    dropped += max(0, len(out) - GUIDE_MAX_SOURCES)
    return out[:GUIDE_MAX_SOURCES], dropped


def _label(src: GuideSource) -> str:
    """来源的显示名：模型给了标题就用它，没给退回域名（比裸 url 好读，也不会撑破排版）。"""
    if src.title:
        return src.title
    rest = src.url.split("://", 1)[-1]
    return rest.split("/", 1)[0] or src.url


def _render(out: GuideOutput) -> str:
    """排成 Markdown —— 前端回看（FinalAnswer）与落盘 summary.md 读的都是这一份。

    块之间空行、**块内不空行**：要点行之间插空行会被 marked 渲染成松散列表（每项套一层 ``<p>``），
    行距翻倍、读起来像几段独立的话，而它们本来就是一条标准下的几个要点。
    """
    blocks: list[str] = []
    if out.topic:
        blocks.append(f"### {out.topic}")
    if out.assumptions:
        blocks.append("按这些假设讲的（不对就说一声）：" + "；".join(out.assumptions) + "。")
    for i, s in enumerate(out.sections, 1):
        head = f"#### {i}. {s.title}" if s.title else f"#### {i}."
        blocks.append("\n".join([head, *(f"- {p}" for p in s.points)]))
    if out.closing:
        blocks.append(out.closing)
    srcs = [s for s in out.sources if s.url]  # 防御：直接调 _render 时（单测 / 脚本）没过清洗
    if srcs:
        blocks.append("\n".join(["**参考来源**", *(f"- [{_label(s)}]({s.url})" for s in srcs)]))
    return "\n\n".join(blocks)


@tool
async def present_guide(
    topic: str = "",
    sections: SectionListArg | None = None,
    assumptions: StrListArg | None = None,
    sources: SourceListArg | None = None,
    closing: str = "",
) -> GuideOutput:
    """终结性：把「这个品类怎么挑」的判断依据排成一张分节指南（无商品卡的那一轮用它收尾）。
    何时调用：这轮答的是选购标准 / 两种做法的取舍 / 从哪下手，正文没有商品清单。
    参数：topic 这份指南讲什么（一句话）；sections 每节一条标准
    [{title, points:[要点]}]，3~5 节、一节 2~4 个要点，**没有依据的标准不写**；assumptions
    缺的事实你按什么估的（「按两人用估的」），让用户一个词就能纠正；sources 用到的公网出处
    [{title, url}]，url 逐字抄 research 的 claim、不要自己编；closing 末尾一句收束（推荐方向 /
    要紧提醒）。要让用户挑一条继续，同一轮再发 ask_user(closes_turn=true) 带选项，别写进本工具。
    """
    await monitor.report_tool_start("present_guide", topic=(topic or "").strip())
    sec = _clean_sections(list(sections or []))
    src, dropped = _clean_sources(list(sources or []))
    notes: list[str] = []
    if dropped:
        notes.append(f"已忽略 {dropped} 条来源（不是 http/https 链接或超出上限），正文别提它们。")
    if not sec:
        # 空指南照样终结、不抛错：这里抛出去只会让模型换个工具再讲一遍，而它手上的内容本来就空。
        notes.append("没有给出任何标准内容（sections 为空），用户看到的只有 topic 与结尾那句。")
        logger.warning("present_guide 收到空 sections topic=%r", topic)
    out = GuideOutput(
        topic=(topic or "").strip(),
        sections=sec,
        assumptions=[a.strip() for a in (assumptions or []) if a and a.strip()],
        sources=src,
        closing=(closing or "").strip(),
        note="；".join(notes),
    )
    out.markdown = _render(out)
    await monitor.report_guide(out.model_dump(exclude={"markdown", "note"}))
    await monitor.report_tool_end(
        "present_guide",
        results=len(out.sections),
        degraded=not out.sections,
        result=out.topic or out.note or "已给出选购指南",
    )
    return out
