"""present_comparison —— 把几件**已展示过**的商品横向摆开：逐件 pros/cons/best_for + 一件推荐。

**为什么单独做一个工具，而不是让模型在收尾文案里写对比。** 对比是有结构的（N 件 × 几个维度 +
一个结论），写进自由文案就只能整段读、没法对齐渲染；前端那张对比表本来就在（平台/价格/槽位/理由
四行都是本地结构化字段），缺的恰恰是「哪件更值、各自适合谁」这一档判断。工具把判断也变成结构，
表才填得进去。

**两条入口，一个实现**（:func:`compare_items`）：

1. **前端对比栏按钮**（确定性路径）：走 ``POST /api/threads/{thread_id}/compare`` —— 用户勾了几件
   点「比一比」，意图已百分之百明确，不需要再让模型判断该不该调、调哪几件。与 ``/api/similar``
   同一个取舍：
   意图确定 + 不需要主环上下文的活，塞进 AgentLoop 只换来几十秒规划开销。
2. **对话里说「帮我比比这几个」**：模型调本工具，终结性（调完即收尾）。不让它调完对比再调一次
   ``shopping_summary`` 把同样的话讲第二遍——那正是 over-loop 的老形态。

**id 幻觉是这里唯一会造成用户可见错误的失败形态**，所以两道都收在函数侧、不靠提示词：入参 id 只
认候选登记表（``hydrate``），归纳模型回的 id 不在入参集合里即整条丢弃、``recommended_item_id``
对不上就置空（宁可不推荐，也不能推一件不在表里的商品——前端按 id 高亮，错位比没有更糟）。

**价格口径不由模型判**：几件里混着到手价与货价时直接比数字会误导，这个判定是确定性的
（看字段在不在），函数侧算好写进 ``note``，不进归纳模型的自由发挥范围。
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from app.agent.invoke import call_structured
from app.agent.llm import get_fast_llm
from app.api import monitor
from app.tools._args import StrListArg
from app.tools._candidates import hydrate
from app.tools._shell import tool
from app.tools.schemas import ItemCandidate
from app.utils.env import env_int

logger = logging.getLogger(__name__)

#: 一次最多比几件。超出截断——对比表横向排版超过 4 列就没法看了，前端也只让勾 2~4 件。
COMPARE_MAX_ITEMS = env_int("COMPARE_MAX_ITEMS", 4)


class ComparisonItem(BaseModel):
    """一件商品在这次对比里的位置。"""

    item_id: str
    pros: list[str] = Field(default_factory=list, description="相对其它几件的优势，每条一句")
    cons: list[str] = Field(default_factory=list, description="相对劣势；没有明显短板就留空")
    best_for: str = Field(default="", description="什么样的人 / 场景该选它，一句话")


class ComparisonOutput(BaseModel):
    """present_comparison 的结构化返回：前端对比表与主环看到的是同一份。"""

    items: list[ComparisonItem] = Field(default_factory=list)
    recommended_item_id: str = Field(default="", description="最推荐的一件；拿不准就留空")
    recommendation_reason: str = Field(default="", description="为什么推荐它，一句话")
    note: str = Field(default="", description="价格口径不一致 / 资料不足等提醒")


class _ComparisonDraft(BaseModel):
    """归纳模型的产出。``note`` 由函数侧填（价格口径是算出来的，不是判断出来的）。"""

    items: list[ComparisonItem] = Field(default_factory=list)
    recommended_item_id: str = ""
    recommendation_reason: str = ""


def _price_kind(c: ItemCandidate) -> str:
    """这件商品当前报的是哪种价：到手价（跑过 shipping_calc）还是货价。"""
    return "landed" if c.landed_usd is not None else "list"


def _mixed_price_note(cands: list[ItemCandidate]) -> str:
    """几件里混着两种价格口径时的提醒（确定性判定，与前端那条 compare-note 同一口径）。"""
    if len({_price_kind(c) for c in cands}) > 1:
        return (
            "这几件的价格口径不一致（有的是含税运到手价、有的只是货价），直接比数字会误导；"
            "要严格比总花费，先把它们都算一遍到手价。"
        )
    return ""


_SYSTEM = """你在帮购物助手做几件商品的横向对比。只依据给定字段作判断，字段没有的信息不要脑补。

规则：
1. 每件都要给 pros / cons / best_for，**相对这几件之间比**，不是孤立夸一句。
2. item_id 必须逐字复制给定的那几个，一个都不能改写或新造。
3. cons 没有明显短板就留空，不要为了对仗硬凑。
4. best_for 写「什么人 / 什么场景该选它」，一句话，别写成广告语。
5. recommended_item_id 只选一件；几件确实各有取舍、分不出高下时留空，并在
   recommendation_reason 里说清楚分歧在哪。留空比硬推一件诚实。"""


def _fmt(c: ItemCandidate) -> str:
    """一件商品喂给归纳模型的行。只投影对比用得上的字段，不整个 dump。"""
    price = (
        f"到手价 ${c.landed_usd:.2f}"
        if c.landed_usd is not None
        else (f"货价 ${c.price_usd:.2f}" if c.price_usd is not None else "价格未知")
    )
    bits = [f"item_id={c.item_id}", f"平台={c.platform}", f"标题={c.title}", price]
    if c.brand:
        bits.append(f"品牌={c.brand}")
    if c.rating is not None:
        bits.append(f"评分={c.rating}（{c.reviews_count or 0} 条评价）")
    if c.category:
        bits.append(f"品类={c.category}")
    if c.pick_reason:
        bits.append(f"此前入选理由={c.pick_reason}")
    return "- " + "；".join(bits)


def _ground(draft: _ComparisonDraft, allowed: list[str]) -> tuple[list[ComparisonItem], str, str]:
    """把归纳结果钉回入参的 id 集合：不在集合里的整条丢、缺席的补空壳、顺序按入参排。

    前端按 item_id 找列、按 recommended_item_id 高亮，**id 错一个就是用户可见的错位**——比没有
    对比更糟。所以这里不做「模糊匹配一下说不定是它」，对不上就是丢。缺席的补空壳而不是省略，
    是为了表格列数恒等于用户勾选的件数：少一列会让人以为那件被系统判为不值一提。
    """
    by_id = {it.item_id: it for it in draft.items if it.item_id in set(allowed)}
    items = [by_id.get(i) or ComparisonItem(item_id=i) for i in allowed]
    rec = draft.recommended_item_id if draft.recommended_item_id in set(allowed) else ""
    reason = draft.recommendation_reason if rec else ""
    return items, rec, reason


async def compare_items(item_ids: list[str]) -> ComparisonOutput:
    """对比的唯一实现：工具入口与 ``POST /api/threads/{thread_id}/compare`` 都调它。

    归纳失败不抛：回一份只有 note 的空对比，让调用方（前端表 / 主 agent）如实说「这次没比出来」，
    而不是把整条链路打挂——用户勾的那几件商品本身还在，表照样看得见。
    """
    ids = [i.strip() for i in item_ids if i and i.strip()][:COMPARE_MAX_ITEMS]
    cands = hydrate(ids)
    if len(cands) < 2:
        return ComparisonOutput(
            note="需要至少 2 件本会话展示过的商品才能对比（拿不到的 item_id 已忽略）。"
        )

    allowed = [c.item_id for c in cands]
    note = _mixed_price_note(cands)
    user = "请对比下面这几件商品：\n" + "\n".join(_fmt(c) for c in cands)
    try:
        draft = await call_structured(
            get_fast_llm(), [("system", _SYSTEM), ("user", user)], _ComparisonDraft
        )
    except Exception:
        logger.warning("present_comparison 归纳失败", exc_info=True)
        return ComparisonOutput(
            items=[ComparisonItem(item_id=i) for i in allowed],
            note="；".join(filter(None, [note, "这次没能比出结论，可以换个问法再试。"])),
        )

    items, rec, reason = _ground(draft, allowed)
    return ComparisonOutput(
        items=items, recommended_item_id=rec, recommendation_reason=reason, note=note
    )


@tool
async def present_comparison(item_ids: StrListArg) -> ComparisonOutput:
    """把 2~4 件**本会话展示过**的商品横向对比：逐件优劣势 + 适合谁 + 推荐哪件。调完即收尾。

    何时调用：用户点名要比较已经给过的几件（「这几个哪个好」「第 1 和第 3 比一比」）。
    参数：item_ids 只能来自你见过的候选 / 清单里的 item_id，不要自己编。
    不要用它找商品或比价格数字——前者用 item_search，后者用 price_compare。
    """
    await monitor.report_tool_start("present_comparison", item_ids=list(item_ids))
    out = await compare_items(list(item_ids))
    picked = next((i.item_id for i in out.items if i.item_id == out.recommended_item_id), "")
    await monitor.report_tool_end(
        "present_comparison",
        results=len(out.items),
        degraded=not out.items,
        result=out.recommendation_reason or out.note or f"已对比 {len(out.items)} 件",
        recommended=picked,
    )
    return out
