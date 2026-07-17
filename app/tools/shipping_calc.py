"""shipping_calc —— 到手价估算（landed cost = 货价 + 国际运费 + 关税）。

「便宜」要看到手价，不是看标价：跨境买东西常被运费 + 关税反超。本工具对已归一到 USD 的
候选，逐条估运费（:func:`app.recall.shipping.estimate_shipping`，按平台发货区域 + 重量）
与关税（:func:`app.recall.duty.estimate_duty`，按品类税率 + 进口国免征额），合成到手价并
按到手价升序排，让模型看到真实的「最划算」。

依赖 ``price_compare`` 先填好 ``price_usd``；没填的条目算不了到手价，原样保留并沉底。
**不 fork**：逐条输入→输出映射，主 loop 直接调。
"""

from __future__ import annotations

import json
from typing import Annotated

from langchain_core.tools import InjectedToolArg, tool
from pydantic import BaseModel

from app.api import monitor
from app.api.context import get_dest_country
from app.recall.duty import estimate_duty
from app.recall.shipping import estimate_shipping
from app.tools._args import StrListArg
from app.tools._candidates import compact_candidates, hydrate, register_updates
from app.tools.schemas import ItemCandidate


class ShippingCalcOutput(BaseModel):
    """shipping_calc 的结构化返回。"""

    dest_country: str
    base_currency: str
    items: list[ItemCandidate]  # 已填 shipping_usd/duty_usd/landed_usd，按到手价升序
    uncosted: list[str]  # 缺 price_usd、无法算到手价的 item_id

    def __str__(self) -> str:
        """喂给模型的紧凑形态：JSON 且丢 url/image_url（见 _candidates.compact_candidates）。"""
        return json.dumps(
            {
                "dest_country": self.dest_country,
                "base_currency": self.base_currency,
                "items": compact_candidates(self.items),
                "uncosted": self.uncosted,
            },
            ensure_ascii=False,
            default=str,
        )


def cost_item(item: ItemCandidate, dest_country: str, base: str = "USD") -> bool:
    """就地给一件候选补运费 / 关税 / 到手价（确定性表查，零 LLM）；缺 ``price_usd`` 返回 False。

    工具主循环与 shopping_summary 的收尾补算共用这一份口径——补搜进来的候选没经过
    price_compare/shipping_calc，landed 为 None；收尾不补算就会「裸价混进到手价总价」还声称
    含税（线上 badcase 4c0ac682：$55.31 里两件是裸价）。
    """
    if item.price_usd is None:
        return False
    ship = estimate_shipping(
        platform=item.platform,
        dest_country=dest_country,
        weight_kg=item.weight_kg,
        item_price_base=item.price_usd,
        base=base,
    )
    duty = estimate_duty(
        price_base=item.price_usd,
        category=item.category,
        dest_country=dest_country,
        base=base,
        # 发货国 == 收货国 → 国内单，无进口关税。不传 platform 就等于把「美国仓发美国」
        # 也当跨境征税（改造前正是如此，只是被 US $800 免征额碰巧掩盖了）。
        platform=item.platform,
    )
    item.shipping_usd = ship.amount
    item.duty_usd = duty.amount
    item.weight_kg = ship.weight_kg  # 回填实际采用的重量（含默认值），便于核对
    item.landed_usd = round(item.price_usd + ship.amount + duty.amount, 2)
    return True


@tool
async def shipping_calc(
    item_ids: StrListArg | None = None,
    dest_country: str = "",
    candidates: Annotated[list[ItemCandidate] | None, InjectedToolArg] = None,
) -> ShippingCalcOutput:
    """估算每件候选商品的到手价（货价 + 国际运费 + 关税）。

    何时调用：**通常不需要**——price_compare 已一步算好到手价（landed_usd）。仅当手里的候选
    缺到手价（如未经 price_compare、或需要换收货国重算）时才单独调本工具。
    参数：
      - item_ids：要算到手价的候选 **item_id 列表**（通常是 price_compare 之后的那批 id）。只传 id
        即可——工具自动按 id 从会话登记表捞回全量候选（已含 price_usd），**不要**重吐候选对象。
      - dest_country：收货国 ISO 码。**通常不必传**——系统已按用户原话 / 会话状态确定，留空即可。
    """
    # 基准币钉死 USD（同 price_compare）：输入 price_usd、输出 *_usd 全是 USD 语义，
    # base 若可被模型改写只会造出字段名与量纲不符的脏数据。
    base = "USD"
    # 候选来源：模型只传 item_ids → 按 id hydrate（含 price_compare 写回的 price_usd）。candidates
    # 为 InjectedToolArg（模型侧不可见）：仅供直接调用 / 单测注入现成候选，绕开登记表。
    if candidates is None:
        candidates = hydrate(list(item_ids or []))
    # 收货国**机制兜底**：模型漏传 / 传错都不影响正确性——planner 早已按「用户原话 > 会话 slots >
    # 长期记忆 > 默认国」确定性判完并写进会话上下文。这里读它，而不是信模型这一次填得对。
    # 收货国决定关税免征额（US $0 / CN $7 / AU $660），判错整条到手价就废了，不能交给 prompt 保证。
    dest_country = (dest_country or "").strip().upper() or get_dest_country()
    await monitor.report_tool_start("shipping_calc", count=len(candidates), dest=dest_country)

    costed: list[ItemCandidate] = []
    uncosted: list[str] = []
    for c in candidates:
        item = c.model_copy()
        if not cost_item(item, dest_country, base=base):
            uncosted.append(item.item_id)
        costed.append(item)

    # 能算到手价的按 landed 升序在前，算不了的沉底。
    costed.sort(
        key=lambda c: (c.landed_usd is None, c.landed_usd if c.landed_usd is not None else 0.0)
    )

    # 把运费/关税/到手价回写会话登记表：下游 item_picker 改按 id hydrate 时，预算硬筛用的是最新
    # landed_usd（而非源头快照的 None，否则续用 price_usd 会漏算运费关税）。
    register_updates(costed)

    out = ShippingCalcOutput(
        dest_country=dest_country.upper(),
        base_currency=base,
        items=costed,
        uncosted=uncosted,
    )
    # 思考结果摘要：到手价（货价+运费+关税）最低的几件（供前端展开看这一步「算出了什么」）。
    priced = [c for c in costed if c.landed_usd is not None]
    if priced:
        sc_lines = [f"到手价（货价+运费+关税）至 {dest_country.upper()}，共 {len(priced)} 件："]
        for c in priced[:3]:
            sc_lines.append(f"· ${c.landed_usd:.2f} {c.title}（{c.platform}）")
        sc_result = "\n".join(sc_lines)
    else:
        sc_result = "无可估到手价的候选（缺货价）"
    await monitor.report_tool_end(
        "shipping_calc", costed=len(costed) - len(uncosted), result=sc_result
    )
    return out
