"""price_compare —— 跨平台比价 + 到手价一步出（汇率归一到 USD，含运费与关税）。

6 个平台报价币种各异（USD/MXN/CLP/IDR/MYR…），不归一就没法比。本工具把每条候选的
原价用 :func:`app.recall.fx.to_base` 折算到统一基准币（默认 USD），随后对留下的 top_n
直接内联估运费 + 关税合成到手价（landed = 货价 + 运费 + 关税），按到手价升序排。

**为什么不再「只折算」**（延迟归因 round2 刀3）：折算与到手价都是 ~0s 的纯计算，拆成
price_compare → shipping_calc 两个工具却要主模型多解码一整轮（实测 4~6s）才能把 id 从
上一个的输出搬进下一个的入参。两步之间无任何模型决策价值，故合为一步；``shipping_calc``
保留，仅供候选缺到手价时单独补算。收货国由机制兜底（planner 已确定性判定），不依赖模型传参。
**不 fork**：一次聚合 + 逐条映射，无并行 / 无深链，主 loop 直接调。

遇到汇率表里没有的币种：不抛错（否则一条脏数据崩掉整次比价），把该条 ``price_usd``
留 ``None`` 并塞进 ``note`` 说明，排序时沉到末尾，让模型自己判断。
"""

from __future__ import annotations

import json
from typing import Annotated

from langchain_core.tools import InjectedToolArg, tool
from pydantic import BaseModel

from app.api import monitor
from app.api.context import get_dest_country
from app.recall.duty import estimate_duty
from app.recall.fx import to_base_or_none
from app.recall.shipping import estimate_shipping
from app.tools._args import StrListArg
from app.tools._candidates import (
    compact_candidates,
    hydrate,
    register_updates,
    registry_snapshot,
)
from app.tools.schemas import ItemCandidate

# 比价后默认只保留前 N 条进入下游（到手价计算），上游剪枝省下游算力。
DEFAULT_TOP_N = 12


class PriceCompareOutput(BaseModel):
    """price_compare 的结构化返回。``ranked`` 已含 landed_usd（到手价），按到手价升序。"""

    base_currency: str
    dest_country: str = ""
    ranked: list[ItemCandidate]  # 已填 price_usd + shipping/duty/landed_usd，按到手价升序
    cheapest_per_platform: dict[str, str]  # platform -> item_id
    skipped: list[str]  # 因未知币种无法折算的 item_id

    def __str__(self) -> str:
        """喂给模型的紧凑形态：JSON 且丢 url/image_url（见 _candidates.compact_candidates）。

        标题截短回显（title_chars）：ranked 的完整标题在上下文里的检索结果中已出现过，
        这里只要短 handle + item_id 够模型对上号；下游取数按 id hydrate 全量。
        """
        return json.dumps(
            {
                "base_currency": self.base_currency,
                "dest_country": self.dest_country,
                "ranked": compact_candidates(self.ranked, title_chars=60),
                "cheapest_per_platform": self.cheapest_per_platform,
                "skipped": self.skipped,
                # 写给模型的直接指令：省一轮 shipping_calc（它只会重算一遍同样的数）。
                "note": "到手价（landed_usd，含运费+关税）已算好，无需再调 shipping_calc，"
                "可直接 item_picker 精挑。",
            },
            ensure_ascii=False,
            default=str,
        )


@tool
async def price_compare(
    item_ids: StrListArg | None = None,
    top_n: int = DEFAULT_TOP_N,
    candidates: Annotated[list[ItemCandidate] | None, InjectedToolArg] = None,
) -> PriceCompareOutput:
    """把多平台候选商品的价格统一折算成 USD，并一步算出到手价（货价+运费+关税）排序。

    何时调用：手里有跨平台候选、要按价格横向比较时（通常在合流 item_search 结果之后）。
    返回的 ranked 已含 landed_usd 到手价——**调完本工具无需再调 shipping_calc**。
    参数：
      - item_ids：**通常不必传**——缺省即本轮全部候选（推荐做法）；只在确实只比其中几件时传 id。
      - top_n：返回给你看的前 N 条（最便宜的），默认 12。**不影响谁被算到手价**——全部候选都会算。
    """
    # 基准币钉死 USD：price_usd / landed_usd 等字段、item_picker 的预算硬筛、前端展示全按
    # USD 语义消费。此前 base 暴露为模型参数，一旦传个 "EUR"，price_usd 里装的就是欧元值、
    # 字段名却还叫 *_usd——量纲错乱会顺着登记表污染整条会话。
    base = "USD"
    # 候选来源：缺省 = 会话登记表里的**全部候选**。模型漏传 / 少传 id 的代价太大——没被传进来的
    # 候选算不到到手价，可 item_picker 照样会从全量池里把它挑出来，于是一批卡里混着「到手价」和
    # 「标价」两种口径（见下面到手价那段注释）。默认全量，模型想精选再传 id。
    # candidates 为 InjectedToolArg（模型侧不可见）：仅供直接调用 / 单测注入现成候选，绕开登记表。
    if candidates is None:
        ids = list(item_ids or [])
        candidates = hydrate(ids) if ids else registry_snapshot()
    await monitor.report_tool_start("price_compare", count=len(candidates), base=base)

    priced: list[ItemCandidate] = []
    skipped: list[str] = []
    for c in candidates:
        item = c.model_copy()
        item.price_usd = to_base_or_none(item.price, item.currency, base)
        if item.price_usd is None:  # 金额缺失或币种未知 → 折不动
            skipped.append(item.item_id)
        priced.append(item)

    # 能折算的按归一价升序在前，未知币种（price_usd=None）沉底。
    priced.sort(
        key=lambda c: (c.price_usd is None, c.price_usd if c.price_usd is not None else 0.0)
    )

    cheapest: dict[str, str] = {}
    for c in priced:
        if c.price_usd is None:
            continue
        if c.platform not in cheapest:  # priced 已升序，首见即该平台最便宜
            cheapest[c.platform] = c.item_id

    # 到手价内联（原 shipping_calc 的活）：**对全量候选**逐条估运费 + 关税，不是只算 top_n。
    #
    # 曾经这里只算 ranked（top_n=12）里的那几条，可 item_picker 精挑走的是**全量**候选池——于是它
    # 挑出的商品里，一部分有到手价、一部分只有平台标价：① 同一批商品卡混口径，用户拿「$28 到手」
    # 和「$9.99 标价」没法比；② 更糟的是 picker 的 _effective_price 就拿这两个数混着排序，排出来
    # 的「最便宜」是假的。到手价是**纯本地计算**（fx 静态表 + 运费 + 关税，零 LLM 零网络），
    # 40 条和 12 条没有成本差别——省这一点算力换来一批口径不一的价格，血亏。
    # top_n 的语义因此收窄成「ranked 里回给模型看几条」，不再决定「谁有到手价」。
    #
    # 收货国机制兜底——planner 早已按「用户原话 > 会话 slots > 长期记忆 > 默认国」确定性判完并写进
    # 会话上下文。
    dest_country = get_dest_country()
    for item in priced:
        if item.price_usd is None:
            continue
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
            platform=item.platform,  # 发货国==收货国 → 国内单免关税（见 shipping_calc 同款注释）
        )
        item.shipping_usd = ship.amount
        item.duty_usd = duty.amount
        item.weight_kg = ship.weight_kg
        item.landed_usd = round(item.price_usd + ship.amount + duty.amount, 2)
    # 「最划算」按到手价说了算：能算 landed 的升序在前，折不动的沉底。
    priced.sort(
        key=lambda c: (c.landed_usd is None, c.landed_usd if c.landed_usd is not None else 0.0)
    )

    # 把 price_usd + 运费/关税/到手价回写会话登记表——**回写全量**，不只是回给模型的那 top_n：
    # 下游 item_picker 在全量候选池上精挑，登记表里漏掉谁，谁挑中了就只能顶着一个没算过税运的
    # 标价上桌（见上面到手价那段注释）。
    register_updates(priced)
    ranked = priced[: max(1, top_n)]  # top_n 只决定「回给模型看几条」

    out = PriceCompareOutput(
        base_currency=base,
        dest_country=dest_country.upper(),
        ranked=ranked,
        cheapest_per_platform=cheapest,
        skipped=skipped,
    )
    # 思考结果摘要：到手价最低的几件（供前端展开看这一步「比出了什么」）。
    if ranked:
        pc_lines = [
            f"按到手价（货价+运费+关税，{base}）升序至 "
            f"{dest_country.upper()}，共 {len(ranked)} 件："
        ]
        for c in ranked[:3]:
            price = f"${c.landed_usd:.2f} 到手" if c.landed_usd is not None else "—"
            pc_lines.append(f"· {price} {c.title}（{c.platform}）")
        pc_result = "\n".join(pc_lines)
    else:
        pc_result = "无可比价候选"
    await monitor.report_tool_end(
        "price_compare", ranked=len(ranked), skipped=len(skipped), result=pc_result
    )
    return out
