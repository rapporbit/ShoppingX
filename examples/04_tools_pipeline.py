"""M4 示例：九大工具的主链路串联（离线，不调 LLM）。

把确定性工具按主 loop 的典型次序手工串一遍，看清数据在工具间怎么流：
  item_search（每平台一次，模拟主 loop 的跨平台 fork 并行）
    → 合流 → price_compare（汇率归一到 USD）
    → shipping_calc（到手价 = 货价 + 运费 + 关税）
    → item_picker（按预算 + 软偏好精挑，给选购理由）

这不是主 AgentLoop（那在 M9 组装）——这里手动编排，纯粹演示工具的输入输出契约与
「渐进填充」的 ItemCandidate 怎样一路被补全。用本地确定性编码 + data/ 真实索引，离线可跑。

运行：uv run python examples/04_tools_pipeline.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.tools.category_insight import category_insight  # noqa: E402
from app.tools.item_picker import item_picker  # noqa: E402
from app.tools.item_search import item_search  # noqa: E402
from app.tools.price_compare import price_compare  # noqa: E402
from app.tools.schemas import ItemCandidate  # noqa: E402
from app.tools.shipping_calc import shipping_calc  # noqa: E402

QUERY = "travel toiletry organizer bag"
PLATFORMS = ["amazon", "ebay", "shopee"]


async def main() -> None:
    # 0) 品类洞察：先摸清行情（爆款 / 价格档 / 评分大盘）。
    insight = await category_insight.ainvoke({"category": QUERY})
    print(f"[category_insight] 命中 {insight.card_count} 卡（{insight.source}），价格档：")
    for t in insight.price_tiers:
        print(f"    {t.tier:8s} ${t.low_usd:.2f} ~ ${t.high_usd:.2f}")

    # 1) 跨平台检索：主 loop 会 fork 并行，这里顺序调演示合流。
    merged: list[ItemCandidate] = []
    for p in PLATFORMS:
        res = await item_search.ainvoke({"query": QUERY, "platform": p, "top_k": 8})
        print(f"[item_search] {p}: 召回 {res.total_recall} 件")
        merged.extend(res.candidates)
    print(f"  合流共 {len(merged)} 件候选")

    # 2) 比价：归一到 USD，排序。
    pc = await price_compare.ainvoke({"candidates": [c.model_dump() for c in merged], "top_n": 15})
    print(f"[price_compare] 归一 {len(pc.ranked)} 件，最便宜：${pc.ranked[0].price_usd:.2f}")

    # 3) 到手价：货价 + 运费 + 关税。
    sc = await shipping_calc.ainvoke({"candidates": [c.model_dump() for c in pc.ranked]})
    cheapest = sc.items[0]
    print(
        f"[shipping_calc] 最低到手价 ${cheapest.landed_usd:.2f} "
        f"(货 ${cheapest.price_usd:.2f} + 运 ${cheapest.shipping_usd:.2f} "
        f"+ 税 ${cheapest.duty_usd:.2f}) @ {cheapest.platform}"
    )

    # 4) 精挑：预算 + 软偏好，给选购理由。
    ip = await item_picker.ainvoke(
        {
            "candidates": [c.model_dump() for c in sc.items],
            "budget_usd": 40,
            "prefer_keywords": ["waterproof", "travel"],
            "top_k": 3,
        }
    )
    print(
        f"[item_picker] 精挑 {len(ip.picks)} 件"
        f"（淘汰 {len(ip.excluded)}、超预算 {len(ip.over_budget)}）："
    )
    for i, c in enumerate(ip.picks, 1):
        print(f"    {i}. [{c.platform}] {c.title[:42]}…  → {c.pick_reason}")

    print("\n（shopping_summary 是终结性 LLM 工具，需真实模型，省略；见 tests/test_tools.py）")


if __name__ == "__main__":
    asyncio.run(main())
