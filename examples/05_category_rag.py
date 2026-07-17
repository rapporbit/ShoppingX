"""M5 示例：CategoryInsight 的 RAG 链路（召回 → 精排 → 结构化提炼）。

演示 category_insight 怎么把一个品类名变成「行家常识」：
  KBClient.search（KNN+BM25 Hybrid 粗排）
    → RerankerClient（cross-encoder 精排，剔跑题卡）
    → 按 card_type 分组提炼 → 结构化 CategoryInsightOutput（给主 loop 结论，不给原文）

数据后端按 env 自动选：配了 OPENSEARCH_HOST 走真 OpenSearch（引擎层 Hybrid），否则走进程内
本地 hybrid 回退——两种都用同一份卡片、算同一个加权融合公式，离线可跑。

前置：先建知识库（产出 data/rag/category_cards.jsonl，已 gitignore）：
    uv run python scripts/build_category_kb.py
运行：
    uv run python examples/05_category_rag.py
    OPENSEARCH_HOST=localhost uv run python examples/05_category_rag.py   # 走真 OpenSearch
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.recall.kb_client import get_kb_client  # noqa: E402
from app.tools.category_insight import category_insight  # noqa: E402

# 覆盖名词类 / 跨语言 / 语义化（关 BM25）三种 query 形态。
CASES = [
    ("luggage", "deep"),
    ("旅行收纳", "quick"),  # 中文 → 别名归一到 travel accessories，靠 KNN 跨语言召回
    ("minimal aesthetic mug", "quick"),  # 语义化 query → 自动关掉 BM25 子路
]


async def main() -> None:
    backend = "OpenSearch（引擎层 Hybrid）" if get_kb_client().remote else "进程内本地 hybrid 回退"
    print(f"知识库后端：{backend}\n")

    for category, depth in CASES:
        out = await category_insight.ainvoke({"category": category, "depth": depth})
        print(f"=== query={category!r}  depth={depth}  source={out.source}")
        print(f"  归一品类: {out.category}  命中卡片: {out.card_count}  置信度: {out.confidence}")
        print(f"  代表款: {out.components[:3]}")
        if out.bestsellers:
            b = out.bestsellers[0]
            print(f"  头部爆款: {b.title}  ${b.price_usd}  {b.rating}★")
        if out.price_tiers:
            tiers = " / ".join(
                f"{t.tier} ${t.low_usd:.0f}-${t.high_usd:.0f}" for t in out.price_tiers
            )
            print(f"  价格档位: {tiers}")
        if out.attributes:
            print(f"  评分分布: {out.attributes[0].distribution}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
