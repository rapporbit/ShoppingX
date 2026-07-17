"""ETL：用 LLM 生成品类的多维度属性分布卡片。

refdocs 设计的 attribute 卡应是「材质：尼龙 60% / 帆布 25%」这种多维度分布，但真实
商品数据集要么没有结构化属性字段、要么没有子品类分组。务实替代：让 LLM 基于训练知识
按品类生成「行家常识」级别的典型属性分布——数值不精确但方向正确，足以给 item_picker
提供品类锚点（"这个品类主流材质是什么""什么价位算中档"）。

卡片 summary 格式与现有 attribute 卡（评分分布）兼容：
``"DimensionName: Value1 NN% / Value2 NN% / ..."``
——``category_insight.py`` 的 ``_extract_attributes()`` 不需要改。

每个品类生成 3-5 张卡（每张一个维度），card_type 仍为 "attribute"。
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from scripts.etl._llm import LLM_SEM, get_etl_llm


class _ValuePct(BaseModel):
    value: str = Field(description="属性取值，如 Nylon / Bluetooth / Foldable")
    pct: int = Field(description="该取值在品类中的近似占比 (0-100)，所有取值合计约 100")


class _Dimension(BaseModel):
    name: str = Field(description="属性维度名，如 Material / Connectivity / Key Features")
    values: list[_ValuePct] = Field(description="该维度的典型取值及占比")


class _CategoryAttributes(BaseModel):
    dimensions: list[_Dimension] = Field(
        description="3-5 个对该品类最重要的选购维度",
        min_length=2,
        max_length=6,
    )


_SYSTEM = """\
You are a product domain expert. Given an e-commerce product category name, \
list the 3-5 most important **shopping dimensions** that buyers care about \
when choosing products in this category.

For each dimension, give the typical value distribution (approximate \
percentages that sum to ~100%). Focus on attributes that help a shopper \
**compare and decide** — skip generic dimensions like "Color" or "Brand" \
unless they are unusually decisive for this category.

Rules:
- Percentages are rough estimates based on common market composition — \
  directional accuracy matters, not decimal precision.
- Use English dimension names and value names.
- CRITICAL: never use "/" or ":" in dimension names or value names — \
  these are reserved separators. Use "&" or "and" instead \
  (e.g. "Polyester and Nylon" not "Polyester / Nylon").
- Keep value names short and simple (2-4 words max).
- If the category is too abstract to have meaningful product attributes \
  (e.g. "gift cards"), return only 2 dimensions with broad strokes.
"""


def _sanitize(s: str) -> str:
    """兜底消毒 LLM 输出：清掉解析保留分隔符 "/" 和 ":"（category_insight 的 _extract_attributes
    靠 ":"/"："切维度名、summary 结构靠 "/" 分取值，维度名/取值里混入会把下游解析带偏）。
    prompt 已要求 LLM 回避，这里再兜一道防偶发违规。折叠空白，去首尾。"""
    return " ".join(s.replace("/", " and ").replace(":", " -").split())


async def _generate_one(llm, category: str, now: str) -> list[dict]:
    """为一个品类生成属性分布卡片草稿列表。"""
    structured = llm.with_structured_output(_CategoryAttributes)
    async with LLM_SEM:
        try:
            result: _CategoryAttributes = await structured.ainvoke(
                [
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": f"Category: {category}"},
                ]
            )
        except Exception as e:
            print(f"  ⚠ {category}: LLM 调用失败 ({type(e).__name__}: {e})")
            return []

    slug = "_".join(category.lower().split())[:48]
    cards: list[dict] = []
    for dim in result.dimensions:
        # 消毒维度名/取值里的保留分隔符；summary 结构本身仍用 "/" ":" 作分隔（下游据此解析）。
        dim_name = _sanitize(dim.name)
        parts = " / ".join(f"{_sanitize(v.value)} {v.pct}%" for v in dim.values)
        summary = f"{dim_name}: {parts}"
        cards.append(
            {
                "card_id": f"{slug}_attr_{dim_name.lower().replace(' ', '_')[:20]}",
                "category": category,
                "card_type": "attribute",
                "summary": summary,
                "raw_evidence": [f"LLM-generated approximate distribution for {category}"],
                "last_updated": now,
                "confidence": 0.6,
            }
        )
    return cards


async def build_llm_attribute_cards(
    categories: list[str],
    *,
    progress: bool = True,
) -> list[dict]:
    """批量生成所有品类的 LLM 属性卡片。返回卡片草稿列表（待 admit + 编码）。"""
    llm = get_etl_llm()
    now = datetime.now(UTC).isoformat(timespec="seconds")

    tasks = [_generate_one(llm, cat, now) for cat in categories]

    all_cards: list[dict] = []
    done = 0
    for coro in asyncio.as_completed(tasks):
        cards = await coro
        all_cards.extend(cards)
        done += 1
        if progress and done % 20 == 0:
            print(f"  LLM 属性卡进度：{done}/{len(categories)}")

    if progress:
        print(f"  LLM 属性卡完成：{len(categories)} 个品类 → {len(all_cards)} 张卡片")
    return all_cards


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

    test_cats = sys.argv[1:] or ["headphones & earbuds", "travel accessories", "running shoes"]
    cards = asyncio.run(build_llm_attribute_cards(test_cats))
    for c in cards:
        print(json.dumps(c, ensure_ascii=False, indent=2))
