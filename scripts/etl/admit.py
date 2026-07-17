"""ETL Step 3：入库门禁（refdocs 13-1 §2.5）。

不是抽出来就能进库——三道门串行，把不合格草卡挡在知识库外，让下游的规则提炼面对的
永远是「格式可解析、置信度够、长度合理」的卡片，而不是替小模型/聚合的自由发挥擦屁股。
"""

from __future__ import annotations

from pydantic import ValidationError

from app.recall.category_kb import CategoryCard

MIN_CONFIDENCE = 0.5
MAX_SUMMARY_LEN = 400


def admit(raw: dict) -> tuple[bool, str]:
    """校验一张卡片草稿，返回 ``(是否放行, 原因)``。"""
    # 门 1：schema 严格校验。
    try:
        card = CategoryCard(**raw)
    except ValidationError as e:
        return False, f"schema 校验失败: {e.errors()[:1]}"

    # 门 2：置信度 + 长度。
    if card.confidence < MIN_CONFIDENCE:
        return False, f"confidence {card.confidence} < {MIN_CONFIDENCE}"
    if len(card.summary) > MAX_SUMMARY_LEN:
        return False, f"summary 超长 {len(card.summary)} > {MAX_SUMMARY_LEN}"

    # 门 3：summary 格式约定校验（按 card_type 各查一个标志符）。
    if card.card_type == "bestseller" and ":" not in card.summary:
        return False, "bestseller summary 缺少品类前缀冒号"
    if card.card_type == "attribute" and "%" not in card.summary:
        return False, "attribute summary 缺少百分比"
    if card.card_type == "price_range" and "$" not in card.summary:
        return False, "price_range summary 缺少价格符"
    if card.card_type == "attribute_schema":
        if "：" not in card.summary:
            return False, "attribute_schema summary 缺少维度前缀"
        # 至少要有「映射来源 + 1 个属性」两条证据，否则是张空骨架卡，没价值。
        if len(card.raw_evidence) < 2:
            return False, "attribute_schema 缺少属性证据"

    return True, "ok"
