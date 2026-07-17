"""召回层共享数据结构。

放在独立模块，避免 ``ann.py`` / ``towers.py`` / M4 的 ``item_search`` 工具之间循环依赖。
"""

from pydantic import BaseModel, Field


class RecallCandidate(BaseModel):
    """一条召回候选商品。

    字段是跨 6 个平台**对齐后**的归一化视图（原始 CSV 各家 schema 不同，由建索引脚本统一）。
    ``score`` 是本次检索的相似度分（内积，向量已归一化 → 等价余弦，越大越像）。
    """

    item_id: str
    platform: str
    title: str
    brand: str = ""
    price: float | None = None
    currency: str = "USD"
    rating: float | None = None
    reviews_count: int | None = None
    category: str = ""
    url: str = ""
    image_url: str = ""
    price_usd: float | None = None  # 建库时预折算（Qdrant filter + 渐进填充）
    score: float = 0.0


class ItemRecord(BaseModel):
    """建索引时落盘到 metadata sidecar 的一条商品记录（不含向量）。

    与 :class:`RecallCandidate` 几乎同构，但不带运行时的 ``score``——
    检索时把命中记录 + 当次得分组装成 ``RecallCandidate`` 回给上层。
    """

    item_id: str
    platform: str
    title: str
    brand: str = ""
    price: float | None = None
    currency: str = "USD"
    rating: float | None = None
    reviews_count: int | None = None
    category: str = ""
    url: str = ""
    image_url: str = ""
    price_usd: float | None = None  # 建库时预折算（供 Qdrant payload filter）
    # 仅建索引阶段用于编码，不回传给模型，故 dump 时排除以缩小 sidecar 体积。
    embed_text: str = Field(default="", exclude=True)
