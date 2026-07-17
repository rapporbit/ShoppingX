"""RAG 品类知识库的卡片 schema 与品类名归一（refdocs 13 / 13-1）。

知识库**不放博主长文**，只放四类**结构化卡片**——每张卡都是「已经提炼好的一段结论」：

- ``bestseller``       爆款卡：这个品类的代表性热卖款（给 item_search 拆 sub-query）。
- ``attribute``        属性卡：典型属性的分布（这里用评分档位分布，给 item_picker 判断品质大盘）。
- ``price_range``      价格卡：便宜/中档/高端三档价位（给 item_picker 判断价格档位）。
- ``attribute_schema`` 属性骨架卡：这个品类**该看哪些维度、每个维度有哪些取值**（材质 /
  特性 / 闭合方式…），来自 Shopify 商品分类法。让 planner 能把「不要塑料的」落到 Material
  维度、item_picker 能按「抗造」对应 Features 取值打分——补上前三类「只有数值、没有语义骨架」
  的认知缺口（refdocs 13 §3.1「属性图谱卡片」的语义维度落地）。

卡片由离线 ETL（``scripts/build_category_kb.py``）从 ``data/rag`` 的真实商品数据聚合产出，
工具运行时只读不写。``content_vector`` 是卡片文本的稠密向量（建库时用同一个
:class:`~app.recall.towers.TowerClient` 编码，保证和在线 query 编码同一向量空间）——
它只服务检索，**不回传给模型**，故 ``model_dump`` 时排除。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

CardType = Literal["bestseller", "attribute", "price_range", "attribute_schema"]


class CategoryCard(BaseModel):
    """知识库里的一张品类卡片（结构化结论 + 支撑证据）。"""

    card_id: str
    category: str  # 已归一的标准品类名（见 normalize_category）
    card_type: CardType
    summary: str  # 一段已提炼好的结论，写法按 card_type 有格式约定（见 ETL）
    # 品类的别名/口语说法（中英混合，ETL 用 LLM 生成，同品类各卡共享一份）。参与 BM25
    # 与向量编码，让「旅行收纳」「packing cubes」这类非标准写法也能锚定到品类。
    aliases: list[str] = Field(default_factory=list)
    raw_evidence: list[str] = Field(default_factory=list)  # 1-3 段支撑原文（不回传给模型）
    last_updated: str = ""  # ISO 时间戳（建库时写入）
    confidence: float = 0.0  # 0-1：样本量 + 措辞确定性自评
    # 卡片文本的稠密向量，仅供本地回退后端做语义召回；不进结构化输出。
    content_vector: list[float] = Field(default_factory=list, exclude=True)

    def search_text(self) -> str:
        """参与编码 / BM25 / 本地 token 召回的文本。

        品类名**重复一次提权**（与 OpenSearch BM25 路径里 ``category^2`` 的字段加权同理）：
        attribute / price_range 卡的 summary 是「评分分布…」「budget…」这类长通用串，不提权
        的话品类信号会被稀释，导致同品类卡召不全。重复品类名让三类卡都稳稳带上品类锚点。
        别名并入文本：非标准写法（中文/口语）的 query 靠它拿到词面与语义两路信号。
        """
        parts = [self.category, self.category, *self.aliases, self.summary]
        return " ".join(p for p in parts if p).strip()


# ---------------------------------------------------------------------------
# 品类名归一（ETL Step 1 + 在线 query 入口共用）
#
# 不同数据源/不同用户对同一品类写法不一（"travel organizer" / "packing cubes" /
# "旅行收纳" 都指向同一品类），但它们应指向同一批卡片。归一表是「运营 + 数据团队
# 共维护的 ground truth」，不让 LLM 现场猜——这里给一份**代表性初值**（我们语料是
# 英文 Amazon 品类，故以英文为主，附少量跨语言别名示意）。
# ---------------------------------------------------------------------------
CATEGORY_ALIASES: dict[str, str] = {
    "travel organizer": "travel accessories",
    "travel organizers": "travel accessories",
    "packing cubes": "travel accessories",
    "旅行收纳": "travel accessories",
    "旅行三件套": "travel accessories",
    "coffee mug": "coffee mugs",
    "mug": "coffee mugs",
    "马克杯": "coffee mugs",
    "咖啡杯": "coffee mugs",
    "running shoe": "running shoes",
    "跑鞋": "running shoes",
    "earbuds": "headphones",
    "earphones": "headphones",
    "耳机": "headphones",
}


def normalize_category(raw: str) -> str:
    """把品类名归一到标准写法：先做大小写/空白清洗，再查别名表。

    查不到别名就返回清洗后的原值（开放词表：新品类不强行映射，留给冷启动兜底）。
    """
    key = " ".join(raw.strip().lower().split())
    return CATEGORY_ALIASES.get(key, key)
