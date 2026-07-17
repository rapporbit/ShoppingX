"""工具层共享的商品候选结构 —— 九大工具之间传递的「通用货币」。

主链路是一条流水线：``item_search`` 召回 → ``price_compare`` 折算 → ``shipping_calc``
算到手价 → ``item_picker`` 精挑。让这几个工具传同一种结构，模型在 Observe 阶段读到的
字段名稳定、跨工具一致，比每个工具各定一套 in/out 模型更省 token、也更不容易让模型搞混。

设计：``ItemCandidate`` 用「渐进填充」——召回阶段只有基础字段（标题/原价/原币种），
比价后补 ``price_usd``，算到手价后补 ``shipping_usd`` / ``duty_usd`` / ``landed_usd``，
精挑后补 ``pick_reason``。没算到的阶段字段在**内部结构里**留 ``None``；喂模型时由
:func:`app.tools._candidates.compact_candidates` 整个丢掉——「是 null」与「不在」对模型同义，
但 ``"shipping_usd": null`` 这类空字段要真金白银花 token（实测占召回返回体 27%），模型照样
一眼看出「这步还没做」。

基准货币统一 **USD**（与 :mod:`app.recall.fx` 默认基准、本仓库数据一致）。
"""

from __future__ import annotations

from pydantic import BaseModel

from app.recall.schemas import RecallCandidate


class ItemCandidate(BaseModel):
    """一件商品候选，贯穿召回→比价→到手价→精挑全流程（字段渐进填充）。"""

    # ---- 召回阶段（item_search 产出）----
    item_id: str
    platform: str
    title: str
    brand: str = ""
    price: float | None = None  # 原始报价（原币种）
    currency: str = "USD"
    rating: float | None = None
    reviews_count: int | None = None
    category: str = ""
    url: str = ""
    image_url: str = ""
    score: float = 0.0  # 召回相似度分
    # 套装槽位**id**（「一套齐」需求专用，见 app.tools._bundle）：这件候选是为哪个子品类槽
    # 搜的。item_search 经 resolve_slot 解析后盖章（dispatch 的 slot_scope / slot 入参），
    # bundle 组合优选按它分组；出前端/卡片时经 slot_display 映射回展示名。旧会话读回的
    # 候选可能还盖着名字章（id 化前的数据），prospective_slot 按名兼容。
    # 非套装轮恒为空串（compact_candidates 的空值过滤会自动把它从模型可见投影里丢掉）。
    slot: str = ""

    # ---- 比价阶段（price_compare 填充）----
    price_usd: float | None = None  # 折算到 USD 的货价

    # ---- 到手价阶段（shipping_calc 填充）----
    shipping_usd: float | None = None
    duty_usd: float | None = None
    landed_usd: float | None = None  # 到手价 = price_usd + shipping_usd + duty_usd
    weight_kg: float | None = None

    # ---- 精挑阶段（item_picker 填充）----
    pick_reason: str = ""  # 入选理由（对应用户的硬约束 / 软偏好）
    # 精挑时是否真命中了用户偏好词（None=没经过 picker）。给 shopping_summary 的 prompt 当判据
    # （「没一件命中 → 别包装成完美匹配」）——曾经这判据靠匹配 pick_reason 里的措辞片段
    # （「正好是你要的…」），_build_reason 改一个字 prompt 就静默失效，故改成结构化布尔。
    pref_matched: bool | None = None
    # 品类一致性分（cross-encoder，picker 相关性门写；None=本轮没打过分）。随 rerank_query
    # 一起回写登记表做**增量缓存**：补搜轮 picker 重跑时，query 没变的候选跳过重复打分。
    rerank_score: float | None = None
    rerank_query: str = ""  # 上述分数对应的干净品类 query（缓存失效判据）

    @classmethod
    def from_recall(cls, rc: RecallCandidate) -> ItemCandidate:
        """从召回层的 :class:`RecallCandidate` 构造（item_search 用）。"""
        return cls(
            item_id=rc.item_id,
            platform=rc.platform,
            title=rc.title,
            brand=rc.brand,
            price=rc.price,
            currency=rc.currency,
            rating=rc.rating,
            reviews_count=rc.reviews_count,
            category=rc.category,
            url=rc.url,
            image_url=rc.image_url,
            price_usd=rc.price_usd,
            score=rc.score,
        )
