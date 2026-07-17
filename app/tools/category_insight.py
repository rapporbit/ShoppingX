"""category_insight —— 品类洞察（两段式 RAG：Hybrid 品类定位 → 按品类结构化取卡）。

检索 / 精挑前先了解一个品类「长什么样」：典型爆款、评分大盘、价格档位，让后续 query
拆解和 item_picker 判断有锚点。这是链路里的**前置认知层**——不是又一个搜商品的工具，
而是给后面所有工具供「这个品类的行家常识」（refdocs 13）。

**两段式 RAG 管线（refdocs 13 §5 的思想 + 结构化重构）：** 知识库按品类组织、下游按
card_type 消费，检索的本质是「定位品类」而非「对 2044 张卡做全局 top-K」：
1. 定位：query 走 KNN+BM25 Hybrid 按品类投票（``KBClient.resolve_category``），得候选品类
   与置信度；歧义时 cross-encoder 只对 2 条「品类名+别名」消歧（``RerankerClient``）。
2. 取卡：按定位到的品类 term 精确取全四类卡（``fetch_cards``）——零漏召、零跨品类污染。
3. 提炼：按 card_type 分组，把多张卡片合并成结构化字段——**给主 loop 的是结论不是原文**
   （raw_evidence 在工具内部消化掉，不回传）。

**为什么是 fork「调用链 ≥ 3」候选（refdocs 13 §6）：** 内部「编码 → Hybrid 定位 →（消歧）→
取卡 → 分组提炼」≥3 跳，且召回的原始卡片大、不该污染主 loop 上下文。``depth="deep"`` 尤其
适合 fork；``depth="quick"`` 主 loop 直接调也行——fork 与否由主 loop 判。

数据后端：配了 ``OPENSEARCH_HOST`` 走真 OpenSearch（``source="opensearch"``），否则走进程内
本地 hybrid 回退（``source="local_kb_fallback"``）——诚实标注当前数据通路。
"""

from __future__ import annotations

import asyncio
import re
from typing import Literal

import numpy as np
from langchain_core.tools import tool
from pydantic import BaseModel

from app.api import monitor
from app.observability import metrics
from app.recall.category_kb import CategoryCard, normalize_category
from app.recall.kb_client import get_kb_client
from app.recall.reranker import get_reranker
from app.recall.semantic_cache import SemanticCache
from app.recall.towers import get_tower_client
from app.utils.env import env_float, env_int

# 品类定位歧义线：top2 置信度 / top1 置信度高于此，视为「两个品类都像」，交给
# cross-encoder 对两个品类的描述文本消歧（rerank 的对象是品类，不再是 30 张长文本卡）。
AMBIGUITY_RATIO = 0.75

# 定位核验闸：投票占比是**相对**信号——知识库压根没收录的品类（保温杯/瑜伽垫/机械键盘）
# 也能在无关品类上凑出 0.37~0.49 的占比，单看它切不开对错。置信低于 GATE 时再花一次
# cross-encoder 判「query ↔ 胜出品类」的**绝对**相关性：金标实测正确定位最低 0.014，
# 库外垃圾 ≈0.0000~0.0001，0.005 落在两者之间（差两个数量级）。低于线判「未收录」返回
# 空卡，让 Agent 转 web_search，而不是把汽车清洁剂当保温杯行情喂给它。
# 仅远程精排真的可用时启用——本地 token 重叠兜底打分量纲不同，闸会误杀一切中文 query。
RESOLVE_CONF_GATE = 0.55
RERANK_RELEVANCE_MIN = 0.005

# 薄数据自报线：卡片数或卡片平均置信度（ETL 按样本量 + 措辞确定性自评）低于线时，返回里
# 明说「库内数据薄、参考价值有限」——RAG 库是脏 CSV 聚合的，样本薄的品类「均价 / 爆款」
# 不该被主 loop 当基准线拿去背书。工具自己诚实，比在 prompt 里教「什么时候别信它」可靠。
THIN_DATA_CARD_MIN = env_int("INSIGHT_THIN_CARD_MIN", 3)
THIN_DATA_CONF_MIN = env_float("INSIGHT_THIN_CONF_MIN", 0.5)

# 品类洞察缓存（E 块）：品类知识弱时效、重复 / 近义查询多，缓存命中即省一整条 Hybrid + 精排。
# 只缓 category_insight，**绝不缓 item_search 的价格 / 库存**（强时效）。阈值 0.92：品类近义才命中、
# 不误把相邻品类当同一个。env 可调。
_cache: SemanticCache[CategoryInsightOutput] = SemanticCache(
    "category_insight",
    max_entries=env_int("CATEGORY_CACHE_MAX", 256),
    ttl=env_float("CATEGORY_CACHE_TTL", 3600.0),
    threshold=env_float("CATEGORY_CACHE_THRESHOLD", 0.92),
)


async def _encode_category(text: str) -> np.ndarray | None:
    """把品类文本编码成向量（复用召回层 embedding）供语义缓存匹配；失败返回 None（只走精确）。"""
    try:
        return await get_tower_client().encode_query(text)
    except Exception:
        return None  # 编码失败不影响主链路：跳过语义层，照常精确查 + 正常检索


class Bestseller(BaseModel):
    """品类里的代表性热卖款（从爆款卡的证据解析）。"""

    title: str
    price_usd: float | None = None
    rating: float | None = None
    why_popular: str = ""


class AttributeDist(BaseModel):
    """一个属性维度的分布（这里是评分档位占比，如 {"4.5★+": 0.58, ...}）。"""

    name: str
    distribution: dict[str, float]


class AttributeSchema(BaseModel):
    """品类的一个选购维度 + 其典型取值（来自 Shopify 属性骨架卡）。

    给 planner 把「不要塑料的」落到 ``Material`` 维度、给 item_picker 按「抗造」对应
    ``Features`` 取值打分——这是认知层的语义骨架（区别于 AttributeDist 的评分数值分布）。
    """

    name: str  # 维度名，如 "Bag/Case material" / "Watch movement"
    values: list[str]  # 该维度的典型取值，如 ["Canvas", "Leather", "Nylon", ...]


class PriceTier(BaseModel):
    """价格档位（从价格卡解析的分位切档）。"""

    tier: str  # budget / mid / premium
    low_usd: float
    high_usd: float


class CategoryInsightOutput(BaseModel):
    """category_insight 的结构化返回（原文证据已在工具内消化，不进此结构）。"""

    category: str
    # 实际定位到的知识库品类 + 定位置信度（0-1，按品类投票的得分占比）。matched_category
    # 为空 = 没定位到（后端挂了/库里真没有），主 loop 据此分辨「没匹配上」vs「品类没数据」，
    # 低置信时可转 web_search 兜底。
    matched_category: str
    resolution_confidence: float
    components: list[str]  # 这个品类的典型代表款/组成（爆款卡提炼）
    bestsellers: list[Bestseller]
    attributes: list[AttributeDist]  # quick 模式为空，deep 才填
    attribute_schema: list[AttributeSchema]  # 选购维度 + 取值（属性骨架卡，quick/deep 都填）
    price_tiers: list[PriceTier]
    card_count: int  # 精排后命中的卡片数
    confidence: float  # 命中卡片的平均置信度
    source: str  # opensearch / local_kb_fallback
    # 数据可信度自报：库内数据薄（卡片少 / 卡片自评置信低）时非空，明说参考价值有限、评价类
    # 结论建议 web_search 口碑佐证。空串 = 数据量正常。
    data_note: str = ""


# ---------------------------------------------------------------------------
# 两段式召回：品类定位 → 结构化取卡（歧义时 cross-encoder 只对品类消歧）
# ---------------------------------------------------------------------------
def _category_text(category: str, cards: list[CategoryCard]) -> str:
    """品类的消歧描述文本：品类名 + 别名（别名各卡共享，取第一张即可）。"""
    aliases = cards[0].aliases if cards else []
    return f"{category} {' '.join(aliases)}".strip()


async def _resolve_and_fetch(query: str) -> tuple[str, float, list[CategoryCard]]:
    """两段式：定位品类 → 取全该品类卡片。返回 ``(定位到的品类, 定位置信度, 卡片)``。

    定位歧义（top2/top1 置信度比 ≥ AMBIGUITY_RATIO）时，把两个候选品类的「品类名+别名」
    交给 cross-encoder 与 query 精排消歧——精排对象从 30 张长文本卡缩到 2 条品类描述，
    且只在歧义时才花这次调用。
    """
    kb = get_kb_client()
    cands = await kb.resolve_category(query, top_n=2)
    if not cands:
        return "", 0.0, []
    ambiguous = len(cands) >= 2 and cands[1][1] >= cands[0][1] * AMBIGUITY_RATIO
    if not ambiguous:
        cat, conf = cands[0]
        cards = await kb.fetch_cards(cat)
        if conf < RESOLVE_CONF_GATE and cards:
            scores, used_remote = await get_reranker().score_detailed(
                query, [_category_text(cat, cards)]
            )
            if used_remote and scores[0] < RERANK_RELEVANCE_MIN:
                return "", conf, []
        return cat, conf, cards
    # 歧义路径：两个候选的卡都取来（也顺便省一次二次 fetch），rerank 定胜者。
    # 胜者分数顺带过核验闸——库外品类常以「两个都不像」的形态落进歧义路径，零额外开销。
    card_sets = await asyncio.gather(*(kb.fetch_cards(cat) for cat, _ in cands))
    texts = [_category_text(cat, cs) for (cat, _), cs in zip(cands, card_sets, strict=True)]
    scores, used_remote = await get_reranker().score_detailed(query, texts)
    win = max(range(len(scores)), key=lambda i: scores[i])
    if used_remote and scores[win] < RERANK_RELEVANCE_MIN:
        return "", cands[win][1], []
    return cands[win][0], cands[win][1], card_sets[win]


async def _recall_cards(query: str, top_k: int) -> list[CategoryCard]:
    """评测口径的薄封装（scripts/eval 直接拿它打分，绕开整条工具）。"""
    _, _, cards = await _resolve_and_fetch(query)
    return cards[:top_k]


# ---------------------------------------------------------------------------
# 分组提炼：把多张卡片合并成结构化字段
# ---------------------------------------------------------------------------
def _group(cards: list[CategoryCard]) -> dict[str, list[CategoryCard]]:
    bag: dict[str, list[CategoryCard]] = {
        "bestseller": [],
        "attribute": [],
        "price_range": [],
        "attribute_schema": [],
    }
    for c in cards:
        bag.setdefault(c.card_type, []).append(c)
    return bag


# 爆款卡 summary 约定："{品类}: 款1 / 款2 / 款3"（款名之间用 " / " 分隔，见 aggregate.py）。
# 按 " / "（带空格）切、而非裸 "/"：真实标题常含裸斜杠（如 "9005/HB3 H11/H9/H8"、"PS4/Pro/Slim"），
# 裸 "/" 会把单个标题切成残片混进 components。带空格的 " / " 才是款名间的真分隔符。
def _extract_components(cards: list[CategoryCard]) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for c in cards:
        if ":" not in c.summary:
            continue
        tail = c.summary.split(":", 1)[1]
        for token in tail.split(" / "):
            name = token.strip().rstrip("…")
            if name and name not in seen:
                seen.add(name)
                found.append(name)
    return found


# 爆款卡 raw_evidence 约定："{title} | ${price} | {stars}★ | 月销 {n}"
_PRICE_RE = re.compile(r"\$([0-9]+(?:\.[0-9]+)?)")
_STAR_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)★")


def _extract_bestsellers(cards: list[CategoryCard], limit: int) -> list[Bestseller]:
    out: list[Bestseller] = []
    for c in cards:
        for line in c.raw_evidence:
            parts = [p.strip() for p in line.split("|")]
            if not parts or not parts[0]:
                continue
            price_m = _PRICE_RE.search(line)
            star_m = _STAR_RE.search(line)
            out.append(
                Bestseller(
                    title=parts[0].rstrip("…"),
                    price_usd=float(price_m.group(1)) if price_m else None,
                    rating=float(star_m.group(1)) if star_m else None,
                    why_popular=parts[-1] if len(parts) > 1 else "",
                )
            )
            if len(out) >= limit:
                return out
    return out


# 属性卡 summary 约定："评分分布：4.5★+ 58% / 4.0–4.5★ 23% / ..."
_PCT_RE = re.compile(r"([^/:：]+?)\s*([0-9]+)%")


def _extract_attributes(cards: list[CategoryCard]) -> list[AttributeDist]:
    out: list[AttributeDist] = []
    for c in cards:
        name = c.summary.split("：", 1)[0].split(":", 1)[0].strip() or "属性"
        dist = {
            label.strip(): round(int(pct) / 100, 4) for label, pct in _PCT_RE.findall(c.summary)
        }
        if dist:
            out.append(AttributeDist(name=name, distribution=dist))
    return out


# 价格卡 summary 约定："budget $lo–$p33 / mid $p33–$p66 / premium $p66–$hi"
_TIER_RE = re.compile(r"(budget|mid|premium)\s*\$([0-9.]+)[–\-]\$([0-9.]+)")


def _extract_price_tiers(cards: list[CategoryCard]) -> list[PriceTier]:
    for c in cards:
        tiers = [
            PriceTier(tier=t, low_usd=float(lo), high_usd=float(hi))
            for t, lo, hi in _TIER_RE.findall(c.summary)
        ]
        if tiers:
            return tiers  # 价格卡通常一张就够，命中即返
    return []


# 属性骨架卡 raw_evidence 约定（半角冒号+空格）："{维度名}: {取值1}, {取值2}, ..."
# provenance 行「映射自 Shopify 品类：…」用全角冒号，天然不被这条匹配，自动跳过。
_ATTR_SCHEMA_RE = re.compile(r"^(.+?): (.+)$")


def _extract_attribute_schema(cards: list[CategoryCard]) -> list[AttributeSchema]:
    out: list[AttributeSchema] = []
    seen: set[str] = set()
    for c in cards:
        for line in c.raw_evidence:
            m = _ATTR_SCHEMA_RE.match(line.strip())
            if not m:
                continue
            name = m.group(1).strip()
            if name in seen:
                continue
            values = [v.strip() for v in m.group(2).split(",") if v.strip()]
            if values:
                seen.add(name)
                out.append(AttributeSchema(name=name, values=values))
    return out


def _thin_data_note(card_count: int, confidence: float) -> str:
    """薄数据自报文案：卡片数或平均置信度低于线时非空。空卡不报（另有「未收录」硬路径）。"""
    if card_count == 0 or (card_count >= THIN_DATA_CARD_MIN and confidence >= THIN_DATA_CONF_MIN):
        return ""
    return (
        f"库内该品类数据薄（卡片 {card_count} 张、平均置信 {confidence:.0%}），"
        "此基准参考价值有限；评价类结论建议以 web_search 口碑佐证，别只拿库内大盘背书。"
    )


def _insight_result(out: CategoryInsightOutput) -> str:
    """把品类洞察的关键发现拼成人读摘要，供前端「思考过程」展开看这一步**查到了什么**——
    而非 ``card_count=8`` 这类元信息。取爆款款名、价位档、Top 选购维度及其典型取值。"""
    lines: list[str] = []
    if not out.matched_category and out.card_count == 0:
        return "知识库未收录该品类（定位相关性过低），建议转 web_search 补充行情"
    if out.data_note:
        lines.append(f"⚠ {out.data_note}")
    if out.matched_category and out.matched_category != out.category:
        lines.append(f"定位品类：{out.matched_category}（置信 {out.resolution_confidence:.0%}）")
    names = "、".join(b.title for b in out.bestsellers[:5] if b.title)
    if names:
        lines.append(f"爆款：{names}")
    if out.price_tiers:
        tiers = "、".join(f"{t.tier} ${t.low_usd:.0f}–${t.high_usd:.0f}" for t in out.price_tiers)
        lines.append(f"价位档：{tiers}")
    for sch in out.attribute_schema[:3]:
        vals = "、".join(sch.values[:6])
        if vals:
            lines.append(f"{sch.name}：{vals}")
    return "\n".join(lines)


@tool
async def category_insight(
    category: str, depth: Literal["quick", "deep"] = "quick"
) -> CategoryInsightOutput:
    """查一个品类的典型爆款 / 评分大盘 / 价格档位 / 选购维度（RAG 品类知识库）。

    返回里的 ``attribute_schema`` 给出这个品类「该看哪些维度、每维有哪些取值」（材质 / 特性 /
    闭合方式…），用来把用户的软硬约束（如「不要塑料的」「要抗造」）落到具体属性上。

    何时调用：检索或精挑前想先了解某品类的行情、典型属性与选购维度时。
    返回的 ``matched_category`` / ``resolution_confidence`` 标明实际定位到的品类与置信度；
    matched_category 为空或置信度很低说明知识库没接住这个品类，可转 web_search 补充。
    ``data_note`` 非空 = 库内该品类数据薄，返回的大盘参考价值有限——评价类结论请以
    web_search 口碑佐证，不要只拿它背书。
    参数：
      - category：品类名 / 关键词（如「luggage」「running shoes」「旅行收纳」）。
      - depth：quick（默认，不算属性分布）/ deep（额外算评分分布，更适合 fork）。
    """
    category = normalize_category(category)
    await monitor.report_tool_start("category_insight", category=category, depth=depth)

    # 两级缓存（E 块）：精确（免编码）→ 语义（近义命中）→ 都未命中才真算。depth 作维度隔离
    # （quick/deep 结果不同，不互相命中）。命中即省一整条 Hybrid 召回 + cross-encoder 精排。
    exact_key = f"{depth}:{category}"
    hit = _cache.get_exact(exact_key)
    if hit is not None:
        metrics.record_cache("category_insight", "exact_hit")
        await monitor.report_tool_end(
            "category_insight",
            card_count=hit.card_count,
            source=hit.source,
            cached=True,
            result=_insight_result(hit),
        )
        return hit
    qvec = await _encode_category(category)
    if qvec is not None:
        sem = _cache.get_semantic(qvec, group=depth)
        if sem is not None:
            metrics.record_cache("category_insight", "semantic_hit")
            _cache.put(exact_key, sem, qvec, group=depth)  # 登记精确键：下次同词走 O(1) 快路径
            await monitor.report_tool_end(
                "category_insight",
                card_count=sem.card_count,
                source=sem.source,
                cached=True,
                result=_insight_result(sem),
            )
            return sem
    metrics.record_cache("category_insight", "miss")

    matched, res_conf, cards = await _resolve_and_fetch(category)
    grouped = _group(cards)

    components = _extract_components(grouped["bestseller"])
    bestsellers = _extract_bestsellers(grouped["bestseller"], limit=5)
    price_tiers = _extract_price_tiers(grouped["price_range"])
    # 取卡已按品类精确圈定，骨架卡天然单品类；fetch 按 confidence 降序，取首卡即最可信一张。
    attribute_schema = _extract_attribute_schema(grouped["attribute_schema"][:1])
    attributes = _extract_attributes(grouped["attribute"]) if depth == "deep" else []

    confidence = round(sum(c.confidence for c in cards) / len(cards), 3) if cards else 0.0
    source = "opensearch" if get_kb_client().remote else "local_kb_fallback"
    out = CategoryInsightOutput(
        category=category,
        matched_category=matched,
        resolution_confidence=res_conf,
        components=components,
        bestsellers=bestsellers,
        attributes=attributes,
        attribute_schema=attribute_schema,
        price_tiers=price_tiers,
        card_count=len(cards),
        confidence=confidence,
        source=source,
        data_note=_thin_data_note(len(cards), confidence),
    )
    # 写回缓存：精确层总写（编码不可用时相同 query 仍走快路径）；语义层在编码可用时一并写。
    _cache.put(exact_key, out, qvec, group=depth)
    await monitor.report_tool_end(
        "category_insight",
        card_count=out.card_count,
        source=out.source,
        result=_insight_result(out),
    )
    return out
