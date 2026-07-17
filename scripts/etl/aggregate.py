"""ETL Step 1+2：从真实 Amazon 商品数据按品类聚合，抽出三类卡片草稿。

refdocs 13-1 §2 的「标准化 → 小模型抽取」两步，在本项目落成**确定性规则聚合**——
我们没有 GPU 也不需要让 LLM 现场编造品类常识，真实商品数据本身就能算出可信的结论：

- ``bestseller``  按 ``boughtInLastMonth`` 月销取头部款 → 代表性热卖清单（整类月销全 0 则不出此卡）
- ``attribute``   按 ``stars`` 算评分档位分布 → 这个品类的品质大盘长什么样
- ``price_range`` 按 ``price`` 分位数切 budget/mid/premium 三档（外界用 p5–p95 裁掉离群尾）

数据源：``data/rag/amazon_products.csv``（asin/title/stars/price/category_id/isBestSeller/
boughtInLastMonth）+ ``amazon_categories.csv``（category_id → 品类名）。品类名经
:func:`~app.recall.category_kb.normalize_category` 归一后再聚合（不同 id 归到同名品类会合并）。

summary 写法**有格式约定**（入库门禁与下游规则提炼都靠它）：
- bestseller： ``"{品类}: {款1} / {款2} / {款3}"``
- attribute： ``"评分分布：4.5★+ a% / 4.0–4.5★ b% / 3.0–4.0★ c% / <3.0★ d%"``
- price_range：``"budget $lo–$p33 / mid $p33–$p66 / premium $p66–$hi"``
"""

from __future__ import annotations

import csv
import heapq
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from app.recall.category_kb import normalize_category

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAG_DIR = PROJECT_ROOT / "data" / "rag"
PRODUCTS_CSV = RAG_DIR / "amazon_products.csv"
CATEGORIES_CSV = RAG_DIR / "amazon_categories.csv"

csv.field_size_limit(10 * 1024 * 1024)

# 品类样本太少算不出靠谱分位/分布，跳过（冷启动品类由在线 WebSearch 兜底，不进库）。
MIN_SAMPLES = 30
# 分级质量门：不同卡对样本量的敏感度不同——p33/p66 分位切档用 30 个价样会噪到失去锚点
# 意义，评分分布次之，爆款卡只看头部无所谓。达不到线的品类**只是不出那张卡**，不整类剔除。
MIN_PRICE_SAMPLES = 100  # price_range 卡：有价样本下限
MIN_RATED_SAMPLES = 50  # attribute（评分分布）卡：有评分样本下限
# 单张卡 summary 里塞的头部款数 / 证据数。
TOP_N = 3
EVIDENCE_N = 5
# 标题在 summary 里截断长度（避免 summary 过长撑爆门禁的长度门）。
TITLE_CLIP = 48


@dataclass
class _CatAgg:
    """单个归一品类的流式聚合器（边读边累计，避免把 1.4M 行全驻留）。"""

    category: str
    count: int = 0
    prices: list[float] = field(default_factory=list)
    # 评分档位计数：[4.5+, 4.0–4.5, 3.0–4.0, <3.0]
    rating_buckets: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    # 头部款最小堆：(bought, seq, title, price, stars)，按月销只留 EVIDENCE_N 个。
    _top: list[tuple] = field(default_factory=list)
    _seq: int = 0

    def add(self, title: str, stars: float | None, price: float | None, bought: int) -> None:
        self.count += 1
        if price is not None and price > 0:
            self.prices.append(price)
        if stars is not None:
            if stars >= 4.5:
                self.rating_buckets[0] += 1
            elif stars >= 4.0:
                self.rating_buckets[1] += 1
            elif stars >= 3.0:
                self.rating_buckets[2] += 1
            else:
                self.rating_buckets[3] += 1
        # seq 做 tie-breaker，避免 bought 相等时去比不可比的 tuple 尾部。
        item = (bought, self._seq, title, price, stars)
        self._seq += 1
        if len(self._top) < EVIDENCE_N:
            heapq.heappush(self._top, item)
        elif item > self._top[0]:
            heapq.heapreplace(self._top, item)

    def top_products(self) -> list[tuple]:
        """头部款按月销降序。"""
        return sorted(self._top, key=lambda x: x[0], reverse=True)


def _load_category_map() -> dict[str, str]:
    """category_id → 品类名。"""
    mapping: dict[str, str] = {}
    with CATEGORIES_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mapping[row["id"].strip()] = row["category_name"].strip()
    return mapping


def _parse_float(raw: str) -> float | None:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _parse_int(raw: str) -> int:
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return 0


def _quantile(sorted_vals: list[float], q: float) -> float:
    """线性插值分位数（避免引 numpy，ETL 脚本零额外依赖）。"""
    if not sorted_vals:
        return 0.0
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def aggregate() -> dict[str, _CatAgg]:
    """流式扫商品表，按归一品类聚合。"""
    id_to_name = _load_category_map()
    aggs: dict[str, _CatAgg] = {}
    with PRODUCTS_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = id_to_name.get((row.get("category_id") or "").strip())
            if not name:
                continue
            cat = normalize_category(name)
            agg = aggs.get(cat)
            if agg is None:
                agg = aggs[cat] = _CatAgg(category=cat)
            agg.add(
                title=(row.get("title") or "").strip(),
                stars=_parse_float(row.get("stars", "")),
                price=_parse_float(row.get("price", "")),
                bought=_parse_int(row.get("boughtInLastMonth", "")),
            )
    return aggs


def _clip(title: str) -> str:
    title = title.strip()
    return title if len(title) <= TITLE_CLIP else title[:TITLE_CLIP].rstrip() + "…"


def _slug(category: str) -> str:
    return "_".join(category.lower().split())[:48]


def cards_for(agg: _CatAgg) -> list[dict]:
    """把一个品类的聚合结果转成三张卡片草稿（dict，待 admit 校验 + 编码向量）。"""
    if agg.count < MIN_SAMPLES:
        return []
    now = datetime.now(UTC).isoformat(timespec="seconds")
    slug = _slug(agg.category)
    tops = agg.top_products()
    # 样本越足越可信，封顶 0.95（诚实地不给满分——毕竟只是单数据源聚合）。
    conf = round(min(0.95, 0.5 + agg.count / 4000), 2)
    cards: list[dict] = []

    # ---- bestseller ----
    # tops 按月销降序，tops[0][0] 即品类内最高月销。整类月销全 0（如 gift cards / computer
    # servers 这类无销量信号的品类）时，「热卖」无从谈起——不出爆款卡，避免拿插入序冒充榜单
    # 误导下游 components/why_popular（attribute / price_range 卡仍照常出）。
    if tops and tops[0][0] > 0:
        names = " / ".join(_clip(t[2]) for t in tops[:TOP_N])
        # 头部款按月销选，价格可能缺失（_parse_float 返回 None）——格式化前先兜底成 "—"，
        # 否则 f"${None:.2f}" 会 TypeError 中断整条建库。
        evidence = [
            f"{_clip(t[2])} | {f'${t[3]:.2f}' if t[3] is not None else '$—'} | "
            f"{t[4] or '—'}★ | 月销 {t[0]}"
            for t in tops[:EVIDENCE_N]
        ]
        cards.append(
            {
                "card_id": f"{slug}_bestseller",
                "category": agg.category,
                "card_type": "bestseller",
                "summary": f"{agg.category}: {names}",
                "raw_evidence": evidence,
                "last_updated": now,
                "confidence": conf,
            }
        )

    # ---- attribute（评分分布）----
    rated = sum(agg.rating_buckets)
    if rated >= MIN_RATED_SAMPLES:
        pct = [round(100 * b / rated) for b in agg.rating_buckets]
        cards.append(
            {
                "card_id": f"{slug}_attribute",
                "category": agg.category,
                "card_type": "attribute",
                "summary": (
                    f"评分分布：4.5★+ {pct[0]}% / 4.0–4.5★ {pct[1]}% / "
                    f"3.0–4.0★ {pct[2]}% / <3.0★ {pct[3]}%"
                ),
                "raw_evidence": [f"{rated} 件有评分样本，月销头部 {len(tops)} 款"],
                "last_updated": now,
                "confidence": conf,
            }
        )

    # ---- price_range（分位切档）----
    if len(agg.prices) >= MIN_PRICE_SAMPLES:
        sp = sorted(agg.prices)
        # budget 下界 / premium 上界用 p5–p95 而非原始 min/max：单个错类目/奢侈离群单品
        # （如手工品类里的 $19400）会把档位上界拉爆几个数量级，让 budget/mid/premium 失真。
        # 裁掉两端 5% 长尾后，三档对「这个品类正常多少钱」才有锚点意义；原始极值留进 evidence。
        lo, hi = _quantile(sp, 0.05), _quantile(sp, 0.95)
        p33, p66 = _quantile(sp, 1 / 3), _quantile(sp, 2 / 3)
        cards.append(
            {
                "card_id": f"{slug}_price_range",
                "category": agg.category,
                "card_type": "price_range",
                "summary": (
                    f"budget ${lo:.2f}–${p33:.2f} / mid ${p33:.2f}–${p66:.2f} / "
                    f"premium ${p66:.2f}–${hi:.2f}"
                ),
                "raw_evidence": [
                    f"{len(sp)} 件有价样本，中位 ${_quantile(sp, 0.5):.2f}，"
                    f"实际区间 ${sp[0]:.2f}–${sp[-1]:.2f}"
                ],
                "last_updated": now,
                "confidence": conf,
            }
        )
    return cards
