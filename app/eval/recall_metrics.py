"""召回评测三指标：Recall@K / MRR / NDCG@K（refdocs 13-1 §5.3）。

这是 CategoryInsight 的「模块级日常体检」——区别于第 8 章端到端的 Rubric 评测（慢、贵、要
judge LLM），召回评测快、便宜、纯结构化指标：改一行召回代码、调一次权重、换一版 reranker，
都能立刻量化「整体是变好还是变差」，避免 case-by-case 看着好、整体悄悄退化。

三者各管一件事：
- ``recall_at_k``：标注的相关卡片，Top-K 里找回了多少（任何召回环节的「底线」）。
- ``mrr``：首条命中排在第几（Top-1 直接喂下游时最关心）。
- ``ndcg_at_k``：排序质量——不只看命中，还看高价值卡片是否靠前。
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Top-K 召回覆盖了多少标注（命中标注数 / 标注总数）。"""
    rel = set(relevant)
    if not rel:
        return 0.0
    hit = set(retrieved[:k]) & rel
    return len(hit) / len(rel)


def mrr(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    """首条相关卡片的倒数排名（第 1 位命中=1.0，第 2 位=0.5，未命中=0）。"""
    rel = set(relevant)
    for i, rid in enumerate(retrieved, start=1):
        if rid in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """NDCG@K：标注按序赋递减增益，DCG/IDCG 归一到 [0,1]，同时看「命中 + 靠前」。"""
    # 标注列表本身按重要性排序：越靠前 gain 越大。
    rel_gain = {rid: len(relevant) - i for i, rid in enumerate(relevant)}
    dcg = sum(rel_gain.get(rid, 0) / math.log2(i + 2) for i, rid in enumerate(retrieved[:k]))
    ideal = sum(rel_gain[rid] / math.log2(i + 2) for i, rid in enumerate(list(relevant)[:k]))
    return dcg / ideal if ideal else 0.0


def aggregate(rows: list[dict], k: int) -> dict[str, float]:
    """对一批 ``{"retrieved": [...], "relevant": [...]}`` 求三指标均值。"""
    if not rows:
        return {f"recall@{k}": 0.0, "mrr": 0.0, f"ndcg@{k}": 0.0}
    n = len(rows)
    return {
        f"recall@{k}": sum(recall_at_k(r["retrieved"], r["relevant"], k) for r in rows) / n,
        "mrr": sum(mrr(r["retrieved"], r["relevant"]) for r in rows) / n,
        f"ndcg@{k}": sum(ndcg_at_k(r["retrieved"], r["relevant"], k) for r in rows) / n,
    }


# ────────────────────────── 分级口径（商品召回 / ESCI 标注）──────────────────────────
# 上面那套是**二值**标注（品类金标只有「相关 / 不相关」）。商品侧的 golden 来自 ESCI，标注天然
# 分三档：Exact（正例）/ Substitute（可替代）/ Complement（配件）。二值 NDCG 会把「召回了同款
# 替代品」和「召回了一堆配件」当成一回事——而「搜手机出配件」恰恰是本仓的历史 bad case，尺子
# 必须能分开它俩。档位取 ESCI 论文的标准 gain（M21 基线 `scripts/train/eval_recall.py` 同款，
# 换 K 不换口径，两份报告可直接对照）。
GRADED_GAIN = {"positive": 3.0, "substitute": 2.0, "complement": 1.0}


def graded_metrics(
    retrieved: Sequence[str],
    positives: Sequence[str],
    substitutes: Sequence[str] = (),
    complements: Sequence[str] = (),
    k: int = 20,
) -> dict[str, float]:
    """单条 query 的分级三指标 + 配件专项。

    - ``recall@k``：**分母只取正例**，且是「该 query 在我们库内的正例数」（qrels 构建时已按库
      内商品过滤）。用全部 ESCI 正例当分母会系统性低估，那个数没有可比性。
    - ``mrr``：首条**正例**的倒数排名——可替代品排第一不算命中，下游直接喂用户的是 Top-1。
    - ``ndcg@k``：分级 gain 的 DCG / 理想 DCG（理想序 = 库内已知相关按 gain 降序）。
    - ``complement_hits@k``：Top-K 里标注为配件的条数，**越低越好**（「搜整机出配件」的直接度量）。
      灵敏度有限（库里绝大多数配件没被 ESCI 标过），当参考不当结论。
    """
    pos, sub, comp = set(positives), set(substitutes), set(complements)
    top_k = list(retrieved[:k])

    def gain_of(item_id: str) -> float:
        if item_id in pos:
            return GRADED_GAIN["positive"]
        if item_id in sub:
            return GRADED_GAIN["substitute"]
        if item_id in comp:
            return GRADED_GAIN["complement"]
        return 0.0

    rr = 0.0
    for i, item_id in enumerate(top_k, start=1):
        if item_id in pos:
            rr = 1.0 / i
            break

    ideal_gains = sorted(
        [GRADED_GAIN["positive"]] * len(pos)
        + [GRADED_GAIN["substitute"]] * len(sub)
        + [GRADED_GAIN["complement"]] * len(comp),
        reverse=True,
    )[:k]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gain_of(x) for x in top_k))
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal_gains))

    return {
        f"recall@{k}": len(pos & set(top_k)) / len(pos) if pos else 0.0,
        "mrr": rr,
        f"ndcg@{k}": dcg / idcg if idcg else 0.0,
        f"complement_hits@{k}": float(len(comp & set(top_k))),
        "has_complement": 1.0 if comp else 0.0,
    }


def aggregate_graded(rows: list[dict], k: int = 20) -> dict[str, float]:
    """对一批 ``{"retrieved", "positives", "substitutes", "complements"}`` 求分级指标均值。

    ``complement_hits@k`` 的分母**只算带配件标注的 query**（没标配件的 query 恒为 0，混进去只会
    把这个数稀释成噪声）。
    """
    keys = [f"recall@{k}", "mrr", f"ndcg@{k}"]
    if not rows:
        empty = dict.fromkeys([*keys, f"complement_hits@{k}"], 0.0)
        return empty | {"queries_with_complement": 0.0}

    n = len(rows)
    per_row = [
        graded_metrics(
            r["retrieved"],
            r["positives"],
            r.get("substitutes") or (),
            r.get("complements") or (),
            k,
        )
        for r in rows
    ]
    n_comp = sum(m["has_complement"] for m in per_row)
    out = {key: sum(m[key] for m in per_row) / n for key in keys}
    out[f"complement_hits@{k}"] = (
        sum(m[f"complement_hits@{k}"] for m in per_row) / n_comp if n_comp else 0.0
    )
    out["queries_with_complement"] = n_comp
    return out
