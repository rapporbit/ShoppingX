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
