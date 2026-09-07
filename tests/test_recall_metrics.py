"""召回指标单测：Recall@K / MRR / NDCG 的确定性断言（不依赖外部服务）。"""

from __future__ import annotations

from app.eval.recall_metrics import (
    aggregate,
    aggregate_graded,
    graded_metrics,
    mrr,
    ndcg_at_k,
    recall_at_k,
)


def test_recall_at_k() -> None:
    # 3 个标注里 Top-3 命中 2 个 → 2/3。
    assert recall_at_k(["a", "x", "b", "c"], ["a", "b", "z"], k=3) == 2 / 3
    # 标注全在 Top-K → 1.0。
    assert recall_at_k(["a", "b"], ["a", "b"], k=5) == 1.0
    # 空标注 → 0（不除零）。
    assert recall_at_k(["a"], [], k=5) == 0.0


def test_mrr() -> None:
    assert mrr(["a", "b"], ["a"]) == 1.0  # 首位命中
    assert mrr(["x", "a"], ["a"]) == 0.5  # 第 2 位
    assert mrr(["x", "y"], ["a"]) == 0.0  # 未命中


def test_ndcg_ordering() -> None:
    relevant = ["a", "b", "c"]  # 重要性 a > b > c
    # 完美顺序 NDCG=1。
    assert ndcg_at_k(["a", "b", "c"], relevant, k=3) == 1.0
    # 把最不重要的排首位，分数下降。
    worse = ndcg_at_k(["c", "b", "a"], relevant, k=3)
    assert 0.0 < worse < 1.0
    # 完全没命中 → 0。
    assert ndcg_at_k(["x", "y"], relevant, k=3) == 0.0


def test_aggregate_means() -> None:
    rows = [
        {"retrieved": ["a", "b"], "relevant": ["a", "b"]},  # recall 1.0
        {"retrieved": ["x", "a"], "relevant": ["a"]},  # recall 1.0, mrr .5
    ]
    out = aggregate(rows, k=2)
    assert out["recall@2"] == 1.0
    assert out["mrr"] == 0.75
    assert "ndcg@2" in out


def test_aggregate_empty() -> None:
    out = aggregate([], k=10)
    assert out == {"recall@10": 0.0, "mrr": 0.0, "ndcg@10": 0.0}


# ---------- 分级口径（商品召回 / ESCI 三档标注）----------
def test_graded_recall_denominator_is_positives_only() -> None:
    """recall 的分母只数正例——把替代品/配件算进去会把「召回对了」稀释成「召回了一半」。"""
    m = graded_metrics(["p1", "s1", "c1"], ["p1", "p2"], ["s1"], ["c1"], k=3)
    assert m["recall@3"] == 0.5  # 2 个正例里命中 1 个，替代/配件不进分母


def test_graded_mrr_ignores_substitutes() -> None:
    """首位是可替代品不算命中：Top-1 是直接端给用户的那件。"""
    assert graded_metrics(["s1", "p1"], ["p1"], ["s1"], k=5)["mrr"] == 0.5


def test_graded_ndcg_separates_substitute_from_complement() -> None:
    """同样是「没召回正例」，召回替代品必须比召回配件分高——二值 NDCG 分不出这两者。"""
    sub_only = graded_metrics(["s1", "s2"], ["p1"], ["s1", "s2"], [], k=2)
    comp_only = graded_metrics(["c1", "c2"], ["p1"], [], ["c1", "c2"], k=2)
    assert sub_only["ndcg@2"] > comp_only["ndcg@2"] > 0.0
    assert sub_only["recall@2"] == comp_only["recall@2"] == 0.0  # recall 上二者无差别


def test_graded_ndcg_perfect_order_is_one() -> None:
    m = graded_metrics(["p1", "p2", "s1", "c1"], ["p1", "p2"], ["s1"], ["c1"], k=4)
    assert m["ndcg@4"] == 1.0
    assert m["recall@4"] == 1.0


def test_graded_respects_k_cutoff() -> None:
    """第 3 位的正例在 K=2 时不算命中——门禁量的是线上真吃进下游的那几条。"""
    assert graded_metrics(["x", "y", "p1"], ["p1"], k=2)["recall@2"] == 0.0
    assert graded_metrics(["x", "y", "p1"], ["p1"], k=3)["recall@3"] == 1.0


def test_aggregate_graded_complement_denominator_is_labeled_queries() -> None:
    """配件专项只在「标了配件的 query」上取均值；没标配件的恒 0，混进去只会把它稀释成噪声。"""
    rows = [
        {"retrieved": ["c1", "c2"], "positives": ["p1"], "complements": ["c1", "c2"]},
        {"retrieved": ["p1"], "positives": ["p1"]},  # 无配件标注，不该进分母
    ]
    out = aggregate_graded(rows, k=2)
    assert out["queries_with_complement"] == 1.0
    assert out["complement_hits@2"] == 2.0  # 2 条 / 1 条 query，而不是 2/2
    assert out["recall@2"] == 0.5  # 一条 0.0 一条 1.0


def test_aggregate_graded_empty() -> None:
    out = aggregate_graded([], k=20)
    assert out["recall@20"] == 0.0
    assert out["queries_with_complement"] == 0.0
