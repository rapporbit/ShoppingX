"""召回指标单测：Recall@K / MRR / NDCG 的确定性断言（不依赖外部服务）。"""

from __future__ import annotations

from app.eval.recall_metrics import aggregate, mrr, ndcg_at_k, recall_at_k


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
