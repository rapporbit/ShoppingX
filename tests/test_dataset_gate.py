"""评测数据门禁的单测：`run_product_recall` 的阈值判定 + `validate_datasets` 的各道检查。

这两个脚本本身就是「拦回退」的闸，所以它们自己更要被拦住——**闸门失灵是静默的**：阈值判反了
只会永远放行，数据集校验漏了一类问题只会永远说「无问题」。这里全部走纯函数，不碰 Qdrant / 网络。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.eval.run_product_recall import any_floor, check_gate, load_qrels
from scripts.eval.validate_datasets import (
    check_category,
    check_qrels,
    check_seeds,
    check_seeds_in_sync,
)

METRICS = {"recall@20": 0.25, "mrr": 0.18, "ndcg@20": 0.15}


def _args(**kw: float | None) -> argparse.Namespace:
    base = {"min_recall": None, "min_mrr": None, "min_ndcg": None}
    return argparse.Namespace(**{**base, **kw})


# ---------- 召回门禁的阈值判定 ----------
def test_gate_passes_when_above_floor() -> None:
    assert check_gate(METRICS, 20, _args(min_recall=0.22, min_mrr=0.15, min_ndcg=0.13)) == []


def test_gate_reports_every_breached_metric() -> None:
    """三条阈值各判各的——只报第一条会让人修完一处又撞第二处。"""
    failed = check_gate(METRICS, 20, _args(min_recall=0.30, min_mrr=0.15, min_ndcg=0.20))
    assert len(failed) == 2
    assert "recall@20" in failed[0] and "ndcg@20" in failed[1]


def test_gate_without_floors_never_blocks() -> None:
    """不传阈值 = 只出表。默认必须是「不拦」，否则谁跑一次报表都被拦一次。"""
    assert check_gate({"recall@20": 0.0, "mrr": 0.0, "ndcg@20": 0.0}, 20, _args()) == []
    assert any_floor(_args()) is False
    assert any_floor(_args(min_mrr=0.1)) is True


def test_gate_boundary_is_not_a_failure() -> None:
    """刚好等于阈值算过——阈值是「不得低于」，卡在边界上反复红是噪声。"""
    metrics = {"recall@20": 0.22, "mrr": 0.5, "ndcg@20": 0.5}
    assert check_gate(metrics, 20, _args(min_recall=0.22)) == []


def test_load_qrels_drops_rows_without_positives(tmp_path: Path) -> None:
    """无正例的行量不出 recall，进了分母只会把指标平白拉低。"""
    path = tmp_path / "q.jsonl"
    path.write_text(
        json.dumps({"query_id": 1, "query": "a", "positives": ["p1"]})
        + "\n"
        + json.dumps({"query_id": 2, "query": "b", "positives": []})
        + "\n",
        encoding="utf-8",
    )
    assert [r["query_id"] for r in load_qrels(path, 0)] == [1]


def test_load_qrels_missing_file_exits(tmp_path: Path) -> None:
    """golden 在 gitignore 的 data/ 下，缺了要给出重建命令而不是抛 FileNotFoundError。"""
    try:
        load_qrels(tmp_path / "nope.jsonl", 0)
    except SystemExit as exc:
        assert "build_esci_pairs" in str(exc)
    else:
        raise AssertionError("缺 golden 时应当 SystemExit")


# ---------- 数据集体检 ----------
def test_qrels_flags_duplicate_and_overlap() -> None:
    rows = [
        {"query_id": 1, "query": "a", "positives": ["p"], "complements": ["p"]},
        {"query_id": 1, "query": "b", "positives": ["x"]},
    ]
    errs = "\n".join(check_qrels(rows))
    assert "query_id 重复" in errs
    assert "正例/配件 标注重叠" in errs  # 同一商品两档 → gain 取决于判定顺序，指标不可复现


def test_qrels_flags_empty_query_and_positives() -> None:
    errs = "\n".join(check_qrels([{"query_id": 7, "query": "  ", "positives": []}]))
    assert "query 为空" in errs and "positives 为空" in errs


def test_qrels_accepts_clean_rows() -> None:
    rows = [{"query_id": 1, "query": "a", "positives": ["p"], "substitutes": ["s"]}]
    assert check_qrels(rows) == []


def test_seeds_flags_turns_query_mismatch() -> None:
    """多轮 case 只对最后一轮打分：query 与 turns[-1] 不一致 = 拿 A 的答案对着 B 的尺子打。"""
    rows = [
        {
            "id": "x1",
            "bucket": "b",
            "intent": "shopping",
            "query": "第一轮",
            "turns": ["第一轮", "第二轮"],
        }
    ]
    assert any("turns[-1] 不一致" in e for e in check_seeds(rows))


def test_seeds_flags_missing_field_and_dup_id_and_blank_prior() -> None:
    rows = [
        {"id": "x1", "bucket": "b", "intent": "shopping", "query": "q", "prior_context": "  "},
        {"id": "x1", "bucket": "", "intent": "shopping", "query": "q2"},
    ]
    errs = "\n".join(check_seeds(rows))
    assert "id 重复" in errs and "缺字段 bucket" in errs and "prior_context 是空串" in errs


def test_seeds_in_sync_catches_jsonl_only_edits() -> None:
    """只改产物不改脚本 = 下次重建就被冲掉（批 1 踩过，三条交易 case 差点丢）。"""
    from scripts.eval.build_eval_queries import QUERIES

    rows = [{"id": q["id"], "query": q["query"]} for q in QUERIES]
    assert check_seeds_in_sync(rows) == []

    rows.append({"id": "手工加的", "query": "只写进了 jsonl"})
    errs = "\n".join(check_seeds_in_sync(rows))
    assert "jsonl 里有、脚本里没有" in errs

    errs = "\n".join(check_seeds_in_sync(rows[:-2]))
    assert "脚本里有、jsonl 里没有" in errs


def test_seeds_in_sync_catches_query_drift() -> None:
    from scripts.eval.build_eval_queries import QUERIES

    rows = [{"id": q["id"], "query": q["query"]} for q in QUERIES]
    rows[0]["query"] = rows[0]["query"] + "（悄悄改了一个字）"
    assert any("query 与脚本不一致" in e for e in check_seeds_in_sync(rows))


def test_category_flags_empty_relevant() -> None:
    errs = "\n".join(check_category([{"query": "背包", "relevant": []}]))
    assert "relevant 为空" in errs
    assert check_category([{"query": "背包", "relevant": ["c1"]}]) == []
