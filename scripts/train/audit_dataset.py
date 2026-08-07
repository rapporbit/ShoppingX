"""训练/评测数据的可用性与可靠性审计——开训之前必须先信得过数据。

分两类查：

**可用性**（能不能拿来训）：结构合规、有无空值、文本长度是否在模型上下文内、query 分布是否
和线上一致。

**可靠性**（信号真不真）：三个致命项——
1. ``train`` / ``eval`` 的 query 泄漏（官方按 query 切 split，但必须自己验一遍）；
2. **同一样本内 pos 与 neg 文本撞车**。ESCI 标注是对 ASIN 标的，而我们库里存在同款不同 ASIN
   （颜色/尺码变体），它们的 ``embed_text`` 可能逐字相同 —— 一个被标 E、一个被标 S，就会出现
   「同一段文本既是正例又是负例」，这是直接互相抵消的矛盾信号，比噪声更毒；
3. 头部集中度与长尾覆盖（refdocs 04-2 §5.5.3：长尾品类每桶 <50 条就训不动）。

用法：``uv run --group train python scripts/train/audit_dataset.py``
"""

from __future__ import annotations

import json
import statistics as st
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = PROJECT_ROOT / "data" / "train" / "esci_train.jsonl"
QRELS_PATH = PROJECT_ROOT / "data" / "train" / "esci_eval_qrels.jsonl"
OUT_PATH = PROJECT_ROOT / "data" / "train" / "audit_report.json"

# BGE-M3 上下文 8192 token；英文按 ~4 字符/token 粗估，留足余量的告警线
LONG_TEXT_CHARS = 2000


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def category_of(text: str) -> str:
    """从 embed_text 里取类目段（含 ' > ' 的那段；字段可能因缺失被跳过，故按内容识别）。"""
    for seg in text.split(" | "):
        if " > " in seg:
            return seg.split(" > ")[-1].strip()
    return "(none)"


def audit_usability(train: list[dict], qrels: list[dict]) -> dict:
    empty_q = sum(1 for r in train if not r["query"].strip())
    no_pos = sum(1 for r in train if not r["pos"])
    no_neg = sum(1 for r in train if not r["neg"])
    texts = [t for r in train for t in (*r["pos"], *r["neg"])]
    lens = [len(t) for t in texts]
    qwords = [len(r["query"].split()) for r in train]
    return {
        "train_rows": len(train),
        "eval_rows": len(qrels),
        "empty_query": empty_q,
        "rows_without_pos": no_pos,
        "rows_without_neg": no_neg,
        "text_len_median": st.median(lens),
        "text_len_p95": sorted(lens)[int(len(lens) * 0.95)],
        "text_over_2000_chars": sum(1 for n in lens if n > LONG_TEXT_CHARS),
        "query_words_median": st.median(qwords),
        "query_words_p95": sorted(qwords)[int(len(qwords) * 0.95)],
        "eval_positives_median": st.median([len(r["positives"]) for r in qrels]),
    }


def audit_reliability(train: list[dict], qrels: list[dict]) -> dict:
    train_q = Counter(r["query"] for r in train)
    eval_q = {r["query"] for r in qrels}
    leaked = set(train_q) & eval_q

    # 致命项：同一条样本里，同一段商品文本既当正例又当负例
    conflict_rows = 0
    conflict_pairs = 0
    for r in train:
        dup = set(r["pos"]) & set(r["neg"])
        if dup:
            conflict_rows += 1
            conflict_pairs += len(dup)

    # 同款不同 ASIN 的普遍程度：正例文本本身有多少是重复的
    pos_texts = [t for r in train for t in r["pos"]]
    pos_uniq = len(set(pos_texts))

    cats = Counter(category_of(r["pos"][0]) for r in train if r["pos"])
    top10 = sum(c for _, c in cats.most_common(max(1, len(cats) // 10)))
    thin = sum(1 for _, c in cats.items() if c < 50)
    return {
        "duplicate_query_texts": sum(1 for q, c in train_q.items() if c > 1),
        "train_eval_query_leak": len(leaked),
        "pos_neg_conflict_rows": conflict_rows,
        "pos_neg_conflict_pairs": conflict_pairs,
        "pos_text_total": len(pos_texts),
        "pos_text_unique": pos_uniq,
        "pos_text_dup_rate": round(1 - pos_uniq / max(len(pos_texts), 1), 4),
        "categories": len(cats),
        "top10pct_category_share": round(top10 / max(len(train), 1), 4),
        "categories_under_50_rows": thin,
        "top5_categories": cats.most_common(5),
    }


def main() -> None:
    train = read_jsonl(TRAIN_PATH)
    qrels = read_jsonl(QRELS_PATH)
    report = {
        "usability": audit_usability(train, qrels),
        "reliability": audit_reliability(train, qrels),
    }
    OUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for section, body in report.items():
        print(f"\n=== {section} ===")
        for k, v in body.items():
            print(f"{k:28} {v}")
    print(f"\n已写 {OUT_PATH}")


if __name__ == "__main__":
    main()
