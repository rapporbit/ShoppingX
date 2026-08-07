"""探针：Amazon ESCI 的人工标注，能覆盖本仓库商品库（ASIN）多少？

**为什么先跑这个再动手。** ESCI 的 ``product_id`` 就是 ASIN，而我们清洗后的 amazon 商品
``item_id`` 也是 ASIN（见 ``data/platforms/clean/by_platform/amazon*.jsonl``）。两边能 join 上
多少，直接决定训练数据的配方：

- 交集大（≥10 万 judgement）→ 正例/难负例都用**真人工标注**，LLM 合成退居补长尾；
  评测集也能满足 refdocs 04-2 §6.1「不许拿爬来的数据当评测集」这条红线。
- 交集小 → ESCI 只能当「同分布外部训练集」，评测得另想办法（回到 §5.5 冷启动合成路线）。

**数据源选型（踩过的坑）。** HF 上 ``tasksource/esci`` 是 examples × products join 好的宽表，
带 ``product_text`` 等大字段共 1.8GB；即便用 parquet 列裁剪去读，走 HTTP range 请求也慢到不可用
（6 分钟没读完第一个 shard）。官方 ``esci-data`` 仓库的 **examples 文件只有 51MB** 且含全部 262 万
条标注——探针要的 ASIN / 标签 / query 全在里面。所以直接顺序下载它，几十秒的事。

用法：``uv run --group train python scripts/train/probe_esci_overlap.py``
"""

from __future__ import annotations

import json
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

ESCI_URL = (
    "https://media.githubusercontent.com/media/amazon-science/esci-data/main/"
    "shopping_queries_dataset/shopping_queries_dataset_examples.parquet"
)
ESCI_LOCAL = Path("data/train/_esci_examples.parquet")

CLEAN_DIR = Path("data/platforms/clean/by_platform")
OUT_PATH = Path("data/train/esci_overlap.json")


def load_our_asins() -> set[str]:
    """本仓库召回库里的 amazon ASIN 全集（主表 + RAG 扩充表）。"""
    asins: set[str] = set()
    for path in sorted(CLEAN_DIR.glob("amazon*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("platform") == "amazon" and rec.get("item_id"):
                    asins.add(rec["item_id"])
        print(f"  读 {path.name}: 累计 {len(asins)} 个 ASIN")
    return asins


def fetch_esci() -> Path:
    """下载官方 examples.parquet（51MB），已存在则复用。"""
    if ESCI_LOCAL.exists():
        print(f"复用已下载的 {ESCI_LOCAL}（{ESCI_LOCAL.stat().st_size / 1e6:.1f} MB）")
        return ESCI_LOCAL
    ESCI_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    print(f"下载 ESCI examples → {ESCI_LOCAL} …")
    tmp = ESCI_LOCAL.with_suffix(".part")
    urllib.request.urlretrieve(ESCI_URL, tmp)  # noqa: S310 — 固定的 https 常量地址
    tmp.rename(ESCI_LOCAL)
    print(f"完成（{ESCI_LOCAL.stat().st_size / 1e6:.1f} MB）")
    return ESCI_LOCAL


def main() -> None:
    ours = load_our_asins()
    print(f"本仓库 amazon ASIN：{len(ours)}\n")

    table = pq.read_table(fetch_esci())
    print(f"ESCI 行数：{table.num_rows}，列：{table.schema.names}\n")
    cols = table.to_pydict()
    splits = cols.get("split") or ["?"] * table.num_rows

    esci_asins: set[str] = set()
    hit_asins: set[str] = set()
    rows_total = 0
    rows_hit = 0
    label_all: Counter[str] = Counter()
    label_hit: Counter[str] = Counter()
    locale_hit: Counter[str] = Counter()
    split_hit: Counter[str] = Counter()
    # query_id → 该 query 在**我们库内**命中的标签集合。用来数「有几条 query 是可训/可评的」
    query_hit_labels: defaultdict[int, set[str]] = defaultdict(set)

    for pid, locale, label, qid, split in zip(
        cols["product_id"],
        cols["product_locale"],
        cols["esci_label"],
        cols["query_id"],
        splits,
        strict=True,
    ):
        rows_total += 1
        esci_asins.add(pid)
        label_all[label] += 1
        if pid in ours:
            rows_hit += 1
            hit_asins.add(pid)
            label_hit[label] += 1
            locale_hit[locale] += 1
            split_hit[split] += 1
            query_hit_labels[qid].add(label)

    # 一条 query 要能用来训练/评测，至少得有一个正例（Exact）落在我们库里
    trainable_queries = sum(1 for labels in query_hit_labels.values() if "E" in labels)

    stats = {
        "our_asins": len(ours),
        "esci_asins": len(esci_asins),
        "hit_asins": len(hit_asins),
        "asin_coverage_of_ours": round(len(hit_asins) / max(len(ours), 1), 4),
        "asin_coverage_of_esci": round(len(hit_asins) / max(len(esci_asins), 1), 4),
        "rows_total": rows_total,
        "rows_hit": rows_hit,
        "row_hit_rate": round(rows_hit / max(rows_total, 1), 4),
        "queries_touched": len(query_hit_labels),
        "queries_with_positive": trainable_queries,
        "label_all": dict(label_all),
        "label_hit": dict(label_hit),
        "locale_hit": dict(locale_hit),
        "split_hit": dict(split_hit),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 结果 ===")
    for k, v in stats.items():
        print(f"{k:26} {v}")
    print(f"\n已写 {OUT_PATH}")


if __name__ == "__main__":
    main()
