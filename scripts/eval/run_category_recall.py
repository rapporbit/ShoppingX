"""跑召回评测：用金标集打 Recall@K / MRR / NDCG，作升级的「客观刹车」。

refdocs 13-1 §5.5 把这套指标接进发版流程：任何对 weights / coarse_k / rerank 阈值的改动，
都先跑一次本地评测过门禁，再合代码。本项目的门禁是**宽松基线冒烟**（不是上线级硬阈值）——
金标集是脚本合成的小集（gitignore，可复现），阈值定得保守，主要防「整体塌方」式回退。

用法：
    uv run python scripts/eval/run_category_recall.py
    uv run python scripts/eval/run_category_recall.py --k 10
退出码：达标 0 / 不达标 1（可挂 CI，但当前定位是冒烟）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv  # noqa: E402

from app.eval.recall_metrics import aggregate  # noqa: E402
from app.tools.category_insight import _recall_cards  # noqa: E402

# 不加载 .env 会静默退化到本地哈希编码（256 维）——对着 1024 维真向量库要么崩、要么
# 算出无意义相似度，评测结果全是假的（与 build_category_kb.py 的 P0 同源）。
load_dotenv()

GOLDEN_PATH = Path("data/eval/category_recall.jsonl")

# 宽松基线（冒烟，非上线硬阈值）：金标 query 是品类自指，召回应稳稳过线。
THRESHOLDS = {"recall": 0.60, "mrr": 0.50, "ndcg": 0.55}


async def main(k: int) -> int:
    if not GOLDEN_PATH.exists():
        print(f"找不到金标集 {GOLDEN_PATH}，请先跑 scripts/eval/build_category_golden.py")
        return 1
    samples = [
        json.loads(line)
        for line in GOLDEN_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows: list[dict] = []
    for s in samples:
        cards = await _recall_cards(s["query"], top_k=k)
        rows.append({"retrieved": [c.card_id for c in cards], "relevant": s["relevant"]})

    metrics = aggregate(rows, k)
    print(f"评测集 {len(samples)} 条，Top-K={k}")
    for name, val in metrics.items():
        print(f"  {name:10s} = {val:.3f}")

    failed = []
    if metrics[f"recall@{k}"] < THRESHOLDS["recall"]:
        failed.append(f"recall@{k} < {THRESHOLDS['recall']}")
    if metrics["mrr"] < THRESHOLDS["mrr"]:
        failed.append(f"mrr < {THRESHOLDS['mrr']}")
    if metrics[f"ndcg@{k}"] < THRESHOLDS["ndcg"]:
        failed.append(f"ndcg@{k} < {THRESHOLDS['ndcg']}")
    if failed:
        print("门禁未过：" + "；".join(failed))
        return 1
    print("门禁通过（宽松基线冒烟）。")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=10, help="Top-K")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.k)))
