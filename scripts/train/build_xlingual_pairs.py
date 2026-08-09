"""构造中英同义 query 对，给跨语言体检当尺子。

ESCI 只有 en/es/jp，全参微调完全可能把 BGE-M3 的中文能力训崩——而 ShoppingX 线上是中文入口，
崩了就是灾难。refdocs 04-2 §6.3 给的两把尺子都在这里备好：

1. **同义跨语言 cosine 通过率**（阈值 ≥75%）：不需要标注集，几百对就能测，秒级出结果。
2. **跨语言 Recall Gap**（阈值 ≤0.05）：中文 query 全库检索的 recall 与英文版的差。

第 2 把尺子刻意做成「同结构 qrels」而不是新脚本——直接喂给 ``eval_on_gpu.py`` 就能出全套指标，
口径与英文侧逐条一致，不给自己留第二套评测代码。同时落一份**同样 query 的英文子集**：拿全量
11364 条的英文分数去比 1500 条的中文分数是不公平对照，两边必须同一批 query。

用法：``uv run --group train python scripts/train/build_xlingual_pairs.py --limit 1500``
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app.agent.llm import get_llm  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "train"
BATCH = 20
CONCURRENCY = 6

PROMPT = """把下面这些电商搜索词翻译成简体中文，保持「用户在搜索框里会怎么打」的口语形态。

规则：
- 品牌名、型号、规格（iPhone 14 / 3.5mm / XL / 4K）保留原文，不要音译也不要翻译
- 不要意译成书面语，不要补主语，不要加解释或标点修饰
- 严格输出 JSON 数组，字符串元素，长度必须与输入完全一致

输入：{items}"""


def _parse_array(text: str, expect: int) -> list[str] | None:
    """剥围栏取 JSON 数组；长度对不上直接判失败——宁可丢一批也不要错位对齐。"""
    m = re.search(r"\[.*]", text, re.S)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(arr, list) or len(arr) != expect:
        return None
    return [str(x).strip() for x in arr]


async def translate(rows: list[dict]) -> list[dict]:
    llm, sem = get_llm(), asyncio.Semaphore(CONCURRENCY)
    batches = [rows[i : i + BATCH] for i in range(0, len(rows), BATCH)]

    async def one(batch: list[dict]) -> list[dict]:
        queries = [r["query"] for r in batch]
        async with sem:
            for _ in range(2):  # 失败重试一次，仍失败则丢弃该批
                resp = await llm.ainvoke(PROMPT.format(items=json.dumps(queries, ensure_ascii=False)))
                zh = _parse_array(str(resp.content), len(queries))
                if zh:
                    return [{**r, "query_zh": z} for r, z in zip(batch, zh, strict=True)]
        print(f"  [warn] 一批 {len(batch)} 条翻译失败，已丢弃")
        return []

    done = await asyncio.gather(*(one(b) for b in batches))
    return [r for part in done for r in part]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = DATA_DIR / "esci_eval_qrels.jsonl"
    rows = [json.loads(x) for x in src.open(encoding="utf-8") if x.strip()]
    random.Random(args.seed).shuffle(rows)
    sample = rows[: args.limit]
    print(f"从 {len(rows)} 条评测 query 采样 {len(sample)} 条，开始翻译…")

    out = await translate(sample)
    print(f"翻译成功 {len(out)}/{len(sample)}")

    files = {
        "xlingual_pairs.jsonl": [
            {"query_id": r["query_id"], "en": r["query"], "zh": r["query_zh"]} for r in out
        ],
        "esci_eval_qrels_zh.jsonl": [{**{k: v for k, v in r.items() if k != "query_zh"}, "query": r["query_zh"]} for r in out],
        "esci_eval_qrels_en_sub.jsonl": [{k: v for k, v in r.items() if k != "query_zh"} for r in out],
    }
    for name, data in files.items():
        path = DATA_DIR / name
        with path.open("w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {len(data):>5} 条 → {name}")

    if out:
        print(f"\n样例: {out[0]['query']}  →  {out[0]['query_zh']}")


if __name__ == "__main__":
    asyncio.run(main())
