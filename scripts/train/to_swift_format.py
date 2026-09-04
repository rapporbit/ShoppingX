"""把训练集转成 ms-swift 的 embedding(infonce) 输入格式。

ms-swift 4.x 的 infonce 任务吃的是 ``messages`` / ``positive_messages`` / ``negative_messages``
三段式（见其 ``MTEBRerankPreprocessor``）。我们直接产最终格式，不依赖它的字段名推断——那层
preprocessor 各版本改过好几轮，绕开它更稳。

**正例展开（``--max-pos``，默认 3）。** 一开始每条 query 只取第一个正例，等于把人工标注的正例
扔掉 74%（平均 3.94 个/query）——而人工标注是本项目最贵的数据。首轮实测 2 epoch 就过拟合
（v3 训 3 epoch 全面差于 2 epoch），根因是样本量只有 3.87 万，展开正好对症。refdocs 04-2 §3.1
也是这个口径：高质量数据一个 epoch 可以见 3-5 次。

**展开必须打散，这是个会静默训歪的坑。** infonce 开了 in-batch negatives（batch 内其他 query 的
正例互相当负例）。同一条 query 展开出的多条样本若落进同一个 batch，它自己的另一个正例就会被
当成负例——**标准的假负样本，而且是我们亲手造的**。按 query 顺序写出必然相邻必然同 batch；
打散后同 batch 概率降到千分之一量级。所以 ``--shuffle`` 默认开，别关。

用法：``uv run --group train python scripts/train/to_swift_format.py --input esci_train_v3.jsonl``
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "train"


def to_row(r: dict, pos: str) -> dict:
    return {
        "messages": [{"role": "user", "content": r["query"]}],
        "positive_messages": [[{"role": "assistant", "content": pos}]],
        "negative_messages": [[{"role": "assistant", "content": n}] for n in r["neg"]],
    }


def expand(rows: list[dict], max_pos: int) -> list[dict]:
    """一条 query × N 个正例 → N 条样本。正例去重，避免同文本重复计入。"""
    out = []
    for r in rows:
        seen: set[str] = set()
        for p in r["pos"][:max_pos]:
            if p not in seen:
                seen.add(p)
                out.append(to_row(r, p))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="esci_train_v3.jsonl")
    ap.add_argument("--prefix", default="swift_v3", help="输出文件前缀")
    ap.add_argument("--val-size", type=int, default=1000, help="切多少条 query 做验证集")
    ap.add_argument("--max-pos", type=int, default=3, help="每条 query 最多展开几个正例")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-shuffle", action="store_true", help="关掉打散（会造假负样本，别用）")
    args = ap.parse_args()

    rows = [json.loads(x) for x in (DATA_DIR / args.input).open(encoding="utf-8") if x.strip()]
    # 按 query_id 排序后切尾部做验证集：同一份数据重跑得到同样的切分，训练/验证不会串。
    # 切分在展开**之前**做——展开后再切会把同一条 query 的不同正例劈到两边，等于验证集泄漏。
    rows.sort(key=lambda r: r["query_id"])
    val_src = rows[-args.val_size :] if args.val_size else []
    train_src = rows[: len(rows) - len(val_src)]

    # 验证集不展开：每条 query 一个正例，跨版本的 eval_loss 才有可比性
    parts = {
        "train": expand(train_src, args.max_pos),
        "val": [to_row(r, r["pos"][0]) for r in val_src],
    }
    if not args.no_shuffle:
        random.Random(args.seed).shuffle(parts["train"])

    for name, part in parts.items():
        if not part:
            continue
        path = DATA_DIR / f"{args.prefix}_{name}.jsonl"
        with path.open("w", encoding="utf-8") as out:
            for r in part:
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
        src_n = len(train_src) if name == "train" else len(val_src)
        print(
            f"{name:6} {src_n:>6} query → {len(part):>6} 条  "
            f"{path.stat().st_size / 1e6:6.1f} MB  → {path.name}"
        )

    sample = parts["train"][0]
    print("\n样例（截断）:")
    print(f"  query    : {sample['messages'][0]['content'][:70]}")
    print(f"  positive : {sample['positive_messages'][0][0]['content'][:70]}")
    print(f"  negatives: {len(sample['negative_messages'])} 条")


if __name__ == "__main__":
    main()
