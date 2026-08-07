"""把 v3 训练集转成 ms-swift 的 embedding(infonce) 输入格式。

ms-swift 4.x 的 infonce 任务吃的是 ``messages`` / ``positive_messages`` / ``negative_messages``
三段式（见其 ``MTEBRerankPreprocessor``）。我们直接产最终格式，不依赖它的字段名推断——那层
preprocessor 各版本改过好几轮，绕开它更稳。

**每条 query 只取一个正例。** infonce 一条样本一个正例，我们平均有 3.94 个；全展开会把样本量
撑到 15 万、训练时间翻四倍，而边际收益未知。先取第一个跑通，展开与否留作后续消融。

用法：``uv run --group train python scripts/train/to_swift_format.py [--input esci_train_v3.jsonl]``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "train"


def to_row(r: dict) -> dict:
    return {
        "messages": [{"role": "user", "content": r["query"]}],
        "positive_messages": [[{"role": "assistant", "content": r["pos"][0]}]],
        "negative_messages": [[{"role": "assistant", "content": n}] for n in r["neg"]],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="esci_train_v3.jsonl")
    ap.add_argument("--prefix", default="swift_v3", help="输出文件前缀")
    ap.add_argument("--val-size", type=int, default=1000, help="切多少条做验证集")
    args = ap.parse_args()

    rows = [json.loads(x) for x in (DATA_DIR / args.input).open(encoding="utf-8") if x.strip()]
    # 按 query_id 排序后切尾部做验证集：同一份数据重跑得到同样的切分，训练/验证不会串
    rows.sort(key=lambda r: r["query_id"])
    val = rows[-args.val_size :] if args.val_size else []
    train = rows[: len(rows) - len(val)]

    for name, part in (("train", train), ("val", val)):
        if not part:
            continue
        path = DATA_DIR / f"{args.prefix}_{name}.jsonl"
        with path.open("w", encoding="utf-8") as out:
            for r in part:
                out.write(json.dumps(to_row(r), ensure_ascii=False) + "\n")
        size_mb = path.stat().st_size / 1e6
        print(f"{name:6} {len(part):>6} 条  {size_mb:6.1f} MB  → {path.name}")

    sample = to_row(train[0])
    print("\n样例（截断）:")
    print(f"  query    : {sample['messages'][0]['content'][:70]}")
    print(f"  positive : {sample['positive_messages'][0][0]['content'][:70]}")
    print(f"  negatives: {len(sample['negative_messages'])} 条")


if __name__ == "__main__":
    main()
