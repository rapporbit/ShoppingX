"""生成阈值标定样本：每个 ESCI 档位（E/S/C/I）各抽一批 (query, 商品文本) 对。

假负闸的阈值不能拍脑袋，得先看这把尺子（cross-encoder）在**人工已判定**的四个档位上打出什么
分布：``E`` 是人工说对的，``I`` 是人工说无关的，两组分数若分得开，阈值才有立足点；若倒挂，说明
闸根本不该上。

商品文本按线上精排口径拼（标题+品牌+品类，小写），与 ``mine_ann_negatives.py`` 落盘的候选文本
同源——标定和实际打分吃的必须是同一种输入。

用法：``uv run --group train python scripts/train/build_calib_pairs.py [--per-label 2000]``
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from qdrant_client import models  # noqa: E402

from app.recall.qdrant_store import COLLECTION, make_client  # noqa: E402

TRAIN_PATH = PROJECT_ROOT / "data" / "train" / "esci_train.jsonl"
OUT_PATH = PROJECT_ROOT / "data" / "train" / "calib_pairs.jsonl"
SRC_TO_LABEL = {"esci_S": "S", "esci_C": "C", "esci_I": "I"}


def fetch_texts(client, item_ids: list[str]) -> dict[str, str]:
    """按 item_id 批量取 payload，拼成线上精排口径的文本。"""
    texts: dict[str, str] = {}
    for i in range(0, len(item_ids), 500):
        batch = item_ids[i : i + 500]
        found, _ = client.scroll(
            COLLECTION,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="item_id", match=models.MatchAny(any=batch))]
            ),
            limit=len(batch),
            with_payload=["item_id", "title", "brand", "category"],
            with_vectors=False,
        )
        for p in found:
            pl = p.payload or {}
            raw = f"{pl.get('title', '')} {pl.get('brand', '')} {pl.get('category', '')}".lower()
            texts[pl.get("item_id", "")] = " ".join(raw.split())
    return texts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-label", type=int, default=2000, help="每档抽多少对")
    args = ap.parse_args()

    rows = [json.loads(x) for x in TRAIN_PATH.open(encoding="utf-8") if x.strip()]
    wanted: dict[str, list[tuple[str, str]]] = {k: [] for k in ("E", "S", "C", "I")}
    for r in rows:
        if len(wanted["E"]) < args.per_label:
            for iid in r["pos_ids"][:1]:  # 每条 query 只取一个正例，避免热门商品刷屏
                wanted["E"].append((r["query"], iid))
        for iid, src in zip(r["neg_ids"], r["neg_src"], strict=True):
            label = SRC_TO_LABEL.get(src)
            if label and len(wanted[label]) < args.per_label:
                wanted[label].append((r["query"], iid))
        if all(len(v) >= args.per_label for v in wanted.values()):
            break

    client = make_client()
    all_ids = sorted({iid for pairs in wanted.values() for _, iid in pairs})
    print(f"需要取文本的商品：{len(all_ids)}")
    texts = fetch_texts(client, all_ids)
    print(f"取到：{len(texts)}")

    n = 0
    with OUT_PATH.open("w", encoding="utf-8") as out:
        for label, pairs in wanted.items():
            for query, iid in pairs:
                text = texts.get(iid)
                if not text:
                    continue
                out.write(
                    json.dumps(
                        {"query": query, "item_id": iid, "text": text, "label": label},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                n += 1
            print(f"  {label}: {len(pairs)} 对")
    print(f"\n共 {n} 对，已写 {OUT_PATH}")


if __name__ == "__main__":
    main()
