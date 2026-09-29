"""增量重编：只重新编码 embed_text 变了的商品，按原 point id 原地覆盖线上 Qdrant。

McAuley 并入后（``scripts/merge_mcauley.py``）约 100 万条商品的编码文本变了（补品牌 / 描述 /
短属性行），其余 37 万条逐字没变，不重编。point id 沿用 ``build_item_index.py`` 的顺排：
``amazon.jsonl``（995 条）在前，``amazon_rag.jsonl`` 第 r 行 → id = 995 + r。

安全闸：
- ``--require-remote`` 语义内置：编码器退化成本地哈希回退、或维度与线上 collection 不一致 → 退出；
- 每个 chunk 写入前先按 id 读回线上 payload 的 item_id，逐条对不上 → 退出（防行号错位覆盖错商品）。
断点续跑：进度行打印已处理到的原始行号，崩了用 ``--start-row`` 接着跑。

用法（在能连到 Qdrant、带 EMBED_* 的环境里）：
    python scripts/reembed_changed.py --orig amazon_rag.jsonl --merged amazon_rag_mcauley.jsonl \\
        [--limit 1000] [--start-row 0]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.recall.qdrant_store import COLLECTION, DENSE_VEC, QdrantRecall  # noqa: E402
from app.recall.schemas import ItemRecord  # noqa: E402
from app.recall.text import embed_text  # noqa: E402
from app.recall.towers import TowerClient  # noqa: E402
from app.utils.clean import CleanItem  # noqa: E402
from scripts.build_item_index import (  # noqa: E402  (导入即 load_dotenv)
    CHUNK_RECORDS,
    DEFAULT_LOCAL_DIM,
    _clean_to_record,
    _encode_records,
    _RateLimiter,
)

ID_OFFSET = 995  # amazon.jsonl 行数：build_item_index 先灌它，amazon_rag 从这里接着排
TOTAL_ROWS = 1_376_253


def _check_ids(recall: QdrantRecall, ids: list[int], recs: list[ItemRecord]) -> None:
    got = recall._client.retrieve(COLLECTION, ids=ids, with_payload=["item_id"])  # noqa: SLF001
    live = {int(p.id): (p.payload or {}).get("item_id") for p in got}
    bad = [
        (i, r.item_id, live.get(i))
        for i, r in zip(ids, recs, strict=True)
        if live.get(i) != r.item_id
    ]
    if bad:
        raise SystemExit(
            f"❌ point id 与 item_id 对不上（前 3 条）：{bad[:3]}，已中止，未写入本 chunk"
        )


async def main(args: argparse.Namespace) -> None:
    tower, recall, limiter = TowerClient(), QdrantRecall(), _RateLimiter()
    if not tower.remote:
        raise SystemExit("❌ EMBED_MODEL 未生效（本地哈希回退），拒绝写线上 Qdrant")
    info = recall._client.get_collection(COLLECTION)  # noqa: SLF001
    live_dim = info.config.params.vectors[DENSE_VEC].size
    print(f"collection={COLLECTION} points={info.points_count:,} dim={live_dim}", flush=True)

    pending: list[tuple[int, ItemRecord]] = []
    stats = {"rows": 0, "changed": 0, "written": 0}
    t0 = time.monotonic()

    async def flush(last_row: int) -> None:
        ids = [ID_OFFSET + r for r, _ in pending]
        recs = [rec for _, rec in pending]
        _check_ids(recall, ids, recs)
        dense = await _encode_records(recs, tower, limiter)
        if dense.shape[1] != live_dim or dense.shape[1] == DEFAULT_LOCAL_DIM:
            raise SystemExit(f"❌ 编码维度 {dense.shape[1]} ≠ 线上 {live_dim}，已中止")
        recall.upsert(recs, dense, ids=ids)
        stats["written"] += len(recs)
        pending.clear()
        el = time.monotonic() - t0
        done = last_row + 1 - args.start_row
        eta = (TOTAL_ROWS - last_row - 1) / (done / el) / 60 if done else 0
        print(
            f"  row={last_row + 1:,}/{TOTAL_ROWS:,} 已写 {stats['written']:,} "
            f"跳过 {stats['rows'] - stats['changed']:,}  {el / 60:.0f}min  ETA {eta:.0f}min",
            flush=True,
        )

    with open(args.orig, encoding="utf-8") as fo, open(args.merged, encoding="utf-8") as fm:
        row = -1
        for row, (lo, lm) in enumerate(zip(fo, fm, strict=True)):
            if row < args.start_row:
                continue
            stats["rows"] += 1
            new = _clean_to_record(CleanItem.model_validate_json(lm))
            if new.embed_text == embed_text(CleanItem.model_validate_json(lo)):
                continue
            stats["changed"] += 1
            pending.append((row, new))
            if len(pending) >= CHUNK_RECORDS:
                await flush(row)
            if args.limit and stats["changed"] >= args.limit:
                break
        if pending:
            await flush(row)
    await tower.aclose()
    print(f"完成：扫描 {stats['rows']:,} 行，重编并覆盖 {stats['written']:,} 条", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="增量重编 embed_text 变了的商品并覆盖线上 point")
    ap.add_argument("--orig", required=True, help="并入前的 amazon_rag.jsonl（算旧编码文本）")
    ap.add_argument("--merged", required=True, help="merge_mcauley.py 的产物")
    ap.add_argument("--start-row", type=int, default=0, help="断点续跑：从这一行开始")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 条变化商品（冒烟用）")
    asyncio.run(main(ap.parse_args()))
