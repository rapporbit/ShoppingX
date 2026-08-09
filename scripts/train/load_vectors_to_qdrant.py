"""把 GPU 上预编码的向量灌进一个**新的** Qdrant collection，用于 A/B 对照。

为什么不直接改线上 collection：换模型 = 换向量空间，灌错了整个召回会静默崩掉（不报错、只是
结果变垃圾）。所以新老各一份，跑完端到端对照再决定切不切。

**payload 完全复用 `build_item_index.py` 的组装逻辑**——payload 里的 price_usd / platform /
rating 是 filter 的依据，自己另写一套迟早和线上不一致。这里只替换"向量从哪来"：不调 embedding
API，改用 `encode_corpus_gpu.py` 导出的 npy，按 item_id 对齐。

用法::

    QDRANT_COLLECTION=shoppingx_items_e10 uv run --group train python \\
        scripts/train/load_vectors_to_qdrant.py --vectors data/train/vectors_e10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from qdrant_client.models import OptimizersConfigDiff  # noqa: E402

from app.recall.qdrant_store import COLLECTION, QdrantRecall  # noqa: E402
from app.utils.clean import PLATFORMS  # noqa: E402
from scripts.build_item_index import _chunks, _iter_platform  # noqa: E402

CHUNK = 2000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", required=True, help="不含扩展名的前缀，如 data/train/vectors_e10")
    ap.add_argument("--platforms", nargs="*", default=list(PLATFORMS))
    args = ap.parse_args()

    vecs = np.load(f"{args.vectors}.npy")
    ids = json.loads(Path(f"{args.vectors}_ids.json").read_text(encoding="utf-8"))
    if len(ids) != len(vecs):
        raise SystemExit(f"❌ 向量 {len(vecs)} 与 id {len(ids)} 数量不一致")
    pos = {iid: i for i, iid in enumerate(ids)}
    print(f"载入向量 {vecs.shape}，目标 collection = {COLLECTION}")
    # 守卫必须比对 **.env 里配置的那个线上名字**，而不是代码里的默认值——本仓库线上是
    # globex_items，代码默认值是 shoppingx_items，照默认值判会让守卫完全失效。
    live = (dotenv_values(PROJECT_ROOT / ".env") or {}).get("QDRANT_COLLECTION") or "shoppingx_items"
    if COLLECTION == live:
        raise SystemExit(
            f"❌ 目标 collection ({COLLECTION}) 就是 .env 里的线上库，拒绝覆盖。\n"
            f"   请用 QDRANT_COLLECTION={live}_e10 这类新名字做 A/B。"
        )

    recall = QdrantRecall()
    recall.ensure_collection(int(vecs.shape[1]), recreate=True)

    # 批量导入前先关掉 HNSW 自动建索引，灌完再打开——Qdrant 官方推荐的姿势。
    # 不这么做的代价我们实测过：138 万点边灌边建索引，把 OrbStack VM 直接打爆，
    # Docker daemon 整个失去响应，灌到 103 万点时全崩。索引留到最后一次性建，
    # 灌库阶段内存占用是平的。
    recall.client.update_collection(
        collection_name=COLLECTION,
        optimizer_config=OptimizersConfigDiff(indexing_threshold=0),
    )
    print("已关闭自动建索引（indexing_threshold=0），灌完后恢复")

    start_id, done, missing, t0 = 0, 0, 0, time.monotonic()
    for platform in args.platforms:
        for chunk in _chunks(_iter_platform(platform), CHUNK):
            # 按 item_id 对齐，不依赖两边的遍历顺序恰好一致——顺序假设是最容易静默错位的地方
            keep, rows = [], []
            for rec in chunk:
                i = pos.get(rec.item_id)
                if i is None:
                    missing += 1
                    continue
                keep.append(i)
                rows.append(rec)
            if not rows:
                continue
            recall.upsert(rows, vecs[keep].astype(np.float32), start_id=start_id)
            start_id += len(rows)
            done += len(rows)
            if done % 50000 < CHUNK:
                el = time.monotonic() - t0
                print(f"  {done:,} 条  {done / el:.0f} 条/s  缺向量 {missing}", flush=True)

    el = time.monotonic() - t0
    print(f"\n灌入完成：{done:,} 条 → {COLLECTION}，耗时 {el / 60:.1f} min，缺向量 {missing} 条")

    # 恢复索引阈值，触发一次性建 HNSW。这一步同样吃资源，但它是单线程后台任务、
    # 内存曲线可控，比边灌边建温和得多。
    print("恢复 indexing_threshold=20000，开始后台建索引…", flush=True)
    recall.client.update_collection(
        collection_name=COLLECTION,
        optimizer_config=OptimizersConfigDiff(indexing_threshold=20000),
    )
    for _ in range(120):  # 最多等 20 分钟
        info = recall.client.get_collection(COLLECTION)
        if str(info.status) .endswith("green"):
            print(f"索引就绪：status={info.status}，points={info.points_count}")
            return
        time.sleep(10)
    print("⚠️ 20 分钟内索引未转 green，请手动确认 collection 状态")


if __name__ == "__main__":
    main()
