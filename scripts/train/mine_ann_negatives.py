"""用当前召回模型挖难负例：语义最像、但**不是**这条 query 的答案的商品。

refdocs 04-2 §5.5.4 明令禁止冷启动期这么干（「自己挖自己的负样本，v0 错的会被 v1 学进去，
错得更深」）。**这条禁令的前提是「没有东西能判定挖出来的对不对」——我们不成立**：

- 第一道闸：ESCI 人工标注。挖出来的候选凡是被标过 ``E``（该 query 的正例）一律剔除。
- 第二道闸：cross-encoder 打分（在 GPU 机器上跑 ``score_negatives.py``）。分数过高的判为
  「疑似真相关但没被标注」，也剔除。这是 refdocs §3.3 假负过滤的替代品——我们没有本地商品图，
  pHash 那条路走不通，改用语义层面的判定。

本脚本只产候选（第一道闸已应用），第二道闸在 GPU 上跑。输出的每条候选都带 ann_rank / ann_score，
便于事后分析「挖到第几名开始就不像了」。

用法：``uv run --group train python scripts/train/mine_ann_negatives.py [--top-k 25]``
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from qdrant_client import models  # noqa: E402

from app.recall.qdrant_store import COLLECTION, DENSE_VEC, make_client  # noqa: E402
from app.recall.towers import TowerClient  # noqa: E402

TRAIN_PATH = PROJECT_ROOT / "data" / "train" / "esci_train.jsonl"
OUT_PATH = PROJECT_ROOT / "data" / "train" / "neg_ann_candidates.jsonl"
BATCH = 128


async def run(rows: list[dict], top_k: int) -> dict[str, int]:
    tower = TowerClient()
    client = make_client()
    n_cand = n_drop_labeled = 0

    with OUT_PATH.open("w", encoding="utf-8") as out:
        for i in range(0, len(rows), BATCH):
            chunk = rows[i : i + BATCH]
            vecs = await tower.encode_texts([r["query"] for r in chunk])
            reqs = [
                models.QueryRequest(
                    query=[float(x) for x in v.ravel()],
                    using=DENSE_VEC,
                    limit=top_k,
                    with_payload=["item_id", "title", "brand", "category"],
                )
                for v in vecs
            ]
            res = client.query_batch_points(COLLECTION, requests=reqs)
            for row, r in zip(chunk, res, strict=True):
                # 已知正例、以及已经被选进训练集的负例，都不再重复挖
                known = set(row["pos_ids"]) | set(row["neg_ids"])
                cands = []
                for rank, p in enumerate(r.points, 1):
                    payload = p.payload or {}
                    iid = payload.get("item_id", "")
                    if iid in known:
                        n_drop_labeled += 1
                        continue
                    # 文本按**线上精排的口径**拼好再落盘（item_search.py::_searchable：
                    # 标题+品牌+品类，小写）。GPU 侧那台机器没有商品库，拼不了，也不该再拼一次
                    # ——两处各拼一遍迟早会漂。
                    text = (
                        f"{payload.get('title', '')} {payload.get('brand', '')} "
                        f"{payload.get('category', '')}"
                    ).lower()
                    cands.append(
                        {
                            "item_id": iid,
                            "text": " ".join(text.split()),
                            "ann_rank": rank,
                            "ann_score": round(float(p.score), 4),
                        }
                    )
                n_cand += len(cands)
                out.write(
                    json.dumps(
                        {"query_id": row["query_id"], "query": row["query"], "candidates": cands},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            print(f"  {min(i + BATCH, len(rows))}/{len(rows)}  候选累计 {n_cand}", flush=True)

    return {"queries": len(rows), "candidates": n_cand, "dropped_known": n_drop_labeled}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=25, help="每条 query 挖多少候选")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（冒烟）")
    args = ap.parse_args()

    rows = [json.loads(x) for x in TRAIN_PATH.open(encoding="utf-8") if x.strip()]
    if args.limit:
        rows = rows[: args.limit]
    print(f"训练 query：{len(rows)}，每条挖 Top-{args.top_k}\n")

    stats = asyncio.run(run(rows, args.top_k))
    print("\n=== 结果 ===")
    for k, v in stats.items():
        print(f"{k:16} {v}")
    print(f"已写 {OUT_PATH}")


if __name__ == "__main__":
    main()
