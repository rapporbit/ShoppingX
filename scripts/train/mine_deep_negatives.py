"""给 reranker 挖训练负例：**按 e15 的排名分层采样**，而不是只取最像的那几个。

**为什么不复用 M21 挖的那批（``neg_ann_candidates.jsonl``）：** 那批是给 embedding 挖的，
只取 top-25，攻的是「近义干扰抑制」。这次要治的病完全不同——体检实测现成 reranker 把候选池
从 20 加深到 1000，池里正例占比从 .33 涨到 .89，而 recall@8 从 .2426 只动到 .2437，**深池里
的正例它一个都捞不出来**。要教会的正是「排在 200 名开外的正例长什么样」，负例就必须覆盖到
那个深度，否则模型连见都没见过。

**分层比例（``--layers``，默认 5/5/5）：**

- ``rank 1-50``    e15 已经排到前面的非正例。模型要学会把它们压下去（浅层精排的活）。
- ``rank 50-200``  中段。
- ``rank 200-500`` 深段背景。没有这一层，模型在深池里就是瞎的。

**ESCI 标注的 S/C 一律直接入选，且不过假负闸。** 人工标注优先于模型判据：S（可替代但不是
要的那个）和 C（互补配件）正是 cross-encoder 最容易打高分的东西，交给闸去筛，最有价值的
hard negative 会被闸得一个不剩。而未标注的采样候选反过来——ESCI 只标了我们库的 9.2%，深池里
大量未标注的真相关商品会被误当负例（M21 实测闸掉率 52.24%），那些必须过闸。

输出只到「候选 + 文本」为止，闸在 GPU 上跑（``score_negatives.py``），组装在
``build_rerank_train.py``。

用法（需 e15 隧道）::

    EMBED_BASE_URL=http://127.0.0.1:8090/v1 QDRANT_COLLECTION=globex_items_e15 \\
        uv run --group train python scripts/train/mine_deep_negatives.py [--limit 0]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
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
CORPUS_PATH = PROJECT_ROOT / "data" / "train" / "corpus.jsonl"
OUT_PATH = PROJECT_ROOT / "data" / "train" / "deep_neg_candidates.jsonl"

TOP_K = 500
BATCH = 128  # 只取 item_id，payload 轻，批可以比导候选时大一倍
SEED = 42
# (下界, 上界, 采样数)。上界取不到，rank 从 1 开始。
LAYERS = [(1, 50, 5), (50, 200, 5), (200, 500, 5)]


def load_corpus() -> dict[str, str]:
    texts: dict[str, str] = {}
    with CORPUS_PATH.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            texts[rec["item_id"]] = rec["text"]
    return texts


def sample_layers(ranked: list[str], exclude: set[str], rng: random.Random) -> list[dict]:
    """按 e15 排名分层采样负例（已剔除该 query 的人工正例）。"""
    out: list[dict] = []
    for lo, hi, n in LAYERS:
        window = enumerate(ranked[lo - 1 : hi - 1], start=lo)
        seg = [(rank, x) for rank, x in window if x not in exclude]
        for rank, item_id in rng.sample(seg, min(n, len(seg))):
            out.append({"item_id": item_id, "rank": rank, "source": f"ann_{lo}_{hi}"})
    return out


async def run(rows: list[dict], corpus: dict[str, str], out_path: Path) -> dict:
    tower = TowerClient()
    client = make_client()
    rng = random.Random(SEED)
    stats = {"queries": 0, "ann": 0, "esci": 0, "no_text": 0}

    with out_path.open("w", encoding="utf-8") as out:
        for i in range(0, len(rows), BATCH):
            chunk = rows[i : i + BATCH]
            vecs = await tower.encode_texts([r["query"] for r in chunk])
            reqs = [
                models.QueryRequest(
                    query=[float(x) for x in v.ravel()],
                    using=DENSE_VEC,
                    limit=TOP_K,
                    # 文本走 corpus 内存映射，别让 Qdrant 吐 GB 级 payload
                    with_payload=["item_id"],
                )
                for v in vecs
            ]
            batch_res = client.query_batch_points(COLLECTION, requests=reqs)
            for row, res in zip(chunk, batch_res, strict=True):
                pos_ids = [p for p in row["pos_ids"] if p in corpus]
                if not pos_ids:
                    continue
                ranked = [(p.payload or {}).get("item_id", "") for p in res.points]
                cands = sample_layers(ranked, set(row["pos_ids"]), rng)
                stats["ann"] += len(cands)
                # ESCI 标注的 S/C/I 负例：直接入选、标记 gated=False（下游不对它们执行假负闸）
                # 合成 query（synth）没有 neg_src，负例全部来自 ANN 深池采样，走同一道假负闸
                neg_ids, neg_src = row.get("neg_ids") or [], row.get("neg_src") or []
                for nid, src in zip(neg_ids, neg_src, strict=True):
                    if src.startswith("esci"):
                        cands.append({"item_id": nid, "rank": 0, "source": src})
                        stats["esci"] += 1
                sized = [c | {"text": corpus.get(c["item_id"], "")} for c in cands]
                dropped = [c for c in sized if not c["text"]]
                stats["no_text"] += len(dropped)
                out.write(
                    json.dumps(
                        {
                            "query_id": row["query_id"],
                            "query": row["query"],
                            "pos": [{"item_id": p, "text": corpus[p]} for p in pos_ids],
                            "candidates": [c for c in sized if c["text"]],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                stats["queries"] += 1
            print(f"  {min(i + BATCH, len(rows))}/{len(rows)}", flush=True)

    await tower.aclose()
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（冒烟用，0=全跑）")
    ap.add_argument(
        "--input",
        default=str(TRAIN_PATH),
        help="训练 query 源。换 synth_train.jsonl 即为 M21 那批 LLM 合成 query（50% 中文）",
    )
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()

    rows = [json.loads(x) for x in Path(args.input).open(encoding="utf-8") if x.strip()]
    if args.limit:
        rows = rows[: args.limit]
    print(f"collection={COLLECTION}  query={len(rows)}  top_k={TOP_K}")
    print("加载 corpus.jsonl…", flush=True)
    corpus = load_corpus()
    print(f"  {len(corpus)} 条商品文本\n", flush=True)

    stats = asyncio.run(run(rows, corpus, Path(args.out)))
    print(f"\n{stats}\n已写 {args.out}")


if __name__ == "__main__":
    main()
