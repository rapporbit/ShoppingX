"""体检第一段：把 e15 召回的 top-K 候选连同文本导出，等着上 GPU 打精排分。

**为什么要有这一步（而不是本地直接 rerank）：** 本机没有卡，1000 条 query × 1000 候选
= 100 万对 cross-encoder 打分，走 API 又慢又贵。沿用 M21 已经跑通的分工——本地挖候选、
GPU 打分、本地算指标（同 ``mine_ann_negatives`` → ``score_negatives`` → ``merge_negatives``）。
分数与指标解耦的好处是：后面调评测口径（换 K、换 gain、换指标）不用重跑 GPU。

**三条口径，都是有意为之：**

1. **候选文本 = ``title brand category`` 小写**，与 ``app/tools/item_picker.py`` 的
   ``_searchable()`` 逐字对齐。体检要量的是**线上那把尺子**，不是理论最优形态；换个更全的
   文本可能更好看，但那不是线上跑的东西。
2. **抽样 1000 条 query，不全量。** 全量 11364 × 1000 候选的文本约 1.7GB，而我们要的只是
   一条深度曲线的形状，1000 条足够；种子固定，前后两次对照可比。
3. **同时导出「品类词 query」**：取该 query 在 top-K 里命中的正例的 ``category`` 众数。这是
   在模拟线上 ``item_picker`` 传给 reranker 的那个粗品类词——而且是**作弊版**（线上的 planner
   看不见正例，只能看 query 猜）。给品类词形态一个上界优势，如果这样它还是输给完整意图句，
   「线上 rerank 的 query 形态不对」这个结论就再无争议。

用法（需要 e15 的 embedding 服务隧道通着）::

    EMBED_BASE_URL=http://127.0.0.1:8090/v1 QDRANT_COLLECTION=globex_items_e15 \\
        uv run --group train python scripts/train/export_rerank_candidates.py --limit 1000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

import asyncio  # noqa: E402

from qdrant_client import models  # noqa: E402

from app.recall.qdrant_store import COLLECTION, DENSE_VEC, make_client  # noqa: E402
from app.recall.towers import TowerClient  # noqa: E402

QRELS_PATH = PROJECT_ROOT / "data" / "train" / "esci_eval_qrels.jsonl"
OUT_PATH = PROJECT_ROOT / "data" / "train" / "rerank_candidates.jsonl"

TOP_K = 1000  # 导最深一档，浅档在评测脚本里按 rank 截断即可，不用重跑
ENCODE_BATCH = 64  # 单条 payload 比 eval_recall 重（带文本），批小一点防单次响应过大
SEED = 42
PAYLOAD_FIELDS = ["item_id", "title", "brand", "category"]
CORPUS_PATH = PROJECT_ROOT / "data" / "train" / "corpus.jsonl"


def load_corpus_texts() -> dict[str, str]:
    """``--text-form embed`` 用：item_id → ``embed_text`` 全库文本（建索引/embedding 训练同源）。

    与 ``_searchable`` 的差别不在长度（实测中位 129 vs 146 字符，相当），在**组织方式**：
    ``embed_text`` 是 ``title | brand | 尾3类 | 描述片段``，``_searchable`` 是 ``title brand
    全路径品类`` 空格拼接。cross-encoder 吃的是 token 序列，这种差别值得量一次——毕竟它
    零训练成本，而且训练该用哪个形态得由这个数来定。
    """
    texts: dict[str, str] = {}
    with CORPUS_PATH.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            texts[rec["item_id"]] = rec["text"]
    return texts


def searchable(payload: dict) -> str:
    """与 item_picker._searchable() 同形：标题 + 品牌 + 品类，小写。"""
    parts = (payload.get("title", ""), payload.get("brand", ""), payload.get("category", ""))
    return " ".join(parts).lower()


def category_query(cands: list[dict], positives: set[str]) -> str:
    """作弊版「品类词 query」：命中正例的 category 众数（拿不到则空串）。"""
    cats = [c["category"] for c in cands if c["item_id"] in positives and c.get("category")]
    return Counter(cats).most_common(1)[0][0] if cats else ""


async def run(rows: list[dict], out_path: Path, corpus: dict[str, str] | None = None) -> dict:
    tower = TowerClient()
    client = make_client()
    n_with_cat = n_missing = 0

    with out_path.open("w", encoding="utf-8") as out:
        for i in range(0, len(rows), ENCODE_BATCH):
            chunk = rows[i : i + ENCODE_BATCH]
            vecs = await tower.encode_texts([r["query"] for r in chunk])
            reqs = [
                models.QueryRequest(
                    query=[float(x) for x in v.ravel()],
                    using=DENSE_VEC,
                    limit=TOP_K,
                    with_payload=PAYLOAD_FIELDS,
                )
                for v in vecs
            ]
            batch_res = client.query_batch_points(COLLECTION, requests=reqs)
            for row, res in zip(chunk, batch_res, strict=True):
                cands = [
                    {
                        "item_id": (p.payload or {}).get("item_id", ""),
                        "category": (p.payload or {}).get("category", ""),
                        "text": searchable(p.payload or {}),
                    }
                    for p in res.points
                ]
                cat_q = category_query(cands, set(row["positives"]))
                n_with_cat += bool(cat_q)
                if corpus is not None:
                    # corpus 缺这个 item_id 就退回 _searchable：不是所有点都在 e15 训练语料里，
                    # 静默丢候选会让两份导出的候选池不同，A/B 就不是同一批东西了。
                    for c in cands:
                        alt = corpus.get(c["item_id"])
                        n_missing += alt is None
                        c["text"] = alt or c["text"]
                slim = [{"item_id": c["item_id"], "text": c["text"]} for c in cands]
                out.write(
                    json.dumps(
                        {
                            "query_id": row["query_id"],
                            "query": row["query"],
                            "category_query": cat_q,
                            # rank 就是列表下标，不另存；text 之外的字段评测段用 qrels 对齐
                            "candidates": slim,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            print(f"  {min(i + ENCODE_BATCH, len(rows))}/{len(rows)}", flush=True)

    await tower.aclose()
    return {
        "queries": len(rows),
        "top_k": TOP_K,
        "with_category_query": n_with_cat,
        "corpus_missing": n_missing,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1000, help="抽样 query 数（0=全跑，慎用）")
    ap.add_argument("--out", default=str(OUT_PATH))
    ap.add_argument(
        "--text-form",
        choices=("searchable", "embed"),
        default="searchable",
        help="候选文本形态：searchable=线上 item_picker 用的，embed=建索引/embedding 训练用的",
    )
    args = ap.parse_args()

    rows = [json.loads(x) for x in QRELS_PATH.open(encoding="utf-8") if x.strip()]
    if args.limit and args.limit < len(rows):
        random.Random(SEED).shuffle(rows)
        rows = rows[: args.limit]
    print(f"collection={COLLECTION}  query={len(rows)}  top_k={TOP_K}  text={args.text_form}\n")

    corpus = None
    if args.text_form == "embed":
        print("加载 corpus.jsonl（138 万条）…", flush=True)
        corpus = load_corpus_texts()
        print(f"  已载入 {len(corpus)} 条商品文本\n", flush=True)

    report = asyncio.run(run(rows, Path(args.out), corpus))
    print(f"\n{report}\n已写 {args.out}")


if __name__ == "__main__":
    main()
