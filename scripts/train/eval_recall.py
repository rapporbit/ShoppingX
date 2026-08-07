"""训练前基线：现有 embedding 在 ESCI holdout 上的召回质量，外加「整机搜出配件」专项。

**这是后面一切对比的参照物，必须在动数据配方之前跑。** 训完模型换成新 endpoint 再跑一次同一个
脚本，两份 JSON 一比就是全部结论。

四条口径：

1. **候选池是全库**（``platform="all"``，137 万条），不是只在标注命中的 12.7 万里检索。未标注的
   商品会挤占名额、算作 miss —— 指标偏保守，但贴近真实检索环境。
2. **K 取 1000**。这一条是实测倒逼的：我们库里 90.8% 的商品没有标注，一条 "$150 laptop" query
   召回的全是语义同样匹配、但没被 ESCI 标过的笔记本，把标注正例挤到 100 名开外——K=100 时
   ``recall=0.29``、``mrr`` 几乎全 0，训练前后的差异根本量不出来。放到 1000 后 ``recall=0.57``，
   区分度才回来。（ESCI 官方 benchmark 是「给定候选列表排序」，本就不是全库检索。）
3. **Recall 分母取「该 query 在我们库内的正例数」**。ESCI 的正例有相当一部分不在我们库里，用全
   部正例当分母会系统性低估，那个数没有可比性。
4. **NDCG 用分级 gain**（E=3 / S=2 / C=1 / 其他=0），这是 ESCI 论文的标准档位；二值 NDCG 会把
   「召回了可替代品」和「召回了完全无关的东西」当成一回事。

专项指标 ``complement_hits_per_query@100``：在带配件标注的 query 上，Top-100 里标注配件的条数。
这是「搜手机出配件」这个 bad case 的直接度量 —— 数越低越好。**它的灵敏度有限**：ESCI 标过的配件
只占我们库极小一部分，Top-100 里真正的配件大多没标注、量不到。当参考不当结论。

用法：``uv run --group train python scripts/train/eval_recall.py [--limit 2000]``
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()  # EMBED_MODEL / EMBED_BASE_URL / QDRANT_* 都从 .env 来

from qdrant_client import models  # noqa: E402

from app.recall.qdrant_store import COLLECTION, DENSE_VEC, make_client  # noqa: E402
from app.recall.towers import TowerClient  # noqa: E402

QRELS_PATH = PROJECT_ROOT / "data" / "train" / "esci_eval_qrels.jsonl"
OUT_PATH = PROJECT_ROOT / "data" / "train" / "baseline_report.json"

# K 取 1000 而不是 100：实测标注正例大量落在 100~1000 名之间（recall@100=0.29 → @1000=0.57）。
# 原因是我们库里 90.8% 的商品没有标注，同类未标注商品会把标注正例挤下去——K 太小时指标几乎全 0，
# 训练前后的差异会被淹没在噪声里，量不出东西来。
TOP_K = 1000
GAIN = {"pos": 3.0, "sub": 2.0, "comp": 1.0}
ENCODE_BATCH = 128  # 批量编码 + 批量检索，比逐条 await 少两个数量级的往返


def dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def score_one(hits: list[str], row: dict) -> dict[str, float]:
    """单条 query 的指标。``hits`` 是按序的 item_id 列表。"""
    pos = set(row["positives"])
    sub = set(row.get("substitutes") or [])
    comp = set(row.get("complements") or [])

    def gain_of(item_id: str) -> float:
        if item_id in pos:
            return GAIN["pos"]
        if item_id in sub:
            return GAIN["sub"]
        if item_id in comp:
            return GAIN["comp"]
        return 0.0

    top100 = set(hits[:100])
    rr = 0.0
    for i, h in enumerate(hits[:100]):
        if h in pos:
            rr = 1.0 / (i + 1)
            break
    # 理想排序 = 库内已知相关商品按 gain 降序（配件也算弱相关，但排在可替代品之后）
    ideal = sorted(
        [GAIN["pos"]] * len(pos) + [GAIN["sub"]] * len(sub) + [GAIN["comp"]] * len(comp),
        reverse=True,
    )[:100]
    ndcg = dcg([gain_of(h) for h in hits[:100]]) / dcg(ideal) if ideal else 0.0
    return {
        "recall@1000": len(pos & set(hits)) / len(pos),
        "recall@100": len(pos & top100) / len(pos),
        "recall@20": len(pos & set(hits[:20])) / len(pos),
        "mrr@100": rr,
        "ndcg@100": ndcg,
        "complement_hits@100": float(len(comp & top100)),
        "has_complement": 1.0 if comp else 0.0,
    }


async def run(rows: list[dict]) -> dict:
    tower = TowerClient()
    client = make_client()
    totals: dict[str, float] = {}
    done = 0

    def search_batch(vecs) -> list[list[str]]:
        """一次请求发一批 query。

        **不要用线程池并发调 QdrantRecall.search**：qdrant-client 的同步 HTTP 连接跨线程复用会
        炸 ``Bad file descriptor``（实测全量跑到一半崩）。批量接口是官方给的正解，服务端并行，
        客户端单线程。
        """
        reqs = [
            models.QueryRequest(
                query=[float(x) for x in v.ravel()],
                using=DENSE_VEC,
                limit=TOP_K,
                with_payload=["item_id"],
            )
            for v in vecs
        ]
        res = client.query_batch_points(COLLECTION, requests=reqs)
        return [[(p.payload or {}).get("item_id", "") for p in r.points] for r in res]

    for i in range(0, len(rows), ENCODE_BATCH):
        chunk = rows[i : i + ENCODE_BATCH]
        vecs = await tower.encode_texts([r["query"] for r in chunk])
        for row, hits in zip(chunk, search_batch(vecs), strict=True):
            for k, v in score_one(hits, row).items():
                totals[k] = totals.get(k, 0.0) + v
        done += len(chunk)
        print(f"  {done}/{len(rows)}  recall@1000≈{totals['recall@1000'] / done:.4f}", flush=True)

    n = len(rows)
    n_comp = totals["has_complement"] or 1.0
    return {
        "eval_queries": n,
        "top_k": TOP_K,
        "recall@1000": round(totals["recall@1000"] / n, 4),
        "recall@100": round(totals["recall@100"] / n, 4),
        "recall@20": round(totals["recall@20"] / n, 4),
        "mrr@100": round(totals["mrr@100"] / n, 4),
        "ndcg@100": round(totals["ndcg@100"] / n, 4),
        "queries_with_complement": int(totals["has_complement"]),
        "complement_hits_per_query@100": round(totals["complement_hits@100"] / n_comp, 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（冒烟用，0=全跑）")
    args = ap.parse_args()

    rows = [json.loads(x) for x in QRELS_PATH.open(encoding="utf-8") if x.strip()]
    if args.limit:
        rows = rows[: args.limit]
    print(f"评测 query：{len(rows)}，候选池：全库\n")

    report = asyncio.run(run(rows))
    OUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== 基线 ===")
    for k, v in report.items():
        print(f"{k:30} {v}")
    print(f"\n已写 {OUT_PATH}")


if __name__ == "__main__":
    main()
