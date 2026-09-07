"""商品召回门禁：在 ESCI golden（M21 那份 qrels）上打 Recall@20 / MRR / NDCG@20，不达标退非零。

**与 `run_category_recall.py` 的分工**：那条量的是品类知识库（OpenSearch，几百条自指金标）；
这条量的是**商品向量召回**（Qdrant dense，1.1 万条人工标注 query）。改 embedding / 索引 /
payload filter / coarse_k 前后各跑一遍，两份 JSON 一比就是结论。

**与 M21 基线 `scripts/train/eval_recall.py` 的分工**：那份是训练侧的**模型对照**（K=1000，
把「排序空间」全摊开，训前训后差异才量得出）；这份是工程侧的**回归门禁**（K=20 = 线上真实吃
进下游的条数，跑得快、能挂 CI）。口径（分级 gain / 分母取库内正例）刻意与它逐字一致，只换 K。

**K=20 时 recall 天然低（≈0.29）**，不是 bug：库里 90.8% 的商品没有 ESCI 标注，同类未标注商品
会把标注正例挤到 20 名开外。所以阈值是**防塌方的下限**，不是「好」的标准，别照着它调优。

用法：
    uv run python scripts/eval/run_product_recall.py --limit 500          # 冒烟
    uv run python scripts/eval/run_product_recall.py                      # 全集（1.1 万条）
    uv run python scripts/eval/run_product_recall.py --collection globex_items_e15 --batch 64
    # 当门禁用（阈值 = 2026-09-07 基线 ×0.9，防塌方，不是「好」的标准）：
    uv run python scripts/eval/run_product_recall.py \
        --min-recall 0.22 --min-mrr 0.15 --min-ndcg 0.13
退出码：达标 0 / 不达标 1 / 数据或服务不可用 2。

基线（2026-09-07，`globex_items` 全集 11364 条 / BGE-M3 API / 约 9 分钟）：
``recall@20=0.2472  mrr=0.1765  ndcg@20=0.1492  complement_hits@20=0.1716``。
与 M21 的 `baseline_report.json`（同一份 golden，K=1000 检索后截前 20）差 0.9pt，是 HNSW 的
``ef`` 随 limit 变大而变大所致——同一个索引，检索 20 条就是比检索 1000 条再截前 20 略差一点。

⚠️ **collection 与 encoder 必须配套**：`--collection` 换成自训权重建的索引（如 e15）时，query
编码也得换成同一份权重的服务（`EMBED_BASE_URL`），否则查的是两个不同的向量空间——指标会掉得
很难看，但**一条报错都没有**。脚本会把两者一起打进报告头，复查时先核这两行。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

# 不加载 .env，TowerClient 会静默退化到本地哈希编码（256 维）——对 1024 维真索引要么崩、
# 要么算出无意义相似度，指标全是假的（与 build_category_kb.py / run_category_recall.py 同源）。
load_dotenv()

from qdrant_client import models  # noqa: E402

from app.eval.recall_metrics import aggregate_graded  # noqa: E402
from app.recall.qdrant_store import COLLECTION, DENSE_VEC, make_client  # noqa: E402
from app.recall.towers import TowerClient  # noqa: E402

QRELS_PATH = PROJECT_ROOT / "data" / "train" / "esci_eval_qrels.jsonl"
OUT_PATH = PROJECT_ROOT / "data" / "eval" / "product_recall_report.json"

# 批量编码 + 批量检索。逐条 await 会多两个数量级的往返；**不要换成线程池并发调同步 client**：
# qdrant-client 的同步 HTTP 连接跨线程复用会炸 Bad file descriptor（M21 全量跑到一半崩过）。
ENCODE_BATCH = 128


def load_qrels(path: Path, limit: int) -> list[dict]:
    if not path.exists():
        raise SystemExit(
            f"[2] 找不到 golden {path}。它是 M21 的产物（data/ 在 .gitignore 里），"
            "重建：uv run --group train python scripts/train/build_esci_pairs.py"
        )
    rows = [json.loads(x) for x in path.open(encoding="utf-8") if x.strip()]
    rows = [r for r in rows if r.get("positives")]  # 无正例的行量不出 recall，直接剔
    return rows[:limit] if limit else rows


def encoder_id() -> str:
    """报告头里的编码器身份——**核指标之前先核这一行**（见模块 docstring 的配套告警）。"""
    return f"{os.environ.get('EMBED_MODEL', '?')} @ {os.environ.get('EMBED_BASE_URL', '?')}"


async def run(
    rows: list[dict],
    collection: str,
    k: int,
    batch: int = ENCODE_BATCH,
    qdrant_timeout: float = 120.0,
) -> dict:
    # 参数名不叫 timeout：它是 Qdrant 的 HTTP 超时，不是这个协程自身的超时（ASYNC109）。
    tower = TowerClient()
    client = make_client(timeout=qdrant_timeout)
    scored: list[dict] = []
    t0 = time.perf_counter()

    def search_batch(vecs) -> list[list[str]]:
        reqs = [
            models.QueryRequest(
                query=[float(x) for x in v.ravel()],
                using=DENSE_VEC,
                limit=k,
                with_payload=["item_id"],
            )
            for v in vecs
        ]
        res = client.query_batch_points(collection, requests=reqs)
        return [[(p.payload or {}).get("item_id", "") for p in r.points] for r in res]

    for i in range(0, len(rows), batch):
        chunk = rows[i : i + batch]
        vecs = await tower.encode_texts([r["query"] for r in chunk])
        for row, hits in zip(chunk, search_batch(vecs), strict=True):
            scored.append(
                {
                    "retrieved": hits,
                    "positives": row["positives"],
                    "substitutes": row.get("substitutes") or [],
                    "complements": row.get("complements") or [],
                }
            )
        print(f"  {len(scored)}/{len(rows)}", flush=True)

    metrics = aggregate_graded(scored, k)
    return {
        "collection": collection,
        "encoder": encoder_id(),
        "top_k": k,
        "eval_queries": len(scored),
        "elapsed_sec": round(time.perf_counter() - t0, 1),
        "metrics": {key: round(val, 4) for key, val in metrics.items()},
    }


def check_gate(metrics: dict[str, float], k: int, args: argparse.Namespace) -> list[str]:
    """返回未达标项（空 = 门禁通过）。三条阈值各自可选，不传即不拦。"""
    checks = [
        (f"recall@{k}", args.min_recall),
        ("mrr", args.min_mrr),
        (f"ndcg@{k}", args.min_ndcg),
    ]
    return [
        f"{name}={metrics[name]:.4f} < {floor}"
        for name, floor in checks
        if floor is not None and metrics[name] < floor
    ]


def any_floor(args: argparse.Namespace) -> bool:
    return any(x is not None for x in (args.min_recall, args.min_mrr, args.min_ndcg))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qrels", type=Path, default=QRELS_PATH, help="golden 路径（jsonl）")
    ap.add_argument("--collection", default=COLLECTION, help="Qdrant collection（默认取 .env）")
    ap.add_argument("--k", type=int, default=20, help="Top-K（线上吃进下游的条数）")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（冒烟用，0=全跑）")
    ap.add_argument("--min-recall", type=float, default=None, help="recall@K 下限，低于则退出码 1")
    ap.add_argument("--min-mrr", type=float, default=None, help="MRR 下限")
    ap.add_argument("--min-ndcg", type=float, default=None, help="NDCG@K 下限")
    ap.add_argument("--out", type=Path, default=OUT_PATH, help="报告落盘路径")
    ap.add_argument("--batch", type=int, default=ENCODE_BATCH, help="每批 query 数")
    ap.add_argument(
        "--timeout", type=float, default=120.0, help="Qdrant 单请求超时秒（冷索引要放宽）"
    )
    args = ap.parse_args()

    rows = load_qrels(args.qrels, args.limit)
    print(f"golden {len(rows)} 条 | collection={args.collection} | K={args.k}\n")

    report = asyncio.run(run(rows, args.collection, args.k, args.batch, args.timeout))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 商品召回 ===")
    print(f"encoder: {report['encoder']}")
    for name, val in report["metrics"].items():
        print(f"  {name:22s} {val}")
    print(f"\n已写 {args.out}")

    failed = check_gate(report["metrics"], args.k, args)
    if failed:
        print("\n[gate] 门禁未过：" + "；".join(failed))
        return 1
    print("\n[gate] 门禁通过。" if any_floor(args) else "\n（未设阈值，仅出表）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
