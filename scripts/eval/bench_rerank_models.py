"""横向对照不同 reranker 在商品精排上的效果：排序口径 + 判别口径 + 延迟，一张表出结论。

**候选池固定、五个模型共用**：用线上 bge-m3 索引（隧道只读）对 N 条 ESCI query 取 top-K，
候选文本按线上 ``item_picker._searchable``（title brand category 小写）拼——量的是线上那把尺子。

**两个口径缺一不可**（教训：排序更好的模型判别反而更差，而线上 rerank 分只用于品类门）：

- 排序：候选池按 rerank 分重排后的 ndcg@8 / recall@8 / mrr，对照 ``embed`` 行（不重排）与
  ``ceiling``（池内正例占比 = 排序能到的上限）。
- 判别：``calib_pairs.jsonl`` 的 E / I 两档人工标注对，算 AUC（阈值无关）+ 线上 FLOOR=0.2
  的误杀率（E 被挡）/ 拦截率（I 被挡）。不同模型分域不同，FLOOR 与模型强耦合，换模型必重标。

两类远程契约：Cohere 同构（siliconflow bge，``RERANKER_ENDPOINT``）与百炼原生
（``input.query/documents`` + ``parameters.top_n``）。分数落 ``data/eval/bench/rerank_<name>.json``

用法::

    uv run python scripts/eval/bench_rerank_models.py --build --n-queries 300 --depth 50
    uv run python scripts/eval/bench_rerank_models.py --models bge_reranker,qwen3_rerank
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from qdrant_client import models  # noqa: E402

from app.eval.recall_metrics import aggregate_graded  # noqa: E402
from app.recall.qdrant_store import COLLECTION, DENSE_VEC, make_client  # noqa: E402
from app.recall.towers import TowerClient  # noqa: E402

QRELS_PATH = PROJECT_ROOT / "data" / "train" / "esci_eval_qrels.jsonl"
CALIB_PATH = PROJECT_ROOT / "data" / "train" / "calib_pairs.jsonl"
BENCH_DIR = PROJECT_ROOT / "data" / "eval" / "bench"
POOL_PATH = BENCH_DIR / "rerank_pool.jsonl"
REPORT_PATH = BENCH_DIR / "rerank_report.json"
DASHSCOPE_RERANK = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"

SEED = 20260922
FLOOR = float(os.environ.get("PICK_RERANK_FLOOR", "0.2"))
CUTS = (8, 20)

MODELS: dict[str, dict] = {
    "bge_reranker": {
        "kind": "cohere",
        "model": os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
        "endpoint": os.environ.get("RERANKER_ENDPOINT", ""),
        "api_key_env": "RERANKER_API_KEY",
    },
    "qwen3_rerank": {"kind": "dashscope", "model": "qwen3-rerank", "api_key_env": "OPENAI_API_KEY"},
    "qwen37_rerank": {
        "kind": "dashscope",
        "model": "qwen3.7-text-rerank",
        "api_key_env": "OPENAI_API_KEY",
    },
}


def searchable(payload: dict) -> str:
    """与 ``app/tools/item_picker._searchable`` 逐字对齐：title brand category 小写。"""
    parts = (payload.get("title", ""), payload.get("brand", ""), payload.get("category", ""))
    return " ".join(parts).lower()


async def build_pool(n_queries: int, depth: int) -> None:
    rng = random.Random(SEED)
    rows = [json.loads(x) for x in QRELS_PATH.open(encoding="utf-8") if x.strip()]
    rows = [r for r in rows if r.get("positives")]
    rng.shuffle(rows)
    rows = rows[:n_queries]
    tower, client = TowerClient(), make_client(timeout=120.0)
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    with POOL_PATH.open("w", encoding="utf-8") as f:
        for i in range(0, len(rows), 64):
            chunk = rows[i : i + 64]
            vecs = await tower.encode_texts([r["query"] for r in chunk])
            reqs = [
                models.QueryRequest(
                    query=[float(x) for x in v.ravel()],
                    using=DENSE_VEC,
                    limit=depth,
                    with_payload=["item_id", "title", "brand", "category"],
                )
                for v in vecs
            ]
            res = client.query_batch_points(COLLECTION, requests=reqs)
            for r, hits in zip(chunk, res, strict=True):
                cands = [
                    {
                        "item_id": (p.payload or {}).get("item_id", ""),
                        "text": searchable(p.payload or {}),
                    }
                    for p in hits.points
                ]
                f.write(json.dumps({**r, "candidates": cands}, ensure_ascii=False) + "\n")
            print(f"  候选池 {min(i + 64, len(rows))}/{len(rows)}", flush=True)
    await tower.aclose()
    print(f"候选池 → {POOL_PATH}")


async def score_remote(
    client: httpx.AsyncClient, spec: dict, query: str, docs: list[str]
) -> tuple[list[float], float]:
    """打一批分，返回 (与 docs 同序的分数, 本次调用秒数)。

    两类契约都回 ``results[{index, relevance_score}]``，散回原位。
    """
    key = os.environ.get(spec["api_key_env"]) or os.environ.get("OPENAI_API_KEY", "")
    headers = {"Authorization": f"Bearer {key}"}
    if spec["kind"] == "cohere":
        url = spec["endpoint"]
        payload: dict = {
            "model": spec["model"],
            "query": query,
            "documents": docs,
            "top_n": len(docs),
            "return_documents": False,
        }
    else:
        url = DASHSCOPE_RERANK
        payload = {
            "model": spec["model"],
            "input": {"query": query, "documents": docs},
            "parameters": {"top_n": len(docs), "return_documents": False},
        }
    for attempt in range(5):
        t0 = time.perf_counter()
        resp = await client.post(url, json=payload, headers=headers, timeout=60)
        if resp.status_code == 429 or resp.status_code >= 500:
            await asyncio.sleep(1.5 * (attempt + 1))
            continue
        resp.raise_for_status()
        break
    body = resp.json()
    results = body.get("results") or body.get("output", {}).get("results") or []
    scores = [0.0] * len(docs)
    for item in results:
        scores[int(item["index"])] = float(item.get("relevance_score", item.get("score", 0.0)))
    return scores, time.perf_counter() - t0


async def score_all(
    spec: dict, jobs: list[tuple[str, list[str]]], concurrency: int
) -> tuple[list, list]:
    sem = asyncio.Semaphore(concurrency)
    out: list[list[float] | None] = [None] * len(jobs)
    lats: list[float] = []

    async def one(client: httpx.AsyncClient, i: int) -> None:
        async with sem:
            out[i], sec = await score_remote(client, spec, *jobs[i])
            lats.append(sec)
            if len(lats) % 50 == 0:
                print(f"    {len(lats)}/{len(jobs)}", flush=True)

    async with httpx.AsyncClient() as client:
        await asyncio.gather(*(one(client, i) for i in range(len(jobs))))
    return out, lats  # type: ignore[return-value]


def rank_metrics(pool: list[dict], scores: list[list[float]] | None) -> dict[str, float]:
    """按分数重排（None = 保持 embed 原序）后算分级指标；另给 ceiling（池内正例占比）。"""
    rows, ceiling = [], []
    for qi, r in enumerate(pool):
        ids = [c["item_id"] for c in r["candidates"]]
        if scores is not None:
            order = sorted(range(len(ids)), key=lambda j: -scores[qi][j])
            ids = [ids[j] for j in order]
        rows.append(
            {
                "retrieved": ids,
                **{k: r.get(k) or [] for k in ("positives", "substitutes", "complements")},
            }
        )
        ceiling.append(len(set(ids) & set(r["positives"])) / len(r["positives"]))
    out: dict[str, float] = {"ceiling_recall": round(sum(ceiling) / len(ceiling), 4)}
    for k in CUTS:
        m = aggregate_graded(rows, k)
        out[f"recall@{k}"], out[f"ndcg@{k}"] = round(m[f"recall@{k}"], 4), round(m[f"ndcg@{k}"], 4)
    out["mrr"] = round(aggregate_graded(rows, CUTS[0])["mrr"], 4)
    return out


def calib_metrics(pairs: list[dict], scores: list[float]) -> dict[str, float]:
    """E 对 I 的 AUC + FLOOR 下的误杀率 / 拦截率；顺带 S、C 两档均分看分域形状。"""
    by = {"E": [], "S": [], "C": [], "I": []}
    for p, s in zip(pairs, scores, strict=True):
        by[p["label"]].append(s)
    e, i = by["E"], by["I"]
    wins = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in e for b in i)
    return {
        "auc_E_vs_I": round(wins / (len(e) * len(i)), 4) if e and i else 0.0,
        f"E_killed@{FLOOR}": round(sum(s < FLOOR for s in e) / len(e), 4) if e else 0.0,
        f"I_blocked@{FLOOR}": round(sum(s < FLOOR for s in i) / len(i), 4) if i else 0.0,
        **{f"mean_{k}": round(sum(v) / len(v), 4) for k, v in by.items() if v},
    }


async def run_model(name: str, pool: list[dict], pairs: list[dict], concurrency: int) -> dict:
    spec = MODELS[name]
    cache = BENCH_DIR / f"rerank_{name}.json"
    if cache.exists():
        saved = json.loads(cache.read_text())
        print(f"  [{name}] 复用已打分缓存")
    else:
        print(f"  [{name}] 候选池打分 {len(pool)} 条 query …", flush=True)
        pool_scores, lats = await score_all(
            spec, [(r["query"], [c["text"] for c in r["candidates"]]) for r in pool], concurrency
        )
        # 标定对按 query 归组，一次调用打一组，省调用次数
        grouped: dict[str, list[int]] = {}
        for idx, p in enumerate(pairs):
            grouped.setdefault(p["query"], []).append(idx)
        jobs = [(q, [pairs[i]["text"] for i in idxs]) for q, idxs in grouped.items()]
        group_scores, _ = await score_all(spec, jobs, concurrency)
        pair_scores = [0.0] * len(pairs)
        for (_, idxs), sc in zip(grouped.items(), group_scores, strict=True):
            for i, s in zip(idxs, sc, strict=True):
                pair_scores[i] = s
        lats.sort()
        saved = {
            "pool_scores": pool_scores,
            "pair_scores": pair_scores,
            "latency_p50_ms": round(1000 * lats[len(lats) // 2], 1),
            "latency_p90_ms": round(1000 * lats[int(len(lats) * 0.9)], 1),
        }
        cache.write_text(json.dumps(saved))
    return {
        "model": spec["model"],
        "rank": rank_metrics(pool, saved["pool_scores"]),
        "calib": calib_metrics(pairs, saved["pair_scores"]),
        "latency_p50_ms": saved["latency_p50_ms"],
        "latency_p90_ms": saved["latency_p90_ms"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="只构建候选池")
    ap.add_argument("--n-queries", type=int, default=300)
    ap.add_argument("--depth", type=int, default=50)
    ap.add_argument("--models", default="bge_reranker,qwen3_rerank,qwen37_rerank")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument(
        "--smoke", type=int, default=0, help="只取前 N 条 query + 前 N×4 对标定对，不写缓存"
    )
    args = ap.parse_args()

    if args.build or not POOL_PATH.exists():
        asyncio.run(build_pool(args.n_queries, args.depth))
        if args.build:
            return
    pool = [json.loads(x) for x in POOL_PATH.open(encoding="utf-8")]
    pairs = [json.loads(x) for x in CALIB_PATH.open(encoding="utf-8")]
    if args.smoke:
        pool, pairs = pool[: args.smoke], pairs[: args.smoke * 4]

    report = json.loads(REPORT_PATH.read_text()) if REPORT_PATH.exists() and not args.smoke else {}
    report["embed"] = {"model": "(no rerank, bge-m3 order)", "rank": rank_metrics(pool, None)}
    for name in args.models.split(","):
        name = name.strip()
        if args.smoke:  # 冒烟绕过缓存：临时改缓存路径
            cache = BENCH_DIR / f"rerank_{name}.json"
            if cache.exists():
                cache.rename(cache.with_suffix(".json.keep"))
        res = asyncio.run(run_model(name, pool, pairs, args.concurrency))
        if args.smoke:
            (BENCH_DIR / f"rerank_{name}.json").unlink(missing_ok=True)
            keep = BENCH_DIR / f"rerank_{name}.json.keep"
            if keep.exists():
                keep.rename(BENCH_DIR / f"rerank_{name}.json")
        report[name] = res
        print(f"  [{name}] rank={res['rank']} calib={res['calib']} p50={res['latency_p50_ms']}ms")
    if not args.smoke:
        REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"报告 → {REPORT_PATH}")


if __name__ == "__main__":
    main()
