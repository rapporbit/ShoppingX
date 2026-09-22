"""横向对照不同 embedding 模型在商品召回上的效果（ESCI 金标，同品类干扰子集，numpy 暴力检索）。

**为什么不用全库 Qdrant**：换模型 = 换向量空间，138 万点每个模型重编一遍要 1.5 亿 token，
而且本机 6333 只是 gcjp 生产库的只读隧道，不能往里灌新 collection。改成子集暴力检索：

- 抽 N 条 query（固定 seed），取它们全部标注商品（E/S/C）作「必含」；
- 干扰项**按品类配对**抽：正例属于哪个 category，就从该 category 的未标注商品抽（每 query
  上限 ``--distractor-cap``），再撒少量随机项。随机干扰对模型太容易、白花钱；同品类干扰才是
  线上 90.8% 未标注同类商品把正例挤出 top-20 的那种难度。
- 子集只做**模型间相对比较**，数值不能和全库口径的 R@20≈0.29 直接对齐。

编码结果落 ``data/eval/bench/emb_<name>.npy``（float16）+ ids，换指标/换 K 不重跑 API。
指标口径与 ``run_product_recall.py`` 逐字一致（分级 gain、分母取库内正例）。

用法::

    uv run python scripts/eval/bench_embed_models.py --build            # 只建子集
    uv run python scripts/eval/bench_embed_models.py --models bge_m3,qwen37_flash
    uv run python scripts/eval/bench_embed_models.py --models qwen37_flash --smoke 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import httpx
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from app.eval.recall_metrics import aggregate_graded  # noqa: E402
from app.recall.text import embed_text  # noqa: E402
from app.utils.clean import CleanItem  # noqa: E402

QRELS_PATH = PROJECT_ROOT / "data" / "train" / "esci_eval_qrels.jsonl"
CLEAN_DIR = PROJECT_ROOT / "data" / "platforms" / "clean" / "by_platform"
BENCH_DIR = PROJECT_ROOT / "data" / "eval" / "bench"
DOCS_PATH = BENCH_DIR / "subset_docs.jsonl"
QUERIES_PATH = BENCH_DIR / "subset_queries.jsonl"
REPORT_PATH = BENCH_DIR / "embed_report.json"

SEED = 20260922
KS = (20, 100, 1000)

# 模型登记表：base_url / api_key 都从 .env 取；dimensions=None 表示不传（原生维度）。
# batch 是单请求条数上限（百炼 embedding 接口一次最多 10 条；siliconflow bge-m3 用 32 保守值）。
MODELS: dict[str, dict] = {
    "bge_m3": {
        "model": "BAAI/bge-m3",
        "base_url_env": "EMBED_BASE_URL",
        "api_key_env": "EMBED_API_KEY",
        "dimensions": None,
        "batch": 32,
    },
    "qwen37_flash": {
        "model": "qwen3.7-text-embedding-flash",
        "base_url_env": "OPENAI_BASE_URL",
        "api_key_env": "OPENAI_API_KEY",
        "dimensions": 1024,
        "batch": 10,
    },
    "qwen37_std": {
        "model": "qwen3.7-text-embedding",
        "base_url_env": "OPENAI_BASE_URL",
        "api_key_env": "OPENAI_API_KEY",
        "dimensions": 1024,
        "batch": 10,
    },
}


def load_qrels() -> list[dict]:
    rows = [json.loads(x) for x in QRELS_PATH.open(encoding="utf-8") if x.strip()]
    return [r for r in rows if r.get("positives")]


def load_corpus() -> tuple[dict[str, str], dict[str, str]]:
    """全库 item_id -> (embed_text, category)。1.38M 行约 1 分钟，只在 --build 时读。"""
    texts: dict[str, str] = {}
    cats: dict[str, str] = {}
    for path in sorted(CLEAN_DIR.glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                item = CleanItem.model_validate_json(line)
                t = embed_text(item)
                if t and item.item_id not in texts:
                    texts[item.item_id] = t
                    cats[item.item_id] = item.category or ""
        print(f"  读 {path.name}: 累计 {len(texts)}", flush=True)
    return texts, cats


def build_subset(n_queries: int, distractor_cap: int, n_random: int) -> None:
    """抽 query + 同品类干扰，落 subset_docs / subset_queries。"""
    rng = random.Random(SEED)
    rows = load_qrels()
    rng.shuffle(rows)
    picked = rows[:n_queries]
    texts, cats = load_corpus()

    labeled: set[str] = set()
    for r in picked:
        for key in ("positives", "substitutes", "complements"):
            labeled.update(i for i in r.get(key) or [] if i in texts)
    # 过滤掉库内不存在文本的标注 id（理论上 qrels 已按库内过滤，这里再保险一次）
    for r in picked:
        for key in ("positives", "substitutes", "complements"):
            r[key] = [i for i in r.get(key) or [] if i in texts]
    picked = [r for r in picked if r["positives"]]

    by_cat: dict[str, list[str]] = defaultdict(list)
    for item_id, cat in cats.items():
        if item_id not in labeled:
            by_cat[cat].append(item_id)
    for ids in by_cat.values():
        rng.shuffle(ids)

    distractors: set[str] = set()
    cursor: dict[str, int] = defaultdict(int)
    for r in picked:
        quota = min(distractor_cap, 5 * len(r["positives"]) + 10)
        pos_cats = [cats[i] for i in r["positives"]]
        for cat in sorted(set(pos_cats)):
            pool = by_cat.get(cat, [])
            take = quota // len(set(pos_cats)) + 1
            start = cursor[cat]
            distractors.update(pool[start : start + take])
            cursor[cat] = start + take
    all_ids = list(texts)
    rng.shuffle(all_ids)
    for item_id in all_ids:
        if n_random <= 0:
            break
        if item_id not in labeled and item_id not in distractors:
            distractors.add(item_id)
            n_random -= 1

    doc_ids = sorted(labeled | distractors)
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    with DOCS_PATH.open("w", encoding="utf-8") as f:
        for item_id in doc_ids:
            f.write(
                json.dumps({"item_id": item_id, "text": texts[item_id]}, ensure_ascii=False) + "\n"
            )
    with QUERIES_PATH.open("w", encoding="utf-8") as f:
        for r in picked:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(
        f"子集：query {len(picked)} / 标注商品 {len(labeled)} / 同品类+随机干扰 {len(distractors)}"
        f" / 文档合计 {len(doc_ids)} → {DOCS_PATH}"
    )


async def encode(spec: dict, texts: list[str], concurrency: int = 8) -> tuple[np.ndarray, dict]:
    """OpenAI 兼容 /embeddings 批量编码；返回 (向量, 统计)。统计含 token 数与调用延迟 p50。"""
    base = os.environ[spec["base_url_env"]].rstrip("/")
    key = os.environ[spec["api_key_env"]]
    sem = asyncio.Semaphore(concurrency)
    out: list[np.ndarray | None] = [None] * len(texts)
    stats = {"tokens": 0, "calls": 0, "latencies": []}

    async def one(client: httpx.AsyncClient, start: int, chunk: list[str]) -> None:
        payload: dict = {"model": spec["model"], "input": chunk}
        if spec["dimensions"]:
            payload["dimensions"] = spec["dimensions"]
        async with sem:
            for attempt in range(5):
                t0 = time.perf_counter()
                try:
                    resp = await client.post(
                        f"{base}/embeddings",
                        json=payload,
                        headers={"Authorization": f"Bearer {key}"},
                        timeout=60,
                    )
                    if resp.status_code == 429 or resp.status_code >= 500:
                        raise httpx.HTTPStatusError("retry", request=resp.request, response=resp)
                    resp.raise_for_status()
                    break
                except (httpx.HTTPError, httpx.TransportError):
                    if attempt == 4:
                        raise
                    await asyncio.sleep(1.5 * (attempt + 1))
            body = resp.json()
            stats["latencies"].append(time.perf_counter() - t0)
            stats["calls"] += 1
            stats["tokens"] += int((body.get("usage") or {}).get("total_tokens") or 0)
            # 按返回顺序落位，不信 item["index"]：百炼接口批量返回时 index 全是 0（2026-09-22 实测）
            for pos, item in enumerate(body["data"]):
                out[start + pos] = np.asarray(item["embedding"], dtype=np.float32)

    async with httpx.AsyncClient() as client:
        b = spec["batch"]
        tasks = [one(client, i, texts[i : i + b]) for i in range(0, len(texts), b)]
        done = 0
        for coro in asyncio.as_completed(tasks):
            await coro
            done += 1
            if done % 200 == 0:
                print(f"    {done}/{len(tasks)} 批", flush=True)
    mat = np.stack(out)  # type: ignore[arg-type]
    mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
    lat = sorted(stats["latencies"])
    summary = {
        "tokens": stats["tokens"],
        "calls": stats["calls"],
        "latency_p50_ms": round(1000 * lat[len(lat) // 2], 1) if lat else None,
    }
    return mat, summary


def evaluate(q_mat: np.ndarray, d_mat: np.ndarray, doc_ids: list[str], queries: list[dict]) -> dict:
    """暴力检索 + 分级指标。子集几万条，一次矩阵乘就够。"""
    sims = q_mat @ d_mat.T  # (Q, D)
    kmax = min(max(KS), len(doc_ids))
    top = np.argpartition(-sims, kmax - 1, axis=1)[:, :kmax]
    rows = []
    for qi, r in enumerate(queries):
        order = top[qi][np.argsort(-sims[qi, top[qi]])]
        rows.append(
            {
                "retrieved": [doc_ids[j] for j in order],
                "positives": r["positives"],
                "substitutes": r.get("substitutes") or [],
                "complements": r.get("complements") or [],
            }
        )
    metrics: dict[str, float] = {}
    for k in KS:
        if k > len(doc_ids):
            continue
        m = aggregate_graded(rows, k)
        metrics[f"recall@{k}"] = round(m[f"recall@{k}"], 4)
        metrics[f"ndcg@{k}"] = round(m[f"ndcg@{k}"], 4)
        if k == KS[0]:
            metrics["mrr"] = round(m["mrr"], 4)
    return metrics


async def run_model(
    name: str, docs: list[dict], queries: list[dict], query_prefix: str, cache: bool = True
) -> dict:
    spec = MODELS[name]
    tag = name + ("_instruct" if query_prefix else "")
    npy = BENCH_DIR / f"emb_{name}.npy"  # 冒烟不读不写缓存，别污染正式产物
    doc_ids = [d["item_id"] for d in docs]
    stats: dict = {}
    if cache and npy.exists():
        d_mat = np.load(npy).astype(np.float32)
        print(f"  [{name}] 复用已编码文档向量 {d_mat.shape}")
    else:
        print(f"  [{name}] 编码 {len(docs)} 条文档 …", flush=True)
        d_mat, stats = await encode(spec, [d["text"] for d in docs])
        if cache:
            np.save(npy, d_mat.astype(np.float16))
    q_mat, qstats = await encode(spec, [query_prefix + q["query"] for q in queries])
    metrics = evaluate(q_mat, d_mat, doc_ids, queries)
    return {
        "model": spec["model"],
        "dimensions": int(d_mat.shape[1]),
        "query_prefix": query_prefix,
        "doc_encode": stats,
        "query_encode": qstats,
        "metrics": metrics,
        "tag": tag,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="只构建子集")
    ap.add_argument("--n-queries", type=int, default=500)
    ap.add_argument("--distractor-cap", type=int, default=80)
    ap.add_argument("--n-random", type=int, default=5000)
    ap.add_argument("--models", default="bge_m3,qwen37_flash")
    ap.add_argument("--smoke", type=int, default=0, help="只取前 N 条 query + 其相关文档冒烟")
    ap.add_argument("--query-prefix", default="", help="query 侧前缀（instruct 变体）")
    args = ap.parse_args()

    if args.build or not DOCS_PATH.exists():
        build_subset(args.n_queries, args.distractor_cap, args.n_random)
        if args.build:
            return

    docs = [json.loads(x) for x in DOCS_PATH.open(encoding="utf-8")]
    queries = [json.loads(x) for x in QUERIES_PATH.open(encoding="utf-8")]
    if args.smoke:
        queries = queries[: args.smoke]
        keep = {
            i for q in queries for k in ("positives", "substitutes", "complements") for i in q[k]
        }
        docs = [d for d in docs if d["item_id"] in keep][:200] + [
            d for d in docs if d["item_id"] not in keep
        ][:300]

    report = json.loads(REPORT_PATH.read_text()) if REPORT_PATH.exists() and not args.smoke else {}
    for name in args.models.split(","):
        t0 = time.perf_counter()
        res = asyncio.run(
            run_model(name.strip(), docs, queries, args.query_prefix, cache=not args.smoke)
        )
        res["elapsed_sec"] = round(time.perf_counter() - t0, 1)
        res["n_queries"], res["n_docs"] = len(queries), len(docs)
        report[res["tag"]] = res
        print(f"  [{res['tag']}] {res['metrics']}  tokens={res['doc_encode'].get('tokens')}")
    if not args.smoke:
        REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"报告 → {REPORT_PATH}")


if __name__ == "__main__":
    main()
