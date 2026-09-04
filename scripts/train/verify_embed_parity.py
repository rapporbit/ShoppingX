"""S1 环境验收：**GPU 机上的本地 bge-m3 与线上 API 的 bge-m3 是不是同一把尺子**。

为什么非验不可：GRPO 的 reward 有 45% 来自「keywords 真打进 Qdrant 搜到了什么」。rollout 一步
要编码几十条 query，走线上 embedding API 既慢又计费，所以 GPU 机上改用本地 bge-m3 推理。可
`globex_items` 那 138 万条商品向量**是线上 API 编的**——query 侧换个实现，两者就可能不在同一个
向量空间里。这种事故不会报错，只会让 reward 静默失真：训练时以为搜得挺准，切回线上全变样。

**同名不等于同权重，更不等于同口径**：pooling（CLS vs mean）、是否 L2 归一化、是否加指令前缀，
任一处不同都会挪动向量。所以这里不比「配置写得一样吗」，直接比**结果**：
1. 同一条 query 两边编码的余弦相似度；
2. 两个向量各自去搜同一个 Qdrant，top-k 命中集合的重合度（**这才是 reward 真正消费的东西**）。

用法（先把 GPU 机的 embed server 转发到本地）::

    ssh -N -L 18095:localhost:8095 huzhouet &
    uv run python scripts/train/verify_embed_parity.py --local-url http://localhost:18095/v1 \\
        --limit 30 --out data/train/embed_parity.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from pathlib import Path

import httpx
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

from app.recall.qdrant_store import COLLECTION, get_recall_client  # noqa: E402
from app.recall.towers import get_tower_client  # noqa: E402

GOLDEN = PROJECT_ROOT / "data" / "train" / "planner_golden.jsonl"


async def local_encode(url: str, text: str) -> list[float]:
    async with httpx.AsyncClient(timeout=60) as cli:
        r = await cli.post(f"{url}/embeddings", json={"input": text})
        r.raise_for_status()
        return r.json()["data"][0]["embedding"]


def overlap(a: list[str], b: list[str], k: int) -> float:
    sa, sb = set(a[:k]), set(b[:k])
    return len(sa & sb) / k if k else 1.0


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-url", default="http://localhost:18095/v1")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--out", default="data/train/embed_parity.json")
    args = ap.parse_args()

    rows = [json.loads(x) for x in GOLDEN.open(encoding="utf-8") if x.strip()][: args.limit]
    queries = [r["text"] for r in rows]

    recall = get_recall_client()
    tower = get_tower_client()
    cos_list, ov_k, ov_5, ov_1 = [], [], [], []
    for q in queries:
        v_api = np.asarray(await tower.encode_query(q), dtype=np.float32)
        v_gpu = np.asarray(await local_encode(args.local_url, q), dtype=np.float32)
        if v_api.shape != v_gpu.shape:
            raise SystemExit(f"❌ 维度不一致：API {v_api.shape} vs GPU {v_gpu.shape}")
        cos = float(v_api @ v_gpu / (np.linalg.norm(v_api) * np.linalg.norm(v_gpu)))
        cos_list.append(cos)
        t_api = [c.item_id for c in recall.search(v_api, top_k=args.top_k)]
        t_gpu = [c.item_id for c in recall.search(v_gpu, top_k=args.top_k)]
        ov_k.append(overlap(t_api, t_gpu, args.top_k))
        ov_5.append(overlap(t_api, t_gpu, 5))
        ov_1.append(overlap(t_api, t_gpu, 1))

    def _s(xs: list[float]) -> dict:
        return {
            "均值": round(statistics.mean(xs), 4),
            "最小": round(min(xs), 4),
            "中位": round(statistics.median(xs), 4),
        }

    report = {
        "样本数": len(queries),
        "collection": COLLECTION,
        "余弦相似度(API vs 本地GPU)": _s(cos_list),
        f"top{args.top_k} 重合率": _s(ov_k),
        "top5 重合率": _s(ov_5),
        "top1 一致率": round(statistics.mean(ov_1), 4),
        "判定": (
            "可用本地编码"
            if statistics.mean(ov_k) >= 0.95 and min(cos_list) >= 0.98
            else "**不等价**：rollout 的 query 编码必须走线上同一实现，否则 reward 失真"
        ),
    }
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
