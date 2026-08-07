"""在 GPU 机器上评测 embedding 模型：全库编码 + 暴力检索 + 与本地基线同口径的指标。

**自包含**：huzhou 上没有本仓库的 ``app`` 包，指标函数在这里重写一份，口径必须与
``scripts/train/eval_recall.py`` 逐条对齐（K=1000、分级 gain E3/S2/C1、Recall 分母取库内正例数），
否则训练前后的数字没法比。

不建 Qdrant：137 万 × 1024 的相似度矩阵乘对 A100 是小活，分块 topk 比架一套向量库省事。

用法::

    CUDA_VISIBLE_DEVICES=5 python eval_on_gpu.py \\
        --model output/bge-m3-esci-v3/vX-xxx/checkpoint-xxx \\
        --corpus corpus.jsonl --qrels esci_eval_qrels.jsonl --out eval_after.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

TOP_K = 1000
GAIN = {"pos": 3.0, "sub": 2.0, "comp": 1.0}
MAX_LEN = 512


def dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def score_one(hits: list[str], row: dict) -> dict[str, float]:
    pos, sub = set(row["positives"]), set(row.get("substitutes") or [])
    comp = set(row.get("complements") or [])

    def gain_of(i: str) -> float:
        return (
            GAIN["pos"]
            if i in pos
            else GAIN["sub"]
            if i in sub
            else GAIN["comp"]
            if i in comp
            else 0.0
        )

    top100 = set(hits[:100])
    rr = next((1.0 / (i + 1) for i, h in enumerate(hits[:100]) if h in pos), 0.0)
    ideal = sorted(
        [GAIN["pos"]] * len(pos) + [GAIN["sub"]] * len(sub) + [GAIN["comp"]] * len(comp),
        reverse=True,
    )[:100]
    return {
        "recall@1000": len(pos & set(hits)) / len(pos),
        "recall@100": len(pos & top100) / len(pos),
        "recall@20": len(pos & set(hits[:20])) / len(pos),
        "mrr@100": rr,
        "ndcg@100": (dcg([gain_of(h) for h in hits[:100]]) / dcg(ideal)) if ideal else 0.0,
        "complement_hits@100": float(len(comp & top100)),
        "has_complement": 1.0 if comp else 0.0,
    }


@torch.inference_mode()
def encode(tok, model, texts: list[str], batch: int, tag: str) -> torch.Tensor:
    """CLS 池化 + L2 归一化（BGE 系列的标准用法），返回 fp16 张量。"""
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(
            texts[i : i + batch],
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        ).to("cuda")
        vec = model(**enc).last_hidden_state[:, 0]
        out.append(F.normalize(vec, dim=-1).half())
        if (i // batch) % 200 == 0:
            print(f"  [{tag}] {min(i + batch, len(texts))}/{len(texts)}", flush=True)
    return torch.cat(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", default="corpus.jsonl")
    ap.add_argument("--qrels", default="esci_eval_qrels.jsonl")
    ap.add_argument("--out", default="eval_after.json")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    corpus = [json.loads(x) for x in Path(args.corpus).open(encoding="utf-8") if x.strip()]
    qrels = [json.loads(x) for x in Path(args.qrels).open(encoding="utf-8") if x.strip()]
    ids = [c["item_id"] for c in corpus]
    print(f"语料 {len(corpus)}，评测 query {len(qrels)}")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.float16).eval().cuda()

    doc = encode(tok, model, [c["text"] for c in corpus], args.batch, "corpus")
    qry = encode(tok, model, [r["query"] for r in qrels], args.batch, "query")

    totals: dict[str, float] = {}
    for i in range(0, len(qry), 64):  # 分块算相似度，避免一次性 11364×137万 的矩阵
        sims = qry[i : i + 64] @ doc.T
        _, idx = sims.topk(TOP_K, dim=-1)
        for row, hit_idx in zip(qrels[i : i + 64], idx.tolist(), strict=True):
            for k, v in score_one([ids[j] for j in hit_idx], row).items():
                totals[k] = totals.get(k, 0.0) + v
        print(f"  [score] {min(i + 64, len(qry))}/{len(qry)}", flush=True)

    n = len(qrels)
    n_comp = totals["has_complement"] or 1.0
    report = {
        "model": args.model,
        "eval_queries": n,
        "corpus_size": len(corpus),
        "top_k": TOP_K,
        **{
            k: round(totals[k] / n, 4)
            for k in ("recall@1000", "recall@100", "recall@20", "mrr@100", "ndcg@100")
        },
        "queries_with_complement": int(totals["has_complement"]),
        "complement_hits_per_query@100": round(totals["complement_hits@100"] / n_comp, 4),
    }
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== 结果 ===")
    for k, v in report.items():
        print(f"{k:30} {v}")


if __name__ == "__main__":
    main()
