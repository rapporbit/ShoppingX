"""体检第二段（跑在 GPU 机）：给导出的候选打 cross-encoder 分，两种 query 形态各打一遍。

**只出分、不算指标。** 指标口径还会反复调（换 K、换 gain、换 recall 还是 ndcg），每调一次都
重跑 100 万对 GPU 打分是纯浪费。分数落盘一次，本地 ``eval_rerank.py`` 想怎么算怎么算。

**两种 query 形态一次打完：**

- ``intent``  —— 用户完整意图句（ESCI 原 query，如 ``"$150 laptop not previews"``）
- ``category`` —— 粗品类词，模拟线上 ``item_picker`` 实际传给 reranker 的东西

这两列分数的差，就是「线上 rerank 的 query 形态值不值得改」的全部证据。

**模型固定为 ``BAAI/bge-reranker-v2-m3``**：它既是线上 siliconflow 跑的那个，也是 M21 假负闸
用过的那个（``score_negatives.py``）。体检的对照锚点必须是**线上现状**，不是某个更强的模型——
否则量出来的提升里，分不清哪些来自换模型、哪些来自换用法。

用法（在 huzhou 上，候选文件先 scp 过去）::

    export HF_ENDPOINT=https://hf-mirror.com
    CUDA_VISIBLE_DEVICES=5 python rerank_candidates.py \\
        --input rerank_candidates.jsonl --output rerank_scores.jsonl --depth 1000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "BAAI/bge-reranker-v2-m3"
MAX_LEN = 320  # 与 score_negatives.py 一致：商品文本 p95 才 223 字符
BATCH = 256


def load_model(name: str = MODEL_NAME) -> tuple:
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name, torch_dtype=torch.float16)
    model.eval().cuda()
    return tok, model


@torch.inference_mode()
def score_pairs(tok, model, query: str, docs: list[str]) -> list[float]:
    """一条 query 对一批候选打分（sigmoid 归一 0-1，与假负闸口径一致）。"""
    out: list[float] = []
    for i in range(0, len(docs), BATCH):
        chunk = docs[i : i + BATCH]
        enc = tok(
            [query] * len(chunk),
            chunk,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        ).to("cuda")
        logits = model(**enc).logits.view(-1).float()
        out.extend(torch.sigmoid(logits).cpu().tolist())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="rerank_candidates.jsonl")
    ap.add_argument("--output", default="rerank_scores.jsonl")
    ap.add_argument("--depth", type=int, default=1000, help="每条 query 只给前 N 个候选打分")
    ap.add_argument("--no-category", action="store_true", help="跳过品类词形态（省一半时间）")
    ap.add_argument("--model", default=MODEL_NAME, help="换成自训 checkpoint 路径即可做 A/B")
    args = ap.parse_args()

    rows = [json.loads(x) for x in Path(args.input).open(encoding="utf-8") if x.strip()]
    print(f"模型：{args.model}")
    tok, model = load_model(args.model)
    t0 = time.time()
    n_pairs = 0

    with Path(args.output).open("w", encoding="utf-8") as out:
        for n, row in enumerate(rows, 1):
            cands = row["candidates"][: args.depth]
            docs = [c["text"] for c in cands]
            rec: dict = {
                "query_id": row["query_id"],
                "item_ids": [c["item_id"] for c in cands],
                "intent": score_pairs(tok, model, row["query"], docs),
            }
            n_pairs += len(docs)
            # 挖负例那份数据带 pos 字段：正例也打一遍分，好让下游做 query 内**相对**闸。
            # 绝对阈值在这份数据上不可用——标定实测正例(E)分数中位仅 .55、p10 低到 .003，
            # 拍一个 0.5 会连人工标注的 S/C 一起闸掉（S p90=.67、C p90=.89），
            # 而那些正是最该留的 hard negative。
            pos = row.get("pos") or []
            if pos:
                rec["pos_ids"] = [p["item_id"] for p in pos]
                rec["pos_scores"] = score_pairs(tok, model, row["query"], [p["text"] for p in pos])
                n_pairs += len(pos)
            cat_q = "" if args.no_category else (row.get("category_query") or "")
            # 品类词缺失（top-K 内一个标注正例都没召回）→ 该条不参与形态 A/B，本地按 null 跳过
            if cat_q:
                rec["category"] = score_pairs(tok, model, cat_q, docs)
                n_pairs += len(docs)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if n % 50 == 0:
                rate = n_pairs / (time.time() - t0)
                print(f"  {n}/{len(rows)}  {n_pairs} 对  {rate:.0f} 对/秒", flush=True)

    dt = time.time() - t0
    print(f"\n打分完成：{n_pairs} 对，{dt / 60:.1f} 分钟，{n_pairs / dt:.0f} 对/秒")
    print(f"已写 {args.output}")


if __name__ == "__main__":
    main()
