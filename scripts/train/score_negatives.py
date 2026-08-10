"""在 GPU 机器上跑 cross-encoder，给 ANN 挖出的候选打分——即「假负样本闸」。

**跑在 huzhou（A100），不是本机。** 本机只有 Qdrant 和数据，没有卡；97 万对走 API 太贵。

两个子命令：

``calibrate``  先标定阈值，再谈过滤。拿 ESCI 已知标注对喂给同一个 cross-encoder：``E``（人工
判定的正例）和 ``I``（人工判定的无关）各一批，看两组分数分布能不能分开、分界在哪。**不许直接
拍一个 0.5 之类的绝对阈值**——这个项目上次就栽在这（槽位 rerank 的绝对阈值门被真实标定证伪、
最后默认关掉）。标定跑完人看一眼分布再定，不自动采纳。

``score``  用定好的阈值给候选打分：高于阈值 → ``drop_false_neg``（疑似真相关但没被标注，不能
当负例）；否则 ``keep``。

用法（在 huzhou 上）::

    export HF_ENDPOINT=https://hf-mirror.com   # huggingface.co 不通，镜像通
    CUDA_VISIBLE_DEVICES=5 python score_negatives.py calibrate --pairs calib_pairs.jsonl
    CUDA_VISIBLE_DEVICES=5 python score_negatives.py score --input neg_ann_candidates.jsonl \\
        --output neg_ann_scored.jsonl --threshold 0.55
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "BAAI/bge-reranker-v2-m3"
MAX_LEN = 320  # 商品文本 p95 才 223 字符，320 token 足够，短了省显存也快
BATCH = 256


def load_model(name: str = MODEL_NAME) -> tuple:
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name, torch_dtype=torch.float16)
    model.eval().cuda()
    return tok, model


@torch.inference_mode()
def score_pairs(tok, model, pairs: list[tuple[str, str]]) -> list[float]:
    """(query, doc) → 相关性分（sigmoid 归一到 0-1，便于定阈值）。"""
    out: list[float] = []
    for i in range(0, len(pairs), BATCH):
        batch = pairs[i : i + BATCH]
        enc = tok(
            [q for q, _ in batch],
            [d for _, d in batch],
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        ).to("cuda")
        logits = model(**enc).logits.view(-1).float()
        out.extend(torch.sigmoid(logits).cpu().tolist())
    return out


def cmd_calibrate(args: argparse.Namespace) -> None:
    rows = [json.loads(x) for x in Path(args.pairs).open(encoding="utf-8") if x.strip()]
    tok, model = load_model(args.model)
    groups: dict[str, list[float]] = {}
    for label in ("E", "S", "C", "I"):
        pairs = [(r["query"], r["text"]) for r in rows if r["label"] == label]
        if not pairs:
            continue
        groups[label] = score_pairs(tok, model, pairs)

    print("\n=== 各标注档位的 cross-encoder 分数分布 ===")
    print(f"{'档':<4}{'n':>7}{'p10':>9}{'中位':>9}{'p90':>9}")
    for label, scores in groups.items():
        s = sorted(scores)
        print(
            f"{label:<4}{len(s):>7}{s[int(len(s) * 0.1)]:>9.4f}"
            f"{st.median(s):>9.4f}{s[int(len(s) * 0.9)]:>9.4f}"
        )
    if "E" in groups and "I" in groups:
        e, i = sorted(groups["E"]), sorted(groups["I"])
        print(f"\nE 的 p10 = {e[int(len(e) * 0.1)]:.4f}（低于它的正例只有 10%）")
        print(f"I 的 p90 = {i[int(len(i) * 0.9)]:.4f}（高于它的无关项只有 10%）")
        print("\n阈值取在这两个数之间才有意义；两者若倒挂，说明这把尺子分不开，闸不该上。")
    Path(args.out).write_text(
        json.dumps({k: v for k, v in groups.items()}, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n原始分数已写 {args.out}")


def cmd_score(args: argparse.Namespace) -> None:
    rows = [json.loads(x) for x in Path(args.input).open(encoding="utf-8") if x.strip()]
    tok, model = load_model(args.model)
    total = kept = dropped = 0

    with Path(args.output).open("w", encoding="utf-8") as out:
        for n, row in enumerate(rows, 1):
            cands = row["candidates"]
            if cands:
                scores = score_pairs(tok, model, [(row["query"], c["text"]) for c in cands])
                for c, s in zip(cands, scores, strict=True):
                    c["rerank_score"] = round(float(s), 4)
                    drop = s >= args.threshold
                    c["verdict"] = "drop_false_neg" if drop else "keep"
                    dropped += drop
                    kept += not drop
                total += len(cands)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            if n % 2000 == 0:
                print(f"  {n}/{len(rows)}  keep {kept} / drop {dropped}", flush=True)

    rate = dropped / total if total else 0.0
    print(f"\n候选 {total}，keep {kept}，闸掉 {dropped}（{rate:.2%}），阈值 {args.threshold}")
    print("闸掉率是要交的数：它量化了不做假负过滤会往训练集里掺多少毒。")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap.add_argument("--model", default=MODEL_NAME, help="换自训 checkpoint 即为新模型重标定")

    c = sub.add_parser("calibrate", help="用 ESCI 标注对标定阈值")
    c.add_argument("--pairs", default="calib_pairs.jsonl")
    c.add_argument("--out", default="calib_scores.json")
    c.set_defaults(func=cmd_calibrate)

    s = sub.add_parser("score", help="给 ANN 候选打分并判定")
    s.add_argument("--input", default="neg_ann_candidates.jsonl")
    s.add_argument("--output", default="neg_ann_scored.jsonl")
    s.add_argument("--threshold", type=float, required=True)
    s.set_defaults(func=cmd_score)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
