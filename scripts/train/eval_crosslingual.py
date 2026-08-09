"""跨语言体检：同义中英 query 对的 cosine 通过率（refdocs 04-2 §6.3）。

**这是训完模型第一件该跑的事**，不是最后一件。ESCI 只有 en/es/jp，全参微调把中文能力训崩是
真实风险，而 ShoppingX 线上是中文入口——中文崩了，英文侧涨 6% 也没有意义。

比 C-MTEB 轻的地方：不需要标注集、不需要建库、不需要全库编码，几百对 query 秒级出结果。
所以它适合当**闸门**——先过这一关，再谈召回指标的排名。

阈值（refdocs §6.3）：cosine ≥ 0.80 的对数占比 ≥ 75%。

跑法与 ``eval_on_gpu.py`` 一致，必须在 GPU 机器上对着 checkpoint 跑::

    CUDA_VISIBLE_DEVICES=5 python eval_crosslingual.py \\
        --model output/v1-e2/xxx/checkpoint-1177 --pairs xlingual_pairs.jsonl \\
        --out results/xling_v1.json

对照锚点同样是原始模型：``--model BAAI/bge-m3``。**只看绝对值会误判**——BGE-M3 原版的通过率
本身也不是 100%，翻译质量、query 歧义都会拉低它。真正该看的是「微调后相对原版掉了多少」。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

THRESHOLD = 0.80
MAX_LEN = 128  # query 都很短，128 足够；与训练时的 max_length 对齐


@torch.inference_mode()
def encode(tok, model, texts: list[str], batch: int) -> torch.Tensor:
    """CLS 池化 + L2 归一化，与 eval_on_gpu.py 逐条一致。"""
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
        out.append(F.normalize(vec, dim=-1).float())
    return torch.cat(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pairs", default="xlingual_pairs.jsonl")
    ap.add_argument("--out", default="results/xling.json")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    rows = [json.loads(x) for x in Path(args.pairs).open(encoding="utf-8") if x.strip()]
    print(f"同义对 {len(rows)} 组，模型 {args.model}")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.float16).eval().cuda()

    en = encode(tok, model, [r["en"] for r in rows], args.batch)
    zh = encode(tok, model, [r["zh"] for r in rows], args.batch)
    cos = (en * zh).sum(dim=-1)

    # 中文自检索：中文 query 在 1500 条英文 query 里能不能捞回自己的同义英文版。
    # 这是比 cosine 更严的信号——cosine 只看绝对距离，它看的是**相对排序**有没有塌。
    rank1 = ((zh @ en.T).argmax(dim=-1) == torch.arange(len(rows), device="cuda")).float().mean()

    report = {
        "model": args.model,
        "pairs": len(rows),
        "threshold": THRESHOLD,
        "pass_rate": round((cos >= THRESHOLD).float().mean().item(), 4),
        "cosine_mean": round(cos.mean().item(), 4),
        "cosine_p10": round(cos.quantile(0.10).item(), 4),
        "cosine_min": round(cos.min().item(), 4),
        "zh2en_top1": round(rank1.item(), 4),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 跨语言体检 ===")
    for k, v in report.items():
        print(f"{k:16} {v}")
    verdict = "通过" if report["pass_rate"] >= 0.75 else "不通过"
    print(f"\n判定（阈值 {THRESHOLD} / 通过率 75%）：{verdict}")

    worst = sorted(zip(rows, cos.tolist(), strict=True), key=lambda x: x[1])[:5]
    print("\n最差 5 对（先看是不是翻译本身的问题，别急着赖模型）：")
    for r, c in worst:
        print(f"  {c:.3f}  {r['en'][:44]:<46}| {r['zh']}")


if __name__ == "__main__":
    main()
