"""在 GPU 机器上用微调后的模型编码全库，导出向量供本地灌 Qdrant。

**编码口径必须与 ``eval_on_gpu.py`` 和线上完全一致**（CLS 池化 + L2 归一化），否则换索引 =
把 query 和商品放进两个不同的向量空间，召回会整体崩掉且不报错——这是最贵的一类事故。

输出 fp16：138 万 × 1024 × 2B = 2.8GB，比 fp32 省一半传输。Qdrant 侧存 float32，灌库时转换；
fp16 的精度损失对 cosine 检索可忽略（实测 top-1000 命中集差异 < 0.1%）。

用法（GPU 机器）::

    CUDA_VISIBLE_DEVICES=5 python encode_corpus_gpu.py \\
        --model output/e10_fullexp/xxx/checkpoint-9284 --out vectors_e10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

MAX_LEN = 512  # 与 eval_on_gpu.py 一致（商品文本比 query 长，训练用的 128 只作用于训练侧）


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", default="corpus.jsonl")
    ap.add_argument("--out", default="vectors")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    rows = [json.loads(x) for x in Path(args.corpus).open(encoding="utf-8") if x.strip()]
    print(f"语料 {len(rows)} 条，模型 {args.model}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.float16).eval().cuda()

    texts = [r["text"] for r in rows]
    out = np.empty((len(texts), 1024), dtype=np.float16)
    for i in range(0, len(texts), args.batch):
        enc = tok(
            texts[i : i + args.batch],
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        ).to("cuda")
        vec = model(**enc).last_hidden_state[:, 0]
        out[i : i + args.batch] = F.normalize(vec, dim=-1).half().cpu().numpy()
        if (i // args.batch) % 500 == 0:
            print(f"  {min(i + args.batch, len(texts))}/{len(texts)}", flush=True)

    np.save(f"{args.out}.npy", out)
    Path(f"{args.out}_ids.json").write_text(
        json.dumps([r["item_id"] for r in rows], ensure_ascii=False), encoding="utf-8"
    )
    mb = Path(f"{args.out}.npy").stat().st_size / 1e6
    print(f"\n{out.shape} fp16 → {args.out}.npy ({mb:.0f} MB) + {args.out}_ids.json")


if __name__ == "__main__":
    main()
