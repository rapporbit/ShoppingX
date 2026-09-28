"""S5-RFT 步 4：多个 adapter 在同一数据集上的 greedy 对照，一次加载跑完。

**为什么不复用 `rollout_env --selftest` 跑五遍**：那样要起五次 vLLM（每次 40~60s 纯加载），
而且五次是五个进程、五份 KV，谁也不保证显存碎片一样。这里一次加载基座、循环换 `LoRARequest`，
五个模型吃同一个引擎、同一批 prompt、同一个 reward 链路——**对照的唯一变量就是权重本身**。

口径与 GRPO 的 dev 评测逐项对齐（greedy / temperature 0 / 同一份 planner_grpo_*.jsonl），
所以这里出的数可以直接和 SFT 0.7901、r3 0.8134、r5 0.8302 并排放。

用法：
  python eval_rft.py --model <base> --data planner_grpo_dev.jsonl \
      --adapters base= sft=/path/ckpt-306 rft=/path/rft grpo_r3=/path/r3 --out eval_dev.json
"""

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rollout_env import Retriever, extract_json, load_reward, score_batch  # noqa: E402

SCORE_BATCH = 256


def build_prompt(r: dict) -> str:
    return (
        f"<|im_start|>system\n{r['messages'][0]['content']}<|im_end|>\n"
        f"<|im_start|>user\n{r['messages'][1]['content']}<|im_end|>\n<|im_start|>assistant\n"
    )


def mean_or_none(vals: list) -> tuple[float | None, int]:
    kept = [v for v in vals if v is not None]
    return (round(st.mean(kept), 4) if kept else None, len(kept))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapters", nargs="+", required=True, help="name=path，path 空 = 基座")
    ap.add_argument("--data", default="planner_grpo_dev.jsonl")
    ap.add_argument("--out", default="eval_rft.json")
    ap.add_argument("--gen-tokens", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--gpu-util", type=float, default=0.6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-lora-rank", type=int, default=32)
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.data, encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]
    prompts = [build_prompt(r) for r in rows]
    golds = [json.loads(r["golden_json"]) for r in rows]
    texts = [r["text"] for r in rows]

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    pairs = [(s.split("=", 1)[0], s.split("=", 1)[1]) for s in args.adapters]
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=args.max_len,
        gpu_memory_utilization=args.gpu_util,
        enable_prefix_caching=True,
        enable_lora=any(p for _, p in pairs),
        # 上限给 32：同一个引擎要同时装 r16（SFT/RFT/GRPO 各轮）和 r32（容量对照组）的 adapter，
        # 写死 16 会让 r32 直接加载失败。设成上限不影响 r16 那些的行为。
        max_lora_rank=args.max_lora_rank,
    )
    sp = SamplingParams(n=1, temperature=0.0, top_p=1.0, max_tokens=args.gen_tokens)
    retriever, reward_mod = Retriever(), load_reward()

    results = {}
    for idx, (name, path) in enumerate(pairs, start=1):
        lora = LoRARequest(name, idx, path) if path else None
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp, lora_request=lora, use_tqdm=False)
        comps = [o.outputs[0].text for o in outs]
        t_gen = time.perf_counter() - t0

        brs = []
        for i in range(0, len(comps), SCORE_BATCH):
            j = min(i + SCORE_BATCH, len(comps))
            brs.extend(score_batch(comps[i:j], golds[i:j], texts[i:j], retriever, reward_mod))

        totals = [b.total for b in brs]
        parse_ok = sum(1 for c in comps if extract_json(c) is not None)
        retr, n_retr = mean_or_none([b.retrieval for b in brs])
        fld, _ = mean_or_none([b.field_score for b in brs])
        fmt, _ = mean_or_none([b.fmt for b in brs])
        econ, n_econ = mean_or_none([b.econ for b in brs])
        results[name] = {
            "reward": round(st.mean(totals), 4),
            "parse 成功率": round(parse_ok / len(comps), 4),
            "retrieval": retr,
            "retrieval 参与": n_retr,
            "field": fld,
            "format": fmt,
            "econ": econ,
            "econ 参与": n_econ,
            "生成 s": round(t_gen, 1),
        }
        print(f"[{name}] {json.dumps(results[name], ensure_ascii=False)}", flush=True)

    payload = {"数据集": args.data, "样本数": len(rows), "口径": "greedy / temperature 0", "结果": results}
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] → {args.out}", flush=True)


if __name__ == "__main__":
    main()
