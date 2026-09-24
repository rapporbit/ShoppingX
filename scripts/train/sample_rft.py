"""S5-RFT 步 1+2：best-of-n 采样 + reward 打分，全量候选落盘。

**为什么采样和选优拆成两个脚本**：拒绝阈值 τ 是要扫曲线的自由参数（「τ 怎么定的」是这条
工作最该被追问的地方），候选与分数全量落盘后，换阈值重选是秒级的事，不必重烧一遍 GPU。

**为什么同一趟还跑一遍 greedy**：RFT 到底有没有料，全看「采 8 条里最好那条」比「贪心解」
好多少——它量的是模型**知道但不常说**的那部分。这个差值是 RFT 的收益上限，比训练后的分数
更早告诉你值不值得训。两遍共用同一个 vLLM 实例与同一个 reward 链路，口径天然一致。

用法（GPU 机，rollout_env 的 embed:8095 / qdrant:6333 要先起）：
  CUDA_VISIBLE_DEVICES=3 python sample_rft.py \
      --model <base> --adapter <sft_ckpt> --data planner_grpo_train.jsonl --out rft_candidates.jsonl
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rollout_env import Retriever, load_reward, score_batch  # noqa: E402

# 打分按批走：12968 条候选一次性发 embedding 会把 HTTP body 顶爆，且失败要整批重来。
SCORE_BATCH = 256


def build_prompt(r: dict) -> str:
    return (
        f"<|im_start|>system\n{r['messages'][0]['content']}<|im_end|>\n"
        f"<|im_start|>user\n{r['messages'][1]['content']}<|im_end|>\n<|im_start|>assistant\n"
    )


def dims(br) -> dict:
    return {"retrieval": br.retrieval, "field": br.field_score, "format": br.fmt, "econ": br.econ}


def score_all(comps, golds, texts, retriever, reward_mod) -> list:
    out = []
    for i in range(0, len(comps), SCORE_BATCH):
        j = min(i + SCORE_BATCH, len(comps))
        out.extend(score_batch(comps[i:j], golds[i:j], texts[i:j], retriever, reward_mod))
        print(f"[score] {j}/{len(comps)}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--data", default="planner_grpo_train.jsonl")
    ap.add_argument("--out", default="rft_candidates.jsonl")
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--gen-tokens", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--gpu-util", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.data, encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]
    prompts = [build_prompt(r) for r in rows]
    golds_row = [json.loads(r["golden_json"]) for r in rows]

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=args.max_len,
        gpu_memory_utilization=args.gpu_util,
        enable_prefix_caching=True,
        enable_lora=bool(args.adapter),
        max_lora_rank=16,
    )
    lora = LoRARequest("sft", 1, args.adapter) if args.adapter else None
    retriever, reward_mod = Retriever(), load_reward()

    def run(n: int, temp: float, tag: str):
        sp = SamplingParams(
            n=n,
            temperature=temp,
            top_p=args.top_p if temp > 0 else 1.0,
            max_tokens=args.gen_tokens,
            seed=args.seed,
        )
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp, lora_request=lora, use_tqdm=True)
        t_gen = time.perf_counter() - t0
        comps, golds, texts = [], [], []
        for r, g, o in zip(rows, golds_row, outs, strict=True):
            for c in o.outputs:
                comps.append(c.text)
                golds.append(g)
                texts.append(r["text"])
        t1 = time.perf_counter()
        brs = score_all(comps, golds, texts, retriever, reward_mod)
        print(f"[{tag}] n={len(comps)} gen={t_gen:.1f}s score={time.perf_counter() - t1:.1f}s", flush=True)
        return comps, brs

    g_comps, g_brs = run(1, 0.0, "greedy")
    s_comps, s_brs = run(args.group_size, args.temperature, "sample")

    k = args.group_size
    with open(args.out, "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            cands = [
                {"text": s_comps[i * k + j], "reward": s_brs[i * k + j].total, "dims": dims(s_brs[i * k + j])}
                for j in range(k)
            ]
            rec = {
                "id": r["id"],
                "text": r["text"],
                "family": r.get("family", ""),
                "greedy": {"text": g_comps[i], "reward": g_brs[i].total, "dims": dims(g_brs[i])},
                "cands": cands,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[done] {len(rows)} 组 × {k} 候选 → {args.out}", flush=True)


if __name__ == "__main__":
    main()
