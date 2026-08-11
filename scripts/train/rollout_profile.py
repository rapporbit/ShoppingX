"""S3 前置件：**rollout profile** —— 量 GRPO 采样这一侧的真实吞吐、延迟与 prefix caching 收益。

为什么这件事值得单独跑一遍再开训：GRPO 每步的时间 = rollout + 训练两段，而 planner 这个任务
**rollout 占大头**（一步要采 batch × group_size 条，训练只是一次 LoRA 反传）。开训前不知道
rollout 多快，就没法判断 group_size / batch 该取多少，也没法估一组实验要跑多久——M23 的时间
估算（3~7 小时/组）全挂在这个数上。

**prefix caching 是这里最大一笔省**：训练用的精简 prompt 里，system 段（947 字符 ≈ 全长的
大半）在所有样本间完全相同，只有末尾的「本轮用户：…」不同。开 prefix caching 后这段前缀的
KV 只算一次。本脚本用同一批 prompt 在 on/off 两种引擎上各跑一遍来量真实差值——不猜、不引用
别人的 benchmark，因为收益完全取决于自家 prompt 的前缀占比。

**自包含**：不 import 本仓库的 app 包（GPU 机上没有）。数据用 `planner_sft_dev.jsonl`，
prompt 拼法与 `eval_planner_format.py` 逐字一致——两把尺子量的必须是同一个东西。

用法（GPU 机上，RL venv）：
    python rollout_profile.py --model /path/to/Qwen3-4B-Instruct-2507 \\
        --data planner_sft_dev.jsonl --adapter ./output/sft_r16/checkpoint-xxx \\
        --num-prompts 32 --group-size 8 --out rollout_profile.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import time
from pathlib import Path

os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")


def build_prompts(data: Path, limit: int) -> list[str]:
    rows = [json.loads(x) for x in data.open(encoding="utf-8") if x.strip()]
    if limit:
        rows = rows[:limit]
    return [
        f"<|im_start|>system\n{r['messages'][0]['content']}<|im_end|>\n"
        f"<|im_start|>user\n{r['messages'][1]['content']}<|im_end|>\n<|im_start|>assistant\n"
        for r in rows
    ]


def prefix_stats(prompts: list[str], model: str) -> dict:
    """量「固定前缀占 prompt 多少」——prefix caching 的收益上界由这个数定死。

    口径是 **token 级最长公共前缀**，不是字符级：KV cache 按 token block 复用，字符级会高估
    （公共字符串的最后一个 token 常因后续字符不同而分裂）。
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    ids = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
    lcp = 0
    while all(len(x) > lcp for x in ids) and len({x[lcp] for x in ids}) == 1:
        lcp += 1
    lens = [len(x) for x in ids]
    return {
        "prompt 数": len(ids),
        "prompt token 中位数": sorted(lens)[len(lens) // 2],
        "最短/最长": [min(lens), max(lens)],
        "公共前缀 token": lcp,
        "公共前缀占比": round(lcp * len(ids) / sum(lens), 4),
        "总 prompt token": sum(lens),
    }


def gpu_credentials() -> dict:
    """凭据：跑分必须能追溯到具体哪张卡。沿用 gpu_profile.py 的做法。"""
    q = "index,name,uuid,memory.total,driver_version"
    try:
        raw = subprocess.run(
            ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
    except Exception as e:  # noqa: BLE001
        raw = f"（nvidia-smi 不可用：{e}）"
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "（未设，全部可见）")
    return {"nvidia-smi 原文": raw.splitlines(), "CUDA_VISIBLE_DEVICES": visible}


def make_llm(model: str, *, prefix_caching: bool, gpu_util: float, max_len: int,
             adapter: str, lora_rank: int):
    from vllm import LLM

    kw = dict(model=model, dtype="bfloat16", max_model_len=max_len,
              gpu_memory_utilization=gpu_util, enable_prefix_caching=prefix_caching,
              enforce_eager=False, disable_log_stats=False)
    if adapter:
        kw.update(enable_lora=True, max_lora_rank=lora_rank)
    return LLM(**kw)


def run_batch(llm, prompts: list[str], *, n: int, max_tokens: int, adapter: str,
              seed: int | None = None) -> dict:
    """跑一批（= GRPO 的一步 rollout），返回 wall / 吞吐。"""
    from vllm import SamplingParams

    sp = SamplingParams(n=n, temperature=1.0, top_p=1.0, max_tokens=max_tokens, seed=seed)
    lora = None
    if adapter:
        from vllm.lora.request import LoRARequest
        lora = LoRARequest("sft", 1, adapter)
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp, lora_request=lora, use_tqdm=False)
    wall = time.perf_counter() - t0
    gen_tok = sum(len(c.token_ids) for o in outs for c in o.outputs)
    prompt_tok = sum(len(o.prompt_token_ids) for o in outs)
    seqs = sum(len(o.outputs) for o in outs)
    return {
        "wall_s": round(wall, 3),
        "序列数": seqs,
        "生成 token": gen_tok,
        "prompt token": prompt_tok,
        "生成吞吐 tok/s": round(gen_tok / wall, 1),
        "每序列均摊 ms": round(wall * 1000 / seqs, 1),
    }


def cache_hit_rate(llm) -> float | None:
    """vLLM V1 的 prefix cache 命中率 = hits / queries（两个都是 **累计 token 计数器**，
    不是比率——直接读 hits 那个数会得到一串六位数，别被它骗了）。拿不到返回 None，如实标。"""
    try:
        counters: dict[str, float] = {}
        for m in llm.get_metrics():
            name = getattr(m, "name", "")
            if "prefix_cache" in name:
                counters[name.replace("vllm:", "").replace("gpu_", "")] = float(
                    getattr(m, "value", 0) or 0
                )
        hits, q = counters.get("prefix_cache_hits"), counters.get("prefix_cache_queries")
        if hits is not None and q:
            return round(hits / q, 4)
    except Exception:  # noqa: BLE001
        return None
    return None


def free_llm(llm) -> None:
    import torch

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)


def gpu_mem_used_mib() -> list[int]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30,
        ).stdout.split()
        return [int(x) for x in out]
    except Exception:  # noqa: BLE001
        return []


def profile_variant(args, prompts: list[str], *, prefix_caching: bool, gpu_util: float,
                    label: str) -> dict:
    """一个变体 = 一个引擎配置。**同一批 prompt 连跑两轮**：第一轮冷（KV 全新算），第二轮热
    （前缀已在 cache 里）。GRPO 多个 epoch 会反复见到同一批 prompt，热轮才是稳态的真实成本。"""
    llm = make_llm(args.model, prefix_caching=prefix_caching, gpu_util=gpu_util,
                   max_len=args.max_len, adapter=args.adapter, lora_rank=args.lora_rank)
    cold = run_batch(llm, prompts, n=args.group_size, max_tokens=args.gen_tokens,
                     adapter=args.adapter, seed=args.seed)
    warm = run_batch(llm, prompts, n=args.group_size, max_tokens=args.gen_tokens,
                     adapter=args.adapter, seed=args.seed)
    single = [
        run_batch(llm, prompts[i: i + 1], n=args.group_size, max_tokens=args.gen_tokens,
                  adapter=args.adapter, seed=args.seed)["wall_s"]
        for i in range(min(5, len(prompts)))
    ]
    res = {
        "prefix_caching": prefix_caching,
        "gpu_memory_utilization": gpu_util,
        "冷轮": cold,
        "热轮": warm,
        "单 prompt×group 的 wall_s（5 次）": single,
        "单条 rollout 延迟中位 s": round(sorted(single)[len(single) // 2], 3),
        "prefix cache 命中率": cache_hit_rate(llm),
        "显存占用 MiB": gpu_mem_used_mib(),
    }
    free_llm(llm)
    print(f"[{label}] 冷 {cold['wall_s']}s / 热 {warm['wall_s']}s / "
          f"吞吐 {warm['生成吞吐 tok/s']} tok/s")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="本地 snapshot 绝对路径（传 HF 名会重下）")
    ap.add_argument("--adapter", default="", help="S2 的 LoRA 目录；不传 = 基座")
    ap.add_argument("--data", default="planner_sft_dev.jsonl")
    ap.add_argument("--num-prompts", type=int, default=32, help="一步 rollout 的 prompt 数")
    ap.add_argument("--group-size", type=int, default=8, help="GRPO 组大小 = 每 prompt 采样数")
    ap.add_argument("--gen-tokens", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--colocate-util", type=float, default=0.45,
                    help="模拟与训练共卡时留给 rollout 的显存比例；设 0 跳过该组")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="rollout_profile.json")
    args = ap.parse_args()

    prompts = build_prompts(Path(args.data), args.num_prompts)
    pstats = prefix_stats(prompts, args.model)
    print(json.dumps(pstats, ensure_ascii=False, indent=2))

    variants = [
        ("cache_on", True, args.gpu_util),
        ("cache_off", False, args.gpu_util),
    ]
    if args.colocate_util > 0:
        variants.append(("cache_on_colocate", True, args.colocate_util))

    results = {}
    for label, pc, util in variants:
        results[label] = profile_variant(args, prompts, prefix_caching=pc, gpu_util=util,
                                         label=label)

    on, off = results["cache_on"]["热轮"], results["cache_off"]["热轮"]
    speedup = round(off["wall_s"] / on["wall_s"], 3) if on["wall_s"] else None
    import vllm

    report = {
        "模型": args.model, "adapter": args.adapter or "（基座）",
        "vllm 版本": vllm.__version__,
        "采样配置": {"num_prompts": args.num_prompts, "group_size": args.group_size,
                     "max_tokens": args.gen_tokens, "temperature": 1.0},
        "前缀分析": pstats,
        "各变体": results,
        "prefix caching 收益（热轮 wall 之比 off/on）": speedup,
        "一步 rollout 成本（热轮，cache_on）": f"{on['wall_s']}s / {on['序列数']} 条",
        "凭据": gpu_credentials(),
    }
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nprefix caching 提速 ×{speedup}；一步 rollout {on['wall_s']}s\n→ {args.out}")


if __name__ == "__main__":
    main()
