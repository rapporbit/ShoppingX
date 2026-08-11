"""在 GPU 机上跑几十步，采集**真实**的显存 / 吞吐 / MFU —— 不跑完整训练。

目的不是训出模型，是拿到「这套配置在这张卡上到底占多少显存、跑多快」的实测数字：
估算能给出量级，但估算不是证据，显存够不够、MFU 多少，只有真跑过才知道。

**自包含**：不 import 本仓库的 `app` 包——GPU 机上没有这套依赖（M21 已经这么分过一次工，
主仓不装 torch/ms-swift，训练环境另建）。只需要 torch / transformers / peft，数据是一个
jsonl（`build_planner_sft.py` 的产物），拷过去即可。

采完输出 JSON + 一张 markdown 表，含 GPU UUID、驱动版本、nvidia-smi 原文与时间戳——
这些是「确实在这台机器上跑过」的凭据，不是手抄的参数。

用法（在 GPU 机上）：
    python gpu_profile.py --data planner_sft_profile.jsonl --model Qwen/Qwen3-4B \\
        --steps 30 --batch 4 --seq 1024
    python gpu_profile.py --mode rollout --model Qwen/Qwen3-4B   # 需要 vllm
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

# 4090 / A100 的 bf16 稠密峰值（TFLOPS，不含稀疏加速）。算 MFU 用，认不出的卡回退 None。
PEAK_TFLOPS = {
    "4090": 165.2, "A100": 312.0, "A800": 312.0, "H100": 989.0,
    "3090": 71.0, "L40": 181.0, "V100": 125.0,
}


def _sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as exc:
        return f"(采集失败: {exc})"


def gpu_env() -> dict:
    """GPU 环境指纹。UUID 与驱动版本是「真在这台机器上跑过」的凭据。"""
    import torch

    q = "index,name,uuid,memory.total,driver_version,pcie.link.gen.current,pcie.link.width.current"
    raw = _sh(["nvidia-smi", f"--query-gpu={q}", "--format=csv"])
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "卡数": torch.cuda.device_count(),
        "卡型号": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "单卡显存GB": round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3, 1
        ) if torch.cuda.is_available() else 0,
        "nvidia_smi_raw": raw,
        "采集时间": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def _peak_tflops(name: str) -> float | None:
    for key, val in PEAK_TFLOPS.items():
        if key.lower() in name.lower():
            return val
    return None


def load_batches(path: Path, tokenizer, batch: int, seq: int, need: int) -> list[dict]:
    """把 SFT jsonl 编成定长 batch。**只在 assistant 段算 loss**——prompt 占了序列的大头，
    对它算 loss 等于让模型去背 system prompt，既没用又会盖过真正要学的输出格式。"""
    import torch

    rows = [json.loads(x) for x in path.open(encoding="utf-8") if x.strip()]
    samples = []
    for r in rows:
        msgs = r["messages"]
        prompt = tokenizer.apply_chat_template(
            msgs[:-1], tokenize=False, add_generation_prompt=True
        )
        full = prompt + msgs[-1]["content"] + (tokenizer.eos_token or "")
        p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        f_ids = tokenizer(full, add_special_tokens=False)["input_ids"][:seq]
        labels = [-100] * min(len(p_ids), len(f_ids)) + f_ids[len(p_ids):]
        samples.append((f_ids, labels[: len(f_ids)]))

    out, pad = [], tokenizer.pad_token_id or 0
    while len(out) < need:
        for i in range(0, len(samples) - batch + 1, batch):
            chunk = samples[i : i + batch]
            width = max(len(x[0]) for x in chunk)
            out.append({
                "input_ids": torch.tensor([x[0] + [pad] * (width - len(x[0])) for x in chunk]),
                "labels": torch.tensor([x[1] + [-100] * (width - len(x[1])) for x in chunk]),
                "attention_mask": torch.tensor(
                    [[1] * len(x[0]) + [0] * (width - len(x[0])) for x in chunk]
                ),
            })
            if len(out) >= need:
                break
        if not samples:
            break
    return out


def profile_train(args) -> dict:
    """LoRA 训练 profile。**刻意不用 Trainer**：它会把数据加载、梯度累积、日志混在一起，
    profile 出来的每步耗时说不清是谁的。手写循环才能把前向 / 反向 / 优化器分开计时。"""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    quant = None
    if args.qlora:  # 4bit：24G 卡上给 8B 及以上留的路，4B 用 bf16 LoRA 更快
        from transformers import BitsAndBytesConfig

        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, quantization_config=quant,
        attn_implementation=args.attn, trust_remote_code=True,
    ).cuda()
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                        "down_proj"],
    ))
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()  # 不加这行，grad ckpt + LoRA 会「没有梯度可回传」
    model.train()

    n_all = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)

    batches = load_batches(Path(args.data), tok, args.batch, args.seq, args.steps + args.warmup)
    torch.cuda.reset_peak_memory_stats()
    times, tokens = [], []

    for i, b in enumerate(batches[: args.steps + args.warmup]):
        b = {k: v.cuda() for k, v in b.items()}
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss = model(**b).loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if i >= args.warmup:  # 前几步含 cudnn autotune 与显存分配，计进去会低估吞吐
            times.append(dt)
            tokens.append(int(b["attention_mask"].sum().item()))

    step_s = sum(times) / len(times)
    tok_s = sum(tokens) / sum(times)
    peak = torch.cuda.max_memory_allocated() / 1024**3
    # LoRA 的 FLOPs/token ≈ 4N（fwd 2N + bwd 2N；基座权重不算梯度，省掉一个 2N）。
    # 开 gradient checkpointing 要多一次前向重算，按 6N 记。
    flops_per_tok = (6 if args.grad_ckpt else 4) * n_all
    peak_tf = _peak_tflops(torch.cuda.get_device_name(0))
    achieved_tf = tok_s * flops_per_tok / 1e12
    return {
        "配置": {
            "模型": args.model, "精度": "4bit-QLoRA" if args.qlora else "bf16-LoRA",
            "lora_r": args.lora_r, "batch": args.batch, "max_seq": args.seq,
            "grad_checkpointing": args.grad_ckpt, "attn": args.attn,
            "总参数B": round(n_all / 1e9, 2), "可训参数M": round(n_trainable / 1e6, 2),
            "可训占比%": round(100 * n_trainable / n_all, 3),
        },
        "实测": {
            "计时步数": len(times),
            "每步秒": round(step_s, 3),
            "tokens/s": round(tok_s, 1),
            "samples/s": round(args.batch / step_s, 2),
            "序列tokens中位": int(sorted(tokens)[len(tokens) // 2] / args.batch),
            "显存峰值GB(torch allocated)": round(peak, 2),
            "显存峰值GB(torch reserved)": round(
                torch.cuda.max_memory_reserved() / 1024**3, 2
            ),
            "实测TFLOPS": round(achieved_tf, 1),
            "MFU%": round(100 * achieved_tf / peak_tf, 1) if peak_tf else None,
            "卡峰值TFLOPS(bf16稠密)": peak_tf,
        },
    }


def profile_rollout(args) -> dict:
    """vLLM 生成 profile。本项目 87% 的 token 是固定 system prompt，所以**必须量一下开不开
    prefix caching 的差别**——这是 GRPO 里最大的一笔省，不量就不知道省没省到。"""
    from vllm import LLM, SamplingParams

    rows = [json.loads(x) for x in Path(args.data).open(encoding="utf-8") if x.strip()]
    prompts = [
        r["messages"][0]["content"] + "\n\n" + r["messages"][1]["content"] for r in rows
    ][: args.rollout_prompts]
    # GRPO 的一步 = 每个 prompt 采 group_size 条。这里照搬那个形状，量出来的才是 rollout 真实开销
    prompts = [p for p in prompts for _ in range(args.group)]
    sp = SamplingParams(temperature=1.0, max_tokens=args.gen_tokens, n=1)

    out = {}
    for cache in ([False, True] if args.compare_cache else [True]):
        llm = LLM(
            model=args.model, dtype="bfloat16", gpu_memory_utilization=args.gpu_util,
            enable_prefix_caching=cache, max_model_len=args.seq + args.gen_tokens,
            enforce_eager=False,
        )
        llm.generate(prompts[: args.group], sp)  # warmup + 把公共前缀灌进 cache
        t0 = time.perf_counter()
        res = llm.generate(prompts, sp)
        dt = time.perf_counter() - t0
        gen = sum(len(o.outputs[0].token_ids) for o in res)
        out[f"prefix_caching={cache}"] = {
            "序列数": len(prompts),
            "总耗时秒": round(dt, 2),
            "生成tokens": gen,
            "生成tokens/s": round(gen / dt, 1),
            "每序列秒": round(dt / len(prompts), 3),
            "折算一个GRPO步秒(8prompt×group)": round(dt / len(prompts) * 8 * args.group, 2),
        }
        del llm
        import gc

        import torch
        gc.collect()
        torch.cuda.empty_cache()
    return out


def to_markdown(doc: dict) -> str:
    lines = ["# GPU profile 实测", "", "## 环境"]
    for k, v in doc["环境"].items():
        if k != "nvidia_smi_raw":
            lines.append(f"- **{k}**：{v}")
    lines += ["", "```", doc["环境"]["nvidia_smi_raw"], "```", ""]
    for section, body in doc.get("结果", {}).items():
        lines += [f"## {section}", "", "| 项 | 值 |", "| --- | --- |"]
        for k, v in body.items():
            if isinstance(v, dict):
                lines += [f"| **{k}** | |"] + [f"| ├ {kk} | {vv} |" for kk, vv in v.items()]
            else:
                lines.append(f"| {k} | {v} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="train", choices=["train", "rollout", "both"])
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--data", default="planner_sft_profile.jsonl")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--qlora", action="store_true", help="4bit 量化（4B 用 bf16 LoRA 更快）")
    # help 里的 % 必须写成 %%：argparse 拿它做格式串，裸 % 会在 --help 时直接抛 ValueError
    ap.add_argument("--grad-ckpt", action="store_true", help="梯度检查点，省显存换约 30%% 速度")
    ap.add_argument("--attn", default="sdpa", help="sdpa / flash_attention_2 / eager")
    ap.add_argument("--group", type=int, default=8, help="GRPO group_size")
    ap.add_argument("--rollout-prompts", type=int, default=16)
    ap.add_argument("--gen-tokens", type=int, default=256)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--compare-cache", action="store_true", help="对比 prefix caching 开/关")
    ap.add_argument("--out", default="gpu_profile")
    args = ap.parse_args()

    doc = {"环境": gpu_env(), "结果": {}}
    if args.mode in ("train", "both"):
        doc["结果"]["训练（LoRA）"] = profile_train(args)
    if args.mode in ("rollout", "both"):
        doc["结果"]["rollout（vLLM）"] = profile_rollout(args)

    Path(f"{args.out}.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md = to_markdown(doc)
    Path(f"{args.out}.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"\n→ {args.out}.json / {args.out}.md")


if __name__ == "__main__":
    main()
