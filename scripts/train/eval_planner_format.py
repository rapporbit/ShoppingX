"""S2 的验收闸：**格式正确率**。refdocs 08-2 §6.1 把 98% 定为切 RL 的前置条件。

为什么这条是硬闸而不是「顺便看看」：GRPO 里 schema parse 失败直接 -1.0 一票否决。格式率
只有 90% 的话，每 10 条 rollout 就有 1 条把整组的优势基线拽下去——组内相对量被格式噪声
主导，模型学到的第一件事会是「怎么把 JSON 吐对」，而不是怎么把品类判对。那件事该在 SFT
阶段用监督信号便宜地解决掉。

**自包含**：不 import 本仓库的 app 包（GPU 机上没有）。schema 从 `planner_sft_meta.json` 读
——枚举值只有一份事实来源，改了域枚举不会漏同步这边。

用法（GPU 机上）：
    python eval_planner_format.py --model Qwen/Qwen3-4B --adapter ./out/checkpoint-xxx \\
        --data planner_sft_dev.jsonl --meta planner_sft_meta.json
    # 不带 --adapter 就是跑基座做对照
    python eval_planner_format.py --model Qwen/Qwen3-4B --data ... --meta ...
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

PASS_LINE = 0.98  # refdocs 08-2 §6.1


def extract_json(text: str) -> dict | None:
    """从生成里抠出 JSON。**允许 ```json 围栏与前后废话**——线上是 structured output 兜底的，
    这里只判「模型有没有能力吐出结构」，不是判它会不会加围栏。判太严会低估真实可用性。"""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def check(obj: dict | None, meta: dict) -> list[str]:
    """返回违规项列表，空 = 格式完全合格。"""
    if obj is None:
        return ["无法解析出 JSON"]
    bad: list[str] = []
    for f in meta["required_fields"]:
        if f not in obj:
            bad.append(f"缺字段:{f}")

    doms = obj.get("domains")
    if doms is not None:
        if not isinstance(doms, list):
            bad.append("domains 非 list")
        else:
            for d in doms:
                if d in meta["forbidden_domains"]:
                    bad.append(f"用了禁用域:{d}")
                elif d not in meta["domains"]:
                    bad.append("域不在枚举内")

    if (kw := obj.get("keywords")) is not None:
        if not isinstance(kw, list) or not all(isinstance(k, str) for k in kw):
            bad.append("keywords 非 list[str]")

    if (ex := obj.get("exclude_terms")) is not None:
        if not isinstance(ex, list):
            bad.append("exclude_terms 非 list")
        else:
            for t in ex:
                if not isinstance(t, dict) or "word" not in t or "evidence" not in t:
                    bad.append("exclude_terms 缺 word/evidence")
                    break

    b = obj.get("budget_amount", "__missing__")
    if b != "__missing__" and b is not None and not isinstance(b, (int, float)):
        bad.append("budget_amount 非数值/null")
    if (c := obj.get("clear_budget")) is not None and not isinstance(c, bool):
        bad.append("clear_budget 非 bool")
    if (cat := obj.get("category")) is not None and not isinstance(cat, str):
        bad.append("category 非 str")
    return bad


def field_accuracy(pred: dict, gold: dict) -> dict:
    """**附加观测，不是验收项**：字段对不对。口径比 reward 侧的 R_field 粗（品类只做子串匹配、
    不给部分分），够看趋势就行——真要比分数以 app/eval/planner_reward.py 为准，别拿这里的数
    去和那边的对照。

    **gold 品类为空 → 该条弃权（返回 None），不算错**。dev 92 条里有 24 条是 S0-2 定的机制
    弃权样本（无上文的追问碎片，品类判不出来），旧口径把它们一律记为「判错」，天花板被压到
    68/92=0.739 —— 实测 0.522 因此被读成「一半都判错」，实际可判样本上是 0.706。
    分母混进无解的题，得到的就不是判定能力。
    """

    def _norm(s: object) -> str:
        return re.sub(r"[\s的]", "", str(s or "")).lower()

    pc, gc = _norm(pred.get("category")), _norm(gold.get("category"))
    pd_, gd = set(pred.get("domains") or []), set(gold.get("domains") or [])
    inter = len(pd_ & gd)
    return {
        "category": (bool(pc and (pc == gc or pc in gc or gc in pc)) if gc else None),
        "domains_f1": (2 * inter / (len(pd_) + len(gd))) if (pd_ or gd) else 1.0,
        "budget": pred.get("budget_amount") == gold.get("budget_amount"),
    }


def generate(args, prompts: list[str]) -> list[str]:
    """优先 vLLM（快得多）；没装就退 transformers。两条路的采样参数保持一致，
    否则「换了后端分数就变」会让这把尺子失去意义。"""
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        LLM = None
    if LLM is not None:
        kw = {
            "model": args.model,
            "dtype": "bfloat16",
            "max_model_len": args.max_len,
            "gpu_memory_utilization": args.gpu_util,
        }
        if args.adapter:
            from vllm.lora.request import LoRARequest

            llm = LLM(**kw, enable_lora=True, max_lora_rank=args.lora_rank)
            out = llm.generate(
                prompts,
                SamplingParams(temperature=0, max_tokens=args.gen_tokens),
                lora_request=LoRARequest("sft", 1, args.adapter),
            )
        else:
            llm = LLM(**kw)
            out = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=args.gen_tokens))
        return [o.outputs[0].text for o in out]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).cuda()
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    outs = []
    for p in prompts:
        ids = tok(p, return_tensors="pt").to("cuda")
        with torch.no_grad():
            g = model.generate(**ids, max_new_tokens=args.gen_tokens, do_sample=False)
        outs.append(tok.decode(g[0][ids["input_ids"].shape[1] :], skip_special_tokens=True))
    return outs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default="", help="LoRA 目录；不传 = 跑基座做对照")
    ap.add_argument("--data", default="planner_sft_dev.jsonl")
    ap.add_argument("--meta", default="planner_sft_meta.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gen-tokens", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--out", default="format_eval.json")
    args = ap.parse_args()

    meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))
    rows = [json.loads(x) for x in Path(args.data).open(encoding="utf-8") if x.strip()]
    if args.limit:
        rows = rows[: args.limit]
    prompts = [
        f"<|im_start|>system\n{r['messages'][0]['content']}<|im_end|>\n"
        f"<|im_start|>user\n{r['messages'][1]['content']}<|im_end|>\n<|im_start|>assistant\n"
        for r in rows
    ]
    gens = generate(args, prompts)

    violations: Counter = Counter()
    ok = 0
    acc = {"category": 0, "domains_f1": 0.0, "budget": 0}
    cat_n = 0  # 品类可判样本数（gold 为空的弃权样本不进分母）
    samples = []
    for r, g in zip(rows, gens, strict=True):
        obj = extract_json(g)
        bad = check(obj, meta)
        if not bad:
            ok += 1
        for b in bad:
            violations[b] += 1
        gold = json.loads(r["messages"][2]["content"])
        if obj:
            a = field_accuracy(obj, gold)
            if a["category"] is not None:
                acc["category"] += a["category"]
                cat_n += 1
            acc["domains_f1"] += a["domains_f1"]
            acc["budget"] += a["budget"]
        if len(samples) < 5 and bad:
            samples.append({"id": r.get("id"), "违规": bad, "生成": g[:300]})

    n = len(rows)
    rate = ok / n
    report = {
        "模型": args.model,
        "adapter": args.adapter or "（基座）",
        "样本数": n,
        "格式正确率": round(rate, 4),
        "验收线": PASS_LINE,
        "判定": "通过" if rate >= PASS_LINE else f"**未过**（差 {round(PASS_LINE - rate, 4)}）",
        "违规分布": dict(violations.most_common()),
        "字段准确（附加观测，非验收）": {
            "category": round(acc["category"] / cat_n, 3) if cat_n else None,
            "category 可判样本": f"{cat_n}/{n}（gold 品类为空的弃权样本不进分母）",
            "domains_f1": round(acc["domains_f1"] / n, 3),
            "budget": round(acc["budget"] / n, 3),
        },
        "失败样例": samples,
    }
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "失败样例"}, ensure_ascii=False, indent=2
        )
    )
    print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()
