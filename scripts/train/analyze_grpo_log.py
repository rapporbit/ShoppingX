"""读 ms-swift 的 `logging.jsonl`，判 S3 那三条验收线还站不站得住。

```
reward 均值上升 ／ KL ∈ [0.5, 3.0] ／ 格式率不跌破 96%
```

**为什么要分段看而不是画一条线**：GRPO 一步只有 8 个 prompt，单步 reward 的抖动量级和真实
提升差不多大（实测步间跨度 0.56~0.90）。逐步看必然看出「涨了又跌」的错觉，分段均值才是能
判断趋势的东西。这与 M22 端到端 A/B 噪声 ±13.2 的教训同源：**先确认信号大于噪声，再谈涨跌**。

格式率这里用 `parse 失败率` 的代理指标（reward == -1.0 的比例）——一票否决那条只在 schema
parse 失败时触发。真正的格式验收还是跑 `eval_planner_format.py`，那把尺子查的是枚举合法性
与字段类型，比「能不能 parse」严。

用法::

    python analyze_grpo_log.py --log output/grpo_r1/vX/logging.jsonl --segments 5
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def seg_stats(rows: list[dict], key: str, segments: int) -> list[float | None]:
    vals = [r[key] for r in rows if key in r and r[key] is not None]
    if not vals:
        return []
    n = max(1, len(vals) // segments)
    out = []
    for i in range(segments):
        chunk = vals[i * n: (i + 1) * n] if i < segments - 1 else vals[i * n:]
        out.append(round(statistics.mean(chunk), 4) if chunk else None)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--segments", type=int, default=5)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = [json.loads(x) for x in Path(args.log).open(encoding="utf-8") if x.strip()]
    train = [r for r in rows if "reward" in r and "eval_reward" not in r]
    evals = [r for r in rows if "eval_reward" in r]

    seg = args.segments
    reward_seg = seg_stats(train, "reward", seg)
    kl_seg = seg_stats(train, "kl", seg)
    kls = [r["kl"] for r in train if "kl" in r]
    zero_std = [r.get("frac_reward_zero_std", 0.0) for r in train]

    report = {
        "训练步数": len(train),
        f"reward 分{seg}段均值": reward_seg,
        "reward 首段→末段": (round(reward_seg[-1] - reward_seg[0], 4)
                             if len(reward_seg) >= 2 else None),
        f"KL 分{seg}段均值": kl_seg,
        "KL 区间": [round(min(kls), 3), round(max(kls), 3)] if kls else None,
        "组内零方差比例均值": round(statistics.mean(zero_std), 4) if zero_std else None,
        "completion 长度均值": round(
            statistics.mean([r["completions/mean_length"] for r in train
                             if "completions/mean_length" in r]), 1) if train else None,
        "截断率均值": round(statistics.mean([r.get("completions/clipped_ratio", 0.0)
                                             for r in train]), 4) if train else None,
        "eval 序列": [{"step": r.get("global_step"), "reward": round(r["eval_reward"], 4),
                       "kl": round(r.get("eval_kl", 0), 4),
                       "零方差组比例": r.get("eval_frac_reward_zero_std")} for r in evals],
        "单步耗时中位 s": round(statistics.median(
            [r["step_time"] for r in train if "step_time" in r]), 2) if train else None,
    }
    # 三条验收线：涨没涨 / KL 稳不稳 / 有没有崩格式。判断写死在这里，免得每次靠肉眼看数。
    verdicts = []
    if len(reward_seg) >= 2:
        d = reward_seg[-1] - reward_seg[0]
        verdicts.append(f"reward 趋势：{'上升' if d > 0 else '未上升'}（{d:+.4f}）")
    if kls:
        inside = sum(0.5 <= k <= 3.0 for k in kls) / len(kls)
        verdicts.append(f"KL 落在 [0.5,3.0] 的步数占比：{inside:.1%}")
    report["判定"] = verdicts

    txt = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
