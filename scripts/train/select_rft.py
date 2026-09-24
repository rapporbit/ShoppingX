"""S5-RFT 步 2 下半：从 best-of-n 候选里选优，产出再训练用的 SFT 数据集。

三条拒绝规则（`--tau` / `--eps` 是唯二自由参数，脚本会先把 τ 的整条曲线扫出来）：
1. **parse 失败**的候选直接出局（reward = -1.0，不可能被选中，这条是兜底说明）；
2. **组内最高分 < τ → 整组丢弃**。教师都没教对的题，不该学自己的坏答案——RFT 的风险正在
   于把模型自己的错误当成监督信号固化下来；
3. **组内 σ < ε → 记为「无信息组」**。8 条几乎一个样，选谁都等于原样复读 SFT，默认仍保留
   （丢了会让训练集偏向难题，分布漂移），但单独统计占比。

并列最高分时取**最短**的那条：同分意味着 reward 认为一样好，那就挑 token 更省的；这也让
tie-break 是确定性的，换台机器重跑能复现。

写盘前用 `extract_json` 规范化一遍再 `json.dumps`，保证训练目标与教师数据同形——不然模型
会把采样带出来的多余空白、前后缀一起学进去。

用法：python select_rft.py --cands rft_candidates.jsonl --src planner_grpo_train.jsonl --tau 0.75
"""

import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rollout_env import extract_json  # noqa: E402


def pick(cands: list[dict]) -> dict:
    """组内选优：先按 reward 降序，同分取更短的 completion。"""
    return sorted(cands, key=lambda c: (-c["reward"], len(c["text"])))[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", default="rft_candidates.jsonl")
    ap.add_argument("--src", default="planner_grpo_train.jsonl")
    ap.add_argument("--out", default="rft_sft_train.jsonl")
    ap.add_argument("--report", default="rft_select_report.json")
    ap.add_argument("--tau", type=float, default=0.75)
    ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--drop-uninformative", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.cands, encoding="utf-8")]
    src = {json.loads(line)["id"]: json.loads(line) for line in open(args.src, encoding="utf-8")}

    greedy = [r["greedy"]["reward"] for r in rows]
    bests = [max(c["reward"] for c in r["cands"]) for r in rows]
    means = [st.mean([c["reward"] for c in r["cands"]]) for r in rows]
    worsts = [min(c["reward"] for c in r["cands"]) for r in rows]
    sigmas = [st.pstdev([c["reward"] for c in r["cands"]]) for r in rows]
    n_parse_fail = sum(1 for r in rows for c in r["cands"] if c["reward"] <= -0.999)

    # τ 曲线：被问「阈值怎么定的」时，答案应该是一条曲线而不是一个拍脑袋的数
    curve = []
    for t in [round(0.50 + 0.05 * i, 2) for i in range(10)]:
        keep = [b for b in bests if b >= t]
        curve.append(
            {
                "tau": t,
                "保留组数": len(keep),
                "保留率": round(len(keep) / len(rows), 4),
                "选中样本均分": round(st.mean(keep), 4) if keep else None,
            }
        )

    kept, dropped_tau, uninformative = [], 0, 0
    for r in rows:
        sigma = st.pstdev([c["reward"] for c in r["cands"]])
        best = pick(r["cands"])
        if best["reward"] < args.tau:
            dropped_tau += 1
            continue
        if sigma < args.eps:
            uninformative += 1
            if args.drop_uninformative:
                continue
        plan = extract_json(best["text"])
        if plan is None:
            continue
        msgs = src[r["id"]]["messages"][:2]
        kept.append(
            {
                "id": r["id"],
                "messages": [*msgs, {"role": "assistant", "content": json.dumps(plan, ensure_ascii=False)}],
            }
        )

    with open(args.out, "w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    report = {
        "组数": len(rows),
        "候选总数": sum(len(r["cands"]) for r in rows),
        "parse 失败候选": n_parse_fail,
        "reward 均值": {
            "greedy": round(st.mean(greedy), 4),
            "采样均值": round(st.mean(means), 4),
            "best-of-n": round(st.mean(bests), 4),
            "worst-of-n": round(st.mean(worsts), 4),
        },
        "best 比 greedy 高的组": sum(1 for b, g in zip(bests, greedy, strict=True) if b > g + 1e-9),
        "best-greedy 增益": round(st.mean(bests) - st.mean(greedy), 4),
        "组内 σ": {
            "均值": round(st.mean(sigmas), 4),
            "中位": round(st.median(sigmas), 4),
            f"σ<{args.eps} 的组": sum(1 for s in sigmas if s < args.eps),
        },
        "选用参数": {"tau": args.tau, "eps": args.eps, "丢无信息组": args.drop_uninformative},
        "τ 曲线": curve,
        "产出": {"保留": len(kept), "被 τ 丢弃": dropped_tau, "无信息组": uninformative, "文件": args.out},
    }
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
