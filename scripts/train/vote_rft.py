"""S5-RFT 优化：**字段级投票合成**，试着把训练目标做得比 best-of-n 那一条还好。

现在的选优是「整条取最优」，但 reward 是四维加权的——总分最高那条完全可能是 category 判错、
靠 keywords 写得漂亮把分拉回来的。结构化输出给了一个通用文本生成没有的机会：**各字段可以
独立取所长，拼出一条八条候选里谁都不是的解**（self-consistency 投票的结构化变体）。

拼法按「这个字段该由谁决定」来定，不是一律投票：
- `category` / `budget_amount` / `clear_budget` / `currency`：**众数**。这几个是判定题，多数
  票是对噪声最稳的估计。
- `domains`：**逐域过半入选**。它是多标签，整体投票会因为组合稀疏而退化成选众数条目。
- `keywords` / `exclude_terms`：**不投票，取组内 retrieval 分最高那条的**。检索词是搜得动搜
  不动的唯一决定项，投票会把它搅成同义词堆砌——那正是 R_econ 要罚的东西。

合成解不在候选里，**分数必须重新打**，不能假设它一定更好：字段各自最优拼起来可能互相矛盾
（品类换了检索词却没换）。所以这个脚本的产出首先是一张对照表，值不值得拿去训练看数说话。

用法（GPU 机，embed:8095 / qdrant:6333 要先起）：
  python vote_rft.py --cands rft_candidates.jsonl --src planner_grpo_train.jsonl
"""

import argparse
import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rollout_env import extract_json, load_reward, score_batch  # noqa: E402
from rollout_env import Retriever  # noqa: E402

SCORE_BATCH = 256
VOTE_KEYS = ("category", "budget_amount", "clear_budget", "currency", "currency_explicit")


def _mode(vals: list):
    """众数；全为 None/缺失时返回 None。并列取先出现的（候选顺序即采样顺序，确定性）。"""
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return Counter(json.dumps(v, ensure_ascii=False, sort_keys=True) for v in vals).most_common(1)[0][0]


def synth(plans: list[dict], best_retr: dict | None) -> dict:
    """字段级合成。`best_retr` 是组内 retrieval 分最高那条的 plan，检索词直接沿用它。"""
    out: dict = {}
    for k in VOTE_KEYS:
        m = _mode([p.get(k) for p in plans])
        if m is not None:
            out[k] = json.loads(m)
    # domains：逐域过半入选；一个都没过半就退回众数条目的 domains
    cnt: Counter = Counter()
    for p in plans:
        for d in p.get("domains") or []:
            cnt[d] += 1
    half = len(plans) / 2
    picked = sorted([d for d, c in cnt.items() if c >= half])
    if not picked:
        m = _mode([p.get("domains") for p in plans])
        picked = json.loads(m) if m else []
    out["domains"] = picked
    src = best_retr or (plans[0] if plans else {})
    out["keywords"] = src.get("keywords") or []
    out["exclude_terms"] = src.get("exclude_terms") or []
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", default="rft_candidates.jsonl")
    ap.add_argument("--src", default="planner_grpo_train.jsonl")
    ap.add_argument("--out", default="rft_vote_scored.jsonl")
    ap.add_argument("--report", default="rft_vote_report.json")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.cands, encoding="utf-8")]
    src = {json.loads(line)["id"]: json.loads(line) for line in open(args.src, encoding="utf-8")}

    synth_texts, golds, texts, bests = [], [], [], []
    for r in rows:
        plans, dims = [], []
        for c in r["cands"]:
            p = extract_json(c["text"])
            if p is not None:
                plans.append(p)
                dims.append(c["dims"])
        if not plans:
            plans, dims = [{}], [{}]
        retr = [(d.get("retrieval") if d.get("retrieval") is not None else -1) for d in dims]
        best_retr = plans[retr.index(max(retr))]
        synth_texts.append(json.dumps(synth(plans, best_retr), ensure_ascii=False))
        golds.append(json.loads(src[r["id"]]["golden_json"]))
        texts.append(r["text"])
        bests.append(max(c["reward"] for c in r["cands"]))

    retriever, reward_mod = Retriever(), load_reward()
    brs = []
    for i in range(0, len(synth_texts), SCORE_BATCH):
        j = min(i + SCORE_BATCH, len(synth_texts))
        brs.extend(score_batch(synth_texts[i:j], golds[i:j], texts[i:j], retriever, reward_mod))
        print(f"[score] {j}/{len(synth_texts)}", flush=True)

    vote = [b.total for b in brs]
    with open(args.out, "w", encoding="utf-8") as f:
        for r, t, b in zip(rows, synth_texts, brs, strict=True):
            f.write(json.dumps({"id": r["id"], "text": t, "reward": b.total}, ensure_ascii=False) + "\n")

    def dmean(attr):
        v = [getattr(b, attr) for b in brs if getattr(b, attr) is not None]
        return round(st.mean(v), 4) if v else None

    report = {
        "组数": len(rows),
        "投票合成": {
            "reward": round(st.mean(vote), 4),
            "retrieval": dmean("retrieval"),
            "field": dmean("field_score"),
            "format": dmean("fmt"),
            "econ": dmean("econ"),
        },
        "best-of-8 对照": round(st.mean(bests), 4),
        "投票 - best": round(st.mean(vote) - st.mean(bests), 4),
        "投票胜过本组 best 的组": sum(1 for v, b in zip(vote, bests, strict=True) if v > b + 1e-9),
    }
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
