"""把 golden 转成 **GRPO 数据集**：只给到 user 这一轮，不给答案。

与 SFT 数据集（`build_planner_sft.py`）的区别只有一处，但很关键：SFT 那份带 assistant 目标，
教的是「照着抄」；GRPO 这份**故意不带**——答案要由 policy 自己采样出来，再由 reward 打分。
golden 不进 prompt、只进 reward 侧的 kwargs。混进 prompt 就是答案泄漏，训出来的分好看得
离谱、一上线全废。

prompt 拼装直接 import `build_planner_sft` 的 `build_system()` / `USER_TMPL` —— **同一份精简
prompt**。SFT 与 GRPO 的输入分布必须逐字一致，否则冷启动学到的形态到了 RL 阶段对不上，
格式正确率会莫名其妙从 100% 掉下来。

`golden` 落成 **JSON 字符串列**而不是嵌套 dict：HF datasets 会给嵌套 dict 推 struct schema，
各行键不齐（弃权样本没有 budget_amount 等）时直接报类型冲突。字符串列在 reward 里 loads
一次即可，稳。

用法::

    uv run python scripts/train/build_grpo_dataset.py --split train
    uv run python scripts/train/build_grpo_dataset.py --split dev
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train.build_planner_sft import USER_TMPL, build_system  # noqa: E402

GOLDEN = PROJECT_ROOT / "data" / "train" / "planner_golden.jsonl"
OUT_DIR = PROJECT_ROOT / "data" / "train"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--keep-review", action="store_true", help="保留三票没谈拢的样本。默认排除——与 SFT 同口径"
    )
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = [json.loads(x) for x in GOLDEN.open(encoding="utf-8") if x.strip()]
    rows = [r for r in rows if r["split"] == args.split]
    if not args.keep_review:
        rows = [r for r in rows if r["status"] != "review"]
    if args.limit:
        rows = rows[: args.limit]

    system = build_system()
    out_path = OUT_DIR / (args.out or f"planner_grpo_{args.split}.jsonl")
    abstain = {"category": 0, "budget": 0}
    with out_path.open("w", encoding="utf-8") as fh:
        for r in rows:
            prior = "".join(f"用户上一轮：{t}\n" for t in r.get("prior_turns") or [])
            gold = r["golden"]
            abstain["category"] += gold.get("category") is None
            abstain["budget"] += bool(gold.get("budget_uncertain"))
            fh.write(
                json.dumps(
                    {
                        "id": r["id"],
                        "messages": [
                            {"role": "system", "content": system},
                            {
                                "role": "user",
                                "content": USER_TMPL.format(prior=prior, text=r["text"]),
                            },
                        ],
                        "golden_json": json.dumps(gold, ensure_ascii=False),
                        "text": r["text"],
                        "family": r.get("family") or "",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"{args.split}: {len(rows)} 条 → {out_path.relative_to(PROJECT_ROOT)}")
    print(
        f"  弃权样本（reward 会跳过对应维度）：category=None {abstain['category']} 条 / "
        f"budget_uncertain {abstain['budget']} 条"
    )


if __name__ == "__main__":
    main()
