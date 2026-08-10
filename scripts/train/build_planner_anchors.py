"""S0-1 源①：从线上真实对话抽取 planner 训练的**分布锚**（不是训练集本身）。

**为什么不当训练集**：`var/globex.db` 里 role=user 的消息总共只有 139 条、去重后 95 条，
其中还有一大半是评测脚本反复跑同几条种子 query 打进去的。这个量级喂不出 4B 的 SFT，
更撑不起 GRPO 的 rollout。它的价值在**分布**——合成数据长什么样，得照着它对齐。

**实测出来的两条分布事实（决定了 S0 后续所有做法）**：

1. **短**。归一化后长度中位数 17 字、p90 31 字。合成数据若写成三行长句的"标准购物意图"，
   就是造了一个线上不存在的分布。
2. **约三分之一是追问轮碎片**（"不要皮革的" / "预算提到600吧" / "算了，粉色的也可以接受"）。
   这类 query **单看无法标注 category**——planner 在线上读的是 `_render_prior_context()`
   拼上本轮消息。所以训练样本必须带 prior 上下文，golden 也必须在同样的上下文下标。
   不这么做，训出来的模型在线上 1/3 的请求上都是 train/serve skew。

产物：``data/train/planner_anchors.jsonl``，每行一条去重后的真实 query + 轮次位置标记，
供 S0-2 的合成脚本当 few-shot 模板与分布校验基准（`planner_quality_gate.py` 读它算相似度）。

用法：``uv run python scripts/train/build_planner_anchors.py``
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = PROJECT_ROOT / "var" / "globex.db"
SEED_PATH = PROJECT_ROOT / "data" / "eval" / "queries.jsonl"
OUT_PATH = PROJECT_ROOT / "data" / "train" / "planner_anchors.jsonl"

# 归一化只吃掉空白与标点：真实 query 里「预算 300」和「预算300」是同一条，
# 但「预算 300」和「预算 500」必须算两条——所以不能做更激进的数字归一。
_NORM_RE = re.compile(r"[\s，。,.、！!？?]+")

# 判「带预算 / 带排除」用的粗规则。只为统计画像，不参与标注（golden 的预算走
# planner.resolve_budget_currency 那套规则解析器，见 ROADMAP M23 S0-2）。
_BUDGET_RE = re.compile(r"预算|不超|以内|以下|块|元|美元|人民币|\$|budget")
_EXCLUDE_RE = re.compile(r"不要|不想|别太|别的|除了|不能|拒绝|讨厌")


def _norm(text: str) -> str:
    return _NORM_RE.sub("", str(text))


def _load_real() -> list[dict]:
    """按 (thread_id, seq) 读真实 user 消息，thread 内第一条算首轮、其余算追问轮。

    轮次位置**只能从库里的 seq 拿**，不能靠"文本短不短"猜——"安卓手机"是首轮，
    "最好能保温12小时以上"是追问轮，长度上分不开。
    """
    if not DB_PATH.exists():
        raise SystemExit(f"找不到 {DB_PATH}；本脚本读的是线上会话库")
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT thread_id, seq, content FROM messages WHERE role='user' ORDER BY thread_id, seq"
    ).fetchall()
    conn.close()

    seen: dict[str, dict] = {}
    first_seq: dict[str, int] = {}
    for thread_id, seq, content in rows:
        first_seq.setdefault(thread_id, seq)
        text = str(content or "").strip()
        if not text or text.startswith("（用户只上传了图片"):  # 纯图轮不是 planner 的输入形态
            continue
        key = _norm(text)
        if key in seen:
            continue
        seen[key] = {
            "text": text,
            "source": "real",
            "thread_id": thread_id,
            "is_followup": seq > first_seq[thread_id],
        }
    return list(seen.values())


def _load_seed() -> list[dict]:
    """33 条种子集：人工写的，风格上是"标准购物意图"，与真实分布互补，一并当模板。"""
    if not SEED_PATH.exists():
        return []
    out: list[dict] = []
    for line in SEED_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        q = json.loads(line)
        turns = q.get("turns") or [q["query"]]
        for i, t in enumerate(turns):
            out.append(
                {
                    "text": t,
                    "source": "seed",
                    "bucket": q.get("bucket", ""),
                    "thread_id": f"seed_{q['id']}",
                    "is_followup": i > 0,
                }
            )
    return out


def main() -> None:
    records = _load_real()
    by_norm = {_norm(r["text"]): r for r in records}
    # 种子集的 query 基本都被评测跑进过 messages 表（实测 33 条一条不剩全命中），所以这里
    # 几乎不新增记录——真正要捞回来的是**它们的 bucket**（人工分的能力维度：多约束精挑 /
    # 跨平台比价 / 到手价 …），S0-1 的分层合成靠它定配额，丢了就只能按品类瞎分。
    for r in _load_seed():
        hit = by_norm.get(_norm(r["text"]))
        if hit is None:
            by_norm[_norm(r["text"])] = r
            records.append(r)
        elif r.get("bucket"):
            hit["bucket"] = r["bucket"]

    for i, r in enumerate(records):
        norm = _norm(r["text"])
        r["id"] = f"anchor_{i:04d}"
        r["length"] = len(norm)
        r["has_budget"] = bool(_BUDGET_RE.search(r["text"]))
        r["has_exclude"] = bool(_EXCLUDE_RE.search(r["text"]))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(records)
    followup = sum(r["is_followup"] for r in records)
    lens = sorted(r["length"] for r in records)
    print(f"锚样本 {n} 条 → {OUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  真实 {sum(r['source'] == 'real' for r in records)} / 种子 {sum(r['source'] == 'seed' for r in records)}")
    print(f"  追问轮 {followup} 条（{followup / n:.1%}）—— 合成数据必须复现这个比例")
    print(f"  长度 中位数 {lens[n // 2]} / p90 {lens[int(n * 0.9)]}")
    print(f"  带预算 {sum(r['has_budget'] for r in records) / n:.1%} / 带排除 {sum(r['has_exclude'] for r in records) / n:.1%}")


if __name__ == "__main__":
    main()
