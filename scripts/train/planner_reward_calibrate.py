"""M23 S1 前置：**标定 reward 的区分度**——它能不能把好答案和坏答案分开。

没标定过的 reward 等于没有。M22 的教训是「尺子不准，后面做得再漂亮也验收不了」，reward 比
评测尺子更狠：它不是量结果，是**直接决定梯度方向**。区分度不够的 reward，GRPO 组内 8 个
rollout 分数挤成一团，优势全是噪声，训练只会原地抖。

做法：同一批 query 上跑三种 plan，看分数分不分得开——
- **strong**：线上 API 模型真实产出（planner 的推理逻辑，不碰工具体的会话副作用）。这是
  4B 要追的天花板，也是 golden 的来源水平。
- **copycat**：keywords 照抄用户原话——RL 最容易找到的捷径，必须被门禁掐住。
- **broken**：域填 global、预算乱改、exclude 编 evidence——schema 合法但字段全错。

**为什么不跑本地 4B**：这台机器没有 GPU（S1 才上 301）。标定的是 reward 函数本身，不是模型，
用 API 模型当 strong 档完全够——要验证的是「三档能不能分开」，不是「谁分高」。

用法：``uv run python scripts/train/planner_reward_calibrate.py --limit 12``
（需要 Qdrant 在跑 + .env 的 EMBED_* 可用）
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# **必须在导入 app.recall 之前加载 .env**：collection 名是模块级常量、导入即求值，晚一步就
# 落回默认的 shoppingx_items，而线上这套库叫 globex_items —— 报的是 404 Collection 不存在，
# 但真正的错在这一行的位置。
from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

from app.eval.planner_reward import compute_reward  # noqa: E402
from app.recall.qdrant_store import get_recall_client  # noqa: E402
from app.recall.towers import get_tower_client  # noqa: E402

GOLDEN = PROJECT_ROOT / "data" / "train" / "planner_golden.jsonl"
TOP_K = 20


async def _retrieve(keywords: list[str]) -> list[str]:
    """把 plan 的 keywords 真打进 Qdrant，取 top20 标题。

    检索词拼成一条 query 编码——与线上 item_search 同口径（个性化靠偏好词并入检索词，
    不走向量融合）。口径不一致的话，标出来的 reward 阈值到了 rollout 里就不作数。
    """
    text = " ".join(str(k) for k in keywords if str(k).strip())
    if not text:
        return []
    vec = await get_tower_client().encode_query(text)
    return [c.title for c in get_recall_client().search(vec, top_k=TOP_K)]


async def _strong_plan(row: dict) -> dict | None:
    """线上 API 模型的真实产出。只复用 prompt + schema，**不碰工具体**——那里面有
    reset_candidates / P_t 写入 / 计费 / AGUI 上报一堆会话副作用，批量跑会互相污染。"""
    from app.agent.llm import get_fast_llm
    from app.tools.planner import PlanOutput, get_planner_prompt

    prior = "".join(f"用户上一轮：{t}\n" for t in row.get("prior_turns") or [])
    try:
        structured = get_fast_llm().with_structured_output(PlanOutput, method="function_calling")
        out = await structured.ainvoke(
            [("system", get_planner_prompt()), ("user", prior + row["text"])]
        )
        return out.model_dump() if hasattr(out, "model_dump") else dict(out)
    except Exception as exc:  # 标定跑批，单条失败不该中断整跑
        print(f"  [warn] {row['id']} strong 产出失败：{type(exc).__name__}")
        return None


def _copycat(row: dict, strong: dict) -> dict:
    """字段全对，只把 keywords 换成照抄原话——单变量，隔离出 R_retrieval 与门禁的作用。"""
    return {**strong, "keywords": [row["text"]]}


def _broken(row: dict, strong: dict) -> dict:
    """schema 合法但判定全错：域填 global、预算 ×10、evidence 现编。"""
    budget = strong.get("budget_amount")
    return {
        **strong,
        "domains": ["global"],
        "category": "其他",
        "budget_amount": (budget * 10) if budget else 9999.0,
        "exclude_terms": [{"term": "plastic", "evidence": "用户说过不要塑料"}],
        "keywords": ["stuff", "things", "good stuff", "nice things", "cool stuff", "items", "buy"],
    }


async def calibrate(rows: list[dict]) -> dict:
    """逐条跑三档，落每条明细 + 汇总。串行跑：标定只要十几条，并发省下的几十秒不值得
    冒「限流把 strong 档打挂、结果不可比」的风险。"""
    out: list[dict] = []
    for i, row in enumerate(rows, 1):
        strong = await _strong_plan(row)
        if strong is None:
            continue
        variants = {
            "strong": strong,
            "copycat": _copycat(row, strong),
            "broken": _broken(row, strong),
        }
        rec = {"id": row["id"], "text": row["text"], "scores": {}, "detail": {}}
        for name, plan in variants.items():
            titles = await _retrieve(plan.get("keywords") or [])
            br = compute_reward(plan, row["golden"], row["text"], titles=titles)
            rec["scores"][name] = br.total
            rec["detail"][name] = {
                "retrieval": br.retrieval, "field": br.field_score,
                "fmt": br.fmt, "econ": br.econ, "penalties": br.penalties,
            }
        out.append(rec)
        print(f"  [{i}/{len(rows)}] {row['text'][:26]:30s} "
              + "  ".join(f"{k}={v:.3f}" for k, v in rec["scores"].items()), flush=True)

    def _stat(name: str) -> dict:
        vals = [r["scores"][name] for r in out]
        return {
            "均值": round(statistics.mean(vals), 4),
            "标准差": round(statistics.pstdev(vals), 4),
            "最低": round(min(vals), 4), "最高": round(max(vals), 4),
        }

    stats = {k: _stat(k) for k in ("strong", "copycat", "broken")}
    gaps = {
        "strong - copycat": round(stats["strong"]["均值"] - stats["copycat"]["均值"], 4),
        "strong - broken": round(stats["strong"]["均值"] - stats["broken"]["均值"], 4),
    }
    # 判据：strong 必须**同时**显著高于两个劣化档。差距小于组内标准差就等于分不开——
    # GRPO 的优势是组内相对量，分不开就只有噪声可学。
    sd = stats["strong"]["标准差"] or 1e-9
    verdict = {
        k: ("区分得开" if v > sd else f"**区分不开**（差 {v} ≤ strong 组内 σ {round(sd, 4)}）")
        for k, v in gaps.items()
    }
    return {"n": len(out), "分档统计": stats, "差距": gaps, "判据": verdict, "明细": out}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--split", default="dev", help="拿哪一档标定（dev 最贴线上真实分布）")
    ap.add_argument("--out", default="planner_reward_calibration.json")
    args = ap.parse_args()

    rows = [json.loads(x) for x in GOLDEN.open(encoding="utf-8") if x.strip()]
    # 只挑「品类与锚都判得出」的样本：弃权样本本来就跳过大半维度，标不出区分度
    pool = [
        r for r in rows
        if r["split"] == args.split
        and r["golden"].get("category") and r["golden"].get("must_have")
    ][: args.limit]
    print(f"标定 {len(pool)} 条（{args.split}），每条跑 strong / copycat / broken 三档\n")

    report = await calibrate(pool)
    path = PROJECT_ROOT / "data" / "train" / args.out
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + json.dumps(
        {k: report[k] for k in ("n", "分档统计", "差距", "判据")}, ensure_ascii=False, indent=2
    ))
    print(f"\n→ {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    asyncio.run(main())
