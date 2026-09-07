"""从 Rubric 高分轨迹蒸馏**成功策略**，过门禁重放后写入策略库（批 4 / 18-4）。

与同目录 ``distill_fewshot.py`` 的分工（两者都从高分轨迹蒸馏，但产物与生命周期完全不同）：

| | distill_fewshot | 本脚本 |
|---|---|---|
| 产物 | ``prompt/few_shot_distilled.yml``（文件） | ``strategies`` 表（库） |
| 形态 | ✅正确 / ❌反例 的范式片段 | ``{trigger, actions, category, evidence}`` 结构化条目 |
| 何时生效 | 每轮都注入（静态） | 触发词命中本轮 query 才注入 |
| 写入前 | 直接落盘 | **必须过门禁重放**（3 条同类 query 不退化） |
| 之后 | 手工维护 | 命中回血 / 连续失败自动淘汰 |

**门禁存在的理由**：蒸馏素材是「高分轨迹」，而高分轨迹里 Agent 做对的事，未必是 LLM 总结出的
那条原因——它很容易把「这次恰好选中了便宜货」讲成一条听着很有道理、照做却会拖慢链路的规矩。
所以候选写进库之前，拿同类 query 真跑一遍：分数不退化才认。judge 有 ±13 的单样本抖动（记忆
rubric-judge-calibration-pitfalls），所以判据是「P0 不新破 + 总分跌幅不超过
``STRATEGY_GATE_MAX_DROP``」，而不是「必须涨」——要求涨等于要求噪声站在自己这边。

用法::

    # 只蒸馏，不入库（dry-run，1 次 judge 调用）
    uv run python scripts/eval/distill_strategies.py --report data/eval/rubric_report_full.json
    # 过门禁并入库（每条候选重放 3 条同类 query，真跑 Agent + judge）
    uv run python scripts/eval/distill_strategies.py --report ... --write --gate-limit 1
    # 看库里现有策略（含退休的）
    uv run python scripts/eval/distill_strategies.py --list
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pydantic import BaseModel, Field  # noqa: E402

from app.eval.trace import extract_tool_calls, load_history  # noqa: E402
from app.memory.strategies import (  # noqa: E402
    Strategy,
    force_strategies,
    get_strategy_store,
    make_slug,
)
from app.utils.env import env_int  # noqa: E402

# 复用 run_rubric 的三件私有工具，**刻意不复制一份**：thread 回零、prior_context 兜底、种子集
# 读取，这三件事一旦两处实现分叉，门禁跑出来的分数就与基线报告不可比——而那正是本脚本唯一的判据。
from scripts.eval.run_rubric import _load_queries, _prior_context, _reset_thread  # noqa: E402

REPORT_PATH = Path("data/eval/rubric_report.json")
#: 蒸馏结果落盘处。**每次蒸馏都写**，且 ``--candidates`` 能读回来跳过蒸馏——否则「先 dry-run 看看，
#: 满意了再 --write」这个必然的用法要付两次蒸馏钱，而且第二次蒸出来的还可能是另一批候选
#: （temperature 再低也不是 0 概率），于是你门禁验的根本不是刚才看过的那几条。
CANDIDATES_PATH = Path("data/eval/strategy_candidates.json")
#: 门禁重放：同一条 query 的总分比基线低超过这个数就算退化。见模块 docstring 对 judge 抖动的说明。
GATE_MAX_DROP = env_int("STRATEGY_GATE_MAX_DROP", 12)
#: 门禁要重放几条同类 query。手册定的 3：再少就是单样本、被 judge 抖动主导；再多就是烧钱。
GATE_REPLAYS = 3


class _Candidate(BaseModel):
    """LLM 蒸馏出的一条策略候选（还没过门禁，所以不是 :class:`Strategy`）。"""

    category: str = Field(description="场景类别，必须从给定的 bucket 清单里原样选一个")
    slug: str = Field(default="", description="原子标识，英文小写下划线，如 landed_cost_first")
    trigger: str = Field(description="什么局面下适用，一句话")
    trigger_keywords: list[str] = Field(
        default_factory=list,
        description="能在**用户原话**里出现的原子词（中英皆可，3~6 个）。这是唯一的匹配抓手",
    )
    actions: list[str] = Field(default_factory=list, description="该怎么做，2~4 条祈使句")


class _CandidateSet(BaseModel):
    strategies: list[_Candidate] = Field(default_factory=list)


_DISTILL_PROMPT = """你在为一个电商购物 Agent 提炼**成功策略**。下面是若干评测高分轨迹\
（含用户 query、场景 bucket、总分、工具调用序列、评分细则里 judge 给的理由）。

请提炼最多 {n} 条策略，每条回答「遇到哪类局面，就该怎么打」：
- trigger 写**局面**（如「用户给了预算，且品类里充斥低价配件」），不要写商品。
- trigger_keywords 是能在**用户原话**里出现的词（如 预算/budget/便宜/配件）——它是确定性匹配\
的唯一抓手，写成只在工具返回里出现的词等于这条策略永远不会被触发。
- actions 是 2~4 条祈使句，说的是**决策方式**（先调什么、什么时候收尾、拿什么当排序依据），\
不要复述具体商品或价格。
- category 必须从这份清单里原样选一个：{buckets}
- 条与条之间去重；只提炼你能从多条轨迹里看出**共性**的，看不出就少给几条。

高分轨迹：
{traces}

只输出一个 json 对象，形如：
{{"strategies": [{{"category": "…", "slug": "…", "trigger": "…", \
"trigger_keywords": ["…"], "actions": ["…"]}}]}}
不要输出 json 以外的任何文字。"""


def _high_score_records(report_path: Path, min_score: float) -> list[dict]:
    """报告里够格当素材的记录（``is_high_score`` 或总分 ≥ ``min_score``）。"""
    if not report_path.exists():
        raise SystemExit(f"找不到评测报告 {report_path}，请先跑 scripts/eval/run_rubric.py")
    records = json.loads(report_path.read_text(encoding="utf-8"))
    return [
        r
        for r in records
        if r.get("ok")
        and (r["result"].get("is_high_score") or r["result"].get("total", 0) >= min_score)
    ]


def _render_trace(rec: dict) -> str:
    """一条高分记录渲染成蒸馏素材：query + bucket + 分 + 工具序列 + judge 的理由。

    工具序列从 ``output/eval_<id>/history.json`` 取，取不到就略过那一行——轨迹产物是
    gitignore 的，换台机器 / 清过 output 就没有，不该因此整条素材作废。
    """
    res = rec["result"]
    tools: list[str] = []
    hist = Path(f"output/eval_{rec['id']}/history.json")
    if hist.exists():
        tools = [c["name"] for c in extract_tool_calls(load_history(hist))]
    why = "；".join(
        s["rationale"][:80] for s in res.get("scores", []) if s.get("tier") in ("P0", "P1")
    )[:400]
    lines = [
        f"- query：{res.get('query', '')}（bucket={rec.get('bucket', '')}，得分 {res['total']}）"
    ]
    if tools:
        lines.append(f"  工具序列：{' → '.join(tools)}")
    if why:
        lines.append(f"  judge 认可的点：{why}")
    return "\n".join(lines)


async def _distill(records: list[dict], top_n: int) -> list[_Candidate]:
    """一次 judge 调用，把全部素材一起给它——分条调用既贵又更容易蒸出重复的条目。"""
    from app.agent.invoke import call_structured
    from app.agent.llm import get_judge_llm

    buckets = sorted({r.get("bucket", "") for r in records if r.get("bucket")})
    out = await call_structured(
        get_judge_llm(),
        _DISTILL_PROMPT.format(
            n=top_n,
            buckets=" / ".join(buckets),
            traces="\n".join(_render_trace(r) for r in records),
        ),
        _CandidateSet,
    )
    return out.strategies[:top_n]


def _gate_queries(records: list[dict], cand: _Candidate, seeds: dict[str, dict]) -> list[dict]:
    """给一条候选挑门禁重放用的「同类 query」：同 bucket 的高分记录，按 id 稳定排序取前 3。

    **同类的判据是 bucket 而不是触发词命中**：触发词是这条策略自己写的，拿它来选考题等于让
    考生自己出卷——它只会选出必然命中的那几条，而门禁真正要验的恰恰是「这条策略在它自称适用
    的整个场景里都不添乱」。
    """
    same = sorted(
        (r for r in records if r.get("bucket") == cand.category and r["id"] in seeds),
        key=lambda r: r["id"],
    )
    return same[:GATE_REPLAYS]


async def _replay_one(rec: dict, seed: dict, cand: _Candidate) -> tuple[float, list[str], int]:
    """强制注入这条候选，重放一条 query，返回 ``(总分, P0 失败清单, 本轮模型调用数)``。

    模型调用数一并回传是为了**把门禁的真实成本打在明面上**：这套机制最容易变质的方式，是有人
    图省事把 ``--gate-limit`` 调大，一条命令悄悄烧掉上百次调用还没人察觉。
    """
    from app.agent.orchestrator import run_agent
    from app.eval.rubric import evaluate

    strategy = Strategy(
        category=cand.category,
        slug=cand.slug or make_slug(cand.trigger),
        trigger=cand.trigger,
        trigger_keywords=list(cand.trigger_keywords),
        actions=list(cand.actions),
    )
    thread_id = f"strategy_gate_{rec['id']}"
    await _reset_thread(thread_id)
    turns: list[str] = seed.get("turns") or [seed["query"]]
    with force_strategies([strategy]):
        for warmup in turns[:-1]:
            await run_agent(warmup, thread_id=thread_id, user_id=None)
        run = await run_agent(turns[-1], thread_id=thread_id, user_id=None)
    scored = await evaluate(
        turns[-1],
        run,
        seed.get("constraints"),
        seed.get("intent", "shopping"),
        True,  # 复用细则缓存：门禁比的是 Agent 的表现，尺子必须与基线那次是同一把
        prior_context=_prior_context(seed),
    )
    return scored.total, list(scored.p0_failures), int(run.get("model_calls") or 0)


async def _gate(
    cand: _Candidate, records: list[dict], seeds: dict[str, dict], *, max_calls: int = 0
) -> bool:
    """门禁：3 条同类 query 重放，P0 不新破且总分跌幅不超过阈值才算过。

    ``max_calls`` 是 Agent 侧模型调用的**硬预算**（0 = 不限）。预算不够再跑一条时**中止并判不过**
    ——不是「跑了两条都 OK 就放行」。门禁的判据是「3 条同类 query 都不退化」，跑了 2 条就放行等于
    偷偷把标准从 3 改成 2，而报告上仍写着「过了门禁」。预算不够就是没验完，没验完就不能入库。
    估算下一条要花多少：取已跑过的最大值（保守），第一条无从估算，一律放行。
    """
    picked = _gate_queries(records, cand, seeds)
    ok = True
    calls = 0
    worst = 0
    for rec in picked:
        if max_calls and calls and calls + worst > max_calls:
            print(
                f"  [中止] 已用 {calls} 次 Agent 调用，再跑一条预计超过预算 {max_calls}——"
                f"门禁只跑了 {picked.index(rec)}/{len(picked)} 条，**判不过**，不入库。"
            )
            return False
        base_total = rec["result"]["total"]
        base_p0 = set(rec["result"].get("p0_failures") or [])
        total, p0, n = await _replay_one(rec, seeds[rec["id"]], cand)
        calls += n
        worst = max(worst, n)
        new_p0 = sorted(set(p0) - base_p0)
        drop = base_total - total
        verdict = "OK"
        if new_p0:
            verdict, ok = f"P0 新破 {'/'.join(new_p0)}", False
        elif drop > GATE_MAX_DROP:
            verdict, ok = f"退化 {drop:.1f} 分（阈值 {GATE_MAX_DROP}）", False
        print(
            f"  {rec['id']:32s} 基线 {base_total:5.1f} → 重放 {total:5.1f}  "
            f"{verdict}（本条 {n} 次模型调用）"
        )
    print(f"  本条候选的门禁共 {calls} 次 Agent 侧模型调用（judge 另计 {len(picked)} 次）")
    return ok


async def _list_strategies() -> None:
    rows = await get_strategy_store().read_all()
    if not rows:
        print("策略库是空的。")
        return
    print(f"策略库共 {len(rows)} 条：")
    for s in sorted(rows, key=lambda x: (x.status, x.dedup_key)):
        print(
            f"  [{s.status:8s}] {s.dedup_key:44s} hp={s.health} hits={s.hits} "
            f"fails={s.consecutive_failures}\n      触发：{s.trigger}\n      "
            + "\n      ".join(f"· {a}" for a in s.actions)
        )


async def main(args: argparse.Namespace) -> int:
    if args.list:
        await _list_strategies()
        return 0

    report_path = Path(args.report)
    records = _high_score_records(report_path, args.min_score)
    if not records:
        print(f"报告 {report_path} 里没有够格的高分轨迹（is_high_score 或 ≥{args.min_score}）。")
        return 1
    print(f"素材：{len(records)} 条高分轨迹，来自 {report_path}")

    if args.candidates:
        cached = json.loads(Path(args.candidates).read_text(encoding="utf-8"))
        cands = [_Candidate.model_validate(c) for c in cached][: args.top_n]
        print(f"复用 {args.candidates} 里的 {len(cands)} 条候选（跳过蒸馏，零 LLM 调用）")
    else:
        cands = await _distill(records, args.top_n)
        CANDIDATES_PATH.parent.mkdir(parents=True, exist_ok=True)
        CANDIDATES_PATH.write_text(
            json.dumps([c.model_dump() for c in cands], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"候选已落 {CANDIDATES_PATH}（下次可 --candidates 复用，不必重蒸）")
    print(f"\n候选共 {len(cands)} 条：\n")
    for c in cands:
        print(f"  [{c.category}] {c.slug or make_slug(c.trigger)}")
        print(f"    触发：{c.trigger}")
        print(f"    触发词：{', '.join(c.trigger_keywords)}")
        for a in c.actions:
            print(f"    · {a}")
    if not args.write:
        print("\n（dry-run：未跑门禁、未入库。加 --write 才会真跑重放并写库）")
        return 0

    store = get_strategy_store()
    seeds = {q["id"]: q for q in _load_queries(None, None)}
    written = 0
    gated = 0
    skipped = 0
    for cand in cands:
        name = f"[{cand.category}] {cand.slug or make_slug(cand.trigger)}"
        picked = _gate_queries(records, cand, seeds)
        if len(picked) < GATE_REPLAYS:
            # **不降标准硬塞**：同类基线不足 3 条就没法判「不退化」，这条候选直接不入库。
            # 它不占 --gate-limit 的名额——那个闸是拦成本的，不该被跑都没跑的候选吃掉。
            print(f"\n{name}：同类基线只有 {len(picked)} 条（需 {GATE_REPLAYS}），跳过不入库。")
            skipped += 1
            continue
        if gated >= args.gate_limit:
            print(f"\n{name}：已达 --gate-limit={args.gate_limit}，本次不跑门禁（成本闸）。")
            skipped += 1
            continue
        gated += 1
        print(f"\n门禁重放：{name}")
        if not await _gate(cand, records, seeds, max_calls=args.max_agent_calls):
            print("  → 未过门禁，不入库。")
            continue
        await store.upsert(
            Strategy(
                category=cand.category,
                slug=cand.slug or make_slug(cand.trigger),
                trigger=cand.trigger,
                trigger_keywords=list(cand.trigger_keywords),
                actions=list(cand.actions),
                evidence=[
                    f"{r['id']}@{r['result']['total']}" for r in _gate_queries(records, cand, seeds)
                ],
                source_report=report_path.name,
            )
        )
        written += 1
        print("  → 过门禁，已入库。")
    print(f"\n入库 {written} 条；跑了门禁 {gated} 条；没跑门禁 {skipped} 条。")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=str, default=str(REPORT_PATH), help="评测报告路径")
    parser.add_argument("--min-score", type=float, default=80.0, help="总分达到多少也算高分素材")
    parser.add_argument("--top-n", type=int, default=3, help="最多蒸馏几条候选")
    parser.add_argument(
        "--write", action="store_true", help="跑门禁重放并写库（默认 dry-run，只蒸馏不跑 Agent）"
    )
    parser.add_argument(
        "--gate-limit",
        type=int,
        default=1,
        help=(
            "最多给几条候选跑门禁。每条要真跑 3 次 Agent + 3 次 judge，默认 1 是**成本闸**："
            "不写这个默认值，一次 --write 就可能悄悄烧掉几十次模型调用。"
        ),
    )
    parser.add_argument(
        "--max-agent-calls",
        type=int,
        default=0,
        help="门禁重放的 Agent 侧模型调用硬预算（0=不限）。预算不够跑完 3 条即判不过，不入库。",
    )
    parser.add_argument(
        "--candidates",
        type=str,
        default="",
        help=f"复用已落盘的候选（默认落在 {CANDIDATES_PATH}），跳过蒸馏那次 LLM 调用",
    )
    parser.add_argument("--list", action="store_true", help="只打印库里现有策略，不蒸馏")
    raise SystemExit(asyncio.run(main(parser.parse_args())))
