"""批 1 验收③：``WORKER_MODE=clone`` vs ``split`` 同框架对照实验。

回答面经 Q13「为什么不用同质 fork」——不靠架构推理，靠同一运行时、同一批 query、同一把尺子
量出来的差。每条 query 记五列：Rubric（总分 + P0 破没破）、工具轮数、token/成本、wall 时间、
**worker 写工具误调次数**。

为什么必须串行（``concurrency`` 固定 1）：wall 是本实验的指标之一，并发跑会让 6 条互抢网关
名额，clone/split 的差异直接被排队噪声淹没。慢是代价，不是 bug。

为什么模式靠外部 env 而不是参数：``app.agent.agents.WORKER_MODE`` 是 import 时读的模块级常量，
一个进程里切不了。所以一个进程只跑一种模式，跑两次。

用法（一遍 = 一个模式一次）：
    WORKER_MODE=split uv run python -u scripts/eval/compare_worker_mode.py --run 1
    WORKER_MODE=clone uv run python -u scripts/eval/compare_worker_mode.py --run 1
产物：``data/eval/worker_mode_<mode>_<run>.json``（gitignore），汇总用 --report 读回来出表。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_rubric import _load_queries, _reset_thread  # noqa: E402

from app.agent.fork_guard import current_fork_depth  # noqa: E402
from app.agent.orchestrator import run_agent  # noqa: E402
from app.agent.tracing import flush_traces  # noqa: E402
from app.eval.rubric import RubricResult, evaluate  # noqa: E402
from app.eval.trace import extract_tool_calls  # noqa: E402
from app.harness.middleware import harness_hook  # noqa: E402

# 六条对照 query：三条跨平台（并行派发的主靶）、一条长链到手价、一条交易（写边界的唯一观测
# 点——clone 下 worker 拿得到 create_order 且 depth_gate 名单里没有它）、一条闲聊（量「小事也
# 派一趟」的多余开销）。刻意不含 q01：它触发 bundle 的 ask_user，评测无人应答会干等 120s，
# 把 wall 这一列彻底污染（见手册 §0.1 L8 验收）。
DEFAULT_QUERIES = [
    "q05_price_compare_samsung",
    "q15_full_chain_kitchen",
    "q17_compare_hp_ink",
    "q06_landed_cost_luggage",
    "tr01_order_needs_confirm",
    "q09_chitchat_capability",
]

# 写工具：worker（depth≥1）调到任何一个都算误调。clone 下没有任何机制拦得住，只有 _order_guard
# 的确认卡把 confirmed=True 退回出卡——那是「没落库」，不是「没越界」。
WRITE_TOOLS = frozenset({"create_order", "cancel_order"})

_COST_RE = re.compile(r"cost thread=(\S+) usd=([\d.]+) in=(\d+) out=(\d+) calls=(\d+)")
_USAGE_RE = re.compile(r"usage thread=(\S+) calls=(\d+).*?cache_read=(\d+) hit=([\d.]+)%")

# 全量工具调用探针：``(depth, tool_name)``。**只有它能看见 worker 的调用**——worker 的
# messages 不回传主 loop，解析主 thread 的轨迹永远数不到子 Agent 调了什么。注册成 pre_tool_call
# 的最低优先级 Hook（在闸之前跑），所以记的是「模型意图调用」的次数，被闸拦下的也算——写工具
# 误调这一列要的正是「模型敢不敢调」，而不是「拦没拦住」。
_PROBE: list[tuple[int, str]] = []


@harness_hook("pre_tool_call", name="__exp_tool_probe", priority=0)
async def _probe(context: dict) -> None:
    """只观测不干预：任何异常都会变成被测流程的行为改变，所以整段吞掉。"""
    try:
        _PROBE.append((current_fork_depth(), str(context.get("tool_name", ""))))
    except Exception:  # noqa: BLE001
        pass
    return None


class _LogTap(logging.Handler):
    """扒 orchestrator 的 cost/usage 日志行——用量只进日志、不在 run_agent 返回值里。

    子 Agent 与主 loop 共用一棵记账树（子继承父 session_dir），所以主 thread 那条 cost 行
    就是**全树**合计，worker 的开销已经含在里面——这正是本实验要比的东西。
    """

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.rows: list[tuple[str, dict[str, float]]] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001 —— 采集器绝不能把被测流程带崩
            return
        if m := _COST_RE.search(msg):
            self.rows.append(
                (
                    m.group(1),
                    {
                        "cost_usd": float(m.group(2)),
                        "input_tokens": int(m.group(3)),
                        "output_tokens": int(m.group(4)),
                        "model_calls": int(m.group(5)),
                    },
                )
            )
        elif m := _USAGE_RE.search(msg):
            self.rows.append((m.group(1), {"cache_hit_rate": float(m.group(4))}))

    def drain(self) -> dict[str, float]:
        """取本条 query 期间的行并清空。

        后写覆盖先写是**刻意**的：多轮 case（``turns``）每轮都会打一条，而记账树每轮结束就
        ``reset``，所以最后一条 = 被打分的那一轮的用量，与 Rubric 的口径对齐。
        """
        merged: dict[str, float] = {}
        for _thread, payload in self.rows:
            merged.update(payload)
        self.rows.clear()
        return merged


async def _run_one(q: dict, tap: _LogTap, use_cache: bool) -> dict:
    """跑一条 query 并采集五列。异常收成一条失败记录——一条炸不该赔掉整批的 token。"""
    qid = q["id"]
    thread_id = f"cmpwm_{qid}"
    _PROBE.clear()
    tap.drain()
    await _reset_thread(thread_id)
    turns: list[str] = q.get("turns") or [q["query"]]
    t0 = time.perf_counter()
    try:
        for warmup in turns[:-1]:
            await run_agent(warmup, thread_id=thread_id)
        run = await run_agent(turns[-1], thread_id=thread_id)
        wall = time.perf_counter() - t0
        scored: RubricResult = await evaluate(
            turns[-1], run, q.get("constraints"), q.get("intent", "shopping"), use_cache
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  [error] {qid:32s} {type(exc).__name__}: {exc}", flush=True)
        return {"id": qid, "ok": False, "error": f"{type(exc).__name__}: {exc}"}

    calls = list(_PROBE)
    worker_calls = [name for depth, name in calls if depth >= 1]
    usage = tap.drain()
    rec = {
        "id": qid,
        "ok": True,
        "total": scored.total,
        "pass": scored.overall_pass,
        "p0_broken": list(scored.p0_failures),
        "p1_violations": list(scored.p1_violations),
        "p2_avg": scored.p2_avg,
        "wall_sec": round(wall, 1),
        "main_tool_calls": len([1 for depth, _ in calls if depth == 0]),
        "worker_tool_calls": len(worker_calls),
        "worker_write_calls": len([n for n in worker_calls if n in WRITE_TOOLS]),
        "dispatches": len([1 for depth, n in calls if depth == 0 and n == "task_dispatch"]),
        "tool_seq": [f"{d}:{n}" for d, n in calls],
        "final_items": len(run.get("items") or []),
        "reply_tool_calls": len(extract_tool_calls(run.get("messages") or [])),
        **usage,
    }
    flag = "PASS" if rec["pass"] else "FAIL"
    print(
        f"  [done] {qid:32s} {flag} {rec['total']:5.1f}  wall={rec['wall_sec']:6.1f}s"
        f"  calls={rec.get('model_calls', 0)}  usd={rec.get('cost_usd', 0)}"
        f"  worker_write={rec['worker_write_calls']}",
        flush=True,
    )
    return rec


async def _run_batch(ids: list[str], mode: str, run_idx: int, use_cache: bool) -> Path:
    tap = _LogTap()
    root = logging.getLogger()
    root.addHandler(tap)
    if root.level > logging.INFO:
        root.setLevel(logging.INFO)
    logging.getLogger("app.agent.orchestrator").setLevel(logging.INFO)
    queries = _load_queries(set(ids), None)
    print(f"[compare] mode={mode} run={run_idx} 共 {len(queries)} 条（串行）", flush=True)
    records: list[dict] = []
    try:
        for q in queries:
            records.append(await _run_one(q, tap, use_cache))
    finally:
        root.removeHandler(tap)
        flush_traces()
    out = Path("data/eval") / f"worker_mode_{mode}_{run_idx}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"mode": mode, "run": run_idx, "queries": ids, "records": records}
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[compare] 写入 {out}", flush=True)
    return out


def _agg(records: list[dict]) -> dict[str, float]:
    ok = [r for r in records if r.get("ok")]
    n = max(len(ok), 1)
    return {
        "n": len(ok),
        "p0_fail": sum(1 for r in ok if r["p0_broken"]),
        "total_avg": round(sum(r["total"] for r in ok) / n, 1),
        "wall_avg": round(sum(r["wall_sec"] for r in ok) / n, 1),
        "model_calls_avg": round(sum(r.get("model_calls", 0) for r in ok) / n, 1),
        "tool_calls_avg": round(
            sum(r["main_tool_calls"] + r["worker_tool_calls"] for r in ok) / n, 1
        ),
        "cost_avg": round(sum(r.get("cost_usd", 0.0) for r in ok) / n, 6),
        "worker_write": sum(r["worker_write_calls"] for r in ok),
        "dispatches": sum(r["dispatches"] for r in ok),
    }


def _report() -> None:
    """把落盘的各遍读回来出对照表。缺文件就跳过——允许边跑边看。"""
    rows: dict[str, list[dict]] = {}
    for path in sorted(Path("data/eval").glob("worker_mode_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.setdefault(payload["mode"], []).extend(payload["records"])
    if not rows:
        print("没有找到 data/eval/worker_mode_*.json")
        return
    head = (
        f"{'mode':7s} {'n':>3s} {'P0破':>5s} {'均分':>6s} {'wall':>7s}"
        f" {'模型调用':>8s} {'工具调用':>8s} {'成本$':>9s} {'写误调':>6s} {'派发':>5s}"
    )
    print(head)
    print("-" * len(head))
    for mode, recs in sorted(rows.items()):
        a = _agg(recs)
        print(
            f"{mode:7s} {a['n']:3d} {a['p0_fail']:5d} {a['total_avg']:6.1f} {a['wall_avg']:6.1f}s"
            f" {a['model_calls_avg']:8.1f} {a['tool_calls_avg']:8.1f} {a['cost_avg']:9.6f}"
            f" {a['worker_write']:6d} {a['dispatches']:5d}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=int, default=1, help="第几遍（产物文件名后缀）")
    parser.add_argument("--only", type=str, default="", help="覆盖默认六条，逗号分隔")
    parser.add_argument("--report", action="store_true", help="只读回已落盘结果出对照表")
    parser.add_argument("--refresh-rubric", action="store_true", help="忽略细则缓存")
    args = parser.parse_args()
    if args.report:
        _report()
        raise SystemExit(0)
    ids = [s.strip() for s in args.only.split(",") if s.strip()] or DEFAULT_QUERIES
    mode = os.environ.get("WORKER_MODE", "split")
    asyncio.run(_run_batch(ids, mode, args.run, use_cache=not args.refresh_rubric))
