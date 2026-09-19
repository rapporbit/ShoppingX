"""SKILL 触发验收：真跑模型，只量「第一批工具调用里有没有那条 ``Skill``」（阶段 S4）。

**为什么必须真跑模型**：阶段 S2 把 skill 正文的加载方式收敛成「模型自觉调 ``Skill(skill=…)``」、
删掉了全部预注入，于是「这轮该读哪份 skill」从一个可以用纯函数断言的注入判定，变成了模型第一
次调用时的一个决策。断言注入表已经无从断起——唯一还量得到的是模型自己发出来的那批工具调用。

**为什么跑到第一次模型调用就停**：分流表要求 skill 与「本轮第一个检索/读取工具」同轮发出，所以
答案在第一批 tool_call 里就已经定了；再往下跑只是把 item_search / research 的钱和时间白烧。截停
点选在 ``post_reflect``（``app/harness/adapter.py`` 的 ``_run_post_reflect``）：它跑在
``on_reasoning`` 之后、工具执行之前，context 里的 ``response_ai_message`` 正是模型这一轮想调什么。

**planner 的钱省不掉**：分流表有两行（intent_grounding=web、bundle_slots ≥2）的判据来自 planner
预置结果，它在第一次模型调用之前由 ``harness/prefill.py`` 跑掉。所以单条成本 = 1 次 planner
（fast 档）+ 1 次主模型，不是零。

**截停用 BaseException 而不是 HookRejectSignal**：后者是「拒绝这次工具调用」的业务语义，会让模型
换个招再来一轮（继续烧钱）；而 ``HarnessMiddleware.run`` 的兜底是 ``except Exception``，普通异常
会被它吞掉只留一行日志。要干净地把整轮掐断，信号必须是 ``Exception`` 之外的东西。

用法：
    TURN_CACHE_ENABLED=0 uv run python scripts/eval/run_skill_trigger.py
    ... --only sd01_multi_constraint,neg03_chitchat   # 只跑几条
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.agent.orchestrator import run_agent  # noqa: E402
from app.agent.tracing import flush_traces  # noqa: E402
from app.api.context import get_thread_id  # noqa: E402
from app.harness.middleware import harness  # noqa: E402
from app.recall.semantic_cache import turn_cache_enabled  # noqa: E402
from scripts.eval.build_skill_trigger import UNCOVERED  # noqa: E402
from scripts.eval.run_rubric import _reset_thread  # noqa: E402

DATASET_PATH = Path("data/eval/skill_trigger.jsonl")
REPORT_PATH = Path("data/eval/skill_trigger_report.json")

#: 单条墙钟上限。只跑一次模型调用，正常在 20s 内回来；这个值兜的是「永远回不来」。
CASE_TIMEOUT_SEC = 180

#: 框架内置 skill 阅读器的工具名（与 ``app.agent.skills.SKILL_VIEWER_TOOL_NAME`` 同一个值）。
SKILL_TOOL = "Skill"


class _StopAfterFirstCall(BaseException):
    """截停信号。继承 ``BaseException`` 的理由见模块 docstring。"""


#: thread_id → 第一批 tool_call。按 thread_id 分桶，因为并发跑时多条 case 共用同一个 hook。
_CAPTURED: dict[str, list[dict[str, Any]]] = {}


def _tool_calls(msg: Any) -> list[dict[str, Any]]:
    """把一条 assistant 消息里的 tool_call 块收成 ``[{name, skill}]``。

    ``skill`` 只对 ``Skill`` 工具有值。入参可能是 dict 也可能是 JSON 串（取决于框架版本怎么
    存），两种都认——与 ``orchestrator._skills_read`` 同一口径。
    """
    out: list[dict[str, Any]] = []
    for block in getattr(msg, "content", []) or []:
        if getattr(block, "type", None) != "tool_call":
            continue
        name = str(getattr(block, "name", "") or "")
        raw = getattr(block, "input", None)
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except ValueError:
                raw = None
        args = raw if isinstance(raw, dict) else {}
        out.append(
            {"name": name, "skill": str(args.get("skill") or "") if name == SKILL_TOOL else ""}
        )
    return out


def install_probe() -> None:
    """把探针装进 harness 注册表。

    **刻意不用模块级 ``@harness_hook`` 装饰器**：注册表是进程全局单例，用装饰器的话「``import``
    一下这个模块」就等于往生产 pipeline 里插了一个会抛 ``BaseException`` 的 hook——测试里导入
    本模块做纯函数断言，就会把别人的 post_reflect 用例一起掐断。装的动作留给 ``main``。

    priority 取 99（最后跑）：让 drift / phase / terminal 三个真 hook 先按原样跑完，尽量少改变
    被测对象的行为——虽然这里已经是本轮最后一件事，但「探针排在执法之后」是更安全的默认。
    """
    harness.register("post_reflect", "skill_trigger_probe", _probe_and_stop, priority=99)


async def _probe_and_stop(context: dict[str, Any]) -> dict[str, Any] | None:
    """记下这一轮模型想调什么，然后把整轮掐断。"""
    thread_id = get_thread_id() or ""
    _CAPTURED[thread_id] = _tool_calls(context.get("response_ai_message"))
    raise _StopAfterFirstCall


def _verdict(case: dict, calls: list[dict[str, Any]]) -> tuple[str, str]:
    """判这一条过没过，返回 ``(verdict, note)``。

    四种判法分开记而不是一个 bool：``MISS``（压根没读）与 ``WRONG_SKILL``（读错了份）要修的是
    两件事——前者是分流表那一行的措辞没被认出来，后者是两份 skill 的 description 撞车。
    """
    read = [c["skill"] for c in calls if c["name"] == SKILL_TOOL and c["skill"]]
    others = [c["name"] for c in calls if c["name"] != SKILL_TOOL]
    expected = case.get("skill")
    if not expected:
        if read:
            return "FALSE_FIRE", f"不该读却读了 {read}"
        return "PASS", ""
    if expected in read:
        # S2 的「同轮发出」口径：skill 该和本轮第一个检索/读取工具一起发，不该白等一次往返。
        return "PASS", "" if others else "同轮没带业务工具（白等一次往返）"
    if read:
        return "WRONG_SKILL", f"读成了 {read}"
    return "MISS", f"第一批只有 {others or '（无工具调用）'}"


async def _run_one(case: dict, user_id: str | None, sem: asyncio.Semaphore) -> dict[str, Any]:
    """跑一条：真发一次模型调用，拿到第一批 tool_call 就掐断。异常收成一条失败记录。"""
    async with sem:
        qid = case["id"]
        thread_id = f"skilltrig_{qid}"
        await _reset_thread(thread_id)
        _CAPTURED.pop(thread_id, None)
        t0 = time.monotonic()
        error = ""
        try:
            await asyncio.wait_for(
                run_agent(case["query"], thread_id=thread_id, user_id=user_id),
                timeout=CASE_TIMEOUT_SEC,
            )
            # 没被掐断 = 模型这一轮一个工具都没调（纯文字直答），calls 会是空表，按 MISS 判。
        except _StopAfterFirstCall:
            pass
        except Exception as exc:  # noqa: BLE001 —— 单条失败不拖垮整批
            error = f"{type(exc).__name__}: {exc}"
        calls = _CAPTURED.pop(thread_id, [])
        verdict, note = ("ERROR", error) if error else _verdict(case, calls)
        elapsed = time.monotonic() - t0
        print(f"  [done] {qid:28s} {verdict:12s} {elapsed:5.1f}s  {note}")
        return {
            "id": qid,
            "skill": case.get("skill"),
            "bucket": case.get("bucket", ""),
            "borderline": bool(case.get("borderline")),
            "query": case["query"],
            "verdict": verdict,
            "note": note,
            "first_round_tools": [c["name"] for c in calls],
            "skills_read": [c["skill"] for c in calls if c["name"] == SKILL_TOOL and c["skill"]],
            "elapsed_sec": round(elapsed, 1),
        }


def _load_cases(only: str, limit: int | None) -> list[dict]:
    if not DATASET_PATH.exists():
        raise SystemExit(f"标注集不存在：{DATASET_PATH}，先跑 build_skill_trigger.py 生成")
    rows = [
        json.loads(line) for line in DATASET_PATH.read_text("utf-8").splitlines() if line.strip()
    ]
    if only:
        wanted = {s.strip() for s in only.split(",") if s.strip()}
        rows = [r for r in rows if r["id"] in wanted]
    return rows[:limit] if limit else rows


def _summarize(results: list[dict]) -> dict[str, Any]:
    """两个主指标分开算：正例的触发率、负例的误触发率。合成一个总分会把两类错互相掩盖。"""
    pos = [r for r in results if r["skill"]]
    neg = [r for r in results if not r["skill"]]
    pos_hit = [r for r in pos if r["verdict"] == "PASS"]
    per_skill: dict[str, dict[str, int]] = {}
    for r in pos:
        row = per_skill.setdefault(r["skill"], {"n": 0, "pass": 0})
        row["n"] += 1
        row["pass"] += r["verdict"] == "PASS"
    return {
        "total": len(results),
        "positive": {
            "n": len(pos),
            "pass": len(pos_hit),
            "rate": round(len(pos_hit) / len(pos), 3) if pos else None,
            "miss": [r["id"] for r in pos if r["verdict"] == "MISS"],
            "wrong_skill": [r["id"] for r in pos if r["verdict"] == "WRONG_SKILL"],
            # 读对了但没跟业务工具同轮发——S2 分流表「和本轮第一个检索/读取工具同一轮发出」的代价面。
            "lone_skill_round": [r["id"] for r in pos_hit if r["note"]],
        },
        "negative": {
            "n": len(neg),
            "pass": sum(1 for r in neg if r["verdict"] == "PASS"),
            "false_fire": [r["id"] for r in neg if r["verdict"] == "FALSE_FIRE"],
        },
        "per_skill": per_skill,
        "errors": [r["id"] for r in results if r["verdict"] == "ERROR"],
        # 边界例单列：它们判错不一定是模型的问题，可能是标注口径本身就模糊。
        "borderline": [
            {"id": r["id"], "verdict": r["verdict"]} for r in results if r["borderline"]
        ],
        "uncovered_skills": UNCOVERED,
    }


async def main(
    only: str, limit: int | None, concurrency: int, user_id: str | None, out: Path
) -> None:
    if turn_cache_enabled():
        raise SystemExit(
            "拒跑：TURN_CACHE_ENABLED 开着，整轮缓存会让第二次跑直接复读、一次模型调用都不发，"
            "探针永远不触发、全判 MISS。请先关掉（TURN_CACHE_ENABLED=0）再跑。"
        )
    cases = _load_cases(only, limit)
    install_probe()
    print(f"[skill_trigger] {len(cases)} 条，并发 {concurrency}，只跑到第一次模型调用")
    sem = asyncio.Semaphore(concurrency)
    results = await asyncio.gather(*(_run_one(c, user_id, sem) for c in cases))
    summary = _summarize(list(results))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"summary": summary, "cases": list(results)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    p, n = summary["positive"], summary["negative"]
    fired = len(n["false_fire"])
    print(f"\n正例触发 {p['pass']}/{p['n']}（{p['rate']}）  负例误触发 {fired}/{n['n']}")
    if p["miss"] or p["wrong_skill"]:
        print(f"  未读：{p['miss']}  读错：{p['wrong_skill']}")
    if p["lone_skill_round"]:
        print(f"  只发了 Skill、没带业务工具（白等一次往返）：{p['lone_skill_round']}")
    if n["false_fire"]:
        print(f"  负例误读：{n['false_fire']}")
    print(f"报告写入 {out}")
    flush_traces()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default="", help="只跑这些 id（逗号分隔）")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条")
    parser.add_argument("--concurrency", type=int, default=3, help="并发条数（按 LLM 配额调）")
    parser.add_argument("--user-id", type=str, default=None, help="评测用户 id（记忆类 case 需要）")
    parser.add_argument("--out", type=Path, default=REPORT_PATH, help="报告落点")
    args = parser.parse_args()
    asyncio.run(main(args.only, args.limit, args.concurrency, args.user_id, args.out))
