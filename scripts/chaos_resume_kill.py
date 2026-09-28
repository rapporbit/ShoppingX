"""按步续跑故障注入：worker 跑到工具半路被 kill -9，看接管方是续跑还是整轮重跑。

用法（需要一台**可清空**的真 Redis）::

    uv run --no-sync python scripts/chaos_resume_kill.py --redis-url redis://localhost:6391/0
    uv run --no-sync python scripts/chaos_resume_kill.py --redis-url ... --no-checkpoint  # 对照组

**测的是真链路，只桩两头。** worker 子进程跑真 ``RedisStreamQueue.consume`` → 真
``app.worker.handle_task`` → 真 ``run_agent``（检查点、harness、事件日志都是生产代码）。桩掉的只有
模型和两个工具：模型按 context 里已有的工具结果决定下一步（先 fast_tool、再 slow_tool、再收尾），
不依赖进程内计数，所以换一个进程接着跑也照样按剧本走；工具每执行一次都记进 Redis，两个进程的
账汇在一处。零 LLM 花费。

**剧本。** 1 条任务 → worker A 跑到 slow_tool（A 里它睡 ``--slow-sec``）→ 确认检查点已在 Redis →
kill -9 A → 起 worker B（slow_tool 立即返回）→ 等终态 done → 汇总：B 发了几次模型调用、哪些工具
被重跑、检查点收尾后删没删、事件流里有没有给崩溃那行补的 ``tool_end(error)``。
"""

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CALLS = "rv:model_calls"
TOOLS = "rv:tool_runs"
TASK_ID = "rv-task-1"
THREAD_ID = "rv-thread-1"


def _child_env(args: argparse.Namespace, workdir: Path) -> dict[str, str]:
    """子进程环境：所有 Redis 指向测试库、DB 用临时 SQLite、关外发观测。显式设值压过 .env。"""
    env = dict(os.environ)
    for key in ("QUEUE_REDIS_URL", "EVENT_REDIS_URL", "DEDUP_REDIS_URL", "BREAKER_REDIS_URL"):
        env[key] = args.redis_url
    env["DATABASE_URL"] = f"sqlite+aiosqlite:///{workdir / 'rv.db'}"
    env["LANGFUSE_ENABLED"] = "false"
    env["RV_OUTPUT_ROOT"] = str(workdir / "output")
    return env


# ── worker 子进程 ─────────────────────────────────────────────────────────────
def _install_stubs(name: str, slow_sec: float, redis_url: str, no_checkpoint: bool) -> None:
    import redis.asyncio as aredis
    from agentscope.credential import OpenAICredential
    from agentscope.message import TextBlock, ToolCallBlock, ToolResultState
    from agentscope.model import ChatResponse, OpenAIChatModel
    from agentscope.tool import FunctionTool, ToolChunk, Toolkit

    import app.agent.agents as ag
    import app.agent.llm as llm
    import app.utils.path_utils as path_utils
    from app.agent import checkpoint
    from app.harness.adapter import HarnessAgentAdapter

    rec = aredis.from_url(redis_url)
    path_utils.OUTPUT_ROOT = Path(os.environ["RV_OUTPUT_ROOT"])
    if no_checkpoint:
        checkpoint.set_client(None)

    async def fast_tool() -> ToolChunk:
        """快工具。"""
        await rec.rpush(TOOLS, json.dumps({"w": name, "tool": "fast_tool"}))
        ok = ToolResultState.SUCCESS
        return ToolChunk(content=[TextBlock(type="text", text="fast-ok")], state=ok)

    async def slow_tool() -> ToolChunk:
        """慢工具。"""
        await rec.rpush(TOOLS, json.dumps({"w": name, "tool": "slow_tool", "t": time.time()}))
        await asyncio.sleep(slow_sec)
        ok = ToolResultState.SUCCESS
        return ToolChunk(content=[TextBlock(type="text", text="slow-ok")], state=ok)

    def _done(messages: list[Any], tool: str) -> bool:
        return any(
            getattr(b, "type", None) == "tool_result" and getattr(b, "name", None) == tool
            for m in messages
            for b in (m.content if isinstance(m.content, list) else [])
        )

    async def _call(*_a: object, messages: list[Any], **_kw: object) -> ChatResponse:
        await rec.rpush(CALLS, json.dumps({"w": name}))
        for i, tool in enumerate(("fast_tool", "slow_tool")):
            if not _done(messages, tool):
                blk = ToolCallBlock(type="tool_call", id=f"{name}-c{i}", name=tool, input="{}")
                return ChatResponse(content=[blk], is_last=True)
        return ChatResponse(content=[TextBlock(type="text", text="rv-done")], is_last=True)

    model = OpenAIChatModel(
        credential=OpenAICredential(api_key="sk-stub", base_url="http://localhost:1/v1"),
        model="stub",
        stream=False,
        max_retries=0,
    )
    model._call_api = _call  # type: ignore[method-assign]
    ag.get_tier_llm = lambda _tier: model  # type: ignore[assignment]
    llm.get_llm = lambda: model  # type: ignore[assignment]
    llm.get_fast_llm = lambda: model  # type: ignore[assignment]

    async def _no_prefill(self: Any, agent: Any) -> None:
        return None

    HarnessAgentAdapter._prefill = _no_prefill  # type: ignore[method-assign]
    real_build = ag.build_toolkit

    async def _toolkit(**kw: Any) -> Toolkit:
        tk = await real_build(**kw)
        await tk.add_tool(FunctionTool(fast_tool, is_read_only=True))
        await tk.add_tool(FunctionTool(slow_tool, is_read_only=True))
        return tk

    ag.build_toolkit = _toolkit  # type: ignore[assignment]


async def run_worker(args: argparse.Namespace) -> None:
    _install_stubs(args.name, args.slow_sec, args.redis_url, args.no_checkpoint)
    from app.db.session import init_db
    from app.harness.setup import setup_harness
    from app.queue import get_task_queue
    from app.worker import handle_task

    await init_db()
    setup_harness()
    await get_task_queue().consume(  # type: ignore[attr-defined]
        args.name,
        handle_task,
        lambda: False,
        1,
        claim_idle_ms=args.claim_idle_ms,
        heartbeat_sec=args.heartbeat,
        lease_ms=args.lease_ms,
    )


# ── 编排进程 ─────────────────────────────────────────────────────────────────
def _spawn(args: argparse.Namespace, name: str, slow_sec: float, env: dict[str, str]) -> Any:
    cmd = [sys.executable, __file__, "worker", "--name", name, "--redis-url", args.redis_url]
    cmd += ["--slow-sec", str(slow_sec), "--claim-idle-ms", str(args.claim_idle_ms)]
    cmd += ["--heartbeat", str(args.heartbeat), "--lease-ms", str(args.lease_ms)]
    if args.no_checkpoint:
        cmd.append("--no-checkpoint")
    log = open(Path(env["RV_OUTPUT_ROOT"]).parent / f"{name}.log", "wb")  # noqa: SIM115
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)


async def _wait(pred: Any, limit: float, what: str) -> None:
    deadline = time.monotonic() + limit
    while time.monotonic() < deadline:
        if await pred():
            return
        await asyncio.sleep(0.2)
    raise SystemExit(f"等不到：{what}（{limit}s）")


async def orchestrate(args: argparse.Namespace) -> dict[str, Any]:
    import redis.asyncio as aredis

    from app.agent.checkpoint import KEY_PREFIX
    from app.queue.ports import IntentTask
    from app.queue.redis_stream import RedisStreamQueue

    client = aredis.from_url(args.redis_url)
    await client.flushdb()
    rq = RedisStreamQueue(client)
    await rq.ensure_group()
    workdir = Path(tempfile.mkdtemp(prefix="rv-"))
    env = _child_env(args, workdir)
    ck_key = f"{KEY_PREFIX}{TASK_ID}"

    async def rows(key: str) -> list[dict[str, Any]]:
        return [json.loads(r) for r in await client.lrange(key, 0, -1)]

    await rq.enqueue(IntentTask.create(task_id=TASK_ID, thread_id=THREAD_ID, query="rv"))
    a = _spawn(args, "wA", args.slow_sec, env)

    async def slow_started() -> bool:
        return any(r["tool"] == "slow_tool" for r in await rows(TOOLS))

    await _wait(slow_started, 60, "worker A 进入 slow_tool")
    ck_before_kill = bool(await client.exists(ck_key))
    os.kill(a.pid, signal.SIGKILL)
    kill_t = time.time()
    b = _spawn(args, "wB", 0.0, env)

    async def finished() -> bool:
        st = await rq.get_status(TASK_ID)
        return st is not None and st.state in ("done", "failed")

    await _wait(finished, args.timeout, "任务终态")
    done_t = time.time()
    b.kill()

    calls, tools = await rows(CALLS), await rows(TOOLS)
    events = [
        json.loads(f[b"data"]) if b"data" in f else {k.decode(): v.decode() for k, v in f.items()}
        for _, f in await client.xrange(f"shoppingx:events:{THREAD_ID}")
    ]
    status = await rq.get_status(TASK_ID)
    await client.aclose()
    by = lambda rs, w: [r.get("tool", "call") for r in rs if r["w"] == w]  # noqa: E731
    return {
        "mode": "no-checkpoint（对照）" if args.no_checkpoint else "checkpoint",
        "final_state": status.state if status else None,
        "checkpoint_in_redis_before_kill": ck_before_kill,
        "model_calls": {"A": len(by(calls, "wA")), "B": len(by(calls, "wB"))},
        "tool_runs": {"A": by(tools, "wA"), "B": by(tools, "wB")},
        "kill_to_done_sec": round(done_t - kill_t, 2),
        "events_tail": [str(e)[:160] for e in events[-8:]],
        "workdir": str(workdir),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("mode", nargs="?", default="run", choices=["run", "worker"])
    ap.add_argument("--name", default="wA")
    ap.add_argument("--redis-url", required=True, help="会被 FLUSHDB，务必是一次性的库")
    ap.add_argument("--slow-sec", type=float, default=120.0)
    ap.add_argument("--no-checkpoint", action="store_true")
    ap.add_argument("--heartbeat", type=float, default=1.0)
    ap.add_argument("--lease-ms", type=int, default=4_000)
    ap.add_argument("--claim-idle-ms", type=int, default=2_000)
    ap.add_argument("--timeout", type=float, default=90.0)
    args = ap.parse_args()
    if args.mode == "worker":
        asyncio.run(run_worker(args))
        return
    print(json.dumps(asyncio.run(orchestrate(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
