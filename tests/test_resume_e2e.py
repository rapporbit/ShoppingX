"""按步续跑端到端：真装配的主 Agent + ``run_agent``，模拟 worker 在工具执行中被硬杀、接管方续跑。

「硬杀」的模拟：取消任务会走 orchestrator 的 finally（那里会删检查点），而真正的 SIGKILL 走不到。
所以取消前先把检查点拷出来，取消后确认它确实被删了，再原样放回去——等价于「进程死在那一刻」。
"""

import asyncio
import contextlib
from typing import Any

import pytest
from agentscope.tool import FunctionTool, Toolkit

from app.agent import checkpoint
from tests.conftest import FakeRedis
from tests.test_orchestrator import _fake_model
from tests.test_resume_from_checkpoint import _Rig


def _wire(monkeypatch: pytest.MonkeyPatch, rig: _Rig) -> None:
    """主 Agent 照常装配，只把模型换成 rig 的脚本、工具箱里加上 rig 的两个工具。"""
    from app.agent import agents as ag
    from app.harness.adapter import HarnessAgentAdapter
    from app.harness.setup import setup_harness

    async def _no_prefill(self: Any, agent: Any) -> None:
        return None

    setup_harness()
    model = _fake_model()
    model._call_api = rig._call  # type: ignore[method-assign]
    monkeypatch.setattr(HarnessAgentAdapter, "_prefill", _no_prefill)
    monkeypatch.setattr(ag, "get_tier_llm", lambda _tier: model)
    monkeypatch.setattr("app.agent.llm.get_llm", lambda: model)
    monkeypatch.setattr("app.agent.llm.get_fast_llm", lambda: model)

    real_build = ag.build_toolkit

    async def _toolkit(**kw: Any) -> Toolkit:
        tk = await real_build(**kw)
        await tk.add_tool(FunctionTool(rig.fast_tool, is_read_only=True))
        await tk.add_tool(FunctionTool(rig.slow_tool, is_read_only=True))
        return tk

    monkeypatch.setattr(ag, "build_toolkit", _toolkit)


def _users(out: dict[str, Any]) -> int:
    return sum(1 for m in out["messages"] if getattr(m, "role", None) == "user")


async def test_run_agent_resumes_after_hard_kill(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.agent.orchestrator import run_agent

    redis = FakeRedis()
    checkpoint.set_client(redis)

    # 对照：同一份脚本不崩溃地跑完一轮，要几次模型调用
    base = _Rig("")
    _wire(monkeypatch, base)
    clean = await run_agent("hi", "t-base", run_id="r-base")

    rig = _Rig("tool")
    _wire(monkeypatch, rig)
    task = asyncio.create_task(run_agent("hi", "t1", run_id="r1"))
    await asyncio.wait_for(rig.crash_point.wait(), 5)
    key = f"{checkpoint.KEY_PREFIX}r1"
    blob = redis.store[key]
    task.cancel()
    with contextlib.suppress(BaseException):
        await task
    assert key not in redis.store  # 取消走得到 finally，检查点已删
    redis.store[key] = blob  # 硬杀：finally 没跑，检查点留了下来

    from app.api import monitor

    closed: list[tuple[str, dict[str, Any]]] = []
    real_end = monitor.report_tool_end

    async def _spy_end(tool: str, **fields: Any) -> None:
        closed.append((tool, fields))
        await real_end(tool, **fields)

    monkeypatch.setattr(monitor, "report_tool_end", _spy_end)
    calls_at_crash = rig.calls
    rig.hang = ""
    out = await run_agent("hi", "t1", run_id="r1")

    # 续跑一开始先把崩溃时在跑的那行关掉（带 error），前端的先进先出配对才不会留一行空转
    assert closed and closed[0][0] == "slow_tool" and closed[0][1].get("error")

    assert rig.runs == {"fast": 1, "slow": 2}  # 快工具不重跑；慢工具只补跑崩在半路那一次
    assert rig.calls - calls_at_crash == base.calls - 2  # 前两次模型调用不再重发
    assert out["final_text"] == clean["final_text"]
    assert _users(out) == _users(clean)  # 没有重复追加用户消息
    assert key not in redis.store  # 续跑收尾后删掉


async def test_no_checkpoint_runs_fresh(monkeypatch: pytest.MonkeyPatch) -> None:
    """没有检查点（首次投递 / 抛异常后的重投）= 原来的整轮跑，行为不变。"""
    from app.agent.orchestrator import run_agent

    checkpoint.set_client(FakeRedis())
    rig = _Rig("")
    _wire(monkeypatch, rig)
    await run_agent("hi", "t2", run_id="r2")
    assert rig.runs == {"fast": 1, "slow": 1}
