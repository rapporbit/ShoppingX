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


class _OrderRig:
    """剧本：登记候选 A1 → create_order(A1) → 收尾。与 ``_wire`` 同形（slow_tool 不上场）。

    模型按 context 里已有的工具结果决定下一步（不靠进程内计数），续跑时照样按剧本走。
    崩溃点由测试在 ``prepare_order_confirmation`` 返回**之后**挂住：卡已落库、工具结果还没写回
    ——最近的检查点是「刚决定调 create_order」那一刻，续跑会把它**再执行一遍**。
    """

    ORDER = {
        "item_ids": ["A1"],
        "recipient_name": "Test Buyer",
        "country": "US",
        "city": "Austin",
        "address_line": "1 Main St",
    }

    def __init__(self, hang: str) -> None:
        self.hang = hang
        self.calls = 0
        self.runs = {"fast": 0, "slow": 0}
        self.crash_point = asyncio.Event()

    async def fast_tool(self) -> Any:
        """登记一件候选。"""
        from agentscope.message import TextBlock, ToolResultState
        from agentscope.tool import ToolChunk

        from app.tools._candidates import register
        from app.tools.schemas import ItemCandidate

        self.runs["fast"] += 1
        register(
            [
                ItemCandidate(
                    item_id="A1",
                    platform="amazon",
                    title="canvas pouch",
                    price=20,
                    currency="USD",
                    rating=4.6,
                )
            ]
        )
        ok = ToolResultState.SUCCESS
        return ToolChunk(content=[TextBlock(type="text", text="registered A1")], state=ok)

    async def slow_tool(self) -> Any:
        """慢工具。"""
        from agentscope.message import TextBlock, ToolResultState
        from agentscope.tool import ToolChunk

        self.runs["slow"] += 1
        if self.hang == "tool":
            self.crash_point.set()
            await asyncio.sleep(30)
        ok = ToolResultState.SUCCESS
        return ToolChunk(content=[TextBlock(type="text", text="slow-ok")], state=ok)

    async def _call(self, *_a: object, messages: list[Any], **_kw: object) -> Any:
        import json

        from agentscope.message import TextBlock, ToolCallBlock
        from agentscope.model import ChatResponse

        self.calls += 1
        done = {
            getattr(b, "name", None)
            for m in messages
            for b in (m.content if isinstance(m.content, list) else [])
            if getattr(b, "type", None) == "tool_result"
        }
        if "fast_tool" not in done:
            blk = ToolCallBlock(type="tool_call", id="f1", name="fast_tool", input="{}")
            return ChatResponse(content=[blk], is_last=True)
        if "create_order" not in done:
            order = ToolCallBlock(
                type="tool_call", id="o1", name="create_order", input=json.dumps(self.ORDER)
            )
            return ChatResponse(content=[order], is_last=True)
        return ChatResponse(content=[TextBlock(type="text", text="done")], is_last=True)


async def test_resume_reruns_create_order_without_second_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """续跑重跑 create_order：同一 run_id + 从检查点恢复的候选登记表 → 复用同一张确认卡。

    登记表若没恢复，hydrate 会去 Qdrant 回源（测试环境没有），create_order 直接报「不在候选里」。
    """
    import app.tools.create_order as co
    from app.agent.orchestrator import run_agent
    from app.trade.repository_sql import confirmation_repository

    redis = FakeRedis()
    checkpoint.set_client(redis)
    prepared: list[str] = []
    real_prepare = co.prepare_order_confirmation

    async def _spy(*a: Any, **kw: Any) -> Any:
        conf = await real_prepare(*a, **kw)
        prepared.append(conf.confirmation_id)
        if rig.hang == "order":  # 卡已落库、工具结果还没写回：在这一刻被硬杀
            rig.crash_point.set()
            await asyncio.sleep(30)
        return conf

    monkeypatch.setattr(co, "prepare_order_confirmation", _spy)
    rig = _OrderRig("order")
    _wire(monkeypatch, rig)  # type: ignore[arg-type]
    task = asyncio.create_task(run_agent("买 A1", "t-order", user_id="u-order", run_id="r-order"))
    await asyncio.wait_for(rig.crash_point.wait(), 5)
    key = f"{checkpoint.KEY_PREFIX}r-order"
    blob = redis.store[key]
    task.cancel()
    with contextlib.suppress(BaseException):
        await task
    redis.store[key] = blob  # 硬杀

    assert len(prepared) == 1  # 崩溃前卡已落库
    rig.hang = ""
    await run_agent("买 A1", "t-order", user_id="u-order", run_id="r-order")

    assert rig.runs["fast"] == 1  # 登记候选那步没重跑，A1 是靠检查点恢复的登记表拿到的
    assert len(prepared) == 2 and prepared[0] == prepared[1]  # 重跑了，但拿回的是同一张卡
    cards = await confirmation_repository().list_by_thread("u-order", "t-order")
    assert len(cards) == 1
