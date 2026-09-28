"""worker 接管后按步续跑的框架前提（ROADMAP「TODO · worker 接管后按步续跑」第一步）。

钉的是**真框架**（AgentScope 2.0.7）行为：一轮跑到半路被砍（模拟 worker 崩溃），把那一刻的
``AgentState`` 序列化再读回，新建 Agent 调 ``reply(None)``：

- 走的是「新 reply」分支（``cur_iter`` 归零、``reply_id`` 换新），但 ``_next_action`` 会从最后一条
  消息里挑出「有调用无结果」的工具调用直接执行——**已完成的工具不重跑，模型不重想**。
- ``cur_iter`` 与 ``reply_id`` 在 ``ReplyStartEvent`` 那一刻写回检查点的值：前者不写回会白送
  一截 ``max_iters``；后者不写回，补跑的块会落进一条新 assistant 消息（``append_context`` 按
  ``reply_id`` 认消息），「一轮一条消息」的结构被拆开。写回后与没崩溃的一轮结构完全一致。

不打网络：模型是按脚本逐次吐响应的 stub。
"""

import asyncio
import contextlib
from typing import Any

import pytest
from agentscope.agent import Agent
from agentscope.event import ReplyStartEvent
from agentscope.message import Msg, TextBlock, ToolCallBlock, ToolResultState
from agentscope.model import ChatResponse
from agentscope.state import AgentState
from agentscope.tool import FunctionTool, ToolChunk, Toolkit

from tests.test_orchestrator import _fake_model, _user


class _Rig:
    """脚本：第 1 次模型调用 → fast_tool；第 2 次 → slow_tool；之后 → 收尾文本。

    ``hang`` 决定崩溃落在哪：``"tool"`` = slow_tool 执行中，``"model"`` = 第 3 次模型调用中。
    """

    def __init__(self, hang: str) -> None:
        self.hang = hang
        self.calls = 0
        self.runs = {"fast": 0, "slow": 0}
        self.crash_point = asyncio.Event()
        self.last_messages: list[Msg] = []

    async def _maybe_hang(self, where: str) -> None:
        if self.hang == where:
            self.crash_point.set()
            await asyncio.sleep(30)

    async def fast_tool(self) -> ToolChunk:
        """快工具。"""
        self.runs["fast"] += 1
        return ToolChunk(
            content=[TextBlock(type="text", text="fast-ok")], state=ToolResultState.SUCCESS
        )

    async def slow_tool(self) -> ToolChunk:
        """慢工具。"""
        self.runs["slow"] += 1
        await self._maybe_hang("tool")
        return ToolChunk(
            content=[TextBlock(type="text", text="slow-ok")], state=ToolResultState.SUCCESS
        )

    async def _call(self, *_a: object, messages: list[Msg], **_kw: object) -> ChatResponse:
        self.calls += 1
        self.last_messages = list(messages)
        if self.calls <= 2:
            name = "fast_tool" if self.calls == 1 else "slow_tool"
            blk = ToolCallBlock(type="tool_call", id=f"c{self.calls}", name=name, input="{}")
            return ChatResponse(content=[blk], is_last=True)
        await self._maybe_hang("model")
        return ChatResponse(content=[TextBlock(type="text", text="done")], is_last=True)

    async def agent(self, state: AgentState | None = None) -> Agent:
        model = _fake_model()
        model._call_api = self._call  # type: ignore[method-assign]
        toolkit = Toolkit()
        await toolkit.add_tool(FunctionTool(self.fast_tool, is_read_only=True))
        await toolkit.add_tool(FunctionTool(self.slow_tool, is_read_only=True))
        return Agent(name="t", system_prompt="sys", model=model, toolkit=toolkit, state=state)


async def _crash(rig: _Rig) -> tuple[str, int]:
    """跑到崩溃点，取检查点（序列化后的 state + cur_iter），然后砍掉任务。"""
    agent = await rig.agent()
    task = asyncio.create_task(_consume(agent.reply_stream(_user("hi"), yield_final_msg=True)))
    await asyncio.wait_for(rig.crash_point.wait(), 5)
    snap, cur_iter = agent.state.model_dump_json(), agent.state.cur_iter
    task.cancel()
    with contextlib.suppress(BaseException):
        await task
    return snap, cur_iter


async def _consume(stream: Any) -> Any:
    final = None
    async for evt in stream:
        final = evt
    return final


async def _resume(rig: _Rig, snap: str, cur_iter: int) -> tuple[Agent, Msg]:
    rig.hang = ""
    state = AgentState.model_validate_json(snap)
    reply_id = state.reply_id
    agent = await rig.agent(state=state)
    final = None
    async for evt in agent.reply_stream(None, yield_final_msg=True):
        if isinstance(evt, ReplyStartEvent):
            # 框架刚把两样都换掉了，写回检查点的值（生产侧同一动作在 HarnessAgentAdapter.on_reply）
            agent.state.reply_context.cur_iter = cur_iter
            agent.state.reply_context.reply_id = reply_id
        final = evt
    assert isinstance(final, Msg)
    return agent, final


@pytest.mark.parametrize(
    ("hang", "calls_after", "slow_runs", "unfinished"),
    [
        ("tool", 3, 2, ["c2"]),  # 砍在工具里：只补跑 c2，模型只多调 1 次（收尾）
        ("model", 4, 1, []),  # 砍在模型调用里：工具一个不重跑，只重发那次模型调用
    ],
)
async def test_resume_continues_from_crash_point(
    hang: str, calls_after: int, slow_runs: int, unfinished: list[str]
) -> None:
    rig = _Rig(hang)
    snap, cur_iter = await _crash(rig)
    restored = AgentState.model_validate_json(snap)
    assert [c.id for c in restored.get_unfinished_tool_calls("t")] == unfinished
    assert cur_iter == (1 if hang == "tool" else 2)

    agent, final = await _resume(rig, snap, cur_iter)

    assert final.get_text_content() == "done"
    assert rig.calls == calls_after  # 整轮重跑会是 2 + 3 = 5 次
    assert rig.runs == {"fast": 1, "slow": slow_runs}
    assert agent.state.cur_iter == 3  # 与没崩溃的一轮一致：两轮工具 + 一次收尾


async def test_resumed_turn_keeps_single_message_shape() -> None:
    """续跑后本轮仍是一条 assistant 消息，格式化后 tool_call 与结果紧挨（OpenAI 协议要求）。"""
    rig = _Rig("tool")
    snap, cur_iter = await _crash(rig)
    agent, _ = await _resume(rig, snap, cur_iter)

    assert [m.role for m in agent.state.context] == ["user", "assistant"]

    msgs = await agent.model.formatter.format(rig.last_messages)
    shape = [
        (m["role"], m.get("tool_call_id") or [t["id"] for t in m.get("tool_calls") or []])
        for m in msgs
    ]
    i = shape.index(("assistant", ["c2"]))
    assert shape[i + 1] == ("tool", "c2")
