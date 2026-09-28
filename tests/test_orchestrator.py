"""L3 验收：AgentScope 主链路（orchestrator / events / permissions）。

**测的是接缝，不是业务**——工具行为、harness 的各 hook、记忆判定各有自己的测试文件。
这里守的是迁移最容易悄悄摔的三处：

1. 收尾取的是**流出去的那条 Msg**（已过输出审核），不是 ``state.context`` 里的原文；
2. 续聊唯一一条腿：有 ``session.json`` 就恢复它，没有就空开局；
3. 写工具是**精准放行**的——没进放行表的工具照样要用户确认（不能靠 BYPASS 一档全开）。
"""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from agentscope.message import Msg, TextBlock, ToolResultBlock
from agentscope.state import AgentState
from agentscope.tool._response import ToolResultState

import app.agent.orchestrator as orch
from app.tools.shopping_summary import ShoppingSummaryOutput, SummaryItem


def _summary_block() -> ToolResultBlock:
    """一条 shopping_summary 的工具结果（JSON 文本，与 _as_tools 的输出形态一致）。"""
    out = ShoppingSummaryOutput(
        summary="为你精选 1 件：帆布旅行包。",
        items=[SummaryItem(item_id="A1", platform="amazon", title="canvas bag")],
    )
    return ToolResultBlock(
        type="tool_result",
        id="c1",
        name="shopping_summary",
        output=out.model_dump_json(),
    )


def _fake_agent(final_text: str, *, context: list[Msg] | None = None) -> Any:
    """假 Agent：吐一条最终 Msg；``context`` 是**本轮**产生的消息，与真实 Agent 一样在
    ``reply_stream`` 里才追加进 state.context（收尾产物只从本轮下标往后找，预置进去会被当上一轮）。
    """
    ctx = context if context is not None else []

    class _FakeAgent:
        def __init__(self) -> None:
            self.state = AgentState()
            self.inputs: list[Msg] = []

        async def reply_stream(self, inputs: Any, yield_final_msg: bool = False) -> Any:
            self.inputs = list(inputs)
            self.state.context.extend(ctx)
            yield Msg(
                name="shoppingx",
                role="assistant",
                content=[TextBlock(type="text", text=final_text)],
            )

    return _FakeAgent()


@pytest.fixture
def patched(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """顶掉会打网络 / 打库的收尾环节，留下主链路本身。"""
    captured: dict[str, Any] = {}

    async def _noop_curate(*_a: Any, **_kw: Any) -> None:
        return None

    async def _task_result(text: str, **kw: Any) -> None:
        captured["final_text"] = text
        captured["items"] = kw.get("items")

    monkeypatch.setattr(orch, "curate_turn", _noop_curate)
    monkeypatch.setattr(orch.monitor, "report_task_result", _task_result)
    return captured


# ---------- _extract_summary ----------


def test_extract_summary_reads_tool_result_json() -> None:
    msg = Msg(name="shoppingx", role="assistant", content=[_summary_block()])
    got = orch._extract_summary([msg])
    assert got is not None and got.items[0].item_id == "A1"


def test_extract_summary_none_when_tool_errored() -> None:
    """工具报错时结果是 ``[error] ...`` 文本——只按工具名认会把错误文案当清单。"""
    block = ToolResultBlock(
        type="tool_result",
        id="c1",
        name="shopping_summary",
        output="[error] ValueError: 没候选",
        state=ToolResultState.ERROR,
    )
    assert orch._extract_summary([Msg(name="a", role="assistant", content=[block])]) is None


def test_extract_summary_none_without_terminal_tool() -> None:
    msg = Msg(name="a", role="assistant", content=[TextBlock(type="text", text="闲聊")])
    assert orch._extract_summary([msg]) is None


# ---------- 会话恢复：唯一一条腿 session.json ----------


async def test_run_agent_starts_fresh_without_session_file(
    monkeypatch: pytest.MonkeyPatch, patched: dict[str, Any]
) -> None:
    """没有 session.json → 空开局：不回放 messages 表，当轮 query 是唯一的输入消息。"""
    agent = _fake_agent("已为你整理好清单。")

    async def _build(**kw: Any) -> Any:
        patched["state_arg"] = kw.get("state")
        return agent, SimpleNamespace()

    monkeypatch.setattr(orch, "build_main_agent", _build)
    await orch.run_agent("买个旅行包", thread_id="as-t1")

    assert patched["state_arg"] is None
    assert [m.role for m in agent.inputs] == ["user"]
    assert agent.inputs[-1].get_text_content().endswith("买个旅行包")


async def test_run_agent_resumes_from_session_file(
    monkeypatch: pytest.MonkeyPatch, patched: dict[str, Any]
) -> None:
    """有 session.json 就恢复它（含 middle_context 里的前几轮原话），当轮只追加一条 user 消息。"""
    from app.api.context import get_prior_queries

    session_dir = orch.ensure_session_dir("as-t2")
    prior = AgentState()
    prior.context = [Msg(name="user", role="user", content=[TextBlock(type="text", text="上轮")])]
    prior.middle_context[orch.PRIOR_QUERIES_KEY] = ["想买旅行包，不要塑料"]
    (session_dir / orch.STATE_FILE).write_text(prior.model_dump_json(), encoding="utf-8")

    agent = _fake_agent("好的。")

    async def _build(**kw: Any) -> Any:
        patched["state_arg"] = kw.get("state")
        patched["prior_seen"] = get_prior_queries()
        return agent, SimpleNamespace()

    monkeypatch.setattr(orch, "build_main_agent", _build)
    await orch.run_agent("接着聊", thread_id="as-t2")

    state_arg = patched["state_arg"]
    assert isinstance(state_arg, AgentState)
    assert state_arg.context[0].get_text_content() == "上轮"
    assert [m.role for m in agent.inputs] == ["user"]
    # 追问轮的 planner 靠前几轮原话重算约束：从 middle_context 读回、进 run 状态
    assert patched["prior_seen"] == ["想买旅行包，不要塑料"]


async def test_run_agent_saves_state_with_queries_for_next_turn(
    monkeypatch: pytest.MonkeyPatch, patched: dict[str, Any]
) -> None:
    """成功收尾 → session.json 落盘，本轮原话追加进 middle_context（跨轮唯一产物）。"""
    ctx = [Msg(name="user", role="user", content=[TextBlock(type="text", text="本轮")])]
    agent = _fake_agent("好的。", context=ctx)

    async def _build(**_kw: Any) -> Any:
        return agent, SimpleNamespace()

    monkeypatch.setattr(orch, "build_main_agent", _build)
    await orch.run_agent("买个包", thread_id="as-t3")

    session_dir = orch.ensure_session_dir("as-t3")
    saved = orch.load_session_state(session_dir)
    assert saved is not None and saved.context[0].get_text_content() == "本轮"
    assert saved.middle_context[orch.PRIOR_QUERIES_KEY] == ["买个包"]
    assert "pt" not in saved.middle_context  # 累积 P_t 已不落盘
    # 老的多份产物一份都不再写
    for legacy in ("agent_state.json", "pt.json", "candidates.json", "history.json"):
        assert not (session_dir / legacy).exists()


def test_load_state_returns_none_on_corrupt_file(tmp_path: Path) -> None:
    """坏掉的 state 只降级成空开局，不能让整轮聊天起不来。"""
    (tmp_path / orch.STATE_FILE).write_text("{ not json", encoding="utf-8")
    assert orch.load_session_state(tmp_path) is None


def test_save_state_is_atomic(tmp_path: Path) -> None:
    """临时文件 + rename：写完不留 .tmp，文件内容能整体读回。"""
    st = AgentState()
    st.middle_context["pt"] = {"category": "x"}
    orch.save_session_state(tmp_path, st)
    assert not list(tmp_path.glob("*.tmp"))
    loaded = orch.load_session_state(tmp_path)
    assert loaded is not None and loaded.middle_context["pt"]["category"] == "x"


# ---------- 收尾：审核后的文本 / 产物 / 取消 ----------


async def test_final_text_comes_from_streamed_msg_not_context(
    monkeypatch: pytest.MonkeyPatch, patched: dict[str, Any]
) -> None:
    """输出审核改写的是**流出去的** Msg（on_reply），state 里留的是原文。

    从 context 尾部取 final_text 等于把未审核文本发给用户、落进产物和历史——这条真摔过一次，
    单测钉死。
    """
    ctx = [
        Msg(
            name="shoppingx",
            role="assistant",
            content=[TextBlock(type="text", text="清单 [系统提示] 内部哨兵")],
        )
    ]
    agent = _fake_agent("清单", context=ctx)

    async def _build(**_kw: Any) -> Any:
        return agent, SimpleNamespace()

    monkeypatch.setattr(orch, "build_main_agent", _build)
    out = await orch.run_agent("买个包", thread_id="as-t4")

    assert out["final_text"] == "清单"
    assert patched["final_text"] == "清单"


async def test_run_agent_writes_artifacts_and_items(
    monkeypatch: pytest.MonkeyPatch, patched: dict[str, Any]
) -> None:
    """终结产物三处复用：商品卡随 task_result 下发、清单文案落 summary.md、结构化落 result.json。"""
    ctx = [
        Msg(name="shoppingx", role="assistant", content=[_summary_block()]),
        Msg(name="shoppingx", role="assistant", content=[TextBlock(type="text", text="给你")]),
    ]
    agent = _fake_agent("给你", context=ctx)

    async def _build(**_kw: Any) -> Any:
        return agent, SimpleNamespace()

    monkeypatch.setattr(orch, "build_main_agent", _build)
    out = await orch.run_agent("买个旅行包", thread_id="as-t5")

    assert out["items"] and out["items"][0]["item_id"] == "A1"
    assert patched["items"][0]["item_id"] == "A1"
    session_dir = orch.ensure_session_dir("as-t5")
    assert (session_dir / "summary.md").read_text(encoding="utf-8") == "为你精选 1 件：帆布旅行包。"
    assert (session_dir / "result.json").exists()
    # 完整轨迹落盘（AgentScope 的 Msg 序列化，供审计 / 排障）
    assert (session_dir / orch.STATE_FILE).exists()


async def test_run_agent_extracts_summary_only_from_this_turn(
    monkeypatch: pytest.MonkeyPatch, patched: dict[str, Any]
) -> None:
    """续聊第二轮用 chat_fallback 收尾时，不能把恢复回来的上一轮 shopping_summary 当本轮产物。

    真踩过：turn 2「帮我下单第一款」缺地址走 chat_fallback，task_result 却带回 turn 1 的清单。
    """
    session_dir = orch.ensure_session_dir("as-t5b")
    prior = AgentState()
    prior.context = [Msg(name="shoppingx", role="assistant", content=[_summary_block()])]
    (session_dir / orch.STATE_FILE).write_text(prior.model_dump_json(), encoding="utf-8")

    agent = _fake_agent("请先填收件信息。")

    async def _build(**kw: Any) -> Any:
        agent.state = kw["state"]  # 与真实装配一致：恢复的 state 直接挂到 agent 上
        return agent, SimpleNamespace()

    monkeypatch.setattr(orch, "build_main_agent", _build)
    out = await orch.run_agent("帮我下单第一款", thread_id="as-t5b")

    assert out["items"] == []
    assert not patched["items"]


async def test_run_agent_reports_cancel_and_reraises(
    monkeypatch: pytest.MonkeyPatch, patched: dict[str, Any]
) -> None:
    """用户取消：先上报 task_cancelled 再重抛，别把 CancelledError 吞成一次「正常收尾」。"""
    import asyncio

    class _CancelAgent:
        def __init__(self) -> None:
            self.state = AgentState()

        async def reply_stream(self, *_a: Any, **_kw: Any) -> Any:
            raise asyncio.CancelledError
            yield  # pragma: no cover - 让它成为 async generator

    async def _build(**_kw: Any) -> Any:
        return _CancelAgent(), SimpleNamespace()

    cancelled = {"n": 0}

    async def _report() -> None:
        cancelled["n"] += 1

    monkeypatch.setattr(orch, "build_main_agent", _build)
    monkeypatch.setattr(orch.monitor, "report_task_cancelled", _report)

    with pytest.raises(asyncio.CancelledError):
        await orch.run_agent("买个包", thread_id="as-t6")
    assert cancelled["n"] == 1


async def test_cancelled_turn_leaves_session_file_untouched(
    monkeypatch: pytest.MonkeyPatch, patched: dict[str, Any]
) -> None:
    """取消 / 超时不写任何会话文件：session.json 字节不变，上一轮那份原样留着。"""
    import asyncio

    session_dir = orch.ensure_session_dir("as-t7")
    prior = AgentState()
    prior.context = [Msg(name="user", role="user", content=[TextBlock(type="text", text="上轮")])]
    (session_dir / orch.STATE_FILE).write_text(prior.model_dump_json(), encoding="utf-8")
    before = (session_dir / orch.STATE_FILE).read_bytes()

    class _CancelAgent:
        def __init__(self) -> None:
            self.state = AgentState()

        async def reply_stream(self, *_a: Any, **_kw: Any) -> Any:
            raise asyncio.CancelledError
            yield  # pragma: no cover

    async def _build(**_kw: Any) -> Any:
        return _CancelAgent(), SimpleNamespace()

    async def _report() -> None:
        pass

    monkeypatch.setattr(orch, "build_main_agent", _build)
    monkeypatch.setattr(orch.monitor, "report_task_cancelled", _report)
    with pytest.raises(asyncio.CancelledError):
        await orch.run_agent("接着聊", thread_id="as-t7")

    assert (session_dir / orch.STATE_FILE).read_bytes() == before
    assert sorted(p.name for p in session_dir.iterdir()) == [orch.STATE_FILE]


# ---------- 事件泵 ----------


async def test_pump_returns_final_msg_and_reports_max_iters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agentscope.event import ReplyEndEvent, ReplyFinishedReason

    from app.agent import events as ev

    errors: list[tuple[str, str]] = []

    async def _report(kind: str, msg: str) -> None:
        errors.append((kind, msg))

    monkeypatch.setattr(ev.monitor, "report_error", _report)

    async def _stream() -> Any:
        yield ReplyEndEvent(
            session_id="s1",
            reply_id="r1",
            finished_reason=ReplyFinishedReason.EXCEED_MAX_ITERS,
        )
        yield Msg(name="a", role="assistant", content=[TextBlock(type="text", text="收尾")])

    final = await ev.pump_events(_stream())
    assert final is not None and final.get_text_content() == "收尾"
    # 超限当错误上报（前端画红条），但**不抛**——此刻上下文里往往已有可用结果。
    assert errors and errors[0][0] == "max_iters"


async def test_pump_does_not_fake_clarification_on_confirm_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """需要确认的事件只记日志：发 clarification_request 会弹一个没人接得住的确认框。"""
    from agentscope.event import RequireUserConfirmEvent

    from app.agent import events as ev

    asked: list[Any] = []

    async def _ask(*a: Any, **kw: Any) -> None:
        asked.append((a, kw))

    monkeypatch.setattr(ev.monitor, "report_clarification_request", _ask)

    async def _stream() -> Any:
        yield RequireUserConfirmEvent(reply_id="r1", tool_calls=[])

    assert await ev.pump_events(_stream()) is None
    assert asked == []


def _hanging_agent() -> Any:
    """模型调用挂 30s 的**真框架** Agent：钉的是框架吞取消这一行为本身，手搓事件流钉不住。"""
    from agentscope.agent import Agent

    model = _fake_model()

    async def _hang(*_a: object, **_kw: object) -> Any:
        await asyncio.sleep(30)

    model._call_api = _hang  # type: ignore[method-assign]
    return Agent(name="t", system_prompt="sys", model=model)


def _user(text: str) -> Msg:
    return Msg(name="user", role="user", content=[TextBlock(type="text", text=text)])


async def test_pump_reraises_cancel_swallowed_by_framework() -> None:
    """用户取消：框架在 reply 内吞掉 CancelledError 并吐英文兜底，泵必须补抛。"""
    from app.agent.events import pump_events

    agent = _hanging_agent()
    task = asyncio.create_task(pump_events(agent.reply_stream(_user("hi"), yield_final_msg=True)))
    await asyncio.sleep(0.2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_pump_turns_swallowed_timeout_into_timeout_error() -> None:
    """外层 asyncio.timeout 到点：同样被框架吞，补抛后由 timeout 自己换成 TimeoutError。"""
    from app.agent.events import pump_events

    agent = _hanging_agent()
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.2):
            await pump_events(agent.reply_stream(_user("hi"), yield_final_msg=True))


async def test_cancel_propagates_through_assembled_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """真装配：终结纪律会在被打断的 reply 上置 retry_nudge，适配器不得借此吞掉 INTERRUPTED。

    曾经的症状：适配器把 ReplyEnd 当「强制再来一轮」吞了，事件泵认不出取消，用户点取消后
    任务记 done、英文兜底文案当回答发出、上下文里还多一条催收尾提示。
    """
    from app.agent import agents as ag
    from app.agent.events import pump_events
    from app.harness.adapter import HarnessAgentAdapter
    from app.harness.setup import setup_harness

    async def _no_prefill(self: Any, agent: Any) -> None:
        return None

    setup_harness()
    hanging = _hanging_agent().model
    monkeypatch.setattr(HarnessAgentAdapter, "_prefill", _no_prefill)
    monkeypatch.setattr(ag, "get_tier_llm", lambda _tier: hanging)
    monkeypatch.setattr("app.agent.llm.get_llm", lambda: hanging)
    monkeypatch.setattr("app.agent.llm.get_fast_llm", lambda: hanging)
    agent, _ = await ag.build_main_agent(original_query="你好")

    task = asyncio.create_task(pump_events(agent.reply_stream(_user("你好"), yield_final_msg=True)))
    await asyncio.sleep(0.3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    users = [m.get_text_content() for m in agent.state.context if m.role == "user"]
    assert users == ["你好"]  # 没有被塞催收尾提示


async def test_pump_interrupted_always_reraises() -> None:
    """INTERRUPTED 即补抛，不看 ``cancelling()``：框架在并发工具批里会 uncancel 把计数清零。"""
    from agentscope.event import ReplyEndEvent, ReplyFinishedReason

    from app.agent import events as ev

    async def _stream() -> Any:
        yield ReplyEndEvent(
            session_id="s1", reply_id="r1", finished_reason=ReplyFinishedReason.INTERRUPTED
        )
        yield Msg(name="a", role="assistant", content=[TextBlock(type="text", text="兜底")])

    with pytest.raises(asyncio.CancelledError):
        await ev.pump_events(_stream())


async def _slow_tool_agent() -> Any:
    """模型第一步调一个睡 30s 的工具（默认 ``is_concurrency_safe=True``，走框架并发批）。"""
    from agentscope.agent import Agent
    from agentscope.message import ToolCallBlock, ToolResultState
    from agentscope.model import ChatResponse
    from agentscope.tool import FunctionTool, ToolChunk, Toolkit

    async def slow_tool() -> ToolChunk:
        """睡 30 秒。"""
        await asyncio.sleep(30)
        return ToolChunk(content=[TextBlock(type="text", text="slept")], state=ToolResultState.SUCCESS)

    model = _fake_model()

    async def _call(*_a: object, **_kw: object) -> ChatResponse:
        blk = ToolCallBlock(type="tool_call", id="c1", name="slow_tool", input="{}")
        return ChatResponse(content=[blk], is_last=True)

    model._call_api = _call  # type: ignore[method-assign]
    toolkit = Toolkit()
    await toolkit.add_tool(FunctionTool(slow_tool, is_read_only=True))
    return Agent(name="t", system_prompt="sys", model=model, toolkit=toolkit)


async def test_cancel_during_concurrent_tool_reraises() -> None:
    """工具执行中取消：框架 uncancel 清零计数后，泵仍须补抛（修前这里被吞、任务正常返回）。"""
    from app.agent.events import pump_events

    agent = await _slow_tool_agent()
    task = asyncio.create_task(pump_events(agent.reply_stream(_user("hi"), yield_final_msg=True)))
    await asyncio.sleep(0.2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_timeout_during_concurrent_tool_becomes_timeout_error() -> None:
    """工具执行中超时：计数被清零后 ``asyncio.timeout`` 仍须把补抛的取消换成 TimeoutError。"""
    from app.agent.events import pump_events

    agent = await _slow_tool_agent()
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.2):
            await pump_events(agent.reply_stream(_user("hi"), yield_final_msg=True))


# ---------- 权限：精准放行，不是 BYPASS ----------


async def test_allow_tools_is_precise_and_idempotent() -> None:
    from agentscope.permission import PermissionBehavior, PermissionEngine

    from app.agent.permissions import allow_tools
    from app.agent.tool_registry import TOOLS_BY_NAME

    state = AgentState()
    allow_tools(state)
    allow_tools(state)  # 幂等：跨轮复用同一个 state 时规则表不许线性膨胀
    assert len(state.permission_context.allow_rules["ask_user"]) == 1

    engine = PermissionEngine(state.permission_context)
    granted = await engine.check_permission(TOOLS_BY_NAME["ask_user"], {})
    assert granted.behavior == PermissionBehavior.ALLOW

    # 没进放行表的写工具照样要问——这正是「不用 BYPASS」买到的东西：create_order
    # 默认是被拦的，作者必须显式决定放不放。
    outsider = TOOLS_BY_NAME["shopping_summary"]
    state2 = AgentState()
    allow_tools(state2, {"ask_user"})
    decision = await PermissionEngine(state2.permission_context).check_permission(outsider, {})
    assert decision.behavior == PermissionBehavior.ASK


async def test_every_write_tool_is_allowlisted() -> None:
    """机制守栏：**每个非只读工具都必须在放行集里**，否则真跑时会被 DEFAULT 模式挂起。

    S3 补 present_guide 时发现 present_comparison 漏了一年——它还有一条 REST 入口（对比栏按钮
    不经 AgentLoop），把「模型在对话里调它会被挂起」这件事遮住了。这条测试把「新增写工具要
    登记」变成红灯，不再靠人记得。
    """
    from app.agent.permissions import DEFAULT_ALLOWED_TOOLS
    from app.agent.tool_registry import TOOLS

    writers = {t.name for t in TOOLS if not getattr(t, "is_read_only", False)}
    assert writers <= DEFAULT_ALLOWED_TOOLS, writers - DEFAULT_ALLOWED_TOOLS


async def test_trade_tools_are_main_only() -> None:
    """TradeAgent 已删：交易工具只在主 Agent 手上。"""
    from app.agent.tool_registry import build_toolkit

    main = await build_toolkit()
    names = {s["function"]["name"] for s in await main.get_tool_schemas()}
    assert {"create_order", "query_order", "cancel_order"} <= names


def _fake_model() -> Any:
    """不打网络的模型（照 test_harness_adapter 的套路）。"""
    from agentscope.credential import OpenAICredential
    from agentscope.model import ChatResponse, OpenAIChatModel

    model = OpenAIChatModel(
        credential=OpenAICredential(api_key="sk-test", base_url="http://localhost:1/v1"),
        model="test-model",
        stream=False,
        max_retries=0,
    )

    async def _call(*_a: object, **_kw: object) -> ChatResponse:
        return ChatResponse(content=[TextBlock(type="text", text="好的")], is_last=True)

    model._call_api = _call  # type: ignore[method-assign]
    return model


async def test_assembly_binds_one_session_per_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """一次 loop = 一个 HarnessSession + 一份工具实例，且工具适配器拿的是**同一个** session。

    这是 L3 装配的核心不变量：AgentScope 把模型钩子与工具钩子拆成两个类，接力通道全靠共享
    session 传。错配的症状不是崩，而是 worker 的断言流进主 loop、并发 worker 互相污染循环
    检测——所以这里钉死身份，不只是「装起来了」。
    """
    from app.agent import agents as ag
    from app.harness.adapter import HarnessToolAdapter

    tiers: list[str] = []

    def _tiered(tier: str) -> Any:
        tiers.append(tier)
        return _fake_model()

    monkeypatch.setattr(ag, "get_tier_llm", _tiered)

    agent, session = await ag.build_main_agent(original_query="买个包")
    # 装配期拿的是**基座档**（默认 fast，关思考）。这条端到端钉住 P0-1 的另一半：档位表说
    # 一套、装配拿另一套时，上面那批单测（只验档位表本身）是不会红的。
    assert tiers == ["fast"]
    names = ["planner", "item_search", "create_order", "shopping_summary"]
    tools = [await agent.toolkit.get_tool(n) for n in names]
    assert all(t is not None for t in tools)
    for tool in tools:
        adapters = [m for m in tool._middlewares if isinstance(m, HarnessToolAdapter)]
        assert len(adapters) == 1
        assert adapters[0]._s is session
    # 写工具已放行（不靠 BYPASS 整档关引擎）
    assert "create_order" in agent.state.permission_context.allow_rules

    # 第二次装配必须是另一套：共享工具实例 = 共享控制面状态。
    agent2, session2 = await ag.build_main_agent()
    assert session2 is not session
    tools2 = [await agent2.toolkit.get_tool(n) for n in names]
    assert {id(t) for t in tools2}.isdisjoint({id(t) for t in tools})


async def test_assembled_agent_runs_with_terminal_discipline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """冒烟：装配出来的 Agent 能真跑 reply，且**终结纪律全程在链上**。

    假模型只会说话、从不调终结工具——真实链路下这正是最常见的失败模式，harness 会一轮轮催
    （``retry_nudge`` 吞掉 ReplyEnd 强制再来一轮），直到撞 ``max_iters`` 才停。所以这里断言
    的是「被催到上限」而不是「一轮就收尾」：如果哪天它一轮就返回了「好的」，说明终结纪律在
    AgentScope 侧掉线了。max_iters 压到 2，免得为验一条接线跑 30 轮。
    """
    from app.agent import agents as ag
    from app.harness.setup import setup_harness

    setup_harness()
    monkeypatch.setattr(ag, "MAIN_MAX_ITERS", 2)
    monkeypatch.setattr(ag, "get_tier_llm", lambda _tier: _fake_model())
    # 第一轮加档也会换模型：适配器按**档位名**去 app.agent.llm 现取（见 _first_round_tier
    # → _resolve_model_tier），所以那条路也得顶掉，否则冒烟测试会真打网络。
    monkeypatch.setattr("app.agent.llm.get_llm", _fake_model)
    monkeypatch.setattr("app.agent.llm.get_fast_llm", _fake_model)
    agent, session = await ag.build_main_agent(original_query="你好")
    msg = Msg(name="user", role="user", content=[TextBlock(type="text", text="你好")])
    out = await agent.reply(msg)

    assert "exceed" in out.get_text_content().lower()
    assert session.round_counter >= 2  # 模型钩子每轮都过了适配器
    # 催收尾的提示确实进了上下文（跟着 state 走，下一轮模型还看得见）
    assert any(
        "终结" in m.get_text_content() or "收尾" in m.get_text_content()
        for m in agent.state.context
    )


# ---------- 本轮 deadline----------
async def test_run_agent_sets_the_turn_deadline(
    monkeypatch: pytest.MonkeyPatch, patched: dict[str, Any]
) -> None:
    """入口给本轮设 deadline，Agent 跑起来时下游读得到（出站点据它收紧自己的超时）。

    与外面那层 ``asyncio.timeout(MAIN_AGENT_TIMEOUT_SEC)`` 是同一个预算，区别只在**谁看得见**：
    timeout 从外面一刀砍下来，出站点对它一无所知；deadline 把同一个数下传到每个出站点。
    这条断了不会报错，只会悄悄退回「谁都没有 deadline」，于是又开始白等——所以钉一下。
    """
    from app.agent.limits import MAIN_AGENT_TIMEOUT_SEC
    from app.api.context import remaining_seconds, reset_deadline

    reset_deadline()
    seen: list[float | None] = []
    agent = _fake_agent("已为你整理好清单。")
    inner = agent.reply_stream

    async def _probe(inputs: Any, yield_final_msg: bool = False) -> Any:
        seen.append(remaining_seconds())
        async for msg in inner(inputs, yield_final_msg=yield_final_msg):
            yield msg

    agent.reply_stream = _probe

    async def _build(**_kw: Any) -> Any:
        return agent, SimpleNamespace()

    monkeypatch.setattr(orch, "build_main_agent", _build)
    await orch.run_agent("买个旅行包", thread_id="as-dl")

    assert seen and seen[0] is not None
    assert 0 < seen[0] <= MAIN_AGENT_TIMEOUT_SEC
    reset_deadline()
