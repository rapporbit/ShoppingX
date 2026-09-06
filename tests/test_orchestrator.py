"""L3 验收：AgentScope 主链路（orchestrator / events / permissions / task_dispatch）。

**测的是接缝，不是业务**——工具行为、harness 的 36 个 hook、记忆判定各有自己的测试文件。
这里守的是迁移最容易悄悄摔的四处：

1. 收尾取的是**流出去的那条 Msg**（已过输出审核），不是 ``state.context`` 里的原文；
2. 续聊两条腿：有 ``agent_state.json`` 就恢复它，没有才回放精简 (q,a)；
3. 写工具是**精准放行**的——没进放行表的工具照样要用户确认（不能靠 BYPASS 一档全开）；
4. ``task_dispatch`` 把子任务的失败转成工具结果，而不是让主 loop 崩。
"""

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
    """假 Agent：吐一条最终 Msg，state.context 里放本轮消息。"""
    ctx = context if context is not None else []

    class _FakeAgent:
        def __init__(self) -> None:
            self.state = AgentState()
            self.state.context = ctx
            self.inputs: list[Msg] = []

        async def reply_stream(self, inputs: Any, yield_final_msg: bool = False) -> Any:
            self.inputs = list(inputs)
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


# ---------- 会话恢复两条腿 ----------


async def test_run_agent_replays_history_when_no_state(
    monkeypatch: pytest.MonkeyPatch, patched: dict[str, Any]
) -> None:
    """没有 agent_state.json 时回放精简 (q,a)，且当轮 query 拼在最后一条 user 消息里。"""
    agent = _fake_agent("已为你整理好清单。")

    async def _build(**kw: Any) -> Any:
        patched["state_arg"] = kw.get("state")
        return agent, SimpleNamespace()

    async def _prior(*_a: Any, **_kw: Any) -> list[tuple[str, str]]:
        return [("user", "上轮问题"), ("assistant", "上轮回答")]

    monkeypatch.setattr(orch, "build_main_agent", _build)
    monkeypatch.setattr(orch, "load_prior_turns", _prior)

    await orch.run_agent("买个旅行包", thread_id="as-t1")

    assert patched["state_arg"] is None  # 没有落盘 state → 走回放腿
    roles = [m.role for m in agent.inputs]
    assert roles == ["user", "assistant", "user"]
    assert agent.inputs[0].get_text_content() == "上轮问题"
    assert agent.inputs[-1].get_text_content().endswith("买个旅行包")


async def test_run_agent_resumes_from_agent_state(
    monkeypatch: pytest.MonkeyPatch, patched: dict[str, Any]
) -> None:
    """有 agent_state.json 就恢复它，并且**不再回放**历史（否则同一段对话进两遍上下文）。"""
    session_dir = orch.ensure_session_dir("as-t2")
    prior = AgentState()
    prior.context = [Msg(name="user", role="user", content=[TextBlock(type="text", text="上轮")])]
    (session_dir / orch.STATE_FILE).write_text(prior.model_dump_json(), encoding="utf-8")

    agent = _fake_agent("好的。")
    replayed = {"called": False}

    async def _build(**kw: Any) -> Any:
        patched["state_arg"] = kw.get("state")
        return agent, SimpleNamespace()

    async def _prior(*_a: Any, **_kw: Any) -> list[tuple[str, str]]:
        replayed["called"] = True
        return [("user", "上轮"), ("assistant", "上轮回答")]

    monkeypatch.setattr(orch, "build_main_agent", _build)
    monkeypatch.setattr(orch, "load_prior_turns", _prior)

    await orch.run_agent("接着聊", thread_id="as-t2")

    state_arg = patched["state_arg"]
    assert isinstance(state_arg, AgentState)
    assert state_arg.context[0].get_text_content() == "上轮"
    assert replayed["called"] is False
    assert [m.role for m in agent.inputs] == ["user"]


async def test_run_agent_saves_state_for_next_turn(
    monkeypatch: pytest.MonkeyPatch, patched: dict[str, Any]
) -> None:
    ctx = [Msg(name="user", role="user", content=[TextBlock(type="text", text="本轮")])]
    agent = _fake_agent("好的。", context=ctx)

    async def _build(**_kw: Any) -> Any:
        return agent, SimpleNamespace()

    monkeypatch.setattr(orch, "build_main_agent", _build)
    await orch.run_agent("买个包", thread_id="as-t3")

    saved = orch._load_state(orch.ensure_session_dir("as-t3"))
    assert saved is not None and saved.context[0].get_text_content() == "本轮"


def test_load_state_returns_none_on_corrupt_file(tmp_path: Path) -> None:
    """坏掉的 state 只降级成「少一段上下文」，不能让整轮聊天起不来。"""
    (tmp_path / orch.STATE_FILE).write_text("{ not json", encoding="utf-8")
    assert orch._load_state(tmp_path) is None


# ---------- 收尾：审核后的文本 / 产物 / 取消 ----------


async def test_final_text_comes_from_streamed_msg_not_context(
    monkeypatch: pytest.MonkeyPatch, patched: dict[str, Any]
) -> None:
    """输出审核改写的是**流出去的** Msg（L4 的 on_reply），state 里留的是原文。

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
    assert (session_dir / "history.json").exists()


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

    # 没进放行表的写工具照样要问——这正是「不用 BYPASS」买到的东西：批 1 的 create_order
    # 默认是被拦的，作者必须显式决定放不放。
    outsider = TOOLS_BY_NAME["shopping_summary"]
    state2 = AgentState()
    allow_tools(state2, {"ask_user"})
    decision = await PermissionEngine(state2.permission_context).check_permission(outsider, {})
    assert decision.behavior == PermissionBehavior.ASK


# ---------- task_dispatch：派发安全四层 ----------


def _chunk_text(chunk: Any) -> str:
    return "".join(b.get("text", "") if isinstance(b, dict) else b.text for b in chunk.content)


async def test_task_dispatch_rejects_platform_user_did_not_enable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """用户没勾的平台不派——派出去也是空军（语料 99.75% 在 amazon）。"""
    from app.agent.dispatch_tool import task_dispatch
    from app.agent.platform_scope import platform_scope

    with platform_scope(["amazon"]):
        chunk = await task_dispatch("在 shopee 上找一个帆布旅行包，预算 300")
    assert chunk.state == ToolResultState.ERROR
    assert "未启用 shopee" in _chunk_text(chunk)


async def test_task_dispatch_turns_worker_failure_into_tool_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """worker 炸了要变成一条工具结果，不能把主 loop 一起带走。"""
    import app.agent.dispatch_tool as dt
    from app.agent.platform_scope import platform_scope

    async def _boom(_kind: str = "search") -> Any:
        raise RuntimeError("worker 起不来")

    monkeypatch.setattr("app.agent.agents.build_worker_agent", _boom)
    with platform_scope(["amazon"]):
        chunk = await dt.task_dispatch("在 amazon 上找帆布旅行包")
    assert chunk.state == ToolResultState.ERROR
    text = _chunk_text(chunk)
    assert "[task_dispatch 错误] RuntimeError" in text and "worker 起不来" in text


async def test_task_dispatch_depth_limited(monkeypatch: pytest.MonkeyPatch) -> None:
    """深度上限：worker 内部再派发要被拒（批 1 起还会由「worker 没有这只工具」结构性兜住）。"""
    from app.agent.dispatch_tool import task_dispatch
    from app.agent.fork_guard import enter_fork

    with enter_fork():
        chunk = await task_dispatch("再派一层")
    assert chunk.state == ToolResultState.ERROR
    assert "深度已达上限" in _chunk_text(chunk)


async def test_task_dispatch_returns_worker_text(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.agent import dispatch_tool as dt
    from app.agent.platform_scope import platform_scope

    class _Worker:
        async def reply(self, _msg: Any) -> Msg:
            return Msg(
                name="w", role="assistant", content=[TextBlock(type="text", text="找到 3 件")]
            )

    async def _build(_kind: str = "search") -> Any:
        return _Worker()

    monkeypatch.setattr("app.agent.agents.build_worker_agent", _build)
    with platform_scope(["amazon"]):
        chunk = await dt.task_dispatch("在 amazon 上找帆布旅行包", "search")
    assert chunk.state == ToolResultState.SUCCESS
    assert _chunk_text(chunk) == "找到 3 件"
    assert chunk.metadata["subagent_type"] == "search"


# ---------- 装配：一 loop 一份控制面 ----------


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

    monkeypatch.setattr(ag, "get_llm", _fake_model)
    monkeypatch.setattr(ag, "get_fast_llm", _fake_model)

    agent, session = await ag.build_main_agent(original_query="买个包")
    names = ["planner", "item_search", "task_dispatch", "shopping_summary"]
    tools = [await agent.toolkit.get_tool(n) for n in names]
    assert all(t is not None for t in tools)
    for tool in tools:
        adapters = [m for m in tool._middlewares if isinstance(m, HarnessToolAdapter)]
        assert len(adapters) == 1
        assert adapters[0]._s is session
    # 写工具已放行（不靠 BYPASS 整档关引擎）
    assert "task_dispatch" in agent.state.permission_context.allow_rules

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
    monkeypatch.setattr(ag, "get_llm", _fake_model)
    # 第一轮 reasoning_boost 会换档：适配器按**档位名**去 app.agent.llm 现取模型（见
    # adapter._resolve_model_tier），所以这里也得把那一处顶掉，否则冒烟测试会真打网络。
    monkeypatch.setattr("app.agent.llm.get_llm", _fake_model)
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
