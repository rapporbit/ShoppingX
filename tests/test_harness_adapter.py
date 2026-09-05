"""L4 harness 适配层：六个 hook_point 在 AgentScope 侧接得对不对。

**测的是接线，不是 hook 的业务**——36 个真实 hook 的行为归 test_harness.py 的 119 个用例管，
这里把全局注册表换成一张只有探针的空表，专测「适配器有没有在正确的位置、带着正确的 context
调用 Pipeline，以及 Pipeline 的产出有没有真正生效」。
"""

import json

import pytest
from agentscope.agent import Agent
from agentscope.credential import OpenAICredential
from agentscope.message import Msg, TextBlock, ToolCallBlock
from agentscope.model import ChatResponse, OpenAIChatModel
from agentscope.tool import FunctionTool, Toolkit
from agentscope.tool._response import ToolChunk, ToolResultState

from app.harness import adapter as adapter_mod
from app.harness.adapter import HarnessAgentAdapter, HarnessSession, HarnessToolAdapter
from app.harness.middleware import HarnessMiddleware, HookRejectSignal

EXEC_LOG: list[str] = []


async def probe_tool(q: str) -> ToolChunk:
    """探针工具。

    参数：
      - q：随便。
    """
    EXEC_LOG.append(q)
    return ToolChunk(
        content=[TextBlock(type="text", text=json.dumps({"candidates": [1, 2, 3], "q": q}))],
        state=ToolResultState.SUCCESS,
    )


@pytest.fixture
def isolated_harness(monkeypatch: pytest.MonkeyPatch) -> HarnessMiddleware:
    """把适配器用的全局注册表换成空表（只装探针 hook）。"""
    fresh = HarnessMiddleware()
    monkeypatch.setattr(adapter_mod, "harness", fresh)
    EXEC_LOG.clear()
    return fresh


def _model(responses: list[ChatResponse]) -> OpenAIChatModel:
    """按顺序吐预设响应的假模型（不打网络）。"""
    model = OpenAIChatModel(
        credential=OpenAICredential(api_key="sk-test", base_url="http://localhost:1/v1"),
        model="test-model",
        stream=False,
        max_retries=0,
    )
    queue = list(responses)

    async def fake_call_api(*_args: object, **_kwargs: object) -> ChatResponse:
        return queue.pop(0) if queue else ChatResponse(content=[], is_last=True)

    model._call_api = fake_call_api  # type: ignore[method-assign]
    return model


def _tool_call(name: str, **kwargs: object) -> ChatResponse:
    return ChatResponse(
        content=[
            ToolCallBlock(type="tool_call", id="c1", name=name, input=json.dumps(kwargs)),
        ],
        is_last=True,
    )


def _text(msg: str) -> ChatResponse:
    return ChatResponse(content=[TextBlock(type="text", text=msg)], is_last=True)


async def _build(session: HarnessSession, responses: list[ChatResponse]) -> Agent:
    toolkit = Toolkit()
    await toolkit.add_tool(
        FunctionTool(
            probe_tool,
            name="item_search",  # 借用真名，好顺带验证「检索类工具」的候选计数信号
            is_read_only=True,
            middlewares=[HarnessToolAdapter(session)],
        ),
    )
    return Agent(
        name="adapter_test",
        system_prompt="你是助手。",
        model=_model(responses),
        toolkit=toolkit,
        middlewares=[HarnessAgentAdapter(session)],
    )


def _user(text: str) -> Msg:
    return Msg(name="user", role="user", content=[TextBlock(type="text", text=text)])


@pytest.mark.asyncio
async def test_all_six_hook_points_fire_with_expected_context(
    isolated_harness: HarnessMiddleware,
) -> None:
    """一轮「调工具 → 收尾」应触发全部五个 loop 内 hook_point，且顺序与 LangChain 版一致。"""
    seen: list[tuple[str, dict]] = []

    def recorder(point: str):  # type: ignore[no-untyped-def]
        async def hook(ctx: dict) -> None:
            seen.append((point, dict(ctx)))
            return None

        return hook

    for point in ("pre_think", "pre_tool_call", "post_tool_call", "post_reflect", "on_session_end"):
        isolated_harness.register(point, point, recorder(point), priority=50)

    session = HarnessSession(original_query="找个背包")
    agent = await _build(session, [_tool_call("item_search", q="backpack"), _text("给你三个选择")])
    reply = await agent.reply(_user("找个背包"))

    points = [p for p, _ in seen]
    assert points.count("pre_think") == 2  # 两次模型调用
    assert points.count("pre_tool_call") == 1
    assert points.count("post_tool_call") == 1
    assert points.count("post_reflect") == 2
    assert points.count("on_session_end") == 1
    # 顺序契约（与 LangChain 版逐字相同，别想当然）：post_reflect 是「模型刚回复完」的反思点，
    # 跑在工具执行**之前**——本轮要不要收尾、这轮回复合不合规，判断的是模型的输出而不是工具的产出。
    assert points[:4] == ["pre_think", "post_reflect", "pre_tool_call", "post_tool_call"]
    assert points[-1] == "on_session_end"

    ctxs = dict(seen)
    assert ctxs["pre_tool_call"]["tool_name"] == "item_search"
    assert ctxs["pre_tool_call"]["tool_args"] == {"q": "backpack"}
    # 检索类工具的候选计数从真实返回里数出来（阶段机据此推进）
    assert ctxs["post_tool_call"]["call_candidates"] == 3
    assert session.fresh_candidates == 3
    assert session.called_tools == {"item_search"}
    assert "给你三个选择" in "".join(b.text for b in reply.content if b.type == "text")


@pytest.mark.asyncio
async def test_pre_tool_call_rejection_blocks_execution(
    isolated_harness: HarnessMiddleware,
) -> None:
    """闸门拒绝 = 工具一次都不许跑，模型收到的是哨兵文案而不是异常。"""

    async def gate(ctx: dict) -> None:
        raise HookRejectSignal("越权：现在不许搜", raw=False)

    isolated_harness.register("pre_tool_call", "test_gate", gate, priority=1)

    session = HarnessSession()
    agent = await _build(session, [_tool_call("item_search", q="x"), _text("好的")])
    await agent.reply(_user("搜一下"))

    assert EXEC_LOG == [], "被拒的调用不该真的执行"
    # 哨兵进了上下文，模型看得见
    texts = [
        getattr(b, "output", "")
        for m in agent.state.context
        for b in m.content
        if getattr(b, "type", None) == "tool_result"
    ]
    flat = json.dumps(texts, ensure_ascii=False, default=str)
    assert "[Harness 拒绝] 越权：现在不许搜" in flat
    # 拒绝不算进展：不记 called_tools
    assert session.called_tools == set()


@pytest.mark.asyncio
async def test_raw_sentinel_is_passed_through_verbatim(
    isolated_harness: HarnessMiddleware,
) -> None:
    """raw=True 的哨兵是写给模型的完整指令，加前缀只会稀释它。"""

    async def gate(ctx: dict) -> None:
        raise HookRejectSignal("你已经拿到候选了，现在去调 item_picker。", raw=True)

    isolated_harness.register("pre_tool_call", "raw_gate", gate, priority=1)

    session = HarnessSession()
    agent = await _build(session, [_tool_call("item_search", q="x"), _text("好")])
    await agent.reply(_user("搜"))

    flat = json.dumps(
        [
            getattr(b, "output", "")
            for m in agent.state.context
            for b in m.content
            if getattr(b, "type", None) == "tool_result"
        ],
        ensure_ascii=False,
        default=str,
    )
    assert "Harness 拒绝" not in flat
    assert "现在去调 item_picker" in flat


@pytest.mark.asyncio
async def test_pre_think_can_rewrite_messages_and_inject(
    isolated_harness: HarnessMiddleware,
) -> None:
    """pre_think 改写的 messages 必须真的进模型视野——这是压缩 / 预算 hint / 纠正注入的唯一通路。"""
    seen_lengths: list[int] = []

    async def injector(ctx: dict) -> dict:
        seen_lengths.append(len(ctx["messages"]))
        ctx["messages"] = [
            *ctx["messages"],
            Msg(name="system", role="system", content=[TextBlock(type="text", text="预算吃紧")]),
        ]
        return ctx

    isolated_harness.register("pre_think", "injector", injector, priority=10)

    captured: list[list[Msg]] = []
    session = HarnessSession()
    agent = await _build(session, [_text("好")])

    original = agent.model._call_api

    async def spy(*args: object, **kwargs: object) -> ChatResponse:
        captured.append(list(kwargs.get("messages") or []))
        return await original(*args, **kwargs)

    agent.model._call_api = spy  # type: ignore[method-assign]
    await agent.reply(_user("你好"))

    assert captured, "模型没被调用"
    texts = [
        "".join(getattr(b, "text", "") for b in m.content)
        for m in captured[0]
        if m.role == "system"
    ]
    assert any("预算吃紧" in t for t in texts), "pre_think 改写的 messages 没进模型"


@pytest.mark.asyncio
async def test_post_tool_call_can_rewrite_result(isolated_harness: HarnessMiddleware) -> None:
    """post_tool_call 改 tool_result（截断 / 收线通告）要写回模型看到的那份。"""

    async def truncator(ctx: dict) -> dict:
        ctx["tool_result"] = "【已截断】" + ctx["tool_result"][:10]
        return ctx

    isolated_harness.register("post_tool_call", "truncator", truncator, priority=10)

    session = HarnessSession()
    agent = await _build(session, [_tool_call("item_search", q="x"), _text("好")])
    await agent.reply(_user("搜"))

    flat = json.dumps(
        [
            getattr(b, "output", "")
            for m in agent.state.context
            for b in m.content
            if getattr(b, "type", None) == "tool_result"
        ],
        ensure_ascii=False,
        default=str,
    )
    assert "【已截断】" in flat
    assert EXEC_LOG == ["x"], "改写发生在执行之后，工具照常跑一次"


@pytest.mark.asyncio
async def test_retry_nudge_forces_another_round(isolated_harness: HarnessMiddleware) -> None:
    """模型没调终结工具就想收尾 → 追加提示、强制再来一轮（框架原生的「吞 ReplyEnd」语义）。"""
    rounds = {"n": 0}

    async def enforcer(ctx: dict) -> dict | None:
        rounds["n"] += 1
        if rounds["n"] == 1 and not ctx.get("response_has_tool_calls"):
            ctx["retry_nudge"] = "请调 shopping_summary 收尾"
            return ctx
        return None

    isolated_harness.register("post_reflect", "enforcer", enforcer, priority=60)

    session = HarnessSession()
    agent = await _build(session, [_text("我直接说结论"), _text("好的，已收尾")])
    reply = await agent.reply(_user("推荐"))

    assert rounds["n"] == 2, "没有被强制拉回第二轮"
    nudges = [
        "".join(getattr(b, "text", "") for b in m.content)
        for m in agent.state.context
        if m.role == "user"
    ]
    assert any("请调 shopping_summary 收尾" in t for t in nudges)
    assert "已收尾" in "".join(b.text for b in reply.content if b.type == "text")


@pytest.mark.asyncio
async def test_session_end_can_rewrite_final_answer(isolated_harness: HarnessMiddleware) -> None:
    """on_session_end 的改写要落到用户看到的那句话上，否则脱敏就是个只会记日志的摆设。"""

    async def redactor(ctx: dict) -> dict:
        ctx["final_answer"] = ctx["final_answer"].replace("13800138000", "1**********")
        return ctx

    isolated_harness.register("on_session_end", "redactor", redactor, priority=10)

    session = HarnessSession()
    agent = await _build(session, [_text("联系方式 13800138000")])
    reply = await agent.reply(_user("给个联系方式"))

    text = "".join(b.text for b in reply.content if b.type == "text")
    assert "13800138000" not in text
    assert "1**********" in text


@pytest.mark.asyncio
async def test_budget_fallback_skips_model_entirely(isolated_harness: HarnessMiddleware) -> None:
    """预算耗尽档：直接把规则兜底的回答当模型输出返回，一次 LLM 都不再付。"""
    called = {"n": 0}

    async def broke(ctx: dict) -> dict:
        ctx["fallback_answer"] = "预算用完了，这是我已经找到的：A / B"
        return ctx

    isolated_harness.register("pre_think", "broke", broke, priority=20)

    session = HarnessSession()
    agent = await _build(session, [_text("不该被调用")])

    async def counting(*_args: object, **_kwargs: object) -> ChatResponse:
        called["n"] += 1
        return _text("不该被调用")

    agent.model._call_api = counting  # type: ignore[method-assign]
    reply = await agent.reply(_user("推荐"))

    assert called["n"] == 0, "fallback 档还去调了模型"
    assert "预算用完了" in "".join(b.text for b in reply.content if b.type == "text")
    # 之后任何工具调用都该被终结闸拦下
    assert session.guard.terminal_reached is True


@pytest.mark.asyncio
async def test_tool_error_is_not_counted_as_progress(isolated_harness: HarnessMiddleware) -> None:
    """工具返回 ERROR = 没真跑过：不记 called_tools、不推信号、不给看门狗续命。"""

    async def boom_tool(q: str) -> ToolChunk:
        """会失败的工具。

        参数：
          - q：随便。
        """
        return ToolChunk(
            content=[TextBlock(type="text", text="[error] ValidationError: 参数不对")],
            state=ToolResultState.ERROR,
        )

    post_seen: list[dict] = []

    async def watcher(ctx: dict) -> None:
        post_seen.append(dict(ctx))
        return None

    isolated_harness.register("post_tool_call", "watcher", watcher, priority=10)

    session = HarnessSession()
    toolkit = Toolkit()
    await toolkit.add_tool(
        FunctionTool(
            boom_tool,
            name="item_search",
            is_read_only=True,
            middlewares=[HarnessToolAdapter(session)],
        ),
    )
    agent = Agent(
        name="err_test",
        system_prompt="你是助手。",
        model=_model([_tool_call("item_search", q="x"), _text("好")]),
        toolkit=toolkit,
        middlewares=[HarnessAgentAdapter(session)],
    )
    before = session.guard.last_progress_at
    await agent.reply(_user("搜"))

    assert session.called_tools == set()
    assert post_seen == [], "ERROR 的调用不该跑 post_tool_call"
    assert session.guard.last_progress_at == before, "失败的调用不该给看门狗续命"
