"""L1 网关闸门：并发上限 / 起点间隔 / 流式持有 slot / 限流降速 / 备用模型可见。

这些行为都不是「调用成功了吗」，而是「发出去之前排得对不对」——所以一律不打真实 API，
用 ``_call_api`` 的替身控制时序，断言的是**时间关系与占用关系**。
"""

import asyncio
import time
from collections.abc import AsyncGenerator

import pytest
from agentscope.credential import OpenAICredential
from agentscope.model import ChatResponse

from app.agent.gateway import GatewayThrottle, ThrottledChatModel


def _model(throttle: GatewayThrottle, role: str = "main") -> ThrottledChatModel:
    return ThrottledChatModel(
        credential=OpenAICredential(api_key="sk-test", base_url="http://localhost:1/v1"),
        model="test-model",
        stream=False,
        max_retries=0,
        throttle=throttle,
        role=role,
        rate_limit_backoff=0.2,
    )


class _Tracker:
    """记录同时在飞的调用数峰值。"""

    def __init__(self) -> None:
        self.inflight = 0
        self.peak = 0
        self.starts: list[float] = []


def _patch_call(model: ThrottledChatModel, tracker: _Tracker, hold: float = 0.05) -> None:
    async def fake_call_api(*_args: object, **_kwargs: object) -> ChatResponse:
        tracker.starts.append(time.monotonic())
        tracker.inflight += 1
        tracker.peak = max(tracker.peak, tracker.inflight)
        try:
            await asyncio.sleep(hold)
            return ChatResponse(content=[], is_last=True)
        finally:
            tracker.inflight -= 1

    model._call_api = fake_call_api  # type: ignore[method-assign]


@pytest.mark.asyncio
async def test_concurrency_cap_is_enforced() -> None:
    throttle = GatewayThrottle(max_concurrency=2, min_interval=0.0)
    model = _model(throttle)
    tracker = _Tracker()
    _patch_call(model, tracker)

    await asyncio.gather(*(model([]) for _ in range(6)))

    assert tracker.peak <= 2, f"并发上限被击穿，峰值 {tracker.peak}"
    assert len(tracker.starts) == 6


@pytest.mark.asyncio
async def test_min_interval_paces_request_starts() -> None:
    """间隔是**起点到起点**：6 个并发请求也得按 0.05s 一个往外放。"""
    throttle = GatewayThrottle(max_concurrency=8, min_interval=0.05)
    model = _model(throttle)
    tracker = _Tracker()
    _patch_call(model, tracker, hold=0.0)

    await asyncio.gather(*(model([]) for _ in range(4)))

    starts = sorted(tracker.starts)
    gaps = [b - a for a, b in zip(starts, starts[1:], strict=False)]
    # 留 20% 余量给事件循环调度抖动
    assert all(g >= 0.04 for g in gaps), f"起点间隔没生效：{gaps}"


@pytest.mark.asyncio
async def test_stream_holds_slot_until_generator_exhausted() -> None:
    """流式返回的是生成器，请求还在飞——slot 必须持有到耗尽，否则并发上限形同虚设。"""
    throttle = GatewayThrottle(max_concurrency=1, min_interval=0.0)
    model = _model(throttle)

    async def fake_stream(*_args: object, **_kwargs: object) -> AsyncGenerator[ChatResponse, None]:
        async def gen() -> AsyncGenerator[ChatResponse, None]:
            yield ChatResponse(content=[], is_last=False)
            yield ChatResponse(content=[], is_last=True)

        return gen()

    model._call_api = fake_stream  # type: ignore[method-assign]

    stream = await model([])
    # 只取第一片，不耗尽：此时唯一的并发位应仍被占着
    first = await anext(stream)
    assert first is not None
    assert throttle._sem.locked(), "流式尚未耗尽，slot 就被提前归还了"

    async for _ in stream:
        pass
    assert not throttle._sem.locked(), "流式耗尽后 slot 没归还"


@pytest.mark.asyncio
async def test_stream_slot_released_on_early_close() -> None:
    """消费者提前 aclose（用户取消 / 上游 break）也要归还，否则一次取消永久扣掉一个并发位。"""
    throttle = GatewayThrottle(max_concurrency=1, min_interval=0.0)
    model = _model(throttle)

    async def fake_stream(*_args: object, **_kwargs: object) -> AsyncGenerator[ChatResponse, None]:
        async def gen() -> AsyncGenerator[ChatResponse, None]:
            for _ in range(5):
                yield ChatResponse(content=[], is_last=False)

        return gen()

    model._call_api = fake_stream  # type: ignore[method-assign]

    stream = await model([])
    await anext(stream)
    await stream.aclose()
    assert not throttle._sem.locked(), "提前关闭后 slot 泄漏"


@pytest.mark.asyncio
async def test_rate_limit_penalizes_next_request() -> None:
    """撞 429 后闸门自己降速——靠退避而不是靠重试硬撞。"""
    throttle = GatewayThrottle(max_concurrency=4, min_interval=0.0)
    model = _model(throttle)

    async def boom(*_args: object, **_kwargs: object) -> ChatResponse:
        raise RuntimeError("Error code: 429 - Too Many Requests")

    model._call_api = boom  # type: ignore[method-assign]

    with pytest.raises(RuntimeError):
        await model([])

    # 下一次可发时刻被推后了 rate_limit_backoff 秒
    assert throttle._next_allowed_at - time.monotonic() > 0.1
    # 且并发位已归还（异常路径不许泄漏 slot）
    assert not throttle._sem.locked()


@pytest.mark.asyncio
async def test_non_rate_limit_error_does_not_penalize() -> None:
    """超时 / 连接断不是「我发太快了」，一起降速会把偶发抖动误判成拥塞。"""
    throttle = GatewayThrottle(max_concurrency=4, min_interval=0.0)
    model = _model(throttle)

    async def boom(*_args: object, **_kwargs: object) -> ChatResponse:
        raise ValueError("schema mismatch")

    model._call_api = boom  # type: ignore[method-assign]

    with pytest.raises(ValueError):
        await model([])
    assert throttle._next_allowed_at - time.monotonic() <= 0.0


@pytest.mark.asyncio
async def test_agent_falls_back_and_reports_agui_event(monkeypatch: pytest.MonkeyPatch) -> None:
    """主模型连撞 429 → Agent 换备用模型 → 前端能看见一条 model_fallback。

    「今天怎么变慢/变笨了」这类问题查不出根因，多半就是降级发生了却没人看见。
    """
    from agentscope.agent import Agent, ModelConfig
    from agentscope.message import Msg, TextBlock
    from agentscope.tool import Toolkit

    from app.api import monitor

    reported: list[str] = []

    async def fake_report(model: str) -> None:
        reported.append(model)

    monkeypatch.setattr(monitor, "report_model_fallback", fake_report)

    throttle = GatewayThrottle(max_concurrency=4, min_interval=0.0)
    main = _model(throttle, role="main")
    fallback = _model(throttle, role="fallback")
    fallback.model = "backup-model"

    attempts: list[str] = []

    async def boom(*_args: object, **_kwargs: object) -> ChatResponse:
        attempts.append("main")
        raise RuntimeError("Error code: 429 - Too Many Requests")

    async def ok(*_args: object, **_kwargs: object) -> ChatResponse:
        attempts.append("fallback")
        return ChatResponse(
            content=[TextBlock(type="text", text="已切到备用模型，这是回答。")],
            is_last=True,
        )

    main._call_api = boom  # type: ignore[method-assign]
    fallback._call_api = ok  # type: ignore[method-assign]

    agent = Agent(
        name="gateway_spike",
        system_prompt="回答用户。",
        model=main,
        toolkit=Toolkit(),
        # max_retries=1 → 主模型总共试 2 次（初始 + 1 次重试）后才换备用
        model_config=ModelConfig(max_retries=1, fallback_model=fallback),
    )
    reply = await agent.reply(
        Msg(name="user", role="user", content=[TextBlock(type="text", text="在吗")]),
    )

    assert attempts.count("main") == 2, f"主模型没按 max_retries 试满就换人：{attempts}"
    assert "fallback" in attempts, "备用模型压根没被调用"
    assert reported == ["backup-model"], f"model_fallback 事件没发出：{reported}"
    text = "".join(b.text for b in reply.content if b.type == "text")
    assert "备用模型" in text
    # 备用模型也走同一个闸门，slot 不许泄漏
    assert not throttle._sem.locked()


# --- 工厂层：get_as_* 与备用模型的启用判据 -------------------------------------


def _clear_factory_caches() -> None:
    from app.agent import llm

    llm._load_params()


def test_as_factories_build_throttled_models(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.agent import llm

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:1/v1")
    monkeypatch.setenv("LLM_MAIN", "main-model")
    monkeypatch.setenv("LLM_MAX_CONCURRENCY", "3")
    _clear_factory_caches()

    main = llm.get_as_llm()
    fast = llm.get_as_fast_llm()
    assert isinstance(main, ThrottledChatModel)
    # 主 / 快档共用同一个闸门实例：网关 RPM 是按 key 算的，分池等于把限流让给运气
    assert main._throttle is fast._throttle is llm.get_gateway_throttle()
    assert main._throttle.max_concurrency == 3
    # 快档默认关思考（同款模型只省 thinking 解码，不换弱模型）
    assert fast.extra_body == {"enable_thinking": False}
    assert main.extra_body is None
    _clear_factory_caches()


def test_fallback_model_disabled_when_unset_or_same(monkeypatch: pytest.MonkeyPatch) -> None:
    """同一个模型当自己的备用毫无意义——不配、或配得与主模型同名，都视为不启用。"""
    from app.agent import llm

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:1/v1")
    monkeypatch.setenv("LLM_MAIN", "main-model")

    monkeypatch.delenv("LLM_FALLBACK_MODEL", raising=False)
    _clear_factory_caches()
    assert llm.get_as_fallback_llm() is None
    assert llm.get_model_config().fallback_model is None

    monkeypatch.setenv("LLM_FALLBACK_MODEL", "main-model")
    _clear_factory_caches()
    assert llm.get_as_fallback_llm() is None

    monkeypatch.setenv("LLM_FALLBACK_MODEL", "backup-model")
    _clear_factory_caches()
    fb = llm.get_as_fallback_llm()
    assert fb is not None and fb.model == "backup-model" and fb._role == "fallback"
    # Agent 层不再叠加重试：模型自己那层已经重试过，两层相乘会把 429 火上浇油
    assert llm.get_model_config().max_retries == 0
    _clear_factory_caches()
