"""阶段 2 第 1 条：LLM 断路器。

断言的全是**口径**，不是「能不能熔断」——熔断本身是 :mod:`app.utils.circuit_breaker` 早就
验过的。这里守的是那三条容易被后来者「顺手统一一下」的分档：429 不计、总超时不计、
首 token 超时计且掐断。三条里任何一条被改成「都算失败」，线上表现是限流高峰时自己把自己熔断，
而测试如果只测「失败会熔断」是看不出来的。
"""

import asyncio
import time
from collections.abc import AsyncGenerator

import httpx
import openai
import pytest
from agentscope.credential import OpenAICredential
from agentscope.model import ChatResponse

from app.agent.gateway import GatewayThrottle, ThrottledChatModel
from app.agent.llm_breaker import (
    CircuitOpenError,
    FirstTokenTimeout,
    counts_as_failure,
    get_llm_breaker,
    reset_llm_breakers,
)


@pytest.fixture(autouse=True)
def _clean_breakers() -> AsyncGenerator[None, None]:
    """断路器是进程级状态，跨用例必须清干净，否则谁先跑谁赢。"""
    reset_llm_breakers()
    yield
    reset_llm_breakers()


@pytest.fixture(autouse=True)
def _tight_thresholds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_BREAKER_ENABLED", "1")
    monkeypatch.setenv("LLM_BREAKER_THRESHOLD", "3")
    monkeypatch.setenv("LLM_BREAKER_RECOVERY_SEC", "0.15")
    monkeypatch.setenv("LLM_FIRST_TOKEN_TIMEOUT", "0")


def _model(model: str = "prov/test-model", stream: bool = False) -> ThrottledChatModel:
    return ThrottledChatModel(
        credential=OpenAICredential(api_key="sk-test", base_url="http://localhost:1/v1"),
        model=model,
        stream=stream,
        max_retries=0,
        throttle=GatewayThrottle(max_concurrency=4),
        rate_limit_backoff=0.0,
    )


class _Calls:
    def __init__(self) -> None:
        self.n = 0


def _patch_raise(model: ThrottledChatModel, exc: BaseException) -> _Calls:
    calls = _Calls()

    async def fake(*_a: object, **_k: object) -> ChatResponse:
        calls.n += 1
        raise exc

    model._call_api = fake  # type: ignore[method-assign]
    return calls


def _timeout_error() -> openai.APITimeoutError:
    return openai.APITimeoutError(request=httpx.Request("POST", "http://localhost:1/v1"))


async def _drain(model: ThrottledChatModel) -> None:
    async for _ in await model([]):
        pass


@pytest.mark.asyncio
async def test_连续故障到阈值后快速失败且不再发请求() -> None:
    model = _model()
    calls = _patch_raise(model, RuntimeError("Error code: 503 - upstream overloaded"))

    for _ in range(3):
        with pytest.raises(RuntimeError):
            await model([])
    assert calls.n == 3

    with pytest.raises(CircuitOpenError):
        await model([])
    assert calls.n == 3, "OPEN 态还发请求 = 快速失败没生效"


@pytest.mark.asyncio
async def test_限流不计入熔断() -> None:
    """429 说明我们发太快，不说明对面坏了——熔断它等于自己把自己关在门外。"""
    model = _model()
    calls = _patch_raise(model, RuntimeError("Error code: 429 - Too Many Requests"))

    for _ in range(10):
        with pytest.raises(RuntimeError):
            await model([])

    assert calls.n == 10
    assert get_llm_breaker("prov/test-model").state == "closed"


@pytest.mark.asyncio
async def test_总超时不计入熔断() -> None:
    """流跑满 LLM_REQUEST_TIMEOUT 说明它慢，不说明它坏。长回答一多就会误熔断。"""
    model = _model()
    calls = _patch_raise(model, _timeout_error())

    for _ in range(10):
        with pytest.raises(openai.APITimeoutError):
            await model([])

    assert calls.n == 10
    assert get_llm_breaker("prov/test-model").state == "closed"


@pytest.mark.asyncio
async def test_开关关掉后完全不熔断(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_BREAKER_ENABLED", "0")
    model = _model()
    calls = _patch_raise(model, RuntimeError("Error code: 500 - boom"))

    for _ in range(6):
        with pytest.raises(RuntimeError):
            await model([])

    assert calls.n == 6
    assert get_llm_breaker("prov/test-model") is None


@pytest.mark.asyncio
async def test_半开探测成功即恢复() -> None:
    model = _model()
    calls = _patch_raise(model, RuntimeError("Error code: 500 - boom"))
    for _ in range(3):
        with pytest.raises(RuntimeError):
            await model([])
    assert get_llm_breaker("prov/test-model").state == "open"

    await asyncio.sleep(0.2)  # 过恢复窗口

    async def ok(*_a: object, **_k: object) -> ChatResponse:
        calls.n += 1
        return ChatResponse(content=[], is_last=True)

    model._call_api = ok  # type: ignore[method-assign]
    await model([])
    assert calls.n == 4
    assert get_llm_breaker("prov/test-model").state == "closed"


@pytest.mark.asyncio
async def test_半开探测撞限流不悬空() -> None:
    """探测撞 429 时若什么都不记，状态会卡在 half_open、allow 从此恒真 = 断路器静默失效。"""
    breaker = get_llm_breaker("prov/test-model")
    model = _model()
    _patch_raise(model, RuntimeError("Error code: 500 - boom"))
    for _ in range(3):
        with pytest.raises(RuntimeError):
            await model([])
    await asyncio.sleep(0.2)

    _patch_raise(model, RuntimeError("Error code: 429 - Too Many Requests"))
    with pytest.raises(RuntimeError):
        await model([])
    assert breaker.state == "open", "探测结果中立却留在 half_open = 悬空"

    # 窗口已过且没被刷新，下一次调用应当还能再探一次。
    calls = _Calls()

    async def ok(*_a: object, **_k: object) -> ChatResponse:
        calls.n += 1
        return ChatResponse(content=[], is_last=True)

    model._call_api = ok  # type: ignore[method-assign]
    await model([])
    assert calls.n == 1 and breaker.state == "closed"


@pytest.mark.asyncio
async def test_按模型分键互不影响() -> None:
    down = _model("prov/down")
    up = _model("prov/up")
    _patch_raise(down, RuntimeError("Error code: 500 - boom"))
    calls = _Calls()

    async def ok(*_a: object, **_k: object) -> ChatResponse:
        calls.n += 1
        return ChatResponse(content=[], is_last=True)

    up._call_api = ok  # type: ignore[method-assign]

    for _ in range(4):
        with pytest.raises((RuntimeError, CircuitOpenError)):
            await down([])
    await up([])

    assert get_llm_breaker("prov/down").state == "open"
    assert get_llm_breaker("prov/up").state == "closed"
    assert calls.n == 1


def test_四类异常的分档() -> None:
    assert counts_as_failure(RuntimeError("Error code: 502 - bad gateway")) is True
    req = httpx.Request("POST", "http://x")
    assert counts_as_failure(openai.APIConnectionError(request=req)) is True
    assert counts_as_failure(FirstTokenTimeout("m", 15.0)) is True
    assert counts_as_failure(_timeout_error()) is False
    assert counts_as_failure(RuntimeError("Error code: 429 - rate limit")) is False
    assert counts_as_failure(asyncio.CancelledError()) is False
    # 4xx 是请求本身有问题，换谁都是 400。
    bad = openai.BadRequestError(
        "Error code: 400", response=httpx.Response(400, request=req), body=None
    )
    assert counts_as_failure(bad) is False
    # max_tokens=5000 这种数字不该被当成 5xx。
    assert counts_as_failure(RuntimeError("invalid max_tokens 5000")) is False


class _StreamState:
    def __init__(self) -> None:
        self.closed = False


def _patch_stream(
    model: ThrottledChatModel, *, first_delay: float = 0.0, tail_delay: float = 0.0
) -> _StreamState:
    """替身流：可分别控制「首片多久到」与「首片之后还要吐多久」。这两段正是两档超时的分界。"""
    from agentscope.message import TextBlock

    state = _StreamState()

    async def gen() -> AsyncGenerator[ChatResponse, None]:
        try:
            await asyncio.sleep(first_delay)
            yield ChatResponse(content=[TextBlock(type="text", text="hi")], is_last=False)
            await asyncio.sleep(tail_delay)
            yield ChatResponse(content=[], is_last=True)
        finally:
            state.closed = True

    async def fake(*_a: object, **_k: object) -> AsyncGenerator[ChatResponse, None]:
        return gen()

    model._call_api = fake  # type: ignore[method-assign]
    return state


@pytest.mark.asyncio
async def test_首token超时掐断并计入(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_FIRST_TOKEN_TIMEOUT", "0.2")
    model = _model(stream=True)
    state = _patch_stream(model, first_delay=5.0)

    started = time.monotonic()
    with pytest.raises(FirstTokenTimeout):
        await _drain(model)
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, "没掐断，等满了上游的慢流"
    assert state.closed, "掐断了却没关流，连接会随请求一起漏"
    assert get_llm_breaker("prov/test-model")._fail_count == 1


@pytest.mark.asyncio
async def test_首token到得快慢流不计(monkeypatch: pytest.MonkeyPatch) -> None:
    """首片 0.05s 就到、之后吐了 0.4s——超过预算但不是故障，这是长回答的常态。"""
    monkeypatch.setenv("LLM_FIRST_TOKEN_TIMEOUT", "0.2")
    model = _model(stream=True)
    _patch_stream(model, first_delay=0.05, tail_delay=0.4)

    for _ in range(3):
        await _drain(model)

    assert get_llm_breaker("prov/test-model").state == "closed"


@pytest.mark.asyncio
async def test_重试不吃掉首token预算(monkeypatch: pytest.MonkeyPatch) -> None:
    """首次尝试超时 0.3s 后重试，第二次立刻出片——预算按**每次尝试**算，不能把前一次的耗时记上。

    错了的话，限流/抖动一重试就会误掐一次本来要成功的调用，还记一次故障，正好绕过「429 不计」。
    这里 patch 的是父类方法，我们自己那层 ``_call_api`` 的打点必须仍在链上。
    """
    from agentscope.message import TextBlock
    from agentscope.model import OpenAIChatModel

    monkeypatch.setenv("LLM_FIRST_TOKEN_TIMEOUT", "0.2")
    calls = _Calls()

    async def gen() -> AsyncGenerator[ChatResponse, None]:
        yield ChatResponse(content=[TextBlock(type="text", text="hi")], is_last=True)

    async def fake(*_a: object, **_k: object) -> AsyncGenerator[ChatResponse, None]:
        calls.n += 1
        if calls.n == 1:
            await asyncio.sleep(0.3)
            raise _timeout_error()
        return gen()

    monkeypatch.setattr(OpenAIChatModel, "_call_api", fake)
    model = ThrottledChatModel(
        credential=OpenAICredential(api_key="sk-test", base_url="http://localhost:1/v1"),
        model="prov/test-model",
        stream=True,
        max_retries=1,
        retry_delay=0.01,
        throttle=GatewayThrottle(max_concurrency=4),
        rate_limit_backoff=0.0,
    )

    await _drain(model)

    assert calls.n == 2
    assert get_llm_breaker("prov/test-model").state == "closed"


@pytest.mark.asyncio
async def test_上游中途关流不计故障(monkeypatch: pytest.MonkeyPatch) -> None:
    """用户点取消 / 上游 aclose 是我们自己的动作，与供应商健不健康无关。"""
    monkeypatch.setenv("LLM_FIRST_TOKEN_TIMEOUT", "0.2")
    model = _model(stream=True)
    _patch_stream(model)

    for _ in range(3):
        stream = await model([])
        async for _chunk in stream:
            break
        await stream.aclose()

    assert get_llm_breaker("prov/test-model").state == "closed"
