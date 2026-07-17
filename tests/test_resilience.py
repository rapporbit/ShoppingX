"""B 块 · 韧性工程的确定性单测：断路器三态机 + 退避重试白名单 + 外呼集成。

- ``CircuitBreaker``：连续失败到阈值 → OPEN（快速失败、不再调真实函数）；恢复窗口后半开探测，
  成功 → CLOSED，失败 → 退回 OPEN。用可控时钟（monkeypatch monotonic）断言时间相关转换。
- ``call_with_retry``：只重试超时 / 5xx，4xx 与其它异常立即抛。
- 集成：reranker 熔断后走本地兜底且**不再发起远程**；web_search 异常降级标 degraded。
"""

from __future__ import annotations

import httpx
import pytest

from app.recall.reranker import RerankerClient
from app.utils.circuit_breaker import CLOSED, OPEN, CircuitBreaker, CircuitOpenError
from app.utils.retry import call_with_retry, is_retryable_http_error


def _boom_fn(exc: Exception):
    async def _f() -> str:
        raise exc

    return _f


async def _ok() -> str:
    return "ok"


# ---------- 断路器三态机 ----------
async def test_breaker_opens_after_threshold() -> None:
    cb = CircuitBreaker("t", failure_threshold=3, recovery_timeout=30.0)
    boom = _boom_fn(httpx.ConnectError("x"))
    for _ in range(3):
        with pytest.raises(httpx.ConnectError):
            await cb.call(boom)
    assert cb.state == OPEN

    # OPEN 后：快速失败、不调用真实函数。
    called = False

    async def _track() -> str:
        nonlocal called
        called = True
        return "y"

    with pytest.raises(CircuitOpenError):
        await cb.call(_track)
    assert called is False  # 真实函数没被调用 = 没干等超时


async def test_breaker_half_open_recovers(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = {"t": 1000.0}
    monkeypatch.setattr("app.utils.circuit_breaker.time.monotonic", lambda: clock["t"])
    cb = CircuitBreaker("t", failure_threshold=1, recovery_timeout=10.0)

    with pytest.raises(httpx.ConnectError):
        await cb.call(_boom_fn(httpx.ConnectError("x")))
    assert cb.state == OPEN

    # 未到恢复窗口：仍快速失败。
    clock["t"] += 5
    with pytest.raises(CircuitOpenError):
        await cb.call(_ok)

    # 过了恢复窗口：半开探测，成功 → CLOSED。
    clock["t"] += 6  # 累计 11 > 10
    assert await cb.call(_ok) == "ok"
    assert cb.state == CLOSED


async def test_breaker_half_open_failure_reopens(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = {"t": 0.0}
    monkeypatch.setattr("app.utils.circuit_breaker.time.monotonic", lambda: clock["t"])
    cb = CircuitBreaker("t", failure_threshold=1, recovery_timeout=10.0)
    boom = _boom_fn(httpx.ConnectError("x"))

    with pytest.raises(httpx.ConnectError):
        await cb.call(boom)
    clock["t"] += 11  # 进入半开

    # 半开探测再次失败 → 退回 OPEN。
    with pytest.raises(httpx.ConnectError):
        await cb.call(boom)
    assert cb.state == OPEN


async def test_breaker_success_resets_fail_count() -> None:
    cb = CircuitBreaker("t", failure_threshold=3, recovery_timeout=30.0)
    boom = _boom_fn(httpx.ConnectError("x"))
    # 失败 2 次（未到阈值 3）。
    for _ in range(2):
        with pytest.raises(httpx.ConnectError):
            await cb.call(boom)
    # 一次成功清零计数。
    assert await cb.call(_ok) == "ok"
    assert cb.state == CLOSED
    # 再失败 2 次仍不熔断（计数已被成功清零，不是累加到 4）。
    for _ in range(2):
        with pytest.raises(httpx.ConnectError):
            await cb.call(boom)
    assert cb.state == CLOSED


# ---------- 退避重试白名单 ----------
def _status_error(code: int) -> httpx.HTTPStatusError:
    req = httpx.Request("GET", "http://x")
    return httpx.HTTPStatusError("e", request=req, response=httpx.Response(code, request=req))


def test_is_retryable_timeout_connection_and_5xx() -> None:
    assert is_retryable_http_error(httpx.ConnectTimeout("x")) is True
    assert is_retryable_http_error(httpx.ConnectError("x")) is True  # 连接被拒 / DNS 抖动
    assert is_retryable_http_error(httpx.ReadError("x")) is True  # 连接被重置
    assert is_retryable_http_error(_status_error(503)) is True
    assert is_retryable_http_error(_status_error(500)) is True
    assert is_retryable_http_error(_status_error(400)) is False  # 4xx 不重试
    assert is_retryable_http_error(_status_error(404)) is False
    assert is_retryable_http_error(httpx.ProxyError("x")) is False  # 配置错（白名单不宽纳）
    assert is_retryable_http_error(ValueError("x")) is False  # 非 HTTP 异常不重试


async def test_call_with_retry_retries_connection_error() -> None:
    calls = 0

    async def _flaky() -> str:
        nonlocal calls
        calls += 1
        if calls < 2:
            raise httpx.ConnectError("refused")  # 连接类瞬时故障也应被重试
        return "ok"

    out = await call_with_retry(_flaky, attempts=3, initial=0.001, max_wait=0.002)
    assert out == "ok"
    assert calls == 2


async def test_call_with_retry_retries_then_succeeds() -> None:
    calls = 0

    async def _flaky() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise httpx.ConnectTimeout("x")
        return "ok"

    out = await call_with_retry(_flaky, attempts=3, initial=0.001, max_wait=0.002)
    assert out == "ok"
    assert calls == 3  # 首次 + 2 次重试


async def test_call_with_retry_no_retry_on_4xx() -> None:
    calls = 0

    async def _f() -> str:
        nonlocal calls
        calls += 1
        raise _status_error(400)

    with pytest.raises(httpx.HTTPStatusError):
        await call_with_retry(_f, attempts=3, initial=0.001)
    assert calls == 1  # 4xx 立即抛、不重试


# ---------- 集成：reranker 熔断后快速失败、走本地兜底 ----------
async def test_reranker_breaker_short_circuits_after_open() -> None:
    rr = RerankerClient(endpoint="http://reranker.invalid/score")
    rr._breaker.failure_threshold = 2  # 调小阈值便于触发
    remote_calls = 0

    async def _boom(query: str, candidates: list[str]) -> list[float]:
        nonlocal remote_calls
        remote_calls += 1
        raise httpx.ConnectError("refused")

    rr._score_remote = _boom  # type: ignore[method-assign]
    cands = ["luggage suitcase", "coffee mug"]

    # 连续失败 2 次 → 熔断；每次都降级到本地、返回可排序结果。
    assert len(await rr.score("q", cands)) == 2
    assert len(await rr.score("q", cands)) == 2
    assert rr._breaker.state == OPEN

    calls_at_open = remote_calls
    # OPEN 后：仍返回本地兜底结果，但**不再发起远程**（快速失败，省掉干等超时）。
    scores = await rr.score("q", cands)
    assert len(scores) == 2
    assert remote_calls == calls_at_open
    # 降级结果就是本地打分。
    assert scores == rr._score_local("q", cands)


# ---------- 集成：web_search 异常降级标 degraded ----------
async def test_web_search_degraded_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.tools.web_search as ws

    ws._breaker.reset()  # 模块级单例，清掉前序测试可能留下的状态
    monkeypatch.setenv("TAVILY_API_KEY", "fake-key")

    async def _boom(*args: object, **kwargs: object) -> object:
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx.AsyncClient, "post", _boom)

    # 直通掉退避重试：ConnectError 现在可重试，真退避会拖慢本用例；这里只验「降级标记」，
    # 重试行为由 retry.py 自己的单元测试覆盖。
    async def _passthrough(fn: object, **kwargs: object) -> object:
        return await fn()  # type: ignore[operator]

    monkeypatch.setattr(ws, "call_with_retry", _passthrough)

    ends: list[dict] = []

    async def _capture_end(tool: str, **fields: object) -> None:
        ends.append(fields)

    async def _noop_start(tool: str, **fields: object) -> None:
        pass

    monkeypatch.setattr(ws.monitor, "report_tool_end", _capture_end)
    monkeypatch.setattr(ws.monitor, "report_tool_start", _noop_start)

    out = await ws.web_search.ainvoke({"query": "测试"})
    assert out.results == []
    assert "调用失败" in out.note  # 降级 note 让模型感知
    assert ends and ends[-1].get("degraded") is True  # 异常降级标了 degraded（修正现状漏标）


async def test_web_search_missing_key_degraded(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.tools.web_search as ws

    ws._breaker.reset()
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    async def _noop_start(tool: str, **fields: object) -> None:
        pass

    async def _noop_end(tool: str, **fields: object) -> None:
        pass

    monkeypatch.setattr(ws.monitor, "report_tool_start", _noop_start)
    monkeypatch.setattr(ws.monitor, "report_tool_end", _noop_end)

    out = await ws.web_search.ainvoke({"query": "测试"})
    assert out.results == []
    assert "TAVILY_API_KEY" in out.note  # 缺 key 的降级路径不变
