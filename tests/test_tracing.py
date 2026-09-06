"""Langfuse 观测接入：安静降级 + 一轮一条 trace 的根 span（确定性，不打网络）。

观测是调试附属品，**任何一步失败都不许反噬主链路**——所以这里测的几乎全是「坏情况下还能不能
照常跑」：没装包、没配 key、client 建不出来、span 起不来，主链路都得一行不差地继续。
"""

from __future__ import annotations

from typing import Any

import pytest

import app.agent.tracing as T


@pytest.fixture(autouse=True)
def _clear_client_cache() -> Any:
    """client 是 lru_cache 的，逐例清掉，免得前一个用例的桩泄漏到后一个。

    收尾时 ``_get_client`` 可能仍是 monkeypatch 换上去的普通函数（fixture 拆解顺序不保证），
    所以取 ``cache_clear`` 要留余地。
    """
    getattr(T._get_client, "cache_clear", lambda: None)()
    yield
    getattr(T._get_client, "cache_clear", lambda: None)()


def test_middlewares_empty_without_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """未启用观测 → 空表。装配处因此不必写 if，主 Agent 与 worker 一视同仁。"""
    monkeypatch.setattr(T, "_get_client", lambda: None)
    assert T.tracing_middlewares() == []


def test_middlewares_carry_the_native_tracer(monkeypatch: pytest.MonkeyPatch) -> None:
    """启用后挂的是框架原生 ``TracingMiddleware``——它打的 ``gen_ai.*`` 属性正是 Langfuse
    的 span 过滤器放行的那一类，两头自动对上，不需要自建 exporter。"""
    from agentscope.middleware import TracingMiddleware

    monkeypatch.setattr(T, "_get_client", lambda: object())
    mws = T.tracing_middlewares()
    assert len(mws) == 1 and isinstance(mws[0], TracingMiddleware)


def test_turn_span_is_noop_without_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """没有 client 时 ``turn_span`` 是个空壳：不抛、trace_id 保持空。"""
    monkeypatch.setattr(T, "_get_client", lambda: None)
    with T.turn_span(session_id="t1", user_id="u1"):
        assert T.current_trace_id() is None


def test_turn_span_swallows_span_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """连起 span 都失败（SDK 版本对不上 / 网络层抽风）也只能降级，不能把主链路带崩。"""

    class _Boom:
        def start_as_current_observation(self, **_kw: Any) -> Any:
            raise RuntimeError("langfuse 抽风")

    monkeypatch.setattr(T, "_get_client", lambda: _Boom())
    with T.turn_span(session_id="t1"):
        pass  # 不抛即通过


def test_scores_skipped_without_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    """本轮没有 trace（未启用观测）时，回注分数直接跳过——不该凭空造一条 trace 出来。"""
    calls: list[Any] = []

    class _Client:
        def create_score(self, **kw: Any) -> None:
            calls.append(kw)

    monkeypatch.setattr(T, "_get_client", lambda: _Client())
    T._current_trace_id.set(None)
    T.record_trace_scores({"cache_hit_rate": 0.9})
    assert calls == []


def test_flush_swallows_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """短命进程退出前的 flush 失败只记日志——评测结论已经算完了，不该因为上报失败而中断。"""

    class _Client:
        def flush(self) -> None:
            raise RuntimeError("网络断了")

    monkeypatch.setattr(T, "_get_client", lambda: _Client())
    T.flush_traces()  # 不抛即通过
