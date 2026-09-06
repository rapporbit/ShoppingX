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


class _FakeSpanClient:
    """能正常起 span 的假 client（span 收尾不吞异常，跟 Langfuse 真实行为一致）。"""

    def start_as_current_observation(self, **_kw: Any) -> Any:
        import contextlib

        @contextlib.contextmanager
        def _span() -> Any:
            yield "span"

        return _span()

    def get_current_trace_id(self) -> str:
        return "trace-1"


def test_turn_span_lets_body_errors_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """with 体里的业务异常必须原样穿透，不能被 "generator didn't stop after throw()" 盖掉。

    真实事故：q03 撞上游内容审核，``openai.APIError`` 被这里吞掉后生成器又 yield 了一次，
    终端只剩 ``RuntimeError``，归因要往上翻 50 行栈。
    """
    monkeypatch.setattr(T, "_get_client", lambda: _FakeSpanClient())

    class _Boom(Exception):
        pass

    with pytest.raises(_Boom):
        with T.turn_span(session_id="t1"):
            raise _Boom("上游内容审核拦截")


def test_turn_span_lets_cancellation_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """取消（``BaseException`` 一族）同样得穿透——否则前端点「取消」会变成一条假 RuntimeError。"""
    import asyncio

    monkeypatch.setattr(T, "_get_client", lambda: _FakeSpanClient())
    with pytest.raises(asyncio.CancelledError):
        with T.turn_span(session_id="t1"):
            raise asyncio.CancelledError()


def test_turn_span_swallows_span_teardown_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """span **收尾**失败仍是观测自身的毛病：吞掉记日志，主链路当没事发生。"""

    class _TeardownBoom(_FakeSpanClient):
        def start_as_current_observation(self, **_kw: Any) -> Any:
            import contextlib

            @contextlib.contextmanager
            def _span() -> Any:
                yield "span"
                raise RuntimeError("上报失败")

            return _span()

    monkeypatch.setattr(T, "_get_client", lambda: _TeardownBoom())
    with T.turn_span(session_id="t1") as span:
        assert span == "span"  # 不抛即通过


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


def test_turn_span_propagates_prompt_version_and_bucket(monkeypatch: pytest.MonkeyPatch) -> None:
    """A/B 归属要**顺着 propagate 通道抹到本轮所有子 span**，不能只设在根上（批 4 / 18-3）。

    只设在根 span 上时，Langfuse 里按 version 聚合会漏掉所有模型 / 工具 span——版本对比表
    看起来「有数据」，但成本与延迟那两列是空的。
    """
    import contextlib

    import langfuse

    captured: dict[str, Any] = {}

    @contextlib.contextmanager
    def _fake_propagate(**kwargs: Any) -> Any:
        captured.update(kwargs)
        yield None

    monkeypatch.setattr(langfuse, "propagate_attributes", _fake_propagate, raising=False)
    monkeypatch.setattr(T, "_get_client", lambda: _FakeSpanClient())
    with T.turn_span(session_id="t1", user_id="u1", prompt_version="1.1.0", ab_bucket=7):
        pass

    assert captured["version"] == "1.1.0"  # Langfuse 原生维度，UI 里可直接切分
    assert captured["metadata"] == {"ab_bucket": 7}
    assert captured["session_id"] == "t1" and captured["user_id"] == "u1"


def test_turn_span_without_ab_info_sends_no_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """没有 A/B 信息时不塞空 metadata：trace 上多一个恒为 None 的维度只会污染聚合。"""
    import contextlib

    import langfuse

    captured: dict[str, Any] = {}

    @contextlib.contextmanager
    def _fake_propagate(**kwargs: Any) -> Any:
        captured.update(kwargs)
        yield None

    monkeypatch.setattr(langfuse, "propagate_attributes", _fake_propagate, raising=False)
    monkeypatch.setattr(T, "_get_client", lambda: _FakeSpanClient())
    with T.turn_span(session_id="t1"):
        pass

    assert captured["metadata"] is None and captured["version"] is None
