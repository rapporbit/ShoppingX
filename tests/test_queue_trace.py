"""跨队列追踪：API 入队 span → traceparent 进消息 → worker 根 span 接上，同一条 trace。

用**真的** Langfuse client + 内存 exporter（不发网络）：要钉的是 Langfuse 的 ``trace_context``
接续口与 OTel 父子关系真的对上了，替身 client 只能证明「我们调了某个方法」。worker 侧跑在一个
全新的 ``contextvars.Context`` 里——模拟另一个进程：OTel 上下文一点都带不过去，只剩消息里那串字。
"""

import contextvars
import json
import logging
import uuid
from dataclasses import replace
from typing import Any

import pytest
import structlog
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.agent import tracing as T
from app.observability.logging import _ContextFieldsFilter, bind_log_context, unbind_log_context
from app.queue.ports import IntentTask


@pytest.fixture
def exporter(monkeypatch: pytest.MonkeyPatch) -> Any:
    from langfuse import Langfuse

    mem = InMemorySpanExporter()
    client = Langfuse(
        public_key=f"pk-test-{uuid.uuid4().hex}",  # 每个用例一个 key：SDK 按 key 缓存资源单例
        secret_key="sk-test",
        host="http://127.0.0.1:9",
        tracer_provider=TracerProvider(),
        span_exporter=mem,
    )
    monkeypatch.setattr(T, "_get_client", lambda: client)
    yield mem, client
    client.shutdown()


def _worker_side(raw: str) -> str | None:
    """「另一个进程」：只拿得到 JSON 串。返回本轮 trace_id。"""
    task = IntentTask.from_dict(json.loads(raw))
    with T.turn_span(session_id=task.thread_id, parent=task.traceparent):
        return T.current_trace_id()


def test_enqueue_and_turn_are_one_trace(exporter: Any) -> None:
    mem, client = exporter
    task = IntentTask.create(task_id="t1", thread_id="th1", query="买个背包")
    with T.enqueue_span(task_id="t1", session_id="th1") as parent:
        raw = json.dumps(replace(task, traceparent=parent.header()).to_dict())

    worker_trace_id = contextvars.Context().run(_worker_side, raw)
    client.flush()
    spans = {s.name: s for s in mem.get_finished_spans()}

    enq, turn = spans["shoppingx.enqueue"], spans["shoppingx.turn"]
    assert parent.sampled and parent.trace_id == f"{enq.context.trace_id:032x}"
    assert turn.context.trace_id == enq.context.trace_id, "两段不在同一条 trace 上"
    assert turn.parent is not None and turn.parent.span_id == enq.context.span_id
    assert worker_trace_id == parent.trace_id, "返回值 / 评分挂回用的 trace_id 与根不一致"


def test_unsampled_parent_keeps_trace_id_but_no_phantom_parent(exporter: Any) -> None:
    """API 侧观测关着（flags=00 的 traceparent）：沿用 trace_id 供日志对账，但不挂父 span。"""
    mem, client = exporter
    parent = T.new_traceparent()
    raw = json.dumps(IntentTask("t2", "th2", "q", traceparent=parent.header()).to_dict())

    assert contextvars.Context().run(_worker_side, raw) == parent.trace_id
    client.flush()
    (turn,) = [s for s in mem.get_finished_spans() if s.name == "shoppingx.turn"]
    assert turn.parent is None or turn.parent.span_id != int(parent.span_id, 16)


def test_enqueue_without_client_still_yields_traceparent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(T, "_get_client", lambda: None)
    with T.enqueue_span(task_id="t3") as parent:
        pass
    assert T.parse_traceparent(parent.header()) == parent and not parent.sampled


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "garbage",
        "01-" + "a" * 32 + "-" + "b" * 16 + "-01",
        "00-" + "0" * 32 + "-" + "b" * 16 + "-01",
    ],
)
def test_parse_rejects_malformed(bad: str) -> None:
    assert T.parse_traceparent(bad) is None


def test_stdlib_log_records_carry_trace_id() -> None:
    """``bind_log_context`` 绑的字段要进 stdlib 日志——全仓日志都走 stdlib，不桥接等于没绑。"""
    record = logging.LogRecord("shoppingx.queue", logging.INFO, __file__, 1, "入队", None, None)
    tokens = bind_log_context(trace_id="ab" * 16, request_id="rq1", user_id="alice")
    try:
        _ContextFieldsFilter().filter(record)
    finally:
        unbind_log_context(tokens)
    assert record.ctx == f" trace_id={'ab' * 16} request_id=rq1"
    assert "alice" not in record.ctx, "user_id 在这条路上没脱敏，不许进"
    assert "trace_id" not in structlog.contextvars.get_contextvars()
