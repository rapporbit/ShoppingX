"""A 块 · 可观测性的确定性单测：Prometheus metrics + structlog 上下文绑定。

- metrics：工具耗时 / 调用计数累积、fork 计数、断路器状态 gauge、/metrics 端点渲染。
- structlog：thread_scope / enter_fork 把 thread_id / user_id / fork_depth 绑进日志上下文。
"""

from __future__ import annotations

from pathlib import Path

import structlog

from app.observability import metrics
from app.observability.logging import configure_logging
from app.utils.circuit_breaker import CircuitBreaker, all_breakers


def _metric_value(name: str, labels: dict[str, str] | None = None) -> float:
    """从默认 registry 取某指标当前值（labels 精确匹配）；取不到返回 0。"""
    from prometheus_client import REGISTRY

    val = REGISTRY.get_sample_value(name, labels or {})
    return float(val) if val is not None else 0.0


# ---------- metrics 累积 ----------
def test_record_tool_accumulates_count_and_duration() -> None:
    before = _metric_value("shoppingx_tool_calls_total", {"tool": "unit_probe", "status": "ok"})
    metrics.record_tool("unit_probe", 0.123, "ok")
    after = _metric_value("shoppingx_tool_calls_total", {"tool": "unit_probe", "status": "ok"})
    assert after == before + 1
    # histogram 的 _count 也 +1。
    cnt = _metric_value("shoppingx_tool_duration_seconds_count", {"tool": "unit_probe"})
    assert cnt >= 1


def test_record_tool_error_status_separately() -> None:
    before = _metric_value("shoppingx_tool_calls_total", {"tool": "unit_probe2", "status": "error"})
    metrics.record_tool("unit_probe2", 0.01, "error")
    after = _metric_value("shoppingx_tool_calls_total", {"tool": "unit_probe2", "status": "error"})
    assert after == before + 1


def test_inc_fork_counts() -> None:
    before = _metric_value("shoppingx_fork_total")
    metrics.inc_fork()
    metrics.inc_fork()
    assert _metric_value("shoppingx_fork_total") == before + 2


def test_set_task_slots_and_active() -> None:
    metrics.set_active_tasks(3)
    metrics.set_task_slots(active=2, limit=8)
    assert _metric_value("shoppingx_active_tasks") == 3
    assert _metric_value("shoppingx_task_slots_active") == 2
    assert _metric_value("shoppingx_task_slots_limit") == 8


# ---------- 断路器状态 gauge ----------
async def test_refresh_circuit_breakers_reflects_state() -> None:
    import httpx

    cb = CircuitBreaker("metrics_probe", failure_threshold=1, recovery_timeout=30.0)
    assert cb in all_breakers()

    async def _boom() -> None:
        raise httpx.ConnectError("x")

    # 熔断它。
    try:
        await cb.call(_boom)
    except httpx.ConnectError:
        pass
    metrics.refresh_circuit_breakers()
    # OPEN = 1。
    assert _metric_value("shoppingx_circuit_breaker_state", {"name": "metrics_probe"}) == 1.0


# ---------- /metrics 端点渲染 ----------
def test_render_returns_prometheus_text() -> None:
    metrics.record_tool("render_probe", 0.05, "ok")
    body, content_type = metrics.render()
    assert b"shoppingx_tool_calls_total" in body
    assert "text/plain" in content_type


# ---------- structlog 上下文绑定 ----------
def test_thread_scope_binds_log_context() -> None:
    from app.utils.thread_ctx import thread_scope

    configure_logging()
    structlog.contextvars.clear_contextvars()

    with thread_scope("thr-xyz", Path("/tmp/x"), user_id="u-7"):
        ctx = structlog.contextvars.get_contextvars()
        assert ctx.get("thread_id") == "thr-xyz"
        assert ctx.get("user_id") == "u-7"
    # 离开作用域后还原。
    assert "thread_id" not in structlog.contextvars.get_contextvars()


def test_enter_fork_binds_depth() -> None:
    from app.agent.fork_guard import enter_fork

    structlog.contextvars.clear_contextvars()
    with enter_fork():
        assert structlog.contextvars.get_contextvars().get("fork_depth") == 1
    assert "fork_depth" not in structlog.contextvars.get_contextvars()
