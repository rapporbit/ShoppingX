"""阶段 6 · SLO 两条：run 成功率的 outcome 分档 + 首事件延迟的计时。

测的是**分档判得对不对**和**计时只记一次**，不测 Prometheus 自己的加法。两条 SLO 的价值全在
口径上——一次 Qdrant 维护该不该把成功率打穿、排队那几秒算不算进「多久有反应」，判错了曲线
再好看也是假的。
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
from agentscope.message import Msg, TextBlock
from agentscope.state import AgentState

import app.agent.orchestrator as orch
from app.api import context as ctx
from app.api import monitor
from app.observability import metrics
from app.utils.dependency import (
    dependency_down_seen,
    mark_dependency_down,
    reset_dependency_down,
)


def _outcome(name: str) -> float:
    return metrics.RUN_OUTCOME.labels(outcome=name)._value.get()


def _fake_agent(final_text: str = "已为你整理好清单。") -> Any:
    class _FakeAgent:
        def __init__(self) -> None:
            self.state = AgentState()

        async def reply_stream(self, inputs: Any, yield_final_msg: bool = False) -> Any:
            yield Msg(
                name="shoppingx",
                role="assistant",
                content=[TextBlock(type="text", text=final_text)],
            )

    return _FakeAgent()


@pytest.fixture
def patched(monkeypatch: pytest.MonkeyPatch) -> None:
    """顶掉会打网络 / 打库的收尾环节，留下主链路本身（与 test_orchestrator 同款）。"""

    async def _noop(*_a: Any, **_kw: Any) -> None:
        return None

    monkeypatch.setattr(orch, "curate_turn", _noop)
    monkeypatch.setattr(orch.monitor, "report_task_result", _noop)


# ---------- SLO 1：run 收尾分档 ----------


async def test_normal_run_counts_as_success(monkeypatch: pytest.MonkeyPatch, patched: None) -> None:
    async def _build(**_kw: Any) -> Any:
        return _fake_agent(), SimpleNamespace()

    monkeypatch.setattr(orch, "build_main_agent", _build)
    before = _outcome("success")
    await orch.run_agent("买个旅行包", thread_id="slo-ok")
    assert _outcome("success") == before + 1


async def test_failure_counts_as_failed(monkeypatch: pytest.MonkeyPatch, patched: None) -> None:
    async def _build(**_kw: Any) -> Any:
        raise RuntimeError("模型挂了")

    monkeypatch.setattr(orch, "build_main_agent", _build)
    before = _outcome("failed")
    with pytest.raises(RuntimeError):
        await orch.run_agent("买个旅行包", thread_id="slo-fail")
    assert _outcome("failed") == before + 1


async def test_user_cancel_is_not_a_failure(monkeypatch: pytest.MonkeyPatch, patched: None) -> None:
    """用户自己掐的不进分母——把它记成 failed 等于「用户越爱按停止，我们的 SLO 越差」。"""

    async def _build(**_kw: Any) -> Any:
        raise asyncio.CancelledError

    monkeypatch.setattr(orch, "build_main_agent", _build)
    before = (_outcome("cancelled"), _outcome("failed"))
    with pytest.raises(asyncio.CancelledError):
        await orch.run_agent("买个旅行包", thread_id="slo-cancel")
    assert (_outcome("cancelled"), _outcome("failed")) == (before[0] + 1, before[1])


async def test_dependency_down_run_is_not_a_failure(
    monkeypatch: pytest.MonkeyPatch, patched: None
) -> None:
    """依赖挂了但 Agent 如实收了尾：记 dependency_rejected，不算 success 也不算 failed。

    这一档单列的理由：Qdrant 维护半小时，期间每条 query 都「跑完了」，记 success 是自欺；
    记 failed 又会把一次计划内维护变成 SLO 事故。
    """

    async def _build(**_kw: Any) -> Any:
        mark_dependency_down()  # 模拟工具壳捕获 DependencyDown
        return _fake_agent("检索服务暂时不可用，我没法给你看库存。"), SimpleNamespace()

    monkeypatch.setattr(orch, "build_main_agent", _build)
    before = (_outcome("dependency_rejected"), _outcome("success"))
    await orch.run_agent("买个旅行包", thread_id="slo-dep")
    assert _outcome("dependency_rejected") == before[0] + 1
    assert _outcome("success") == before[1]


async def test_dependency_flag_survives_child_context() -> None:
    """工具跑在 create_task 出来的子 context 里，旗子必须写得回来。

    这是可变盒子形态存在的唯一理由：换成裸 ``ContextVar[bool]``，下面这个断言会挂——而线上
    表现是「依赖挂了整半天，SLO 曲线一切正常」。
    """
    reset_dependency_down()
    await asyncio.create_task(asyncio.sleep(0))  # 确认 fixture 之外没有残留

    async def _tool() -> None:
        mark_dependency_down()

    await asyncio.create_task(_tool())
    assert dependency_down_seen() is True


def test_slo_denominator_excludes_the_two_non_failures() -> None:
    assert metrics.SLO_DENOMINATOR == {"success", "failed"}
    assert "cancelled" not in metrics.SLO_DENOMINATOR
    assert "dependency_rejected" not in metrics.SLO_DENOMINATOR


def test_slo_targets_are_published_as_gauges() -> None:
    """目标线进指标：看板 / 告警规则不必各自硬编码 0.99 与 3s。"""
    assert metrics.SLO_TARGET.labels(slo="run_success_rate")._value.get() == 0.99
    assert metrics.SLO_TARGET.labels(slo="first_event_p95_seconds")._value.get() == 3.0


# ---------- SLO 2：首事件延迟 ----------


def test_first_event_latency_is_taken_once() -> None:
    """「首」事件只有一个：第二次取必须是 None，否则每轮思考都会被记成一次首事件。"""
    ctx.begin_first_event_timer(time.time() - 1.0)
    first = ctx.take_first_event_latency()
    assert first is not None and first >= 1.0
    assert ctx.take_first_event_latency() is None


def test_no_timer_means_no_measurement() -> None:
    """离线脚本 / 单测直调 monitor 时没开计时盒——不该凭空记一个从 0 算起的数。"""
    ctx._first_event_var.set(None)
    assert ctx.take_first_event_latency() is None


def test_enqueued_at_is_the_start_point() -> None:
    """排队等的那几秒算进首事件延迟：用户不关心他等的是队列还是模型。"""
    queued = datetime.now(UTC).timestamp() - 5.0
    ctx.begin_first_event_timer(
        orch._parse_enqueued_at(datetime.fromtimestamp(queued, UTC).isoformat())
    )
    latency = ctx.take_first_event_latency()
    assert latency is not None and latency >= 5.0


def test_unparsable_enqueued_at_falls_back_to_now() -> None:
    """老消息没这个字段 / 格式变了：按此刻起算，不是记一个 1970 年起的天文数字。"""
    assert orch._parse_enqueued_at("") is None
    assert orch._parse_enqueued_at("上周二") is None
    ctx.begin_first_event_timer(orch._parse_enqueued_at("上周二"))
    latency = ctx.take_first_event_latency()
    assert latency is not None and latency < 1.0


async def test_assistant_call_records_the_latency(monkeypatch: pytest.MonkeyPatch) -> None:
    """终点是第一条 assistant_call —— 用户第一次看见「有反应了」的那一刻。"""
    seen: list[float] = []
    monkeypatch.setattr(metrics, "record_first_event", seen.append)
    monkeypatch.setattr(monitor, "_emit", lambda *_a, **_kw: asyncio.sleep(0))

    ctx.begin_first_event_timer(time.time() - 2.0)
    await monitor.report_assistant_call()
    await monitor.report_assistant_call()  # 第二轮思考不该再记
    assert len(seen) == 1 and seen[0] >= 2.0


def test_negative_latency_is_dropped() -> None:
    """时钟回拨（跨机部署 + NTP 校正）算出来的负数没有意义，丢掉而不是记成 0。"""
    before = metrics.FIRST_EVENT_LATENCY._sum.get()
    metrics.record_first_event(-3.0)
    assert metrics.FIRST_EVENT_LATENCY._sum.get() == before
