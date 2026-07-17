"""Prometheus 系统级 metrics（A 块）——QPS / 延迟分位 / 错误率 / 熔断状态可看板化。

**为什么是 metrics 而不是再上一套 trace。** 调用树级追踪 Langfuse v4 已经有了（它本身基于
OpenTelemetry、把多 fork 归并成一棵树，见 ``app/agent/tracing.py``）。真正缺的是**系统级聚合
指标**——「过去 5 分钟 item_search 的 P99 是多少」「哪个外呼错误率在涨」「断路器熔了几个」。
这些是 Langfuse 的单条 trace 答不了、却是定位延迟回归（如之前的 295s）最需要的。

**打点位置。**
- 工具耗时 / 调用数：``HarnessAgentMiddleware.awrap_tool_call`` 包住工具执行处（一处覆盖全部工具）。
- fork 数：``monitor.report_fork``。
- 运行时 gauge（活跃任务 / 任务槽 / 断路器状态）：在 ``/metrics`` 被 scrape 时即时刷新——
  这些是「当前值」，scrape 那一刻读最准，不必实时维护。

不引重型 exporter / pushgateway：进程内 ``prometheus_client`` 暴露 ``/metrics`` 文本端点即可，
Prometheus 来拉。规模上来再谈远端写。
"""

from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from app.utils.circuit_breaker import HALF_OPEN, OPEN, all_breakers

# 工具耗时分布：桶覆盖「几十毫秒的本地工具」到「几十秒的跨平台检索」，便于看 P50/P95/P99。
TOOL_DURATION = Histogram(
    "shoppingx_tool_duration_seconds",
    "单个工具执行耗时（秒）",
    ["tool"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
)
# 工具调用计数：status=ok/error，可算错误率。
TOOL_CALLS = Counter("shoppingx_tool_calls_total", "工具调用次数", ["tool", "status"])
# fork 子任务派发数。
FORK_TOTAL = Counter("shoppingx_fork_total", "派发的同质子 Agent 数")
# 运行时 gauge（scrape 时刷新）。
ACTIVE_TASKS = Gauge("shoppingx_active_tasks", "当前活跃的主 AgentLoop 任务数")
TASK_SLOT_ACTIVE = Gauge("shoppingx_task_slots_active", "已占用的任务并发槽数")
TASK_SLOT_LIMIT = Gauge("shoppingx_task_slots_limit", "任务并发槽上限")
# 断路器状态：0=CLOSED（正常）/ 1=OPEN（熔断）/ 2=HALF_OPEN（探测）。
CIRCUIT_STATE = Gauge("shoppingx_circuit_breaker_state", "断路器状态(0关合/1断开/2半开)", ["name"])
# 缓存命中/未命中：result=exact_hit / semantic_hit / miss，可算命中率（E 块）。
CACHE_EVENTS = Counter("shoppingx_cache_events_total", "缓存命中/未命中", ["cache", "result"])
# LLM 累计成本（美元，F 块 FinOps）：每个任务收尾把该任务全树成本加进来，可看总开销 / 均值。
TOKEN_COST = Counter("shoppingx_llm_cost_usd_total", "LLM 累计成本(美元)")
# 任务收尾时的预算档位：result=ok / soft / hard，可看多少比例任务撞到预算闸（F 块）。
BUDGET_OUTCOME = Counter("shoppingx_budget_outcome_total", "任务收尾时的预算档位", ["result"])
# 安全事件：kind=tool_not_allowed / prompt_injection_filtered / output_api_key …（refdocs 16-6）。
# 这条曲线平时应该恒为 0——一旦抬头就说明有人在试探，或者我们的某个数据源被污染了。
SECURITY_EVENTS = Counter("shoppingx_security_events_total", "安全护栏拦截事件", ["kind"])
# 预算档位切换：tier=lite / minimal / fallback（refdocs 16-4 §7，降级率 = 有切换的任务 / 总任务）。
BUDGET_TIER_CHANGE = Counter("shoppingx_budget_tier_change_total", "模型路由降级次数", ["tier"])
# 任务排队：kind=normal / heavy，分别记「直接拿到槽」与「排过队」（refdocs 16-5 §2）。
TASK_QUEUED = Counter("shoppingx_task_queued_total", "任务入队等待次数", ["kind"])
TASK_REJECTED = Counter("shoppingx_task_rejected_total", "任务被拒次数", ["reason"])
QUEUE_PENDING = Gauge("shoppingx_queue_pending", "当前排队等待的任务数", ["kind"])
# 硬闸事件：outcome=reject（拒绝工具调用）/ escape（效率闸连拒达阈值后逃生放行）。
# 「哪堵墙被模型撞得最多」是闸立错位置的直接信号——某 gate 的 escape 抬头说明它依据的
# 上游判定常失准；reject 持续高但零 escape 的多半是安全闸被反复试探，值得翻 trace。
GATE_EVENTS = Counter("shoppingx_gate_events_total", "硬闸拒绝/逃生事件", ["gate", "outcome"])

_STATE_CODE = {OPEN: 1, HALF_OPEN: 2}  # 其余（CLOSED）记 0


def record_tool(tool: str, duration_seconds: float, status: str) -> None:
    """记一次工具执行的耗时与成败。``status`` 取 ``ok`` / ``error``。"""
    TOOL_DURATION.labels(tool=tool).observe(duration_seconds)
    TOOL_CALLS.labels(tool=tool, status=status).inc()


def inc_fork() -> None:
    """记一次 fork 派发。"""
    FORK_TOTAL.inc()


def record_cache(cache: str, result: str) -> None:
    """记一次缓存查询结果。``result`` 取 ``exact_hit`` / ``semantic_hit`` / ``miss``。"""
    CACHE_EVENTS.labels(cache=cache, result=result).inc()


def record_cost(cost_usd: float, outcome: str) -> None:
    """任务收尾记一次成本与预算档位（F 块）。``outcome`` 取 ``ok`` / ``soft`` / ``hard``。"""
    if cost_usd > 0:
        TOKEN_COST.inc(cost_usd)
    BUDGET_OUTCOME.labels(result=outcome).inc()


def record_security_event(kind: str) -> None:
    """记一次安全护栏拦截（L1 白名单 / L3 内容过滤 / L4 输出脱敏）。"""
    SECURITY_EVENTS.labels(kind=kind).inc()


def record_gate_event(gate: str, outcome: str) -> None:
    """记一次硬闸事件。``outcome`` 取 ``reject`` / ``escape``。"""
    GATE_EVENTS.labels(gate=gate, outcome=outcome).inc()


def record_tier_change(tier: str) -> None:
    """记一次模型路由降级（``lite`` / ``minimal`` / ``fallback``）。"""
    BUDGET_TIER_CHANGE.labels(tier=tier).inc()


def record_task_queued(kind: str) -> None:
    """记一次任务入队等待（槽位已满、进了有界队列）。``kind`` 取 ``normal`` / ``heavy``。"""
    TASK_QUEUED.labels(kind=kind).inc()


def record_task_rejected(reason: str) -> None:
    """记一次任务被拒。

    ``reason`` 取 ``queue_full`` / ``duplicate`` / ``already_running`` / ``quota_exhausted``。
    """
    TASK_REJECTED.labels(reason=reason).inc()


def set_queue_pending(kind: str, n: int) -> None:
    """刷新某类队列的排队深度 gauge（在 /metrics scrape 时由 server 调用）。"""
    QUEUE_PENDING.labels(kind=kind).set(n)


def set_task_slots(active: int, limit: int) -> None:
    """刷新任务并发槽 gauge（在 /metrics scrape 时由 server 调用）。"""
    TASK_SLOT_ACTIVE.set(active)
    TASK_SLOT_LIMIT.set(limit)


def set_active_tasks(n: int) -> None:
    """刷新活跃任务 gauge（在 /metrics scrape 时由 server 调用）。"""
    ACTIVE_TASKS.set(n)


def refresh_circuit_breakers() -> None:
    """遍历所有断路器、把当前状态写进 gauge（在 /metrics scrape 时调用）。"""
    for cb in all_breakers():
        CIRCUIT_STATE.labels(name=cb.name).set(_STATE_CODE.get(cb.state, 0))


def render() -> tuple[bytes, str]:
    """渲染 Prometheus 文本格式，返回 ``(body, content_type)`` 供 /metrics 端点直接回。"""
    return generate_latest(), CONTENT_TYPE_LATEST
