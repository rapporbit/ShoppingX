"""工具 RT 告警的确定性单测（refdocs 16-3 §6）。

告警器的价值全在「什么时候该响、什么时候该闭嘴」，所以测的重点是状态机与窗口语义，而不是 webhook
能不能发出去：滑动时间窗真的按时间裁剪、样本不足时不用分位数糊弄、hard_ms 兜住低 QPS、滞回防抖动、
冷却防刷屏、断路器 HALF_OPEN 不算恢复。时间轴全部注入，不 sleep。
"""

from __future__ import annotations

import pytest

from app.observability import alerts
from app.utils import circuit_breaker as cb_mod
from app.utils.circuit_breaker import CircuitBreaker

_TOOL = "shipping_calc"
_RULE = alerts._RULES_BY_TOOL[_TOOL]

# 耗时档位一律**从规则派生**，不写死毫秒数——阈值是拿真实基线校准出来的（见 DEFAULT_RULES 注释），
# 每次重新校准都会变。写死魔数的话，调一次阈值就得回来修一堆测试。
_FIRE_MS = _RULE.p95_threshold_ms * 2  # 明确越线（且低于 hard_ms，避免混进 hard 分支）
_BAND_MS = _RULE.p95_threshold_ms * 0.96  # 滞回带内：低于阈值，但高于阈值的 90%
_CLEAR_MS = _RULE.p95_threshold_ms * 0.8  # 确实降下来了


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch: pytest.MonkeyPatch):
    """每个用例都从干净的窗口 / 状态机出发，且冷却设长（默认不发 ongoing）。

    断路器注册表是**进程级全局**：别的测试文件建的、还停在 OPEN 的断路器会混进 ``check_rules()``
    的返回，让「本用例只该有一条告警」的断言随测试顺序飘。这里换一份空注册表隔离掉。
    """
    alerts.reset()
    monkeypatch.setattr(cb_mod, "_BREAKERS", {})
    monkeypatch.setenv("ALERT_COOLDOWN_SEC", "900")
    yield
    alerts.reset()


def _feed(n: int, ms: float, now: float, tool: str = _TOOL, trace_id: str | None = None) -> None:
    for _ in range(n):
        alerts.record_tool_sample(tool, ms, trace_id=trace_id, now=now)


def _keys(found: list[alerts.Alert]) -> dict[str, str]:
    return {a.key: a.level for a in found}


# ---------- 规则与工具集的一致性 ----------
def test_every_rule_targets_a_real_tool() -> None:
    """规则里的 tool 名必须真的在 FULL_TOOL_SET 里。

    名字对不上的规则是**死规则**：``record_tool_sample`` 按名字过滤，采不到任何样本，于是永远不会
    告警——而且毫无征兆，看起来一切正常。工具改名 / 拆分时这个测试会红，比线上某天「怎么从来没报过」
    要便宜得多。
    """
    from app.agent.tool_registry import FULL_TOOL_SET

    real = {t.name for t in FULL_TOOL_SET}
    unknown = {r.tool for r in alerts.DEFAULT_RULES} - real
    assert not unknown, f"这些规则指向不存在的工具，永远不会告警：{sorted(unknown)}"


def test_fork_meta_tools_are_covered() -> None:
    """两个 fork 元工具都要有规则。

    跨平台检索实际走的是 ``parallel_dispatch_tool``——只给 ``dispatch_tool`` 配规则，最花时间的
    那条路径（一次 fork = 一整棵子 AgentLoop）反而没人盯着。
    """
    ruled = {r.tool for r in alerts.DEFAULT_RULES}
    assert {"dispatch_tool", "parallel_dispatch_tool"} <= ruled


# ---------- 分位数 ----------
def test_percentile_nearest_rank() -> None:
    assert alerts.percentile([], 0.95) == 0.0
    assert alerts.percentile([5.0], 0.95) == 5.0
    # 1..100 的 P95 = 第 95 个（nearest-rank，非插值）。
    assert alerts.percentile([float(i) for i in range(1, 101)], 0.95) == 95.0


# ---------- 窗口与样本量 ----------
def test_no_alert_when_samples_below_min() -> None:
    """样本不足 min_samples 时不算分位数——10 个样本的 P95 就是 max，一次慢调用就会误报。"""
    _feed(_RULE.min_samples - 1, ms=_FIRE_MS, now=100.0)
    assert alerts.check_rules(now=100.0) == []


def test_p95_breach_fires_once_then_cooldown_silences() -> None:
    _feed(_RULE.min_samples, ms=_FIRE_MS, now=100.0)
    first = alerts.check_rules(now=100.0)
    assert _keys(first) == {_TOOL: "firing"}
    assert f"P95={_FIRE_MS:.0f}ms" in first[0].detail
    # 冷却期内重复越线 → 闭嘴，不刷屏。
    assert alerts.check_rules(now=200.0) == []


def test_ongoing_after_cooldown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALERT_COOLDOWN_SEC", "60")
    _feed(_RULE.min_samples, ms=_FIRE_MS, now=100.0)
    assert _keys(alerts.check_rules(now=100.0)) == {_TOOL: "firing"}
    assert alerts.check_rules(now=130.0) == []  # 冷却未过
    assert _keys(alerts.check_rules(now=170.0)) == {_TOOL: "ongoing"}  # 过了冷却，周期性提醒


def test_samples_expire_by_time_not_by_count() -> None:
    """window_sec 必须真的按时间裁剪——refdocs 原文只按条数截断，低 QPS 下算的是几小时前的事故。"""
    _feed(_RULE.min_samples, ms=_FIRE_MS, now=100.0)
    assert _keys(alerts.check_rules(now=100.0)) == {_TOOL: "firing"}
    # 越过 300s 窗口，旧样本全部过期 → 窗口空 → 判恢复。
    assert _keys(alerts.check_rules(now=100.0 + _RULE.window_sec + 1)) == {_TOOL: "resolved"}
    assert len(alerts._windows[_TOOL]) == 0  # 过期样本就地 popleft，不滞留


def test_hysteresis_keeps_firing_between_90pct_and_threshold() -> None:
    """跌回阈值以下但仍在 90% 带内 → 不报恢复，防止贴着阈值线触发/恢复反复刷屏。"""
    _feed(_RULE.min_samples, ms=_FIRE_MS, now=100.0)
    assert _keys(alerts.check_rules(now=100.0)) == {_TOOL: "firing"}

    # 换一批更快的样本（清窗口不清状态机——正在告警中）。
    alerts._windows[_TOOL].clear()
    _feed(_RULE.min_samples, ms=_BAND_MS, now=110.0)  # 低于阈值，但仍在 90% 滞回带内
    assert alerts.check_rules(now=110.0) == []  # 滞回带内：既不恢复也不重复告警

    alerts._windows[_TOOL].clear()
    _feed(_RULE.min_samples, ms=_CLEAR_MS, now=120.0)  # 跌破 90% 带，真的降下来了
    assert _keys(alerts.check_rules(now=120.0)) == {_TOOL: "resolved"}


def test_hysteresis_band_stays_silent_even_after_cooldown(monkeypatch: pytest.MonkeyPatch) -> None:
    """滞回带里冷却过期也不该发 ongoing——值已经低于阈值了，再报「仍在越线」是自相矛盾的。

    状态机若写成两态（breached / not breached），带内的值会被当成「仍越线」，冷却一过就推一条
    detail 写着「P95=480ms > 阈值 500ms」的持续告警。480 并不大于 500。
    """
    monkeypatch.setenv("ALERT_COOLDOWN_SEC", "60")
    _feed(_RULE.min_samples, ms=_FIRE_MS, now=100.0)
    assert _keys(alerts.check_rules(now=100.0)) == {_TOOL: "firing"}

    alerts._windows[_TOOL].clear()
    _feed(_RULE.min_samples, ms=_BAND_MS, now=110.0)  # 滞回带内
    assert alerts.check_rules(now=300.0) == []  # 冷却早过了，仍然闭嘴
    assert alerts._states[_TOOL].firing is True  # 但告警状态保持，没被误判成恢复


# ---------- hard_ms 绝对阈值 ----------
def test_hard_threshold_fires_without_enough_samples() -> None:
    """夜里没流量、分位数哑火时，「一次调用花了 6 秒」本身就该有人知道。"""
    assert _RULE.hard_ms is not None
    _feed(1, ms=_RULE.hard_ms + 1, now=100.0)
    found = alerts.check_rules(now=100.0)
    assert _keys(found) == {_TOOL: "firing"}
    assert "硬阈值" in found[0].detail


# ---------- 采样边界 ----------
def test_unregistered_tool_is_not_sampled() -> None:
    """ask_user 故意没有规则（它阻塞等用户，慢是语义）——不该被采样。"""
    alerts.record_tool_sample("ask_user", 99_999.0, now=100.0)
    assert "ask_user" not in alerts._windows


def test_alert_carries_slowest_call_trace_id() -> None:
    _feed(_RULE.min_samples, ms=_FIRE_MS, now=100.0, trace_id="fast-trace")
    alerts.record_tool_sample(_TOOL, _FIRE_MS * 1.4, trace_id="slow-trace", now=100.0)
    found = alerts.check_rules(now=100.0)
    assert found[0].trace_id == "slow-trace"  # 挑最慢那次，点开直接看根因


def test_trace_url_uses_project_id_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com")
    monkeypatch.delenv("LANGFUSE_PROJECT_ID", raising=False)
    text = alerts.format_alert(alerts.Alert("firing", "k", "t", "d", trace_id="abc"))
    assert "https://us.cloud.langfuse.com/trace/abc" in text

    monkeypatch.setenv("LANGFUSE_PROJECT_ID", "proj-1")
    text = alerts.format_alert(alerts.Alert("firing", "k", "t", "d", trace_id="abc"))
    assert "https://us.cloud.langfuse.com/project/proj-1/traces/abc" in text


def test_trace_url_absent_when_no_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGFUSE_BASE_URL", raising=False)
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)
    text = alerts.format_alert(alerts.Alert("firing", "k", "t", "d", trace_id="abc"))
    assert "trace" not in text.lower()


# ---------- 断路器规则 ----------
def test_breaker_open_fires_and_half_open_is_not_resolved() -> None:
    cb = CircuitBreaker("probe_dep", failure_threshold=1, recovery_timeout=0.0)
    cb.record_failure()  # 阈值 1 → 立刻 OPEN
    assert _keys(alerts.check_rules(now=100.0))["breaker:probe_dep"] == "firing"

    # 恢复窗口到点 → allow() 推进到 HALF_OPEN。探测还没成功，不该报「已恢复」。
    assert cb.allow() is True
    assert cb.state == "half_open"
    assert alerts.check_rules(now=110.0) == []

    cb.record_success()  # 探测成功 → CLOSED
    assert _keys(alerts.check_rules(now=120.0))["breaker:probe_dep"] == "resolved"


# ---------- 安全事件规则 ----------
def test_security_alert_on_delta_only() -> None:
    from app.observability.metrics import SECURITY_EVENTS

    alerts._security_deltas()  # 吞掉存量（模拟启动时的基线快照）
    assert alerts._evaluate_security() == []

    SECURITY_EVENTS.labels(kind="unit_probe_kind").inc()
    found = alerts._evaluate_security()
    assert any("unit_probe_kind" in a.key for a in found)
    # 计数没再涨 → 不重复报（Counter 是单调的，累计值本身不构成信号）。
    assert alerts._evaluate_security() == []


# ---------- 通知降级 ----------
async def test_send_alert_without_webhook_only_logs(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """没配 webhook 也一定落 error 日志——refdocs 原文直接 return，告警就凭空消失了。"""
    monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)
    with caplog.at_level("ERROR", logger="shoppingx.alerts"):
        await alerts.send_alert(alerts.Alert("firing", "k", "工具变慢", "P95 超标"))
    assert "工具变慢" in caplog.text


async def test_check_and_notify_survives_rule_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """规则评估炸了也不能让后台轮询 task 死掉——告警是观测附属品，不反噬主链路。"""
    monkeypatch.setattr(alerts, "check_rules", lambda: 1 / 0)
    await alerts.check_and_notify()  # 不抛即通过


def test_webhook_payload_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALERT_WEBHOOK_KIND", "slack")
    assert alerts._webhook_payload("x") == {"text": "x"}
    monkeypatch.setenv("ALERT_WEBHOOK_KIND", "dingtalk")
    assert alerts._webhook_payload("x") == {"msgtype": "text", "text": {"content": "x"}}
