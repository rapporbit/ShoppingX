"""工具 RT 告警（refdocs 16-3 §6）——把「能看」的指标变成「会报」的告警。

`/metrics` 是被动的：Prometheus 不来拉就没人知道 ``item_search`` 的 P95 涨了一倍。这里补上主动
通知的那一半：后台每 ``ALERT_CHECK_INTERVAL_SEC`` 秒评估一次规则，越线推 webhook（钉钉 / Slack），
没配 webhook 就落 ``logger.error``——**永远不静默**。

**与 refdocs §6 示例代码的四处出入**（原样抄会得到一个不工作的告警器）：

1. 原文 ``AlertRule.window_minutes`` 是**死字段**——``deque(maxlen=200)`` 按条数截断，不按时间。
   低 QPS 下这 200 条可能横跨几小时，算出来的分位数是几小时前的事故。这里按 ``window_sec``
   真正做时间窗裁剪，``maxlen`` 只作内存上界。
2. 原文 ``min_samples=10`` 配 P99 等于「max 值告警」——10 个样本的 99 分位就是最大值，任何一次
   慢调用都触发。这里改用 **P95 + min_samples=20**，并另设 ``hard_ms`` 绝对阈值兜住低 QPS
   （样本不够时单次超长调用照样报，见 :meth:`_evaluate_rt`）。
3. 原文 ``check_alerts()`` **没有调用者**，抄下来就是死代码。这里由 ``server.py`` 的 lifespan
   起后台轮询 task 驱动。
4. 原文 ``send_alert`` 里 ``if not webhook_url: return``——没配 webhook 告警就凭空消失。

**只喂 status=ok 的样本进 RT 窗口。** 失败调用的耗时（尤其断路器 OPEN 时的快速失败，~0ms）会把
P95 拉低，反而在故障最严重时报「已恢复」。错误面由 ``TOOL_CALLS{status}`` 指标与断路器规则覆盖。

**告警消息带 trace 链接。** 窗口里连 trace_id 一起存，触发时挑最慢那次调用的 trace，直接给出
Langfuse 链接——refdocs 16-3 §5 的「打开面板 → 筛选 → 找 trace」三步压成一次点击。

**边界（诚实标注）**：样本窗是**进程内**的。``uvicorn --workers N`` 下各进程各算各的 P95，同一次
抖动会推 N 条告警、每条只看到 1/N 样本。当前部署是单进程，够用；真上多 worker 就得把窗口挪去
Redis，或退回 Prometheus + Alertmanager（``/metrics`` 已就绪，改造成本不高）。
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from collections import deque
from dataclasses import dataclass

from app.agent.tracing import trace_url
from app.observability.metrics import SECURITY_EVENTS
from app.utils.circuit_breaker import CLOSED, all_breakers
from app.utils.env import env_bool, env_float

logger = logging.getLogger("shoppingx.alerts")

# 单个工具的样本上界：只防内存无界，真正的窗口边界是 rule.window_sec 的时间裁剪。
_MAX_SAMPLES = 2000
# 滞回带：触发后要跌回阈值的 90% 以下才算「恢复」。防 P95 贴着阈值线抖动导致触发/恢复刷屏。
_RESOLVE_RATIO = 0.9


@dataclass(frozen=True)
class AlertRule:
    """一条工具 RT 规则。

    ``p95_threshold_ms``：窗口内 P95 越过它即触发。

    ``hard_ms``：绝对阈值。窗口内**任意一次**调用超过它就触发，不看样本量——补住「夜里没流量、
    分位数因样本不足而哑火」的洞。设为 None 表示不启用。
    """

    tool: str
    p95_threshold_ms: float
    hard_ms: float | None = None
    window_sec: float = 300.0
    min_samples: int = 20


# ── 阈值怎么来的：`uv run python scripts/eval/tool_rt_baseline.py` 跑真实种子集量出来的 ──
#
# **告警阈值是观测的结论，不是观测的输入。** refdocs 给的 item_search=3000ms 是本地 mock 数据的
# 水位；实测本项目的 item_search P95 只有 470ms（Qdrant 本地 + reranker 命中缓存），照抄那个数字
# 等于「要慢 6 倍才报警」——规则形同虚设。反过来 planner / parallel_dispatch_tool 动辄几十秒，
# 用 refdocs 的秒级阈值会天天响，响到没人再看。**两个方向都错。**
#
# 下面每条注释里的实测值来自 17 条真实 query（数据落 data/eval/tool_rt_baseline.json）。
# 阈值取「P95 × 1.8」并向上取整；样本 < 20 的工具其 P95 统计上接近 max，额外放宽。
#
# **不设规则的工具，以及为什么：**
# - ask_user：实测 P50 = 120_018ms，因为它 asyncio.wait_for 等用户回复、等满 ASK_USER_TIMEOUT_SEC
#   才返回。**它的慢是语义不是故障**，设规则等于每次澄清都告警。这条只有量了基线才知道。
# - forget_preference：极低频（用户主动撤回偏好才调），攒不出统计意义的窗口。
DEFAULT_RULES: tuple[AlertRule, ...] = (
    # 外呼型（跨网，最该盯）。
    AlertRule("item_search", p95_threshold_ms=1_000, hard_ms=8_000),  # 实测 P95 470 / P99 1888
    AlertRule("category_insight", p95_threshold_ms=3_000, hard_ms=10_000),  # 实测 P95 1287
    AlertRule("web_search", p95_threshold_ms=5_000, hard_ms=15_000),  # 实测 P95 2391（Tavily 外网）
    # 本地计算型：查表 + 算术，慢了说明数据结构或 IO 出了问题。实测都在 20ms 以内。
    AlertRule("price_compare", p95_threshold_ms=500, hard_ms=3_000),  # 实测 P95 20
    AlertRule("shipping_calc", p95_threshold_ms=500, hard_ms=3_000),  # 实测 P95 11
    AlertRule("item_picker", p95_threshold_ms=2_000, hard_ms=8_000),  # 实测 P95 461
    # LLM 型：内部还要跑一次模型调用，基线本就高，阈值跟着抬。
    AlertRule("planner", p95_threshold_ms=45_000, hard_ms=90_000),  # 实测 P95 23_539
    AlertRule("shopping_summary", p95_threshold_ms=25_000, hard_ms=60_000),  # 实测 P95 11_900
    AlertRule("chat_fallback", p95_threshold_ms=12_000, hard_ms=30_000),  # 实测 P95 5_953
    # 元工具：一次 dispatch = 一整棵子 AgentLoop 跑完，阈值按整轮墙钟给。
    # parallel_dispatch_tool 才是跨平台检索实际走的那个（实测 P95 39_930，14 次调用）；
    # dispatch_tool（串行单发）在这 17 条 query 里**一次都没被调用**，阈值按并行版的一半估，
    # 真跑起来再校准。别只给 dispatch_tool 配规则——最花时间的那条路径反而没人盯着。
    AlertRule("parallel_dispatch_tool", p95_threshold_ms=75_000, hard_ms=180_000),
    AlertRule("dispatch_tool", p95_threshold_ms=40_000, hard_ms=120_000),
)

_RULES_BY_TOOL: dict[str, AlertRule] = {r.tool: r for r in DEFAULT_RULES}


@dataclass(frozen=True)
class Alert:
    """一条待通知的告警。``level`` 取 ``firing`` / ``ongoing`` / ``resolved``。"""

    level: str
    key: str
    title: str
    detail: str = ""
    trace_id: str | None = None


@dataclass
class _RuleState:
    """规则的告警状态机：是否正在告警 + 上次通知时刻（用于冷却）。"""

    firing: bool = False
    last_notified_at: float = 0.0


# 样本 = (单调时钟时刻, 耗时毫秒, 当时的 trace_id)。trace_id 让告警能直接给出「最慢那次」的链接。
_Sample = tuple[float, float, str | None]
_windows: dict[str, deque[_Sample]] = {}
_states: dict[str, _RuleState] = {}
# 安全事件计数的上次快照：Counter 是单调递增的，告警看的是**周期内增量**而非累计值。
_security_seen: dict[str, float] = {}


# ────────────────────────────── 采样 ──────────────────────────────
def record_tool_sample(
    tool: str, duration_ms: float, trace_id: str | None = None, now: float | None = None
) -> None:
    """记一次**成功**的工具调用耗时。由 middleware 的打点处调用，与 metrics 同一个计时。

    只登记 ``DEFAULT_RULES`` 里有规则的工具——没规则的工具存了也没人算，白占内存。
    ``now`` 仅供测试注入时间轴，生产不传。
    """
    if tool not in _RULES_BY_TOOL:
        return
    win = _windows.setdefault(tool, deque(maxlen=_MAX_SAMPLES))
    win.append((time.monotonic() if now is None else now, duration_ms, trace_id))


def reset() -> None:
    """清空全部窗口与告警状态（测试用；生产不调）。"""
    _windows.clear()
    _states.clear()
    _security_seen.clear()


# ────────────────────────────── 分位数 ──────────────────────────────
def percentile(values: list[float], q: float) -> float:
    """nearest-rank 分位数：``sorted[ceil(q*n)-1]``。

    不引 numpy——样本量有上界（2000），排序开销可忽略，而且这是在后台 task 里跑。空列表返回 0。
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = math.ceil(q * len(ordered)) - 1
    return ordered[max(0, min(idx, len(ordered) - 1))]


def _fresh_samples(tool: str, window_sec: float, now: float) -> list[_Sample]:
    """裁掉窗口外的旧样本并返回窗口内的。就地 popleft，避免窗口长期滞留过期数据。"""
    win = _windows.get(tool)
    if not win:
        return []
    cutoff = now - window_sec
    while win and win[0][0] < cutoff:
        win.popleft()
    return list(win)


# ────────────────────────────── 规则评估 ──────────────────────────────
def _transition(key: str, breached: bool, cleared: bool, now: float, cooldown: float) -> str | None:
    """告警状态机：返回该发的通知级别，或 None（什么都不发）。

    **三态而非两态**：``breached``（越线，该报）与 ``cleared``（确实降下来了，该报恢复）不是互补的
    ——中间那条滞回带（阈值的 90%~100%）两者皆假，语义是「保持现状、什么都不发」。写成两态的话，
    带内的值会被当成「仍在越线」，冷却一过就推一条 ``ongoing``，detail 里还会打出
    「P95=760ms > 阈值 800ms」这种自相矛盾的话。

    - 未告警 + 越线 → ``firing``
    - 已告警 + 确实回落 → ``resolved``
    - 已告警 + 仍越线 + 过了冷却 → ``ongoing``（周期性提醒，不刷屏）
    - 已告警 + 落在滞回带 → None
    """
    st = _states.setdefault(key, _RuleState())
    if not st.firing:
        if breached:
            st.firing, st.last_notified_at = True, now
            return "firing"
        return None
    if cleared:
        st.firing = False
        return "resolved"
    if breached and now - st.last_notified_at >= cooldown:
        st.last_notified_at = now
        return "ongoing"
    return None


def _evaluate_rt(rule: AlertRule, now: float, cooldown: float) -> Alert | None:
    """评估一条工具 RT 规则。P95 超阈 **或** 窗口内出现单次超 ``hard_ms`` 的调用即算越线。

    ``hard_ms`` 那一支不看样本量：低 QPS 时分位数没有统计意义，但「一次调用花了 30 秒」本身
    就是需要有人知道的事实。
    """
    samples = _fresh_samples(rule.tool, rule.window_sec, now)
    durations = [d for _, d, _ in samples]

    p95 = percentile(durations, 0.95) if len(durations) >= rule.min_samples else 0.0
    slowest = max(durations) if durations else 0.0
    hard_hit = rule.hard_ms is not None and slowest > rule.hard_ms

    breached = p95 > rule.p95_threshold_ms or hard_hit
    # 滞回：跌回阈值 90% 以下才算真的恢复；hard_ms 还在命中期间一律不判恢复。
    # 两者之间是滞回带 —— 既不报警也不报恢复，防 P95 贴着阈值线来回抖导致刷屏。
    cleared = not hard_hit and p95 <= rule.p95_threshold_ms * _RESOLVE_RATIO

    level = _transition(rule.tool, breached, cleared, now, cooldown)
    if level is None:
        return None

    if level == "resolved":
        return Alert("resolved", rule.tool, f"{rule.tool} RT 已恢复", f"P95={p95:.0f}ms")

    # 挑最慢那次的 trace：点开就是 refdocs §5 的 Step 3「展开 Trace 树找出问题的 Span」。
    slowest_trace = max(samples, key=lambda s: s[1])[2] if samples else None
    reason = (
        f"单次 {slowest:.0f}ms > 硬阈值 {rule.hard_ms:.0f}ms"
        if hard_hit
        else f"P95={p95:.0f}ms > 阈值 {rule.p95_threshold_ms:.0f}ms"
    )
    detail = (
        f"{reason}\n窗口 {rule.window_sec / 60:.0f}min / 样本 {len(durations)} 次 / "
        f"最慢 {slowest:.0f}ms"
    )
    return Alert(level, rule.tool, f"{rule.tool} 响应变慢", detail, slowest_trace)


def _evaluate_breakers(now: float, cooldown: float) -> list[Alert]:
    """断路器非 CLOSED 即告警——外呼已经在快速失败了，人必须立刻知道。

    与 RT 规则互补：RT 看「慢」，断路器看「错」。reranker 只是慢（不报错）时断路器不会响，
    正是 RT 规则该抓的；反之外呼直接 5xx 时 RT 窗口里根本没有 ok 样本，只有断路器能报。

    判恢复用 ``state == CLOSED`` 而非 ``!= OPEN``：HALF_OPEN 是「放一次探测出去试试」，还没好，
    此时报恢复会让人以为没事了、结果探测失败又立刻熔回去。
    """
    out: list[Alert] = []
    for cb in all_breakers():
        key = f"breaker:{cb.name}"
        state = cb.state
        # 断路器状态是离散的，没有滞回带：非 CLOSED 即告警，CLOSED 即恢复。
        level = _transition(key, state != CLOSED, state == CLOSED, now, cooldown)
        if level == "resolved":
            out.append(Alert("resolved", key, f"断路器 {cb.name} 已闭合", "外呼恢复正常"))
        elif level is not None:
            detail = f"当前状态 {state.upper()}，{cb.name} 外呼被快速拒绝"
            out.append(Alert(level, key, f"断路器 {cb.name} 熔断", detail))
    return out


def _security_deltas() -> dict[str, float]:
    """读 SECURITY_EVENTS 各 kind 的**增量**（Counter 单调递增，累计值本身不构成告警信号）。"""
    deltas: dict[str, float] = {}
    for metric in SECURITY_EVENTS.collect():
        for sample in metric.samples:
            if not sample.name.endswith("_total"):
                continue
            kind = sample.labels.get("kind", "?")
            delta = sample.value - _security_seen.get(kind, 0.0)
            _security_seen[kind] = sample.value
            if delta > 0:
                deltas[kind] = delta
    return deltas


def _evaluate_security() -> list[Alert]:
    """安全护栏拦截数一旦抬头就告警。

    ``metrics.py`` 自己的注释说得明白：这条曲线平时恒为 0，抬头就意味着有人在试探、或某个数据源
    被污染了。所以这里不设阈值、不做状态机——**有增量就报**，一次都不该被平滑掉。
    """
    return [
        Alert("firing", f"security:{kind}", f"安全护栏拦截 {kind}", f"本周期新增 {delta:.0f} 次")
        for kind, delta in _security_deltas().items()
    ]


def check_rules(now: float | None = None) -> list[Alert]:
    """评估全部规则，返回本周期该发的通知。纯计算（只推进内部状态机），无 IO，可单测。"""
    now = time.monotonic() if now is None else now
    cooldown = env_float("ALERT_COOLDOWN_SEC", 900.0)
    alerts = [a for r in DEFAULT_RULES if (a := _evaluate_rt(r, now, cooldown)) is not None]
    alerts.extend(_evaluate_breakers(now, cooldown))
    alerts.extend(_evaluate_security())
    return alerts


# ────────────────────────────── 通知 ──────────────────────────────
_LEVEL_ICON = {"firing": "🔴", "ongoing": "⏱", "resolved": "✅"}


def format_alert(alert: Alert) -> str:
    """渲染成一段人读的文本（钉钉 / Slack / 日志共用）。"""
    lines = [f"{_LEVEL_ICON.get(alert.level, '•')} [ShoppingX] {alert.title}"]
    if alert.detail:
        lines.append(alert.detail)
    if url := trace_url(alert.trace_id):
        lines.append(f"最慢一次的 trace: {url}")
    return "\n".join(lines)


def _webhook_payload(text: str) -> dict[str, object]:
    """按 ``ALERT_WEBHOOK_KIND`` 适配 body 形状——钉钉和 Slack 的 text 字段不是一回事。"""
    kind = os.environ.get("ALERT_WEBHOOK_KIND", "dingtalk").strip().lower()
    if kind == "slack":
        return {"text": text}
    if kind == "generic":
        return {"message": text}
    return {"msgtype": "text", "text": {"content": text}}


async def send_alert(alert: Alert) -> None:
    """推送一条告警。**没配 webhook 也一定会落日志**——告警绝不能凭空消失。

    webhook 失败同样降级回日志。整个链路吞异常：告警是观测附属品，不能反噬主链路
    （与 ``tracing.py`` 的安静降级同一口径）。
    """
    text = format_alert(alert)
    log = logger.info if alert.level == "resolved" else logger.error
    log("%s", text)

    url = os.environ.get("ALERT_WEBHOOK_URL")
    if not url:
        return
    try:
        import httpx

        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(url, json=_webhook_payload(text))
    except Exception:
        logger.warning("告警 webhook 推送失败（已落日志，不影响主链路）", exc_info=True)


async def check_and_notify() -> None:
    """跑一轮规则评估并推送。异常全吞，保证后台轮询 task 不会因为一次故障而死掉。"""
    try:
        alerts = check_rules()
    except Exception:
        logger.warning("告警规则评估失败，跳过本轮", exc_info=True)
        return
    for alert in alerts:
        await send_alert(alert)


async def alert_loop() -> None:
    """后台轮询：每 ``ALERT_CHECK_INTERVAL_SEC`` 秒评估一次。由 server lifespan 起、shutdown 取消。

    先睡后跑：进程刚起来时窗口是空的，立刻评估没有意义，还会让 ``_security_seen`` 把启动前的
    历史计数当成增量误报一次。
    """
    interval = env_float("ALERT_CHECK_INTERVAL_SEC", 60.0)
    _security_deltas()  # 吞掉启动瞬间的存量计数，只报此后的新增
    while True:
        try:
            await asyncio.sleep(interval)
            await check_and_notify()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("告警轮询异常，继续下一轮", exc_info=True)


def alerts_enabled() -> bool:
    """告警总开关（默认开）。关掉只是不起后台 task，采样照常（无副作用、内存有界）。"""
    return env_bool("ALERT_ENABLED", True)
