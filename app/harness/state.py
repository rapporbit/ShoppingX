"""GuardState：一个 Agent 实例独享的控制面状态。

Hook 是**模块级函数**、全局注册，天然无处安放「这个 Agent 实例搜了几次」这类状态。所以状态集中
放这里，由 :class:`HarnessAgentAdapter` 每实例新建一个，经 ``context["_guard"]`` 传给 Hook。

**绝不跨 Agent 实例复用**——否则会话之间计数串台（LoopDetector 窗口、检索计数、终结标记都是
per-loop 语义）。跨 fork 树共享的量（fork 轮数 / 全树检索总量 / token 成本）另有 ContextVar +
session_dir 聚合兜底，不在这里。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.harness.budgets import DEFAULT_RETRIEVAL_CAP, TREE_RETRIEVAL_BUDGET
from app.harness.loop_detector import LoopDetector
from app.harness.truncation import MAX_TOOL_RESULT_TOKENS


@dataclass
class GuardState:
    """单个 AgentLoop（主或子）的控制面状态。"""

    # ── 可调参数（构造时注入，便于测试与不同 loop 用不同口径）──
    max_tool_tokens: int = MAX_TOOL_RESULT_TOKENS
    retrieval_cap: int = DEFAULT_RETRIEVAL_CAP
    tree_retrieval_cap: int = TREE_RETRIEVAL_BUDGET
    loop_window: int = 6
    loop_threshold: int = 4

    # ── 运行时状态 ──
    detector: LoopDetector = field(init=False)
    #: per-instance 检索计数（无 session 作用域时的回退口径）
    retrieval_count: int = 0
    #: per-instance item_search 计数
    item_search_calls: int = 0
    #: 主 loop 本轮是否已调过终结工具（终结硬停闸用）
    terminal_reached: bool = False
    #: 置上 ``terminal_reached`` 时所处的 ``think_step``（= 批次 id，-1 为未置位）。
    #: 同一条 AI 消息发出的并行调用共享一个 think_step，框架按 ``is_concurrency_safe`` 把它们
    #: 合成一个 concurrent 批用 gather 并发跑——布尔位在这种交错下会让「谁先跑完谁拦死兄弟调用」，
    #: 出几张确认卡取决于事件循环调度。记下批次号，闸才分得清「兄弟调用」与「收尾后的新动作」。
    terminal_step: int = -1
    #: 本次模型调用内已因「纯文字收尾」重发过几次（上限 MAX_TERMINAL_NUDGE_RETRIES）。
    #: 由适配器在每次 ``on_model_call`` 开头清零——配额是 per-call，不是 per-loop。
    terminal_nudge_retries: int = 0
    #: Think 计步：每次请求模型即一次 Think，供前端「思考中（第 N 步）」展示
    think_step: int = 0
    #: 上一次生效的预算档位（main/lite/minimal/fallback）。只为「降档时上报一次 metric」去重——
    #: 一个 20 轮的任务不去重会把 minimal 记 15 次，降级率统计直接失真。
    last_tier: str = "main"
    #: 已发过的阶段收线通告（"search_close" / "picks_close" / "reuse_skip"）——同一通告一个 loop
    #: 只发一次，免得并行两条 item_search 都非空时结果尾部重复两遍。回退 / 补搜时由
    #: phase_rollback / refine_backfill 摘掉 "search_close" 重新武装（新一轮检索需要重新收线）。
    notified_transitions: set[str] = field(default_factory=set)
    #: 硬闸拒绝计数（"{gate}:{escape_key}" → 次数），统一逃生门用（middleware._try_escape）：
    #: 效率闸对同一目标连拒达到阈值即放行——模型的反复坚持是「上游判定（如 planner 误判 reuse）
    #: 可能错了」的强信号，墙必须带门，否则死锁（2026-07-14 线上：reuse 误判 + item_search
    #: 拦死 27 轮）。安全闸（白名单/深度/预算/终结）不接入，永远硬。
    gate_reject_counts: dict[str, int] = field(default_factory=dict)
    #: 逃生门的**批次快照**（step, 快照计数）：同一条 AI 消息里的并行工具调用（同 think_step）
    #: 用批次起点的计数统一裁决——要拦全拦、要放全放，且同批多次拒绝只记一次「连拒」。
    #: 否则同批 4 个并行调用会前 2 个被拒攒计数、后 2 个触发逃生放行——谁被拦由**到达顺序**
    #: 决定，与需求无关（badcase 4c0ac682：恰好拦掉唯一需要补搜的槽）。
    escape_snapshot_step: int = -1
    escape_snapshot: dict[str, int] = field(default_factory=dict)
    #: liveness 看门狗（hooks/watchdog.py）：上次「实质进展」时刻（monotonic；0=未开表）。
    #: 只有工具**真实执行**成功才更新（awrap_tool_call）——被闸拦下 / memo 回放不算。
    last_progress_at: float = 0.0
    #: 看门狗收敛指令的发出时刻（monotonic；0=未武装）。真实进展会复位。
    watchdog_nudged_at: float = 0.0

    def __post_init__(self) -> None:
        self.detector = LoopDetector(window=self.loop_window, threshold=self.loop_threshold)


def guard_of(context: dict[str, Any]) -> GuardState | None:
    """从 Hook 的 context 里取本 loop 的 GuardState；没有（单测裸跑 Hook）返回 None。"""
    guard = context.get("_guard")
    return guard if isinstance(guard, GuardState) else None
