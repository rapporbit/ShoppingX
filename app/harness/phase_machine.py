"""四阶段对话状态机：PLANNING → SEARCHING → COMPARING → CONCLUDING。

**定位：遥测 + 提示的依据，不是执法者**（harness 重构第三段）。早期版本按阶段白名单在执行层
拒绝「当前阶段不可用」的工具，但阶段表守护的是效率不是安全，且其依据的上游判定（planner
意图）是假设不是承诺——planner 误判 reuse 曾把 item_search 拦死 27 轮（2026-07-14 线上）。
白名单禁令已拆除，现在阶段机只负责三件事：

- **遥测**：记录主 loop 走到购物漏斗哪一步（日志 / Langfuse trace 的观测依据）。
- **prompt 提示**：``transition_notice`` 按阶段边沿给模型指路（「检索收线」「直接收尾」），
  只打动机不打机制。
- **少数安全底线的判据**：``phase_check`` 仍用「阶段还在 PLANNING」判定 shopping_summary
  不许交卷（本轮未规划/精挑），那是精确事实判定，不是白名单。

效率约束改由预算兜：复用轮小预算（``REUSE_RETRIEVAL_BUDGET``）、全树检索预算、fork 预算、
token 预算，见 ``hooks/tool_gates.py``——预算永不为 0，模型总有一条能走的路。

仅主 loop（depth 0）维护阶段。子 Agent（depth ≥ 1）由深度闸管权限，与阶段机无关。
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from enum import Enum
from typing import Any

from app.harness.state import GuardState

logger = logging.getLogger("shoppingx.harness.phase")


class Phase(Enum):
    PLANNING = "planning"
    SEARCHING = "searching"
    COMPARING = "comparing"
    CONCLUDING = "concluding"


TRANSITION_SIGNALS: dict[Phase, dict[str, Phase]] = {
    Phase.PLANNING: {"planner_output_ready": Phase.SEARCHING},
    Phase.SEARCHING: {"candidates_available": Phase.COMPARING},
    Phase.COMPARING: {"picks_ready": Phase.CONCLUDING},
}


class PhaseStateMachine:
    """单会话的阶段状态机实例。

    每个主 loop Agent 实例持有一个。子 Agent 不创建自己的状态机（由深度闸管权限）。
    """

    def __init__(self, initial: Phase = Phase.PLANNING) -> None:
        self._phase = initial
        self._no_progress_rounds = 0
        self._regressed_this_round = False

    @property
    def phase(self) -> Phase:
        return self._phase

    def try_transition(self, signal: str) -> bool:
        """尝试按信号推进阶段。成功返回 True 并重置无进展计数。

        本轮发生过回退（:meth:`regress`）则一律拒绝——「回退后同轮不得再前进」是
        状态机自己的不变量，不靠钩子排序或 context 字段清零的手抄纪律维持。
        """
        if self._regressed_this_round:
            logger.info("Phase transition blocked（本轮已回退）: signal=%s", signal)
            return False
        conds = TRANSITION_SIGNALS.get(self._phase, {})
        target = conds.get(signal)
        if target is None:
            return False
        old = self._phase
        self._phase = target
        self._no_progress_rounds = 0
        logger.info("Phase transition: %s → %s (signal=%s)", old.value, target.value, signal)
        return True

    def set_phase(self, phase: Phase) -> None:
        """强制设定阶段——只用于**前进**类特殊场景（reuse 跳过检索 / 强制收尾）。

        往回退一律走 :meth:`regress`：回退不是一次赋值而是一个事务（同轮闭锁 + 进展计数
        清零 + 通告重武装 + 直搜解锁），散装 set_phase 会漏掉状态回收、回退被静默吞掉。
        """
        old = self._phase
        self._phase = phase
        self._no_progress_rounds = 0
        logger.warning("Phase forced: %s → %s", old.value, phase.value)

    def regress(self, to: Phase, *, reason: str, context: dict[str, Any] | None = None) -> None:
        """回退事务：状态回收是回退语义的一部分，由本体一次做完，钩子不得手抄。

        做四件事（曾经散在钩子里各自手抄、漏一处即静默失效——「同轮 40 号钩子吞回退」）：

        1. 阶段设回 ``to``，并**闭锁本轮前进**：同一次 post_reflect 里排在后面的转移钩子
           凭旧计数把阶段推回去 = 回退当场被吞（下一轮由 :meth:`begin_round` 解锁）。
        2. 进展计数清零：``context["total_candidates"]`` 就地清零（同轮后续钩子读到 0）+
           置 ``reset_fresh_candidates``（middleware 把跨轮累计一并清掉）——已判定「这池子
           不够用」，它就不再是「本轮已搜到货」的进展信号。
        3. 重新武装「检索收线」通告：回退即新一轮检索，搜到货后仍需当场指路。
        4. 解锁一次直搜（postfork 棘轮）：初搜若走了并行 fork，search_authority 闸会把
           直调 item_search 拦死，回退指路就成了空话。

        调用方专属的收尾（refine_backfill 的 mode=augment 触发闩、rollback 的注入指令）
        留在调用方——那些是各闸的触发语义，不是回退语义。
        """
        old = self._phase
        self._phase = to
        self._no_progress_rounds = 0
        self._regressed_this_round = True
        logger.warning("Phase regress: %s → %s（%s）", old.value, to.value, reason)
        if context is None:
            return
        context["total_candidates"] = 0
        context["reset_fresh_candidates"] = True
        guard = context.get("_guard")
        if isinstance(guard, GuardState):
            guard.notified_transitions.discard("search_close")
            guard.postfork_search_grants += 1

    def begin_round(self) -> None:
        """新一轮 post_reflect 的边界（middleware 每轮开头 tick）：解除回退闭锁。"""
        self._regressed_this_round = False

    def record_no_progress(self) -> int:
        """记录一轮无进展，返回连续无进展轮数。"""
        self._no_progress_rounds += 1
        return self._no_progress_rounds

    def reset_no_progress(self) -> None:
        self._no_progress_rounds = 0

    @property
    def no_progress_rounds(self) -> int:
        return self._no_progress_rounds

    def reset(self) -> None:
        self._phase = Phase.PLANNING
        self._no_progress_rounds = 0
        self._regressed_this_round = False


# ContextVar：主 loop 开始时 set，子 loop 通过 copy_context 自动继承快照（但子不应读写）。
_current_phase_machine: ContextVar[PhaseStateMachine | None] = ContextVar(
    "current_phase_machine", default=None
)


def get_phase_machine() -> PhaseStateMachine | None:
    return _current_phase_machine.get()


def set_phase_machine(machine: PhaseStateMachine) -> None:
    _current_phase_machine.set(machine)


def reset_phase_machine() -> None:
    _current_phase_machine.set(None)
