"""HarnessSession：一次 AgentLoop 的控制面状态，被 Agent 适配器与 Tool 适配器**共享**。

AgentScope 把模型钩子与工具钩子拆成 ``MiddlewareBase`` 与 ``ToolMiddlewareBase`` 两个类，状态必须
外提到这里——否则 pre_tool_call 攒的断言流不到 post_reflect 的 ``assertion_handler`` 手里。
三条接力通道：inject（Hook 注入 → 下一次 pre_think）/ assertions（工具边界 → 本轮 post_reflect）
/ GuardState（各闸的 per-loop 计数）。``collect_call_signals`` 把一次工具调用的真实返回折成阶段信号
并更新 session（不从全局状态反推）。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from agentscope.message import Msg, TextBlock

from app.agent.token_budget import tree_snapshot
from app.harness.hooks.drift import DriftState
from app.harness.signals import _SEARCH_TOOLS, _as_opt_int, _count_candidates, _count_picks
from app.harness.state import GuardState
from app.tools._diagnostics import consume_diagnostics

logger = logging.getLogger("shoppingx.harness.session")


class HarnessSession:
    """一次 AgentLoop 的控制面状态，被 Agent 适配器与 Tool 适配器**共享**。

    AgentScope 把模型钩子与工具钩子拆成了 ``MiddlewareBase`` 与 ``ToolMiddlewareBase`` 两个类，
    于是状态必须外提到这里——否则 pre_tool_call 攒的断言就流不到 post_reflect 的
    ``assertion_handler`` 手里，三条接力通道（inject / assertions / retry_nudge）一条都不能少。
    """

    def __init__(
        self,
        *,
        original_query: str = "",
        image_paths: Sequence[str] = (),
        guard: GuardState | None = None,
    ) -> None:
        self.original_query = original_query
        # 本轮参考图（文件名）。开局预置要按它决定看不看图，见 harness.prefill。
        self.image_paths: tuple[str, ...] = tuple(image_paths)
        # 开局预置只做一次：on_reply 每轮都进，但预置是「这次任务开始」的动作。
        self.prefilled = False
        self.guard = guard if guard is not None else GuardState()
        self.round_counter = 0
        self.called_tools: set[str] = set()
        self.drift_state = DriftState()
        self.recent_actions: list[str] = []
        # 接力通道 1：Hook 产出的注入消息 → 下一次 pre_think 消费
        self.pending_inject: list[dict[str, str]] = []
        # 接力通道 2：pre/post_tool_call 攒的断言失败 → 本轮 post_reflect 消费
        self.pending_assertions: list[dict[str, Any]] = []
        # 接力通道 3：GuardState（上面的 self.guard）
        self.last_total_tokens = 0
        self.planner_done = False
        self.picker_attempted = False
        self.last_picks = 0
        self.last_must_hits: int | None = None
        self.last_oncat: int | None = None
        self.last_offcat: int | None = None
        self.last_excluded: int | None = None
        self.last_over_budget: int | None = None
        self.fresh_candidates = 0
        # 检索类工具成功返回后武装；下一次 pre_think 自动比价精挑（harness.autopick）
        self.autopick_armed = False
        # post_reflect 判定「该催收尾」后挂在这里，由 on_reply 在 ReplyEnd 时兑现
        self.retry_nudge: str | None = None

    def base_context(self) -> dict[str, Any]:
        return {
            "original_query": self.original_query,
            "round_number": self.round_counter,
            "called_tools": set(self.called_tools),
            "_drift_state": self.drift_state,
            "_guard": self.guard,
        }

    def recent_actions_summary(self) -> str:
        return "; ".join(self.recent_actions[-9:]) if self.recent_actions else ""

    def collect(self, ctx: dict[str, Any]) -> None:
        """把 Hook 产出的注入与断言收进接力通道。"""
        inject = ctx.get("inject_messages")
        if inject:
            self.pending_inject.extend(inject)
        failed = ctx.get("assertions_failed")
        if failed:
            self.pending_assertions.extend(failed)

    def consume_inject(self) -> list[Msg]:
        if not self.pending_inject:
            return []
        msgs = [
            Msg(name="system", role="system", content=[TextBlock(type="text", text=m["content"])])
            for m in self.pending_inject
            if m.get("content")
        ]
        self.pending_inject.clear()
        return msgs

    def track_token_delta(self) -> None:
        """本次模型调用的 token 增量喂给漂移检测（信号 4：成本失控）。"""
        snap = tree_snapshot()
        if not snap:
            return
        total = int(snap.get("input_tokens", 0)) + int(snap.get("output_tokens", 0))
        delta = total - self.last_total_tokens
        self.last_total_tokens = total
        if delta > 0:
            self.drift_state.token_history.append(delta)


def collect_call_signals(s: HarnessSession, tool_name: str, result_text: str) -> dict[str, Any]:
    """从工具的**真实返回**里数阶段信号（不从全局状态反推——见 ``signals`` 的两条原则）。"""
    call_candidates = 0
    call_picks = 0
    call_must_hits: int | None = None
    call_oncat: int | None = None
    call_offcat: int | None = None
    call_excluded: int | None = None
    call_over_budget: int | None = None
    call_filtered_out: list[Any] = []
    call_filtered_price_only = False
    if tool_name == "planner":
        s.planner_done = True
    elif tool_name == "item_picker":
        s.picker_attempted = True
        # 诊断走结构化侧信道（picker 返回前登记），不从模型可见文本里正则抠——
        # 文本截断 / 格式变化都伤不到信号。
        diag = consume_diagnostics("item_picker")
        if diag is not None:
            s.last_picks = call_picks = _as_opt_int(diag.get("picks")) or 0
            s.last_must_hits = call_must_hits = _as_opt_int(diag.get("must_have_hits"))
            s.last_oncat = call_oncat = _as_opt_int(diag.get("oncat_count"))
            s.last_offcat = call_offcat = _as_opt_int(diag.get("offcat_count"))
            s.last_excluded = call_excluded = _as_opt_int(diag.get("excluded_count"))
            s.last_over_budget = call_over_budget = _as_opt_int(diag.get("over_budget_count"))
        else:
            # 侧信道意外空：picks 退回文本兜底，三个诊断字段退化为 None =「不适用」，
            # 补搜闸 fail-open 不误触发（失效方向中性）。
            logger.warning("item_picker 诊断侧信道为空，picks 退回文本解析兜底")
            s.last_picks = call_picks = _count_picks(result_text)
            s.last_must_hits = s.last_oncat = s.last_offcat = None
            s.last_excluded = s.last_over_budget = None
    elif tool_name in _SEARCH_TOOLS:
        call_candidates = _count_candidates(result_text)
        s.fresh_candidates += call_candidates
        if tool_name == "item_search":
            # 探测召回的结论（「库里有但被硬条件挡了」）同样走侧信道，不从模型可见文本正则抠。
            # 只有真探测到东西时工具才登记，故 None＝这次没被挡住任何货。
            diag = consume_diagnostics("item_search")
            if diag is not None:
                blocked = diag.get("filtered_out")
                call_filtered_out = list(blocked) if isinstance(blocked, list) else []
                call_filtered_price_only = bool(diag.get("filtered_price_only"))
    return {
        "call_candidates": call_candidates,
        "call_filtered_out": call_filtered_out,
        "call_filtered_price_only": call_filtered_price_only,
        "call_picks": call_picks,
        "call_must_hits": call_must_hits,
        "call_oncat": call_oncat,
        "call_offcat": call_offcat,
        "call_excluded": call_excluded,
        "call_over_budget": call_over_budget,
    }
