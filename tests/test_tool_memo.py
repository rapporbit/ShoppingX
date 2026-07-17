"""同参数重复调用回放（tool_memo）+ 阶段转移通告补边（picks_ready / reuse 跳转 / tasks 提示）。

对应延迟审计第三轮的三处残留浪费：静默转移让模型撞阶段哨兵白耗一轮、同参数重复检索真执行、
无比价诉求的轮次照跑 price_compare / shipping_calc。
"""

from pathlib import Path

import pytest

from app.api.context import set_retrieval_mode, set_session_tasks
from app.harness.hooks.phase_transition import append_transition_notice
from app.harness.hooks.tool_memo import record_tool_result, replay_duplicate_call
from app.harness.middleware import HookRejectSignal
from app.harness.phase_machine import (
    Phase,
    PhaseStateMachine,
    reset_phase_machine,
    set_phase_machine,
)
from app.harness.state import GuardState
from app.utils.thread_ctx import thread_scope

pytestmark = pytest.mark.anyio

_ARGS = {"platform": "amazon", "query": "camping cookware"}
_RESULT = '{"candidates": [{"item_id": "B01"}]}'


async def _record(guard: GuardState, name: str = "item_search", args: dict | None = None) -> None:
    ctx = {"_guard": guard, "tool_name": name, "tool_args": args or dict(_ARGS)}
    ctx["tool_result"] = _RESULT
    await record_tool_result(ctx)


class TestToolMemo:
    async def test_same_args_replayed_without_execution(self) -> None:
        """已成功执行过的 (tool, args) 再调 → 回放缓存结果 + 提示，不真执行。"""
        guard = GuardState()
        await _record(guard)
        with pytest.raises(HookRejectSignal) as ei:
            await replay_duplicate_call(
                {"_guard": guard, "tool_name": "item_search", "tool_args": dict(_ARGS)}
            )
        assert _RESULT in ei.value.reason  # 回放的是当时的完整结果
        assert "复用" in ei.value.reason  # 且明说未重新执行、别再原样重试
        assert ei.value.raw

    async def test_arg_order_insensitive(self) -> None:
        """指纹按 sort_keys 序列化——参数顺序不同不该骗过回放。"""
        guard = GuardState()
        await _record(guard, args={"query": "camping cookware", "platform": "amazon"})
        with pytest.raises(HookRejectSignal):
            await replay_duplicate_call(
                {"_guard": guard, "tool_name": "item_search", "tool_args": dict(_ARGS)}
            )

    async def test_different_args_not_replayed(self) -> None:
        guard = GuardState()
        await _record(guard)
        assert (
            await replay_duplicate_call(
                {"_guard": guard, "tool_name": "item_search", "tool_args": {"query": "别的"}}
            )
            is None
        )

    async def test_non_idempotent_tool_not_cached(self) -> None:
        """planner / item_picker 有副作用或依赖轮内可变状态，不进回放缓存。"""
        guard = GuardState()
        await _record(guard, name="planner", args={"intent": "买锅"})
        assert (
            await replay_duplicate_call(
                {"_guard": guard, "tool_name": "planner", "tool_args": {"intent": "买锅"}}
            )
            is None
        )

    async def test_replays_feed_loop_detector(self) -> None:
        """回放不走 post_tool_call，LoopDetector 必须在回放路径手动喂——刷到阈值要给打转提示。"""
        guard = GuardState()
        await _record(guard)
        reasons: list[str] = []
        for _ in range(guard.loop_threshold):
            with pytest.raises(HookRejectSignal) as ei:
                await replay_duplicate_call(
                    {"_guard": guard, "tool_name": "item_search", "tool_args": dict(_ARGS)}
                )
            reasons.append(ei.value.reason)
        assert "重复调用" in reasons[-1]  # 第 threshold 次回放附上换思路 / 收尾提示


class TestTransitionNotices:
    """收线通告缀在**触发转移的工具结果**尾部（post_tool_call）——post_reflect 的 inject
    通道晚一轮，模型在读到通告前就已决定下一步（perf-audit-r3 实测连发 item_search 撞哨兵）。"""

    def teardown_method(self) -> None:
        reset_phase_machine()

    async def _notice(self, machine: PhaseStateMachine, ctx: dict) -> str:
        set_phase_machine(machine)
        ctx.setdefault("_guard", GuardState())
        ctx.setdefault("tool_result", "{}")
        out = await append_transition_notice(ctx)
        return (out or ctx)["tool_result"]

    async def test_search_close_rides_tool_result(self) -> None:
        """首个非空检索结果尾部当场缀「检索收线」；同 loop 第二次不重复。"""
        machine = PhaseStateMachine(initial=Phase.SEARCHING)
        guard = GuardState()
        ctx = {"_guard": guard, "tool_name": "item_search", "call_candidates": 10}
        result = await self._notice(machine, ctx)
        assert "[阶段推进]" in result and "item_search" in result
        ctx2 = {"_guard": guard, "tool_name": "item_search", "call_candidates": 5}
        assert "[阶段推进]" not in await self._notice(machine, ctx2)

    async def test_picks_close_rides_picker_result(self) -> None:
        machine = PhaseStateMachine(initial=Phase.COMPARING)
        ctx = {"tool_name": "item_picker", "call_picks": 5}
        result = await self._notice(machine, ctx)
        assert "price_compare" in result  # 点名别再调
        assert "shopping_summary" in result  # 并指路收尾

    async def test_reuse_skip_rides_planner_result(self, tmp_path: Path) -> None:
        with thread_scope("t-notice-reuse", tmp_path):
            set_retrieval_mode("reuse")
            machine = PhaseStateMachine()  # PLANNING：planner 刚返回、转移尚未发生
            ctx = {"tool_name": "planner"}
            result = await self._notice(machine, ctx)
            assert "复用上一轮候选" in result and "item_picker" in result

    async def test_thin_reuse_picks_redirects_to_research(self, tmp_path: Path) -> None:
        """薄复用（<3 件）不发「去收尾」，改指路「重新检索」——refine_backfill 马上退阶段。"""
        with thread_scope("t-notice-thin", tmp_path):
            set_retrieval_mode("reuse")
            machine = PhaseStateMachine(initial=Phase.COMPARING)
            ctx = {"tool_name": "item_picker", "call_picks": 1}
            result = await self._notice(machine, ctx)
            assert "[阶段回退]" in result and "重新检索" in result
            assert "shopping_summary" not in result

    async def test_tasks_hint_when_no_price_demand(self, tmp_path: Path) -> None:
        """planner 判定无比价 / 到手价诉求 → 收线通告附带「无需 price_compare」动机提示。"""
        with thread_scope("t-tasks-rec", tmp_path):
            set_session_tasks(["recommend"])
            machine = PhaseStateMachine(initial=Phase.SEARCHING)
            ctx = {"tool_name": "item_search", "call_candidates": 8}
            assert "无需 price_compare" in await self._notice(machine, ctx)

    async def test_no_tasks_hint_when_price_compare_requested(self, tmp_path: Path) -> None:
        """用户真要比价（tasks 含 price_compare）→ 绝不提示跳过。"""
        with thread_scope("t-tasks-pc", tmp_path):
            set_session_tasks(["recommend", "price_compare"])
            machine = PhaseStateMachine(initial=Phase.SEARCHING)
            ctx = {"tool_name": "item_search", "call_candidates": 8}
            assert "无需 price_compare" not in await self._notice(machine, ctx)

    async def test_no_tasks_hint_when_tasks_unknown(self, tmp_path: Path) -> None:
        """planner 没落 tasks（判不出 / 老会话）→ 安全侧不提示，宁可多调不误伤。"""
        with thread_scope("t-tasks-none", tmp_path):
            machine = PhaseStateMachine(initial=Phase.SEARCHING)
            ctx = {"tool_name": "item_search", "call_candidates": 8}
            assert "无需 price_compare" not in await self._notice(machine, ctx)
