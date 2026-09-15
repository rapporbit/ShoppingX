"""阶段转移通告补边（picks_ready / tasks 提示）。

对应延迟审计第三轮的残留浪费：静默转移让模型撞阶段哨兵白耗一轮、无比价诉求的轮次照跑
price_compare / shipping_calc。（同文件曾有 tool_memo 回放测试，随该 hook 2026-09-15 一起删。）
"""

from pathlib import Path

import pytest

from app.api.context import set_session_tasks
from app.harness.hooks.progress import append_transition_notice
from app.harness.phase_machine import (
    Phase,
    PhaseStateMachine,
    reset_phase_machine,
    set_phase_machine,
)
from app.harness.state import GuardState
from app.utils.thread_ctx import thread_scope

pytestmark = pytest.mark.anyio



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

    async def test_search_close_points_to_autopick(self, tmp_path: Path) -> None:
        """round3 刀 2：自动比价精挑开着时，收线通告指路「等系统结果、直接收尾」，不再让模型
        自己走 price_compare / item_picker，「无需 price_compare」动机提示也随之失去意义。"""
        with thread_scope("t-tasks-auto", tmp_path):
            set_session_tasks(["recommend"])
            machine = PhaseStateMachine(initial=Phase.SEARCHING)
            ctx = {"tool_name": "item_search", "call_candidates": 8}
            out = await self._notice(machine, ctx)
            assert "自动完成比价" in out and "无需 price_compare" not in out

    async def test_tasks_hint_when_no_price_demand(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """planner 判定无比价 / 到手价诉求 → 收线通告附带「无需 price_compare」动机提示。
        （关掉自动比价精挑时的口径；开着时见 test_search_close_points_to_autopick。）"""
        monkeypatch.setenv("AUTOPICK", "0")
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
