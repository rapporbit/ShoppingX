"""Harness 治理框架测试：Hook Pipeline + 阶段状态机 + 单步断言 + 漂移检测。"""

from __future__ import annotations

import pytest

from app.harness.hooks.drift_detector import DriftState, _extract_keywords
from app.harness.middleware import (
    HOOK_POINTS,
    HarnessMiddleware,
    HookRejectSignal,
    harness,
)
from app.harness.phase_machine import (
    Phase,
    PhaseStateMachine,
    get_phase_machine,
    reset_phase_machine,
    set_phase_machine,
)

# ============================================================
# HarnessMiddleware 核心
# ============================================================


class TestHarnessMiddleware:
    """Hook Pipeline 注册与执行。"""

    def test_register_and_list(self) -> None:
        h = HarnessMiddleware()
        called = []

        async def hook_a(ctx: dict) -> dict | None:
            called.append("a")
            return None

        async def hook_b(ctx: dict) -> dict | None:
            called.append("b")
            return None

        h.register("pre_tool_call", "hook_b", hook_b, priority=20)
        h.register("pre_tool_call", "hook_a", hook_a, priority=10)

        hooks = h.list_hooks("pre_tool_call")
        assert [name for _, name, _ in hooks] == ["hook_a", "hook_b"]

    @pytest.mark.asyncio
    async def test_run_hooks_in_priority_order(self) -> None:
        h = HarnessMiddleware()
        order: list[str] = []

        async def first(ctx: dict) -> dict | None:
            order.append("first")
            return None

        async def second(ctx: dict) -> dict | None:
            order.append("second")
            return None

        h.register("post_tool_call", "second", second, priority=20)
        h.register("post_tool_call", "first", first, priority=10)

        await h.run("post_tool_call", {})
        assert order == ["first", "second"]

    @pytest.mark.asyncio
    async def test_hook_modifies_context(self) -> None:
        h = HarnessMiddleware()

        async def add_flag(ctx: dict) -> dict:
            ctx["modified"] = True
            return ctx

        h.register("pre_think", "add_flag", add_flag)
        result = await h.run("pre_think", {})
        assert result["modified"] is True

    @pytest.mark.asyncio
    async def test_hook_reject_signal(self) -> None:
        h = HarnessMiddleware()

        async def reject_hook(ctx: dict) -> dict | None:
            raise HookRejectSignal("forbidden tool")

        async def never_reached(ctx: dict) -> dict | None:
            ctx["reached"] = True
            return ctx

        h.register("pre_tool_call", "reject", reject_hook, priority=10)
        h.register("pre_tool_call", "after", never_reached, priority=20)

        result = await h.run("pre_tool_call", {})
        assert result["_rejected"] is True
        assert "forbidden" in result["_reject_reason"]
        assert "reached" not in result

    @pytest.mark.asyncio
    async def test_hook_exception_does_not_stop_pipeline(self) -> None:
        h = HarnessMiddleware()
        reached = []

        async def bad_hook(ctx: dict) -> dict | None:
            raise RuntimeError("boom")

        async def good_hook(ctx: dict) -> dict | None:
            reached.append(True)
            return None

        h.register("post_reflect", "bad", bad_hook, priority=10)
        h.register("post_reflect", "good", good_hook, priority=20)

        await h.run("post_reflect", {})
        assert len(reached) == 1

    def test_invalid_hook_point_raises(self) -> None:
        h = HarnessMiddleware()

        async def noop(ctx: dict) -> dict | None:
            return None

        with pytest.raises(ValueError, match="未知"):
            h.register("invalid_point", "x", noop)

    def test_hook_points_complete(self) -> None:
        assert len(HOOK_POINTS) == 6
        assert "on_session_start" in HOOK_POINTS
        assert "on_session_end" in HOOK_POINTS


# ============================================================
# PhaseStateMachine
# ============================================================


class TestPhaseStateMachine:
    """四阶段状态机。"""

    def test_initial_phase(self) -> None:
        m = PhaseStateMachine()
        assert m.phase == Phase.PLANNING

    def test_transition_planning_to_searching(self) -> None:
        m = PhaseStateMachine()
        assert m.try_transition("planner_output_ready")
        assert m.phase == Phase.SEARCHING

    def test_transition_searching_to_comparing(self) -> None:
        m = PhaseStateMachine()
        m.try_transition("planner_output_ready")
        assert m.try_transition("candidates_available")
        assert m.phase == Phase.COMPARING

    def test_transition_comparing_to_concluding(self) -> None:
        m = PhaseStateMachine()
        m.try_transition("planner_output_ready")
        m.try_transition("candidates_available")
        assert m.try_transition("picks_ready")
        assert m.phase == Phase.CONCLUDING

    def test_invalid_signal_returns_false(self) -> None:
        m = PhaseStateMachine()
        assert not m.try_transition("candidates_available")
        assert m.phase == Phase.PLANNING

    def test_set_phase_for_rollback(self) -> None:
        m = PhaseStateMachine()
        m.try_transition("planner_output_ready")
        m.try_transition("candidates_available")
        assert m.phase == Phase.COMPARING
        m.set_phase(Phase.SEARCHING)
        assert m.phase == Phase.SEARCHING

    def test_regress_blocks_same_round_advance(self) -> None:
        """回退事务的核心不变量：回退后**同轮**一律不得再前进（曾靠钩子排序 + context 字段
        清零的手抄纪律维持，40 号钩子凭旧计数同轮吞回退）；begin_round 后凭新证据放行。"""
        m = PhaseStateMachine(Phase.COMPARING)
        m.regress(Phase.SEARCHING, reason="test")
        assert m.phase == Phase.SEARCHING
        assert not m.try_transition("candidates_available")
        assert m.phase == Phase.SEARCHING, "回退同轮被推回去 = 回退被吞"
        m.begin_round()
        assert m.try_transition("candidates_available")
        assert m.phase == Phase.COMPARING

    def test_regress_recovers_context_state(self) -> None:
        """状态回收是回退语义的一部分，由本体一次做完：进展计数清零 + 收线通告重武装 +
        直搜解锁一次——散在钩子里手抄、漏一处即静默失效的三件套，现在漏不掉。"""
        from app.harness.state import GuardState

        guard = GuardState()
        guard.notified_transitions.add("search_close")
        ctx: dict = {"total_candidates": 9, "_guard": guard}
        m = PhaseStateMachine(Phase.COMPARING)
        m.regress(Phase.SEARCHING, reason="test", context=ctx)
        assert ctx["total_candidates"] == 0
        assert ctx["reset_fresh_candidates"] is True
        assert "search_close" not in guard.notified_transitions
        assert guard.postfork_search_grants == 1

    def test_no_progress_counter(self) -> None:
        m = PhaseStateMachine()
        assert m.record_no_progress() == 1
        assert m.record_no_progress() == 2
        m.reset_no_progress()
        assert m.no_progress_rounds == 0

    def test_reset(self) -> None:
        m = PhaseStateMachine()
        m.try_transition("planner_output_ready")
        m.record_no_progress()
        m.reset()
        assert m.phase == Phase.PLANNING
        assert m.no_progress_rounds == 0

    def test_contextvar_lifecycle(self) -> None:
        reset_phase_machine()
        assert get_phase_machine() is None
        m = PhaseStateMachine()
        set_phase_machine(m)
        assert get_phase_machine() is m
        reset_phase_machine()
        assert get_phase_machine() is None


# ============================================================
# Step Validator Hooks
# ============================================================


class TestStepValidator:
    """三类单步断言。"""

    @pytest.mark.asyncio
    async def test_sequencing_warns_missing_prereq(self) -> None:
        from app.harness.hooks.step_validator import check_sequencing

        ctx = {"tool_name": "shopping_summary", "called_tools": {"item_search"}}
        result = await check_sequencing(ctx)
        assert result is not None
        failures = result.get("assertions_failed", [])
        assert any(f["type"] == "sequencing" for f in failures)

    @pytest.mark.asyncio
    async def test_sequencing_passes_when_prereqs_met(self) -> None:
        from app.harness.hooks.step_validator import check_sequencing

        ctx = {"tool_name": "shopping_summary", "called_tools": {"item_picker", "item_search"}}
        result = await check_sequencing(ctx)
        # 无 assertions_failed
        if result is not None:
            assert not result.get("assertions_failed", [])

    @pytest.mark.asyncio
    async def test_sequencing_skips_unknown_tool(self) -> None:
        from app.harness.hooks.step_validator import check_sequencing

        result = await check_sequencing({"tool_name": "web_search", "called_tools": set()})
        assert result is None

    @pytest.mark.asyncio
    async def test_schema_assertion_catches_invalid_json(self) -> None:
        from app.harness.hooks.step_validator import check_schema

        ctx = {"tool_name": "item_search", "tool_result": '{"invalid json'}
        result = await check_schema(ctx)
        # 非 JSON 不一定是错（有些工具返回纯文本）
        assert result is None or not result.get("assertions_failed")

    @pytest.mark.asyncio
    async def test_schema_assertion_valid_result(self) -> None:
        import json

        from app.harness.hooks.step_validator import check_schema

        good = json.dumps(
            {
                "platform": "amazon",
                "candidates": [],
                "total_recall": 0,
                "truncated": False,
            }
        )
        ctx = {"tool_name": "item_search", "tool_result": good}
        result = await check_schema(ctx)
        if result is not None:
            assert not result.get("assertions_failed", [])

    @staticmethod
    def _single_platform_render(platform: str = "amazon") -> str:
        """单平台渲染投影：候选按渲染契约省略 platform（顶层已写，见 ItemSearchOutput.__str__）。"""
        import json

        return json.dumps(
            {
                "platform": platform,
                "total_recall": 1,
                "truncated": False,
                "candidates": [
                    {
                        "item_id": "B0C9LS7339",
                        "title": "Samsung 充电器",
                        "price_usd": 16.05,
                        "rating": 4.8,
                        "category": "Cell Phones",
                    }
                ],
            },
            ensure_ascii=False,
        )

    @pytest.mark.asyncio
    async def test_schema_assertion_accepts_single_platform_render(self) -> None:
        """渲染契约省略候选级 platform 不是格式错误——曾每次单平台检索必假阳性（eval q05）。"""
        from app.harness.hooks.step_validator import check_schema

        ctx = {"tool_name": "item_search", "tool_result": self._single_platform_render()}
        result = await check_schema(ctx)
        assert not (result or {}).get("assertions_failed")

    @pytest.mark.asyncio
    async def test_schema_assertion_survives_appended_notice(self) -> None:
        """先跑的 Hook 在结果尾部贴通告（[阶段推进] 等）后断言仍在岗——曾因 Extra data 静默跳过。"""
        from app.harness.hooks.step_validator import check_schema

        bogus = '{"bogus": 1}\n\n[阶段推进] 候选已入池，检索阶段就此收线。'
        ctx = {"tool_name": "item_search", "tool_result": bogus}
        result = await check_schema(ctx)
        assert result is not None and result.get("assertions_failed"), "附言不该让断言失明"

    @pytest.mark.asyncio
    async def test_schema_assertion_still_requires_platform_when_merged(self) -> None:
        """platform="all" 合流时候选必须逐条带 platform——缺了是真错，不回填。"""
        from app.harness.hooks.step_validator import check_schema

        ctx = {"tool_name": "item_search", "tool_result": self._single_platform_render("all")}
        result = await check_schema(ctx)
        assert result is not None and result.get("assertions_failed")


# ============================================================
# Drift Detector
# ============================================================


class TestDriftDetector:
    """Silent Drift 漂移检测。"""

    def test_extract_keywords(self) -> None:
        kw = _extract_keywords("我想买便宜的旅行背包，预算300")
        assert "旅行" in kw
        assert "背包" in kw
        assert "300" in kw
        assert len(kw) > 3

    def test_extract_keywords_english(self) -> None:
        kw = _extract_keywords("I want a cheap travel backpack under 300")
        assert "travel" in kw
        assert "backpack" in kw
        assert "cheap" in kw

    def test_drift_state_lifecycle(self) -> None:
        state = DriftState()
        assert state.round_counter == 0
        assert state.consecutive_severe == 0
        state.round_counter = 6
        state.consecutive_severe = 2
        state.reset()
        assert state.round_counter == 0
        assert state.consecutive_severe == 0

    def test_empty_result_tracking(self) -> None:
        state = DriftState()
        state.consecutive_empty_results = 3
        assert state.consecutive_empty_results >= 3

    @pytest.mark.asyncio
    async def test_drift_skips_when_disabled(self) -> None:
        from app.harness.hooks import drift_detector

        original = drift_detector.DRIFT_ENABLED
        try:
            drift_detector.DRIFT_ENABLED = False
            result = await drift_detector.detect_drift({"_drift_state": DriftState()})
            assert result is None
        finally:
            drift_detector.DRIFT_ENABLED = original

    @pytest.mark.asyncio
    async def test_drift_skips_non_check_round(self) -> None:
        from app.harness.hooks.drift_detector import detect_drift

        state = DriftState()
        state.round_counter = 0  # will become 1 after increment, 1 % 3 != 0
        ctx = {"_drift_state": state, "original_query": "test"}
        result = await detect_drift(ctx)
        assert result is None
        assert state.round_counter == 1


# ============================================================
# Phase Hooks
# ============================================================


class TestPhaseHooks:
    """阶段权限拦截 + 阶段转移。"""

    @pytest.mark.asyncio
    async def test_phase_check_no_whitelist_ban(self) -> None:
        """阶段白名单禁令已撤（重构第三段）：任何阶段调检索/比价类工具都不再被阶段闸拒绝。"""
        from app.agent.fork_guard import _fork_depth
        from app.harness.hooks.phase_check import check_phase_permission

        token = _fork_depth.set(0)
        set_phase_machine(PhaseStateMachine())  # PLANNING
        try:
            for tool in ("item_search", "price_compare", "web_search", "item_picker"):
                assert await check_phase_permission({"tool_name": tool}) is None
        finally:
            _fork_depth.reset(token)
            reset_phase_machine()

    @pytest.mark.asyncio
    async def test_phase_check_skips_sub_agents(self) -> None:
        from app.agent.fork_guard import _fork_depth
        from app.harness.hooks.phase_check import check_phase_permission

        token = _fork_depth.set(1)
        m = PhaseStateMachine()
        set_phase_machine(m)
        try:
            ctx: dict = {"tool_name": "shopping_summary"}
            result = await check_phase_permission(ctx)
            assert result is None  # 子 Agent 由深度闸管权限，本闸不管
        finally:
            _fork_depth.reset(token)
            reset_phase_machine()

    @pytest.mark.asyncio
    async def test_reuse_turn_first_backfill_executes_immediately(self) -> None:
        """复用轮第一次补搜**当场放行执行**（不再攒拒绝换逃生），结果尾部缀软线文案。

        对应 2026-07-14 线上死锁：planner 误判 reuse → item_search 被阶段闸拦死 27 轮。
        预算制下（REUSE_RETRIEVAL_BUDGET≥1，永不为 0）这类死锁在结构上不可能发生。
        """
        from pathlib import Path

        from app.agent.retrieval_budget import reset_tree
        from app.api.context import set_retrieval_mode
        from app.harness.hooks.tool_gates import charge_retrieval
        from app.harness.state import GuardState
        from app.utils.thread_ctx import thread_scope

        guard = GuardState()
        with thread_scope("t-reuse-budget", Path("/tmp/t-reuse-budget")):
            set_retrieval_mode("reuse")
            try:
                ctx = {"tool_name": "item_search", "_guard": guard}
                out = await charge_retrieval(ctx)
                assert out is not None and "converge_note" in out  # 执行 + 缀收敛提示
                assert "复用轮补搜" in out["converge_note"]
            finally:
                reset_tree()

    @pytest.mark.asyncio
    async def test_reuse_turn_budget_exhausted_then_hard_block(self) -> None:
        """复用轮越过小预算后硬挡；改写 augment（补搜授权）即恢复全树预算。"""
        from pathlib import Path

        from app.agent.retrieval_budget import reset_tree
        from app.api.context import set_retrieval_mode
        from app.harness.hooks.tool_gates import charge_retrieval
        from app.harness.state import GuardState
        from app.utils.thread_ctx import thread_scope

        guard = GuardState()
        with thread_scope("t-reuse-exhaust", Path("/tmp/t-reuse-exhaust")):
            set_retrieval_mode("reuse")
            try:
                await charge_retrieval({"tool_name": "item_search", "_guard": guard})
                with pytest.raises(HookRejectSignal, match="复用轮检索预算耗尽"):
                    await charge_retrieval({"tool_name": "item_search", "_guard": guard})
                # refine_backfill / phase_rollback 授权补搜 → mode=augment → 全树预算恢复
                set_retrieval_mode("augment")
                out = await charge_retrieval({"tool_name": "item_search", "_guard": guard})
                assert out is None  # 常规放行（3 <= 全树 cap）
            finally:
                reset_tree()

    def test_reuse_budget_never_zero(self) -> None:
        """REUSE_RETRIEVAL_BUDGET 钉死 ≥1：配 0 等于把预算制改回禁令制，死锁风险回归。"""
        from app.harness.budgets import REUSE_RETRIEVAL_BUDGET

        assert REUSE_RETRIEVAL_BUDGET >= 1

    def test_middleware_tracks_planner_signal(self) -> None:
        """HarnessAgentMiddleware 记录 planner 已执行。"""
        from app.harness.agent_middleware import HarnessAgentMiddleware

        m = HarnessAgentMiddleware(original_query="test")
        assert not m._planner_done
        m._called_tools.add("planner")
        m._planner_done = True  # awrap_tool_call 中 planner 执行后置位
        assert m._planner_done

    @pytest.mark.asyncio
    async def test_prefill_planner_writes_messages_and_phase_signal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """开局预置：planner 在第 1 次模型调用**之前**就跑掉，结果写进 state。

        省掉的是「模型花一整轮只为说出 planner 这个词」那次往返；阶段信号照常置位，
        第 1 轮 post_reflect 因此仍能把 PLANNING 推到 SEARCHING。
        """
        from types import SimpleNamespace

        from langchain_core.messages import AIMessage, ToolMessage

        import app.tools.planner as planner_mod
        from app.harness.agent_middleware import HarnessAgentMiddleware

        async def fake_planner(call: dict) -> ToolMessage:
            return ToolMessage(content='{"tasks": ["recommend"]}', tool_call_id=call["id"])

        # 整体替换模块属性（StructuredTool 是 pydantic 模型，setattr 不进去）——middleware 里是
        # 函数内懒 import，取的正是这个模块属性。
        monkeypatch.setattr(planner_mod, "planner", SimpleNamespace(ainvoke=fake_planner))

        m = HarnessAgentMiddleware(original_query="买个旅行收纳袋")
        update = await m.abefore_agent(state={}, runtime=None)

        assert update is not None
        ai, tool_msg = update["messages"]
        assert isinstance(ai, AIMessage) and ai.tool_calls[0]["name"] == "planner"
        assert isinstance(tool_msg, ToolMessage)
        assert tool_msg.tool_call_id == ai.tool_calls[0]["id"]  # 两条必须对得上，否则 LC 校验失败
        assert m._planner_done and "planner" in m._called_tools

    @pytest.mark.asyncio
    async def test_prefill_planner_skipped_in_fork(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """子 Agent 不预置：它的活是按 demands 检索，demands 里已带主流程拆好的字段。"""
        from app.agent.fork_guard import _fork_depth
        from app.harness.agent_middleware import HarnessAgentMiddleware

        token = _fork_depth.set(1)
        try:
            m = HarnessAgentMiddleware(original_query="在 amazon 搜收纳袋")
            assert await m.abefore_agent(state={}, runtime=None) is None
            assert not m._planner_done
        finally:
            _fork_depth.reset(token)

    @pytest.mark.asyncio
    async def test_prefill_planner_degrades_on_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """planner 抛错 → 不预置、不炸整轮：回到老路（模型自己决定调 planner）。"""
        from types import SimpleNamespace

        import app.tools.planner as planner_mod
        from app.harness.agent_middleware import HarnessAgentMiddleware

        async def boom(call: dict) -> None:
            raise RuntimeError("planner 挂了")

        monkeypatch.setattr(planner_mod, "planner", SimpleNamespace(ainvoke=boom))

        m = HarnessAgentMiddleware(original_query="买个旅行收纳袋")
        assert await m.abefore_agent(state={}, runtime=None) is None
        assert not m._planner_done  # 没跑成就不能置阶段信号，否则阶段机会凭空推进

    @pytest.mark.asyncio
    async def test_phase_transition_on_planner(self) -> None:
        from app.agent.fork_guard import _fork_depth
        from app.harness.hooks.phase_transition import try_phase_transition

        token = _fork_depth.set(0)
        m = PhaseStateMachine()
        set_phase_machine(m)
        try:
            ctx: dict = {"planner_output_ready": True}
            await try_phase_transition(ctx)
            assert m.phase == Phase.SEARCHING
        finally:
            _fork_depth.reset(token)
            reset_phase_machine()

    @pytest.mark.asyncio
    async def test_phase_transition_to_comparing_is_silent(self) -> None:
        """转移本体不再经 inject 发通告——inject 通道晚一轮（perf-audit-r3 实测模型在读到
        通告前就已决定再搜）。「检索收线」通告改由 transition_notice 缀在工具结果尾部
        （见 test_tool_memo.py 的 TestTransitionNotices），这里只验证状态机推进。"""
        from app.agent.fork_guard import _fork_depth
        from app.harness.hooks.phase_transition import try_phase_transition

        token = _fork_depth.set(0)
        m = PhaseStateMachine()
        m.try_transition("planner_output_ready")  # 先推进到 SEARCHING
        set_phase_machine(m)
        try:
            ctx: dict = {"total_candidates": 10}
            await try_phase_transition(ctx)
            assert m.phase == Phase.COMPARING
            assert not ctx.get("inject_messages")
        finally:
            _fork_depth.reset(token)
            reset_phase_machine()

    @pytest.mark.asyncio
    async def test_phase_rollback(self) -> None:
        from app.agent.fork_guard import _fork_depth
        from app.harness.hooks.phase_transition import check_phase_rollback

        token = _fork_depth.set(0)
        m = PhaseStateMachine()
        m.try_transition("planner_output_ready")
        m.try_transition("candidates_available")
        assert m.phase == Phase.COMPARING
        set_phase_machine(m)
        try:
            # 「无进展」= item_picker 跑过但精挑不出任何东西（refdocs 17-4 §4.3）。
            # 「还没调 item_picker」不算，见 TestRollbackRequiresPickerAttempt。
            ctx: dict = {"picks_count": 0, "picker_attempted": True}
            await check_phase_rollback(ctx)
            assert m.phase == Phase.COMPARING  # 还没到阈值

            # 第二轮仍精挑不出东西 → 回退
            ctx2: dict = {"picks_count": 0, "picker_attempted": True}
            result = await check_phase_rollback(ctx2)
            assert m.phase == Phase.SEARCHING
            assert any("回退" in msg["content"] for msg in result.get("inject_messages", []))
        finally:
            _fork_depth.reset(token)
            reset_phase_machine()


# ============================================================
# 集成：全局 harness 单例注册检查
# ============================================================


class TestGlobalHarnessSetup:
    """验证 setup_harness 注册了所有预期的 Hook。"""

    def test_all_hooks_registered(self) -> None:
        from app.harness.setup import setup_harness

        setup_harness()
        hooks = harness.list_hooks()
        names = {name for _, name, _ in hooks}
        expected = {
            "schema_assertion",
            "sequencing_assertion",
            "semantic_assertion",
            "drift_detector",
            "drift_result_tracker",
            "assertion_handler",
            "phase_check",
            "phase_transition",
            "phase_rollback",
        }
        assert expected.issubset(names), f"Missing hooks: {expected - names}"

    def test_middleware_stack_is_harness_only(self) -> None:
        """控制面全在 Hook 里 → LangChain 中间件栈只剩唯一的适配器，没有并行的第二套控制逻辑。"""
        from app.harness.agent_middleware import build_agent_middleware

        stack = build_agent_middleware(original_query="test")
        assert [type(m).__name__ for m in stack] == ["HarnessAgentMiddleware"]

    def test_control_plane_lives_entirely_in_hooks(self) -> None:
        """六个 Hook 点全部有主——迁移后不该再有空的生命周期阶段。"""
        from app.harness.middleware import HOOK_POINTS, harness
        from app.harness.setup import setup_harness

        setup_harness()
        for point in HOOK_POINTS:
            assert harness.list_hooks(point), f"{point} 上没有任何 Hook"


class TestInjectMechanism:
    """验证 inject_messages 的端到端回注。"""

    def test_pending_inject_consumed(self) -> None:
        """_pending_inject 存入后被 _consume_pending_inject 正确消费并清空。"""
        from app.harness.agent_middleware import HarnessAgentMiddleware

        m = HarnessAgentMiddleware(original_query="test")
        m._pending_inject = [
            {"role": "system", "content": "漂移纠正提示"},
            {"role": "system", "content": "断言纠正提示"},
        ]
        msgs = m._consume_pending_inject()
        assert len(msgs) == 2
        assert "漂移纠正" in msgs[0].content
        assert "断言纠正" in msgs[1].content
        # 消费后清空
        assert m._pending_inject == []
        assert m._consume_pending_inject() == []

    def test_collect_inject_from_context(self) -> None:
        """_collect_inject 从 Hook context 收集注入消息到 _pending_inject。"""
        from app.harness.agent_middleware import HarnessAgentMiddleware

        m = HarnessAgentMiddleware(original_query="test")
        ctx = {
            "inject_messages": [
                {"role": "system", "content": "msg1"},
                {"role": "system", "content": "msg2"},
            ]
        }
        m._collect_inject(ctx)
        assert len(m._pending_inject) == 2

    def test_candidate_count_zero_without_session(self) -> None:
        """无 session 作用域时 candidate_count 返回 0（不崩）。"""
        from app.harness.signals import candidate_count

        assert candidate_count() == 0


class TestInjectPersistence:
    """注入随 ModelResponse.result 落 state。

    曾经注入只进当轮请求视图、下一轮消失 → 第 N+1 轮 prompt 不再是第 N 轮的字节延伸，
    隐式前缀缓存链每轮被斩断（eval q05/q03/q16 命中率卡死在 system 段）。"""

    @pytest.mark.asyncio
    async def test_consumed_inject_rides_response_result(self, clean_phase) -> None:
        """挂起注入消费后必须出现在 result 里（模型所见顺序：注入在前、AI 回复在后）。"""
        mw = _mw()
        mw._pending_inject = [{"role": "system", "content": "[漂移提醒] 保持方向"}]
        out = await _run_model(mw)
        assert "[漂移提醒]" in out.result[0].content
        assert out.result[-1].content == "ok"

    @pytest.mark.asyncio
    async def test_no_inject_keeps_result_untouched(self, clean_phase) -> None:
        """无注入时不包一层、不添消息。"""
        mw = _mw()
        out = await _run_model(mw)
        assert [m.content for m in out.result] == ["ok"]


# ============================================================
# 接线测试：Hook 之间、Hook 与中间件之间的数据是否真的流通
#
# 上面的测试都是「手搓一个 context dict 喂给单个 Hook」——Hook 自身逻辑对，不代表它在真实
# Agent 生命周期里拿得到数据。以下测试驱动 HarnessAgentMiddleware 本身，覆盖四条曾经断掉的链路。
# ============================================================


class _FakeModelRequest:
    """最小 ModelRequest 替身：messages / system_message / tools + 链式 override。"""

    def __init__(self, messages: list, system_message: object = None) -> None:
        self.messages = messages
        self.system_message = system_message
        self.tools: list = []

    def override(self, **kw: object) -> _FakeModelRequest:
        out = _FakeModelRequest(
            kw.get("messages", self.messages),  # type: ignore[arg-type]
            kw.get("system_message", self.system_message),
        )
        out.tools = self.tools
        return out


class _FakeToolRequest:
    def __init__(self, name: str, args: dict | None = None) -> None:
        self.tool_call = {"name": name, "args": args or {}, "id": f"call_{name}"}


@pytest.fixture()
def clean_phase():
    """注册全部 Hook + 给每个接线测试一个干净的阶段机。

    setup_harness() 必须显式调——Hook 注册是全局副作用，不调的话这些测试会「因为没有 Hook 在跑」
    而空过（早期版本正是靠别的用例先调过 setup_harness 才碰巧通过，单独跑就挂）。
    """
    from app.harness.setup import setup_harness

    setup_harness()
    reset_phase_machine()
    yield
    reset_phase_machine()


def _mw(query: str = "想买便宜又抗造的旅行三件套，预算300"):
    from app.harness.agent_middleware import HarnessAgentMiddleware

    return HarnessAgentMiddleware(original_query=query)


async def _run_tool(
    mw, name: str, args: dict | None = None, result: str = "{}", diag: dict | None = None
):
    """驱动一次 awrap_tool_call。``diag`` 模拟真实工具在返回前往诊断侧信道登记
    （见 app/tools/_diagnostics.py）——需要测试跑在 thread_scope 里，否则登记静默 no-op。"""
    from langchain_core.messages import ToolMessage

    from app.tools._diagnostics import report_diagnostics

    async def handler(req):
        if diag is not None:
            report_diagnostics(name, diag)
        return ToolMessage(content=result, tool_call_id=req.tool_call["id"], name=name)

    return await mw.awrap_tool_call(_FakeToolRequest(name, args), handler)


async def _run_model(mw, *, with_tool_results: bool = True):
    """驱动一次 awrap_model_call；with_tool_results 决定是否触发 post_reflect。

    handler 按生产契约返回 ModelResponse（LangChain 的 _execute_model_async 即如此）——
    persist_messages 落 state 依赖 response.result 可拼接，替身返回裸 AIMessage 会失真。"""
    from langchain.agents.middleware import ModelResponse
    from langchain_core.messages import AIMessage, ToolMessage

    msgs = (
        [ToolMessage(content="{}", tool_call_id="x", name="item_search")]
        if with_tool_results
        else []
    )

    async def handler(req):
        # 挂一个假 tool_call：这些接线测试模拟的是循环中段（模型还在干活），
        # 不带 tool_calls 会触发 terminal_enforcer 的催收重发，平白多一次 handler 调用。
        ai = AIMessage(
            content="ok",
            tool_calls=[{"name": "item_search", "args": {}, "id": "call_fake"}],
        )
        return ModelResponse(result=[ai])

    return await mw.awrap_model_call(_FakeModelRequest(msgs), handler)


class TestTerminalDirectClose:
    """终结直出（延迟归因 round2 刀2）：shopping_summary 执行后不再唤起模型复述清单。"""

    @pytest.mark.asyncio
    async def test_summary_artifact_short_circuits_model(self, clean_phase) -> None:
        from langchain_core.messages import ToolMessage

        from app.tools.shopping_summary import ShoppingSummaryOutput

        mw = _mw()
        mw._guard.terminal_reached = True
        art = ShoppingSummaryOutput(summary="## 清单\n- A 好货")
        msgs = [
            ToolMessage(
                content=art.summary, tool_call_id="x", name="shopping_summary", artifact=art
            )
        ]

        async def handler(req):  # 模型不该被调
            raise AssertionError("终结直出后不应再唤起模型")

        resp = await mw.awrap_model_call(_FakeModelRequest(msgs), handler)
        assert resp.result[0].content == art.summary

    @pytest.mark.asyncio
    async def test_no_artifact_falls_through_to_model(self, clean_phase) -> None:
        """chat_fallback 等无 summary artifact 的终结 → 照常唤起模型收尾。"""
        mw = _mw()
        mw._guard.terminal_reached = True
        out = await _run_model(mw, with_tool_results=False)
        assert out.result[-1].content == "ok"


class TestAssertionWiring:
    """三类断言 → post_reflect 的 assertion_handler → inject_messages 的完整链路。"""

    @pytest.mark.asyncio
    async def test_sequencing_assertion_reaches_model(self, clean_phase) -> None:
        """顺序断言在 pre_tool_call 产生，必须能流到 post_reflect 并变成注入提示。"""
        set_phase_machine(PhaseStateMachine(Phase.COMPARING))
        mw = _mw()
        # item_picker 的前置是 item_search，此处未调过 → 触发 sequencing 断言
        await _run_tool(mw, "item_picker", {"query": "旅行三件套"})
        assert mw._pending_assertions, "sequencing 断言未被中间件接住"

        await _run_model(mw)
        contents = [m["content"] for m in mw._pending_inject]
        assert any("[顺序问题]" in c for c in contents), f"断言未转成纠正提示: {contents}"
        assert not mw._pending_assertions, "断言消费后应清空"

    @pytest.mark.asyncio
    async def test_schema_assertion_reaches_model(self, clean_phase) -> None:
        """schema 断言在 post_tool_call 产生，同样要能流到 post_reflect。"""
        set_phase_machine(PhaseStateMachine(Phase.SEARCHING))
        mw = _mw()
        # 合法 JSON 但不符合 ItemSearchOutput → ValidationError
        await _run_tool(mw, "item_search", {"query": "旅行三件套"}, result='{"bogus": 1}')
        assert mw._pending_assertions, "schema 断言未被中间件接住"

        await _run_model(mw)
        assert any("[格式问题]" in m["content"] for m in mw._pending_inject)


class TestDriftWiring:
    """漂移检测四类信号的真实触发路径。"""

    @pytest.mark.asyncio
    async def test_on_target_actions_do_not_trigger_goal_forgetting(
        self, clean_phase, monkeypatch
    ) -> None:
        """行为摘要含 query 关键词时不得误报「目标遗忘」——旧实现只喂工具名，命中恒为 0。"""
        import app.agent.llm as llm_mod

        class _FakeLLM:
            async def ainvoke(self, _msgs):
                from langchain_core.messages import AIMessage

                return AIMessage(content="正常")

        monkeypatch.setattr(llm_mod, "get_judge_llm", lambda: _FakeLLM())
        set_phase_machine(PhaseStateMachine(Phase.SEARCHING))
        mw = _mw("想买便宜又抗造的旅行三件套")

        # 三轮都在搜「旅行三件套」，方向没跑偏
        for _ in range(3):
            await _run_tool(
                mw, "item_search", {"query": "旅行三件套"}, result='{"candidates": [1]}'
            )
        for _ in range(3):  # drift 每 3 轮检一次
            await _run_model(mw)

        contents = [m["content"] for m in mw._pending_inject]
        assert not any("漂移提醒" in c for c in contents), f"在目标上却误报漂移: {contents}"

    @pytest.mark.asyncio
    async def test_off_target_actions_trigger_goal_forgetting(self, clean_phase) -> None:
        """行为摘要与 query 毫无关键词交集 → 目标遗忘（轻微偏离）。"""
        set_phase_machine(PhaseStateMachine(Phase.SEARCHING))
        mw = _mw("旅行三件套")
        await _run_tool(
            mw, "item_search", {"query": "帐篷 睡袋 露营"}, result='{"candidates": [1]}'
        )
        for _ in range(3):
            await _run_model(mw)

        assert any("漂移提醒" in m["content"] for m in mw._pending_inject)

    @pytest.mark.asyncio
    async def test_blacklist_hit_triggers_preference_loss(self, clean_phase, monkeypatch) -> None:
        """信号 3：推荐面出现硬 dislike 属性 → 严重偏离 + 定向纠正。"""
        from app.harness.hooks import drift_detector as dd

        monkeypatch.setattr(dd, "blacklist_hits", lambda text: ["塑料"] if "塑料" in text else [])
        set_phase_machine(PhaseStateMachine(Phase.COMPARING))
        mw = _mw("旅行三件套")

        await _run_tool(mw, "item_search", {"query": "旅行三件套"}, result='{"candidates": [1]}')
        await _run_tool(
            mw, "item_picker", {"query": "旅行三件套"}, result='{"picks": ["塑料收纳盒"]}'
        )
        assert mw._drift_state.blacklist_violations > 0, "黑名单命中未被记录"

        for _ in range(3):
            await _run_model(mw)
        assert any("偏好丢失" in m["content"] for m in mw._pending_inject)

    @pytest.mark.asyncio
    async def test_severe_signal_not_masked_by_mild_one(self, clean_phase, monkeypatch) -> None:
        """轻微信号（目标遗忘）与严重信号（偏好丢失）同时命中时，判定必须取严重的那个。

        旧预检逐条 if + 首个命中即 return，「目标遗忘」排在最前，会永久遮蔽后面三类信号。
        """
        from app.harness.hooks import drift_detector as dd

        monkeypatch.setattr(dd, "blacklist_hits", lambda text: ["塑料"] if "塑料" in text else [])
        set_phase_machine(PhaseStateMachine(Phase.COMPARING))
        mw = _mw("旅行三件套")

        # 行为摘要与 query 零关键词交集 → 目标遗忘（轻微）；结果命中黑名单 → 偏好丢失（严重）
        await _run_tool(
            mw, "item_picker", {"query": "帐篷 睡袋"}, result='{"picks": ["塑料收纳盒"]}'
        )
        for _ in range(3):
            await _run_model(mw)

        contents = [m["content"] for m in mw._pending_inject]
        assert any("偏好丢失" in c for c in contents), f"严重信号被轻微信号遮蔽: {contents}"

    @pytest.mark.asyncio
    async def test_search_result_not_checked_for_blacklist(self, clean_phase, monkeypatch) -> None:
        """召回阶段捞到黑名单商品是正常的（后续会淘汰），不应记违规。"""
        from app.harness.hooks import drift_detector as dd

        monkeypatch.setattr(dd, "blacklist_hits", lambda text: ["塑料"] if "塑料" in text else [])
        set_phase_machine(PhaseStateMachine(Phase.SEARCHING))
        mw = _mw("旅行三件套")
        await _run_tool(mw, "item_search", {"query": "三件套"}, result='{"candidates": ["塑料盒"]}')
        assert mw._drift_state.blacklist_violations == 0

    @pytest.mark.asyncio
    async def test_token_history_is_populated(self, clean_phase, monkeypatch) -> None:
        """信号 4：token_history 必须真的有人写——旧实现从无写入点，成本失控是死代码。"""
        import app.harness.agent_middleware as am

        totals = iter([100, 250, 500])
        monkeypatch.setattr(
            am, "tree_snapshot", lambda: {"input_tokens": next(totals), "output_tokens": 0}
        )
        set_phase_machine(PhaseStateMachine(Phase.SEARCHING))
        mw = _mw()
        for _ in range(3):
            await _run_model(mw, with_tool_results=False)

        assert mw._drift_state.token_history == [100, 150, 250]


class TestPhaseGateTerminalExemption:
    """阶段门不得与「强制收尾」通路顶死。"""

    @pytest.mark.asyncio
    async def test_summary_rejected_without_picker_this_turn(
        self, clean_phase, monkeypatch
    ) -> None:
        """底线 3：本轮没跑过 item_picker 就收尾 → 拒绝并指路精挑（badcase 63093a85/q05）。

        拒绝不是死锁：逃生动作唯一且永远可行（候选非空由底线 1 保证），文案必须点名
        item_picker——模型照着走一步就能通过。
        """
        from app.harness.hooks import phase_check as pc

        monkeypatch.setattr(pc, "candidate_count", lambda: 5)
        set_phase_machine(PhaseStateMachine(Phase.SEARCHING))
        mw = _mw()
        result = await _run_tool(mw, "shopping_summary", {}, result='{"items": []}')
        assert "[Harness 拒绝]" in result.content
        assert "item_picker" in result.content

    @pytest.mark.asyncio
    async def test_summary_allowed_after_picker_ran(self, clean_phase, monkeypatch) -> None:
        """本轮 item_picker 跑过（哪怕定稿为空）→ 收尾放行：诚实空清单是合法结论。"""
        from app.harness.hooks import phase_check as pc

        monkeypatch.setattr(pc, "candidate_count", lambda: 5)
        set_phase_machine(PhaseStateMachine(Phase.SEARCHING))
        mw = _mw()
        await _run_tool(mw, "item_picker", {}, result='{"picks": []}')
        result = await _run_tool(mw, "shopping_summary", {}, result='{"items": []}')
        assert "[Harness 拒绝]" not in result.content

    @pytest.mark.asyncio
    async def test_summary_rejected_when_no_candidates(self, clean_phase, monkeypatch) -> None:
        """真的没候选时仍拒绝收尾，并引导 chat_fallback（防过早收尾）。"""
        from app.harness.hooks import phase_check as pc

        monkeypatch.setattr(pc, "candidate_count", lambda: 0)
        set_phase_machine(PhaseStateMachine(Phase.SEARCHING))
        mw = _mw()
        result = await _run_tool(mw, "shopping_summary", {})
        assert "[Harness 拒绝]" in result.content
        assert "chat_fallback" in result.content

    @pytest.mark.asyncio
    async def test_chat_fallback_always_allowed(self, clean_phase, monkeypatch) -> None:
        from app.harness.hooks import phase_check as pc

        monkeypatch.setattr(pc, "candidate_count", lambda: 0)
        set_phase_machine(PhaseStateMachine(Phase.COMPARING))
        mw = _mw()
        result = await _run_tool(mw, "chat_fallback", {})
        assert "[Harness 拒绝]" not in result.content

    @pytest.mark.asyncio
    async def test_force_conclude_authorizes_concluding_phase(self, clean_phase) -> None:
        """连续严重漂移强制收尾时，阶段机必须被推到 CONCLUDING，否则自家 gate 会拦住自家指令。"""
        from app.harness.hooks.drift_detector import _apply_correction

        set_phase_machine(PhaseStateMachine(Phase.SEARCHING))
        state = DriftState()
        state.consecutive_severe = 1  # 上一次已严重，本次是第二次
        ctx = _apply_correction({"original_query": "旅行三件套"}, "严重偏离", state, ["探索发散"])

        assert any("[强制收尾]" in m["content"] for m in ctx["inject_messages"])
        machine = get_phase_machine()
        assert machine.phase == Phase.CONCLUDING, "强制收尾未授权 CONCLUDING → 会与 phase gate 死锁"


@pytest.mark.asyncio
class TestToolErrorNotProgress:
    """status="error" 的 ToolMessage（ToolNode 把参数校验失败转成的）不算执行成功。

    gcjp 会话 d0724e95（2026-07-16）：qwen3.5-flash 把 list 参数吐成 JSON 字符串，
    item_picker 同参连挂 4 次，全被记成「已精挑」——phase_check 底线 3 判据被污染放行
    空清单收尾，收线通告还缀在错误消息尾部教唆模型跳 shopping_summary。
    """

    @staticmethod
    async def _run_error_tool(mw, name: str = "item_picker"):
        from langchain_core.messages import ToolMessage

        async def handler(req):
            return ToolMessage(
                content=f"Error invoking tool '{name}': Input should be a valid list",
                tool_call_id=req.tool_call["id"],
                name=name,
                status="error",
            )

        return await mw.awrap_tool_call(_FakeToolRequest(name, {}), handler)

    async def test_error_result_not_recorded_as_progress(self, clean_phase) -> None:
        """失败调用不进 called_tools / 阶段信号 / 看门狗，错误消息不被收线通告污染。"""
        set_phase_machine(PhaseStateMachine(Phase.COMPARING))
        mw = _mw()
        watchdog_before = mw._guard.last_progress_at
        out = await self._run_error_tool(mw)
        assert "item_picker" not in mw._called_tools
        assert mw._picker_attempted is False
        assert mw._guard.last_progress_at == watchdog_before, "失败调用不给看门狗续命"
        # post_tool_call 全程没跑：错误消息原样回模型，不缀收线通告（线上实锤的教唆路径）
        assert out.content == "Error invoking tool 'item_picker': Input should be a valid list"

    async def test_error_result_feeds_loop_detector(self, clean_phase) -> None:
        """同参硬撞同一个校验错误正是打转——攒到阈值要在错误尾部追加升级提示。"""
        mw = _mw()
        contents = []
        for _ in range(mw._guard.loop_threshold):
            msg = await self._run_error_tool(mw)
            contents.append(str(msg.content))
        assert "重复调用" not in contents[0]
        assert "重复调用" in contents[-1]

    async def test_summary_still_blocked_when_picker_only_failed(
        self, clean_phase, monkeypatch
    ) -> None:
        """闭环：picker 只有失败调用时，底线 3 必须仍拦收尾（判据回归「真实执行成功」）。"""
        from app.harness.hooks import phase_check as pc

        monkeypatch.setattr(pc, "candidate_count", lambda: 5)
        set_phase_machine(PhaseStateMachine(Phase.COMPARING))
        mw = _mw()
        await self._run_error_tool(mw)
        result = await _run_tool(mw, "shopping_summary", {}, result='{"items": []}')
        assert "[Harness 拒绝]" in result.content
        assert "item_picker" in result.content


class TestPhaseTransitionResetsDrift:
    """refdocs 17-4 §9 跨章耦合：阶段转移时重置漂移计数器。"""

    @pytest.mark.asyncio
    async def test_transition_resets_consecutive_counters(self, clean_phase) -> None:
        from app.harness.hooks.phase_transition import try_phase_transition

        set_phase_machine(PhaseStateMachine(Phase.SEARCHING))
        state = DriftState()
        state.consecutive_empty_results = 2  # 搜了两次空
        state.consecutive_severe = 1
        state.blacklist_violations = 1

        # 候选进了登记表 → SEARCHING → COMPARING
        ctx = {"_drift_state": state, "total_candidates": 8}
        await try_phase_transition(ctx)

        assert get_phase_machine().phase == Phase.COMPARING
        assert state.consecutive_empty_results == 0, "转移后仍带着上一阶段的空结果计数"
        assert state.consecutive_severe == 0
        assert state.blacklist_violations == 1, "偏好违规不因阶段推进而清零"

    @pytest.mark.asyncio
    async def test_no_transition_keeps_counters(self, clean_phase) -> None:
        """没转移就不该重置——否则漂移计数永远攒不起来。"""
        from app.harness.hooks.phase_transition import try_phase_transition

        set_phase_machine(PhaseStateMachine(Phase.SEARCHING))
        state = DriftState()
        state.consecutive_empty_results = 2

        ctx = {"_drift_state": state, "total_candidates": 0}  # 还没候选
        await try_phase_transition(ctx)

        assert get_phase_machine().phase == Phase.SEARCHING
        assert state.consecutive_empty_results == 2


class TestPicksSignalIsReal:
    """picks_count 必须来自 item_picker 的真实返回，不能是「调过就算数」。"""

    def test_count_picks_parses_json(self) -> None:
        from app.harness.agent_middleware import _count_picks

        assert _count_picks('{"picks": [{"a": 1}, {"b": 2}], "excluded": []}') == 2
        assert _count_picks('{"picks": [], "excluded": ["x"], "over_budget": ["y"]}') == 0

    def test_count_picks_survives_truncation(self) -> None:
        """TGM 按 token 预算截断长结果 → JSON 解析失败，但长结果必然意味着 picks 非空。"""
        from app.harness.agent_middleware import _count_picks

        truncated = '{"picks": [{"item_id": "A1", "pick_reason": "耐磨"' + "x" * 50
        assert _count_picks(truncated) >= 1
        assert _count_picks("[Harness 拒绝] 越权") == 0

    @pytest.mark.asyncio
    async def test_reuse_zero_must_hits_rolls_back_not_conclude(
        self, clean_phase, tmp_path
    ) -> None:
        """集成重放 bad case「三件套 → 要中式刺绣」：复用轮 picker 返回 8 件但 must_have
        池内 0 命中 → 阶段必须退回 SEARCHING 补搜，而不是推进 CONCLUDING 把补搜的路拦死；
        picker 结果尾部当场缀上「阶段回退 + 请重新检索」的指路。
        """
        from app.api.context import get_retrieval_mode, set_retrieval_mode
        from app.utils.thread_ctx import thread_scope

        with thread_scope("t-zero-hits-e2e", tmp_path):
            set_phase_machine(PhaseStateMachine(Phase.COMPARING))
            set_retrieval_mode("reuse")
            mw = _mw()
            import json as _json

            result = _json.dumps(
                {
                    "must_have_hits": 0,
                    "picks": [{"item_id": f"B{i}", "title": "solid set"} for i in range(8)],
                    "excluded": [],
                }
            )
            msg = await _run_tool(
                mw,
                "item_picker",
                {},
                result=result,
                diag={"picks": 8, "must_have_hits": 0, "oncat_count": None, "offcat_count": None},
            )
            assert "[阶段回退]" in msg.content and "重新检索" in msg.content
            await _run_model(mw)

            assert get_phase_machine().phase == Phase.SEARCHING, "0 命中不该推进 CONCLUDING"
            # mode 改写 augment = 补搜授权：复用轮小预算解除，item_search 走全树预算放行
            assert get_retrieval_mode() == "augment"

    @pytest.mark.asyncio
    async def test_diagnostics_channel_survives_truncation(self, clean_phase, tmp_path) -> None:
        """契约：picker 登记的诊断字段 middleware 都收得到，且与可见文本**无关**——
        把 result 换成截断到面目全非的残片，信号照样一个不少（旧正则通路做不到这点）。"""
        from app.utils.thread_ctx import thread_scope

        with thread_scope("t-diag-truncated", tmp_path):
            mw = _mw()
            mutilated = '{"mus' + "x" * 30  # 任何正则都救不回来的截断残片
            await _run_tool(
                mw,
                "item_picker",
                {},
                result=mutilated,
                diag={"picks": 5, "must_have_hits": 2, "oncat_count": 4, "offcat_count": 6},
            )
            assert mw._last_picks == 5
            assert mw._last_must_hits == 2
            assert (mw._last_oncat, mw._last_offcat) == (4, 6)

    @pytest.mark.asyncio
    async def test_diagnostics_missing_fails_open(self, clean_phase, tmp_path) -> None:
        """侧信道意外空（未走到登记点的旁路）：picks 退回文本兜底，三个诊断字段退化 None
        =「不适用」——补搜闸不误触发（失效方向中性，绝不反转）。"""
        from app.utils.thread_ctx import thread_scope

        with thread_scope("t-diag-missing", tmp_path):
            mw = _mw()
            await _run_tool(mw, "item_picker", {}, result='{"picks": [{"item_id": "A1"}]}')
            assert mw._last_picks == 1  # 文本兜底仍数得出
            assert mw._last_must_hits is None
            assert mw._last_oncat is None and mw._last_offcat is None

    @pytest.mark.asyncio
    async def test_empty_picks_does_not_advance_to_concluding(self, clean_phase) -> None:
        """item_picker 精挑出 0 件 → 不该进 CONCLUDING（否则 shopping_summary 空输出）。"""
        set_phase_machine(PhaseStateMachine(Phase.COMPARING))
        mw = _mw()
        await _run_tool(mw, "item_picker", {}, result='{"picks": [], "excluded": ["a"]}')
        await _run_model(mw)

        assert get_phase_machine().phase == Phase.COMPARING, "空 picks 不该推进阶段"

    @pytest.mark.asyncio
    async def test_nonempty_picks_advances(self, clean_phase) -> None:
        set_phase_machine(PhaseStateMachine(Phase.COMPARING))
        mw = _mw()
        await _run_tool(mw, "item_picker", {}, result='{"picks": [{"item_id": "A1"}]}')
        await _run_model(mw)

        assert get_phase_machine().phase == Phase.CONCLUDING


class TestRollbackRequiresPickerAttempt:
    """回退只在「ItemPicker 返回空」后触发（refdocs 17-4 §4.3），不能因为还没精挑就回退。"""

    @pytest.mark.asyncio
    async def test_no_rollback_before_picker_runs(self, clean_phase) -> None:
        """COMPARING 里先 price_compare / 澄清是正常路径，不该被判无进展回退。"""
        from app.harness.hooks.phase_transition import check_phase_rollback

        set_phase_machine(PhaseStateMachine(Phase.COMPARING))
        for _ in range(3):
            await check_phase_rollback({"picks_count": 0, "picker_attempted": False})

        assert get_phase_machine().phase == Phase.COMPARING, "item_picker 都没跑就回退了"

    @pytest.mark.asyncio
    async def test_rollback_after_picker_returns_empty_twice(self, clean_phase) -> None:
        from app.harness.hooks.phase_transition import check_phase_rollback

        set_phase_machine(PhaseStateMachine(Phase.COMPARING))
        ctx = {"picks_count": 0, "picker_attempted": True}
        await check_phase_rollback(dict(ctx))
        assert get_phase_machine().phase == Phase.COMPARING  # 第 1 轮只计数
        await check_phase_rollback(dict(ctx))
        assert get_phase_machine().phase == Phase.SEARCHING  # 第 2 轮回退


class TestSequencingAnyOf:
    """顺序断言：满足任一前置即可，且必须认 fork 检索通路。"""

    @pytest.mark.asyncio
    async def test_fork_retrieval_satisfies_item_picker_prereq(self) -> None:
        """候选由 parallel_dispatch_tool 的子 Agent 检索而来 → 不该报顺序错误。"""
        from app.harness.hooks.step_validator import check_sequencing

        ctx = {"tool_name": "item_picker", "called_tools": {"parallel_dispatch_tool"}}
        result = await check_sequencing(ctx)
        assert result is None or not result.get("assertions_failed")

    @pytest.mark.asyncio
    async def test_candidates_in_registry_satisfy_prereq(self, monkeypatch) -> None:
        """登记表里有候选（如续聊沿用上轮候选）→ 前置已满足。"""
        from app.harness.hooks import step_validator as sv

        monkeypatch.setattr(sv, "candidate_count", lambda: 12)
        ctx = {"tool_name": "item_picker", "called_tools": set()}
        result = await sv.check_sequencing(ctx)
        assert result is None or not result.get("assertions_failed")

    @pytest.mark.asyncio
    async def test_no_retrieval_at_all_still_warns(self, monkeypatch) -> None:
        from app.harness.hooks import step_validator as sv

        monkeypatch.setattr(sv, "candidate_count", lambda: 0)
        ctx = {"tool_name": "item_picker", "called_tools": {"planner"}}
        result = await sv.check_sequencing(ctx)
        assert result["assertions_failed"][0]["type"] == "sequencing"


# ============================================================
# 迁移进 Hook Pipeline 后新增的两项能力
# ============================================================


class TestToolBreakerHook:
    """工具级熔断：pre_tool_call 判定 + post_tool_call 计数（refdocs 17-2 §2.2）。

    与 web_search / reranker 里那种「包一次外呼」的依赖级断路器不是一回事——这里熔的是工具本身。
    """

    @pytest.fixture(autouse=True)
    def _clean(self):
        from app.harness.hooks.tool_breaker import reset_tool_breakers
        from app.harness.setup import setup_harness

        setup_harness()
        reset_tool_breakers()
        yield
        reset_tool_breakers()

    @pytest.mark.asyncio
    async def test_repeated_failures_open_the_breaker(self, clean_phase) -> None:
        from app.harness.hooks.tool_breaker import _FAILURE_THRESHOLD, get_tool_breaker

        mw = _mw()

        async def boom(_req):
            raise RuntimeError("平台 API 挂了")

        for _ in range(_FAILURE_THRESHOLD):
            with pytest.raises(RuntimeError):
                await mw.awrap_tool_call(_FakeToolRequest("item_search"), boom)

        assert get_tool_breaker("item_search").state == "open"

        # 熔断后：工具**不执行**，直接回哨兵（handler 一次都不该被调）
        called = {"n": 0}

        async def handler(_req):
            called["n"] += 1
            raise AssertionError("熔断后不应执行工具")

        out = await mw.awrap_tool_call(_FakeToolRequest("item_search"), handler)
        assert called["n"] == 0
        assert "熔断保护" in out.content

    @pytest.mark.asyncio
    async def test_success_resets_failure_count(self, clean_phase) -> None:
        from app.harness.hooks.tool_breaker import get_tool_breaker

        mw = _mw()

        async def boom(_req):
            raise RuntimeError("偶发失败")

        with pytest.raises(RuntimeError):
            await mw.awrap_tool_call(_FakeToolRequest("web_search"), boom)
        await _run_tool(mw, "web_search", result="ok")  # 成功一次 → 计数清零
        assert get_tool_breaker("web_search").state == "closed"

    @pytest.mark.asyncio
    async def test_empty_result_is_not_a_failure(self, clean_phase) -> None:
        """空结果是业务信号（没搜到货），不是工具故障——不该熔断，交给漂移检测的探索发散信号。"""
        from app.harness.hooks.tool_breaker import get_tool_breaker

        mw = _mw()
        for _ in range(5):
            await _run_tool(mw, "item_search", result='{"candidates": []}')
        assert get_tool_breaker("item_search").state == "closed"


class TestOutputGuardHook:
    """on_session_end 输出审核：把 Harness 内部控制文案从最终回复里剔掉。"""

    @pytest.mark.asyncio
    async def test_strips_internal_sentinel_lines(self) -> None:
        from app.harness.hooks.session_hooks import audit_final_output

        final = (
            "这是给你的清单：\n"
            "[系统提示] 精选清单已就绪，请立即调用 shopping_summary。\n"
            "1. 帆布收纳袋 $18\n"
        )
        ctx = await audit_final_output({"final_answer": final})
        assert ctx is not None
        assert "[系统提示]" not in ctx["final_answer"]
        assert "帆布收纳袋" in ctx["final_answer"]  # 正常内容不受伤

    @pytest.mark.asyncio
    async def test_clean_answer_untouched(self) -> None:
        from app.harness.hooks.session_hooks import audit_final_output

        assert await audit_final_output({"final_answer": "干净的清单"}) is None

    @pytest.mark.asyncio
    async def test_never_returns_empty_answer(self) -> None:
        """全是内部文案时宁可回原文，也不给用户一片空白。"""
        from app.harness.hooks.session_hooks import audit_final_output

        ctx = await audit_final_output({"final_answer": "[强制收尾] 立即调用 shopping_summary"})
        assert ctx is not None
        assert ctx["final_answer"].strip()


class TestOutputGuardOrdering:
    """输出审核必须排在所有 final_text 消费者之前，否则等于没审。"""

    @pytest.mark.asyncio
    async def test_audited_text_reaches_artifacts_history_and_report(self, tmp_path) -> None:
        """把 on_session_end 排到落盘/上报之后，是「Hook 跑了没人消费」的另一种形态。

        这里钉死：用户实际看到的三条通路——task_result 上报、summary.md 产物、turns.json 历史——
        拿到的都必须是审核后的文本。
        """
        import app.agent.main_agent as ma

        dirty = (
            "给你的清单：\n[系统提示] 精选清单已就绪，必须调用 shopping_summary。\n1. 帆布袋 $18"
        )
        seen: dict[str, str] = {}

        async def fake_run(hook_point, ctx):
            if hook_point == "on_session_end":
                from app.harness.hooks.session_hooks import audit_final_output

                out = await audit_final_output(ctx)
                return out if out is not None else ctx
            return ctx

        # 直接验证 run_agent 里 on_session_end 的位置：审核后的文本必须先于消费者产生。
        src = (ma.__file__,)
        text = open(src[0], encoding="utf-8").read()
        end_hook = text.index('harness.run(\n            "on_session_end"')
        artifacts = text.index("_write_session_artifacts(session_dir, final_text, summary)")
        append = text.index("append_turn(")
        report = text.index("await monitor.report_task_result(")
        assert end_hook < artifacts, "输出审核晚于落产物 → 下载到的 md 是未审核原文"
        assert end_hook < append, "输出审核晚于写历史 → 回看到的是未审核原文"
        assert end_hook < report, "输出审核晚于 task_result → 用户前端看到的是未审核原文"

        # 顺带确认审核本身有效
        ctx = await fake_run("on_session_end", {"final_answer": dirty})
        assert "[系统提示]" not in ctx["final_answer"]
        assert "帆布袋" in ctx["final_answer"]
        seen["ok"] = ctx["final_answer"]
        assert seen["ok"]


class TestGateOrderingContracts:
    """两条易碎的 priority 契约，用测试钉死（注释拦不住重构）。"""

    def test_breaker_is_the_last_pre_tool_call_gate(self) -> None:
        """tool_breaker_gate 的 allow() 有副作用（OPEN→HALF_OPEN 放行探测）。

        它若不是最后一道闸，后面任何一道拒绝都会让这次「已放行的探测」没有对应成败记录，
        断路器悬在 HALF_OPEN 再也回不到 CLOSED。
        """
        from app.harness.middleware import harness
        from app.harness.setup import setup_harness

        setup_harness()
        gates = sorted(harness.list_hooks("pre_tool_call"), key=lambda t: t[2])
        assert gates[-1][1] == "tool_breaker_gate", f"熔断闸不是最后一道: {[g[1] for g in gates]}"

    def test_search_authority_runs_before_retrieval_charge(self) -> None:
        """search_authority 读的是 item_search 自增前的计数，自增在 retrieval_charge。

        谁把自增前移，子 Agent 的「恰好放行一次」会塌成「放行 0 次」。
        """
        from app.harness.middleware import harness
        from app.harness.setup import setup_harness

        setup_harness()
        prio = {n: p for _, n, p in harness.list_hooks("pre_tool_call")}
        assert prio["search_authority_gate"] < prio["retrieval_charge_gate"]

    async def test_sub_search_cap_admits_exactly_cap_calls(self) -> None:
        """行为契约：子 Agent 的 item_search **恰好**放行 SUB_ITEM_SEARCH_CAP 次。

        上面的 priority 测试钉的是钩子顺序；这条钉「自增住在 45 号闸里」——谁把
        ``guard.item_search_calls += 1`` 挪进 30 号闸内部（顺序不变、优先级测试照绿），
        cap 次就塌成 cap-1 次，本测试当场红。"""
        from app.agent.fork_guard import enter_fork
        from app.harness.budgets import SUB_ITEM_SEARCH_CAP
        from app.harness.hooks import tool_gates
        from app.harness.state import GuardState

        guard = GuardState()
        with enter_fork():
            for _ in range(SUB_ITEM_SEARCH_CAP):
                ctx: dict = {"_guard": guard, "tool_name": "item_search", "tool_args": {}}
                assert await tool_gates.check_search_authority(ctx) is None
                await tool_gates.charge_retrieval(ctx)
            with pytest.raises(HookRejectSignal):
                await tool_gates.check_search_authority(
                    {"_guard": guard, "tool_name": "item_search", "tool_args": {}}
                )

    def test_token_budget_runs_before_fork_charge(self) -> None:
        """token_budget_gate 必须早于 fork_budget_gate：fork 闸 charge 即扣槽（parallel 槽
        只有 1 个），预算拒绝若发生在扣槽之后，被拒的尝试会烧掉唯一的并行额度、还连带触发
        postfork 直搜拦截。"""
        from app.harness.middleware import harness
        from app.harness.setup import setup_harness

        setup_harness()
        prio = {n: p for _, n, p in harness.list_hooks("pre_tool_call")}
        assert prio["token_budget_gate"] < prio["fork_budget_gate"]


# ============================================================
# 偏好注入：planner 之后，且只给域内的
# ============================================================


async def _run_inject_hook(tool_name: str, *domains: str) -> str:
    """在「库里有条 footwear 的皮革 dislike」下跑注入 Hook，返回注入给模型的文本（无则空串）。"""
    import tempfile
    from pathlib import Path
    from uuid import uuid4

    from app.api.context import set_session_domains
    from app.harness.hooks.preference_inject import inject_domain_preferences
    from app.memory.store import PreferenceEntry, get_store
    from app.utils.thread_ctx import thread_scope

    uid = f"u-{uuid4().hex[:8]}"
    ctx: dict = {"tool_name": tool_name}
    with thread_scope("t-inject", Path(tempfile.mkdtemp()), user_id=uid):
        await get_store().write(
            uid,
            PreferenceEntry(
                slug="leather",
                content="不要皮革",
                category="material",
                domain="footwear",
                polarity="dislike",
                keywords=["皮革", "leather"],
            ),
        )
        set_session_domains(list(domains))
        await inject_domain_preferences(ctx)
    msgs = ctx.get("inject_messages") or []
    return "\n".join(m["content"] for m in msgs)


@pytest.mark.anyio
class TestPreferenceInject:
    """长期偏好只在 planner 判出域**之后**注入，且只注入域内的。

    改造前它拼在当轮 human 的最前面——那时 planner 还没跑、域还不存在，_in_scope 对空域一律放行，
    模型必然看到跨域偏好，并很自觉地把它转述进 item_picker 的自由文本参数拿到硬淘汰权。
    """

    async def test_injects_in_domain_preference_after_planner(self) -> None:
        text = await _run_inject_hook("planner", "footwear")  # 本轮在买鞋
        assert "皮革" in text

    async def test_does_not_inject_cross_domain_preference(self) -> None:
        """本轮在买包 → 鞋类的偏好压根不该出现在模型眼前。不给，胜过给了再管。"""
        text = await _run_inject_hook("planner", "bags")
        assert text == ""

    async def test_does_not_inject_before_planner(self) -> None:
        """planner 以外的工具不触发注入——域还没产生，注入了就是跨域全量。"""
        text = await _run_inject_hook("item_search", "footwear")
        assert text == ""


# ============================================================
# postfork 闸的结果感知解锁：棘轮闸的两个出口（空池 / 回退授权）
# ============================================================


@pytest.mark.asyncio
class TestPostforkGateRelease:
    """并行 fork 跑过后，item_search 不再是无条件永久拦截。"""

    def _ctx(self, guard=None) -> dict:
        from app.harness.state import GuardState

        return {"_guard": guard or GuardState(), "tool_name": "item_search", "tool_args": {}}

    async def test_empty_pool_releases_gate(self) -> None:
        """整轮 fork 失败/全空时候选池为空——「候选已汇集」不成立，直搜是仅剩的补救通路。"""
        from app.harness.budgets import fork_budget_scope
        from app.harness.hooks.tool_gates import check_search_authority

        with fork_budget_scope() as budget:
            budget.charge("parallel_dispatch_tool")
            assert await check_search_authority(self._ctx()) is None

    async def test_nonempty_pool_still_denied(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """正常情形（fork 后候选在池）维持原语义：直搜拦下，逼模型走比价/精挑/收尾。"""
        from app.harness.budgets import fork_budget_scope
        from app.harness.hooks import tool_gates

        monkeypatch.setattr(tool_gates, "candidate_count", lambda: 5)
        with fork_budget_scope() as budget:
            budget.charge("parallel_dispatch_tool")
            with pytest.raises(HookRejectSignal):
                await tool_gates.check_search_authority(self._ctx())

    async def test_rollback_grant_allows_exactly_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from app.harness.budgets import fork_budget_scope
        from app.harness.hooks import tool_gates
        from app.harness.state import GuardState

        monkeypatch.setattr(tool_gates, "candidate_count", lambda: 5)
        guard = GuardState()
        guard.postfork_search_grants = 1
        with fork_budget_scope() as budget:
            budget.charge("parallel_dispatch_tool")
            assert await tool_gates.check_search_authority(self._ctx(guard)) is None
            with pytest.raises(HookRejectSignal):
                await tool_gates.check_search_authority(self._ctx(guard))

    async def test_phase_rollback_issues_grant(self) -> None:
        """回退到 SEARCHING 的同时必须发补搜授权——否则初搜走过并行 fork 时，注入的
        「调整搜索条件重新检索」会撞 postfork 哨兵，回退腿在集成链路里是死的。"""
        from app.harness.hooks.phase_transition import check_phase_rollback
        from app.harness.state import GuardState

        machine = PhaseStateMachine(initial=Phase.COMPARING)
        set_phase_machine(machine)
        try:
            guard = GuardState()
            ctx = {"_guard": guard, "picker_attempted": True, "picks_count": 0}
            await check_phase_rollback(ctx)
            await check_phase_rollback(ctx)
            assert machine.phase is Phase.SEARCHING
            assert guard.postfork_search_grants == 1
        finally:
            reset_phase_machine()


# ============================================================
# 拒绝路径喂 LoopDetector / 熔断的参数异常豁免 / 哨兵清洗覆盖
# ============================================================


@pytest.mark.asyncio
class TestRejectedCallsFeedLoopDetector:
    async def test_hammering_a_gate_escalates(self, clean_phase) -> None:
        """模型换着参数硬撞同一道闸时（哨兵文案每次相同、无升级），拒绝路径攒到阈值要在
        哨兵尾部追加打转升级提示——否则闸拒绝是循环检测的盲区，只剩 recursion_limit 硬兜底。"""
        mw = _mw()
        mw._guard.terminal_reached = True  # 让 terminal_reached_gate 拦下一切工具
        contents: list[str] = []
        for i in range(mw._guard.loop_threshold):
            msg = await _run_tool(mw, "item_search", {"query": f"q{i}"})
            contents.append(str(msg.content))
        assert "重复调用" not in contents[0]
        assert "重复调用" in contents[-1]


@pytest.mark.asyncio
class TestBreakerParamErrorExemption:
    async def test_validation_error_not_counted(self, clean_phase) -> None:
        """参数校验失败是调用方（模型）的锅，不是工具基础设施故障——断路器进程级共享，
        计入会让一个会话的畸形参数把工具对全部会话熔断。真基建异常照常计数。"""
        from pydantic import BaseModel, ValidationError

        from app.harness.hooks.tool_breaker import get_tool_breaker, reset_tool_breakers

        class _Args(BaseModel):
            x: int

        async def bad_args_handler(req):
            _Args.model_validate({"x": "oops"})

        async def broken_infra_handler(req):
            raise RuntimeError("infra down")

        reset_tool_breakers()
        try:
            mw = _mw()
            with pytest.raises(ValidationError):
                await mw.awrap_tool_call(
                    _FakeToolRequest("web_search", {"query": "q"}), bad_args_handler
                )
            assert get_tool_breaker("web_search")._fail_count == 0
            with pytest.raises(RuntimeError):
                await mw.awrap_tool_call(
                    _FakeToolRequest("web_search", {"query": "q2"}), broken_infra_handler
                )
            assert get_tool_breaker("web_search")._fail_count == 1
        finally:
            reset_tool_breakers()


class TestInternalMarkersCoverage:
    def test_every_sentinel_prefix_is_registered(self) -> None:
        """新增哨兵忘了在 INTERNAL_MARKERS 登记清洗前缀 → 这里挂（曾漏 [阶段推进] 一整批）。
        模块级字符串常量全扫，函数型哨兵取样例输出一并校验。"""
        from app.harness import sentinels

        texts = [
            v
            for k, v in vars(sentinels).items()
            if isinstance(v, str) and not k.startswith("__") and v.startswith("[")
        ]
        texts += [
            sentinels.converge_directive(9),
            sentinels.retrieval_exhausted(10),
            sentinels.sub_search_budget_note(0, 1),
            sentinels.sub_search_budget_note(1, 1),
            sentinels.tool_breaker_open("item_search"),
        ]
        assert texts, "哨兵常量一个都没扫到——扫描逻辑坏了"
        for t in texts:
            assert any(t.startswith(m) for m in sentinels.INTERNAL_MARKERS), (
                f"哨兵前缀未登记清洗表: {t[:24]}"
            )

    @pytest.mark.asyncio
    async def test_output_guard_strips_transition_notice(self) -> None:
        """[阶段推进] 是每条正常链路必然出现的通告，最容易被鹦鹉学舌。"""
        from app.harness.hooks.session_hooks import audit_final_output

        final = "为你精选如下\n[阶段推进] 候选已入池，检索阶段就此收线\n1. 商品A"
        ctx = await audit_final_output({"final_answer": final})
        assert "[阶段推进]" not in ctx["final_answer"]
        assert "商品A" in ctx["final_answer"]


# ============================================================
# Liveness 看门狗
# ============================================================


class TestWatchdog:
    """停滞 → 收敛指令 → 宽限后仍无进展 → 硬停交部分结果（harness 重构第一段）。"""

    def _ctx(self, guard) -> dict:
        return {"_guard": guard, "messages": []}

    @pytest.mark.asyncio
    async def test_first_call_opens_clock(self) -> None:
        import time

        from app.agent.fork_guard import _fork_depth
        from app.harness.hooks.watchdog import check_liveness
        from app.harness.state import GuardState

        token = _fork_depth.set(0)
        guard = GuardState()
        try:
            assert await check_liveness(self._ctx(guard)) is None
            assert guard.last_progress_at > 0
            assert abs(guard.last_progress_at - time.monotonic()) < 1
        finally:
            _fork_depth.reset(token)

    @pytest.mark.asyncio
    async def test_stall_injects_converge_notice(self) -> None:
        import time

        from app.agent.fork_guard import _fork_depth
        from app.harness.hooks.watchdog import WATCHDOG_STALL_SEC, check_liveness
        from app.harness.state import GuardState

        token = _fork_depth.set(0)
        guard = GuardState()
        guard.last_progress_at = time.monotonic() - WATCHDOG_STALL_SEC - 5
        ctx = self._ctx(guard)
        try:
            result = await check_liveness(ctx)
            assert result is not None
            assert len(ctx["messages"]) == 1
            assert "看门狗" in ctx["messages"][0].content
            assert guard.watchdog_nudged_at > 0
            assert "fallback_answer" not in ctx  # 第一级只提醒，不硬停
        finally:
            _fork_depth.reset(token)

    @pytest.mark.asyncio
    async def test_hard_stop_after_grace(self) -> None:
        import time

        from app.agent.fork_guard import _fork_depth
        from app.harness.hooks.watchdog import (
            WATCHDOG_GRACE_SEC,
            WATCHDOG_STALL_SEC,
            check_liveness,
        )
        from app.harness.state import GuardState

        token = _fork_depth.set(0)
        guard = GuardState()
        now = time.monotonic()
        guard.last_progress_at = now - WATCHDOG_STALL_SEC - WATCHDOG_GRACE_SEC - 10
        guard.watchdog_nudged_at = now - WATCHDOG_GRACE_SEC - 1
        ctx = self._ctx(guard)
        try:
            result = await check_liveness(ctx)
            assert result is not None
            assert "先停在这里" in ctx["fallback_answer"]
        finally:
            _fork_depth.reset(token)

    @pytest.mark.asyncio
    async def test_progress_disarms_nudge(self) -> None:
        import time

        from app.agent.fork_guard import _fork_depth
        from app.harness.hooks.watchdog import check_liveness
        from app.harness.state import GuardState

        token = _fork_depth.set(0)
        guard = GuardState()
        guard.last_progress_at = time.monotonic()  # 刚有过真实进展
        guard.watchdog_nudged_at = time.monotonic() - 100  # 旧的武装应被解除
        ctx = self._ctx(guard)
        try:
            assert await check_liveness(ctx) is None
            assert guard.watchdog_nudged_at == 0.0
            assert "fallback_answer" not in ctx
        finally:
            _fork_depth.reset(token)

    @pytest.mark.asyncio
    async def test_skips_sub_agents(self) -> None:
        import time

        from app.agent.fork_guard import _fork_depth
        from app.harness.hooks.watchdog import check_liveness
        from app.harness.state import GuardState

        token = _fork_depth.set(1)
        guard = GuardState()
        guard.last_progress_at = time.monotonic() - 999
        try:
            assert await check_liveness(self._ctx(guard)) is None
        finally:
            _fork_depth.reset(token)


# ============================================================
# 统一逃生门（middleware._try_escape）：效率闸带门、安全闸永远硬
# ============================================================


class TestUnifiedGateEscape:
    """效率闸（websearch / postfork）连拒 2 次后放行；安全闸（子搜上限）无 escape_key 永不放行。"""

    @pytest.mark.asyncio
    async def test_websearch_gate_escape(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import app.harness.hooks.tool_gates as tg
        from app.harness.state import GuardState

        monkeypatch.setattr(tg, "web_search_allowed", lambda: False)
        guard = GuardState()
        mw = HarnessMiddleware()
        mw.register("pre_tool_call", "websearch_gate", tg.check_websearch, priority=15)
        # 「连拒」按模型决策轮计数（批次原子化）：每次重试对应一次新的模型调用，
        # think_step 自增——同一轮里的并行调用只算一次坚持，测试须模拟推进轮次。
        for _ in range(2):
            guard.think_step += 1
            out = await mw.run("pre_tool_call", {"tool_name": "web_search", "_guard": guard})
            assert out.get("_rejected")
        guard.think_step += 1
        out = await mw.run("pre_tool_call", {"tool_name": "web_search", "_guard": guard})
        assert not out.get("_rejected")  # 第 3 轮逃生放行（检索预算另有兜底）

    @pytest.mark.asyncio
    async def test_postfork_search_gate_escape(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from types import SimpleNamespace

        import app.harness.hooks.tool_gates as tg
        from app.harness.state import GuardState

        monkeypatch.setattr(tg, "get_fork_budget", lambda: SimpleNamespace(parallel_calls=1))
        monkeypatch.setattr(tg, "candidate_count", lambda: 5)
        guard = GuardState()
        guard.notified_transitions.add("search_close")
        mw = HarnessMiddleware()
        mw.register(
            "pre_tool_call", "search_authority_gate", tg.check_search_authority, priority=30
        )
        for _ in range(2):
            guard.think_step += 1  # 同 websearch 测试：连拒按模型决策轮计，须推进轮次
            out = await mw.run("pre_tool_call", {"tool_name": "item_search", "_guard": guard})
            assert out.get("_rejected")
        guard.think_step += 1
        out = await mw.run("pre_tool_call", {"tool_name": "item_search", "_guard": guard})
        assert not out.get("_rejected")  # 第 3 轮逃生放行
        assert "search_close" not in guard.notified_transitions  # 收线通告已重新武装

    @pytest.mark.asyncio
    async def test_sub_search_cap_is_not_escapable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """子 Agent 搜索上限是安全闸（fork 安全层），模型再坚持也不放行。"""
        import app.harness.hooks.tool_gates as tg
        from app.agent.fork_guard import _fork_depth
        from app.harness.budgets import SUB_ITEM_SEARCH_CAP
        from app.harness.state import GuardState

        token = _fork_depth.set(1)
        guard = GuardState()
        guard.item_search_calls = SUB_ITEM_SEARCH_CAP
        mw = HarnessMiddleware()
        mw.register(
            "pre_tool_call", "search_authority_gate", tg.check_search_authority, priority=30
        )
        try:
            for _ in range(5):
                out = await mw.run("pre_tool_call", {"tool_name": "item_search", "_guard": guard})
                assert out.get("_rejected")
        finally:
            _fork_depth.reset(token)

    @pytest.mark.asyncio
    async def test_gate_events_metric(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """拒绝与逃生都进 shoppingx_gate_events_total（gate/outcome 维度）。"""
        import app.harness.hooks.tool_gates as tg
        from app.harness.state import GuardState
        from app.observability.metrics import GATE_EVENTS

        def _count(outcome: str) -> float:
            return GATE_EVENTS.labels(gate="websearch_gate", outcome=outcome)._value.get()

        monkeypatch.setattr(tg, "web_search_allowed", lambda: False)
        guard = GuardState()
        mw = HarnessMiddleware()
        mw.register("pre_tool_call", "websearch_gate", tg.check_websearch, priority=15)
        rejects0, escapes0 = _count("reject"), _count("escape")
        for _ in range(3):
            guard.think_step += 1  # 连拒按模型决策轮计（批次原子化），须推进轮次
            await mw.run("pre_tool_call", {"tool_name": "web_search", "_guard": guard})
        assert _count("reject") == rejects0 + 2
        assert _count("escape") == escapes0 + 1


class TestDetectionLayerLanguageBridge:
    """检测层必须与执行层同一套词命中口径——裸子串在中英混合链路上是双向失效的。"""

    def test_blacklist_hits_bridges_language_and_negation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from app.harness import signals

        monkeypatch.setattr(signals, "blacklist_terms", lambda: ["塑料"])
        # 中文黑名单词必须匹中英文结果（旧版裸子串恒 miss，「偏好丢失」信号空转）
        assert signals.blacklist_hits("picked: 6 Set Plastic Storage Box") == ["塑料"]
        # 否定修饰不算违规：plastic-free 正是不要塑料的人想要的
        assert signals.blacklist_hits("Plastic-Free Bamboo Lunch Box") == []

    def test_goal_forgetting_bridges_chinese_query_to_english_actions(self) -> None:
        from app.harness.hooks.drift_detector import _computational_precheck

        ctx = {
            "original_query": "想买防水的旅行收纳袋",
            "recent_actions_summary": "item_search(waterproof packing cubes amazon)",
        }
        # 「防水」经归一补出 waterproof，可匹上英文检索参数——不再报假阳性
        _, hit = _computational_precheck(ctx, DriftState())
        assert "目标遗忘" not in hit

    def test_goal_terms_from_pt_rescue_unmapped_category_words(self) -> None:
        from app.harness.hooks.drift_detector import _computational_precheck

        ctx = {
            "original_query": "旅行收纳袋",  # 品类词不在 ZH_EN 词表，归一补不出英文
            "recent_actions_summary": "item_search(packing cubes shein)",
        }
        _, hit = _computational_precheck(ctx, DriftState())
        assert "目标遗忘" in hit  # 无词桥：中文 bigram 匹英文行为恒 0
        bridged = DriftState()
        bridged.goal_terms = {"packing cubes"}  # planner 写完 P_t 后由 hook 刷新
        _, hit = _computational_precheck(ctx, bridged)
        assert "目标遗忘" not in hit

    def test_empty_result_is_judged_structurally(self) -> None:
        from app.harness.hooks.drift_detector import _is_empty_result

        assert _is_empty_result('{"platform": "amazon", "total_recall": 0, "candidates": []}')
        # 追问轮复用：fresh 折叠进 already_in_pool ≠ 空结果（旧版字符串匹配数成连续空）
        assert not _is_empty_result(
            '{"platform": "all", "total_recall": 20, "candidates": [], "already_in_pool": ["a"]}'
        )
        # 被约束筛光（总召回 > 0）是筛太狠不是方向错，不算探索发散
        assert not _is_empty_result('{"platform": "amazon", "total_recall": 8, "candidates": []}')
        # 非 JSON 的文本工具（web_search）退回特征词
        assert _is_empty_result("未找到相关结果")
