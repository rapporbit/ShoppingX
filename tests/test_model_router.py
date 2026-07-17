"""四档模型路由降级的确定性单测（refdocs 16-4 §3 / §6）。

用可控的 usage_metadata + 已知费率把全树成本推到各个档位，断言：分档边界、降档只上报一次
metric、minimal 档收走成本放大器工具、fallback 档不调 LLM 且不编造商品。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from app.agent import model_router as mr
from app.agent import token_budget as tb
from app.agent.model_router import Tier
from app.harness.hooks.context_compress import route_by_budget
from app.harness.hooks.tool_gates import check_token_budget
from app.harness.middleware import HookRejectSignal
from app.harness.state import GuardState
from app.tools.schemas import ItemCandidate
from app.utils.thread_ctx import thread_scope


@pytest.fixture
def budget_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """费率固定成「1 美元 = 1M input token」，预算上限 1 美元 → 花掉 N% 的 token 即耗掉 N% 预算。"""
    monkeypatch.setenv("TOKEN_PRICE_INPUT", "1.0")
    monkeypatch.setenv("TOKEN_PRICE_OUTPUT", "0.0")
    monkeypatch.setenv("TOKEN_PRICE_CACHE_READ", "0.0")
    monkeypatch.setenv("TOKEN_BUDGET_USD", "1.0")


def _spend(fraction: float, mid: str) -> None:
    """烧掉预算的 ``fraction`` 比例。"""
    tb.charge_tree_usage(
        [
            AIMessage(
                content="",
                id=mid,
                usage_metadata={
                    "input_tokens": int(1_000_000 * fraction),
                    "output_tokens": 0,
                    "total_tokens": int(1_000_000 * fraction),
                    "input_token_details": {"cache_read": 0},
                },
            )
        ]
    )


# ============================================================
# 分档
# ============================================================


class TestTierBoundaries:
    def test_no_budget_scope_is_main(self) -> None:
        """无 session 作用域（单测 / 离线脚本）绝不能凭空降级。"""
        assert mr.current_tier() is Tier.MAIN

    def test_budget_disabled_is_main(self, tmp_path: Path, monkeypatch: Any) -> None:
        monkeypatch.setenv("TOKEN_BUDGET_USD", "0")
        with thread_scope("t", tmp_path):
            tb.reset_tree()
            _spend(10.0, "m1")  # 花爆也不降级——没配预算 = 预算充裕
            assert mr.current_tier() is Tier.MAIN
            tb.reset_tree()

    def test_tiers_step_down_as_budget_burns(self, tmp_path: Path, budget_env: None) -> None:
        with thread_scope("t", tmp_path):
            tb.reset_tree()
            assert mr.current_tier() is Tier.MAIN  # 剩 100%

            _spend(0.40, "m1")  # 剩 60% > 50%
            assert mr.current_tier() is Tier.MAIN

            _spend(0.15, "m2")  # 剩 45% ∈ (20%, 50%]
            assert mr.current_tier() is Tier.LITE

            _spend(0.30, "m3")  # 剩 15% ∈ (5%, 20%]
            assert mr.current_tier() is Tier.MINIMAL

            _spend(0.12, "m4")  # 剩 3% ≤ 5%
            assert mr.current_tier() is Tier.FALLBACK
            tb.reset_tree()

    def test_tier_is_ordered(self) -> None:
        """IntEnum 可比大小——``tier >= Tier.MINIMAL`` 是预算闸的判据。"""
        assert Tier.MAIN < Tier.LITE < Tier.MINIMAL < Tier.FALLBACK

    def test_remaining_ratio_floors_at_zero(self, tmp_path: Path, budget_env: None) -> None:
        with thread_scope("t", tmp_path):
            tb.reset_tree()
            _spend(3.0, "m1")  # 花掉 300% 预算
            assert tb.remaining_ratio() == 0.0  # 不为负
            tb.reset_tree()


# ============================================================
# tier_model：换哪个模型
# ============================================================


class TestTierModel:
    def test_main_and_fallback_have_no_override(self) -> None:
        """MAIN=不覆盖（用原模型）；FALLBACK=根本不调 LLM。两者都返回 None。"""
        assert mr.tier_model(Tier.MAIN) is None
        assert mr.tier_model(Tier.FALLBACK) is None

    def test_lite_and_minimal_use_cheap_model(self, monkeypatch: Any) -> None:
        sentinel = object()
        monkeypatch.setattr(mr, "_lite_llm", lambda: sentinel)
        assert mr.tier_model(Tier.LITE) is sentinel
        assert mr.tier_model(Tier.MINIMAL) is sentinel


# ============================================================
# fallback 兜底回答
# ============================================================


class TestFallbackAnswer:
    def test_uses_real_candidates(self, monkeypatch: Any) -> None:
        cands = [
            ItemCandidate(item_id="a1", platform="amazon", title="Travel Cube Set", price_usd=29.9),
            ItemCandidate(
                item_id="s1", platform="shein", title="Packing Organizer", price_usd=12.5
            ),
        ]
        monkeypatch.setattr(mr, "_top_candidates", lambda n: cands[:n])
        answer = mr.build_fallback_answer("买旅行三件套")
        assert "Travel Cube Set" in answer
        assert "$29.90" in answer
        assert "amazon" in answer

    def test_no_candidates_never_fabricates(self, monkeypatch: Any) -> None:
        """连候选都没有时，宁可如实说没有——绝不编造商品（system prompt 的 P0 红线）。"""
        monkeypatch.setattr(mr, "_top_candidates", lambda n: [])
        answer = mr.build_fallback_answer("买旅行三件套")
        assert "编造商品不是选项" in answer
        assert "买旅行三件套" in answer

    def test_candidate_without_price_does_not_crash(self, monkeypatch: Any) -> None:
        cands = [ItemCandidate(item_id="a1", platform="walmart", title="Bag", price_usd=None)]
        monkeypatch.setattr(mr, "_top_candidates", lambda n: cands)
        assert "价格待确认" in mr.build_fallback_answer("q")

    def test_top_candidates_without_session_returns_empty(self) -> None:
        assert mr._top_candidates(3) == []

    def test_minimal_hint_is_strippable_by_output_guard(self) -> None:
        """MINIMAL_HINT 可能被模型抄进最终回答——它的前缀必须在内部文案标记表里。"""
        from app.harness.hooks.session_hooks import _INTERNAL_MARKERS

        assert any(mr.MINIMAL_HINT.startswith(marker) for marker in _INTERNAL_MARKERS)


# ============================================================
# budget_router Hook：决策写进 context，适配器执行
# ============================================================


class TestBudgetRouterHook:
    @pytest.mark.asyncio
    async def test_main_tier_does_nothing(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(mr, "current_tier", lambda: Tier.MAIN)
        assert await route_by_budget({"_guard": GuardState(), "messages": []}) is None

    @pytest.mark.asyncio
    async def test_lite_tier_sets_model_override_without_hint(self, monkeypatch: Any) -> None:
        sentinel = object()
        monkeypatch.setattr(mr, "current_tier", lambda: Tier.LITE)
        monkeypatch.setattr(mr, "tier_model", lambda t: sentinel)
        ctx: dict[str, Any] = {"_guard": GuardState(), "messages": []}
        out = await route_by_budget(ctx)
        assert out is not None
        assert out["model_override"] is sentinel
        assert out["messages"] == []  # lite 档不注入 hint，只是悄悄换个便宜模型

    @pytest.mark.asyncio
    async def test_minimal_tier_injects_hint_and_switches_model(self, monkeypatch: Any) -> None:
        sentinel = object()
        monkeypatch.setattr(mr, "current_tier", lambda: Tier.MINIMAL)
        monkeypatch.setattr(mr, "tier_model", lambda t: sentinel)
        ctx: dict[str, Any] = {"_guard": GuardState(), "messages": []}
        out = await route_by_budget(ctx)
        assert out is not None
        assert out["model_override"] is sentinel
        assert len(out["messages"]) == 1
        assert "预算提醒" in out["messages"][0].content
        # hint 须同步登记 persist_messages（随 ModelResponse 落 state，不然只活一轮还斩缓存链）
        assert out["persist_messages"] == out["messages"]

    @pytest.mark.asyncio
    async def test_minimal_hint_injected_once_per_entry(self, monkeypatch: Any) -> None:
        """hint 落 state 后长驻历史——留在 minimal 档的后续轮次不再重复注入。"""
        sentinel = object()
        monkeypatch.setattr(mr, "current_tier", lambda: Tier.MINIMAL)
        monkeypatch.setattr(mr, "tier_model", lambda t: sentinel)
        guard = GuardState()
        first: dict[str, Any] = {"_guard": guard, "messages": []}
        await route_by_budget(first)
        assert len(first["messages"]) == 1
        second: dict[str, Any] = {"_guard": guard, "messages": []}
        out = await route_by_budget(second)
        assert out is not None
        assert out["model_override"] is sentinel  # 换模型每轮都要（override 不落 state）
        assert second["messages"] == []  # hint 不重复

    @pytest.mark.asyncio
    async def test_fallback_tier_yields_answer_and_no_model(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(mr, "current_tier", lambda: Tier.FALLBACK)
        monkeypatch.setattr(mr, "build_fallback_answer", lambda q: f"兜底:{q}")
        ctx: dict[str, Any] = {"_guard": GuardState(), "messages": [], "original_query": "买包"}
        out = await route_by_budget(ctx)
        assert out is not None
        assert out["fallback_answer"] == "兜底:买包"
        assert "model_override" not in out  # 根本不调 LLM，谈何换模型

    @pytest.mark.asyncio
    async def test_tier_change_reported_once_per_tier(self, monkeypatch: Any) -> None:
        """一个 20 轮的任务不去重会把 minimal 记 15 次，降级率统计直接失真。"""
        recorded: list[str] = []
        monkeypatch.setattr(mr, "current_tier", lambda: Tier.LITE)
        monkeypatch.setattr(mr, "tier_model", lambda t: object())
        monkeypatch.setattr(
            "app.harness.hooks.context_compress.metrics.record_tier_change", recorded.append
        )
        guard = GuardState()
        for _ in range(3):
            await route_by_budget({"_guard": guard, "messages": []})
        assert recorded == ["lite"]
        assert guard.last_tier == "lite"


# ============================================================
# 预算闸：minimal 档就开始收走成本放大器工具
# ============================================================


class TestAdapterWiring:
    """Hook 决策 → 适配器执行的接线（Hook 逻辑对，不代表它在真实 Agent 生命周期里生效过）。"""

    @staticmethod
    def _fake_request() -> Any:
        class _Req:
            def __init__(self) -> None:
                self.messages: list[Any] = []
                self.system_message = None
                self.tools: list[Any] = []
                self.model: Any = "原始模型"

            def override(self, **kw: Any) -> Any:
                out = _Req()
                out.messages = kw.get("messages", self.messages)
                out.system_message = kw.get("system_message", self.system_message)
                out.model = kw.get("model", self.model)
                return out

        return _Req()

    @pytest.mark.asyncio
    async def test_fallback_skips_the_model_call_entirely(self, monkeypatch: Any) -> None:
        """FALLBACK 档的全部意义：一次 LLM 都不调。handler 被调用即为失败。"""
        from app.harness.agent_middleware import HarnessAgentMiddleware
        from app.harness.setup import setup_harness

        setup_harness()
        monkeypatch.setattr(mr, "current_tier", lambda: Tier.FALLBACK)
        monkeypatch.setattr(mr, "build_fallback_answer", lambda q: "已用尽预算的兜底清单")

        calls: list[Any] = []

        async def handler(req: Any) -> Any:
            calls.append(req)
            raise AssertionError("fallback 档不该调用模型")

        mw = HarnessAgentMiddleware(original_query="买包")
        resp = await mw.awrap_model_call(self._fake_request(), handler)

        assert calls == []
        assert resp.result[0].content == "已用尽预算的兜底清单"
        assert not resp.result[0].tool_calls  # 无 tool_calls → AgentLoop 自然终止
        # 置位终结标记：万一 loop 还想调工具，terminal_reached_gate 会拦下
        assert mw._guard.terminal_reached is True

    @pytest.mark.asyncio
    async def test_lite_tier_overrides_the_model_on_request(self, monkeypatch: Any) -> None:
        from langchain_core.messages import AIMessage as _AI

        from app.harness.agent_middleware import HarnessAgentMiddleware
        from app.harness.setup import setup_harness

        setup_harness()
        sentinel = object()
        monkeypatch.setattr(mr, "current_tier", lambda: Tier.LITE)
        monkeypatch.setattr(mr, "tier_model", lambda t: sentinel)

        seen: list[Any] = []

        async def handler(req: Any) -> Any:
            seen.append(req.model)
            return _AI(content="ok")

        mw = HarnessAgentMiddleware(original_query="买包")
        await mw.awrap_model_call(self._fake_request(), handler)
        assert seen == [sentinel]  # 降档模型真的传到了模型调用上

    @pytest.mark.asyncio
    async def test_main_tier_leaves_the_model_untouched(self, monkeypatch: Any) -> None:
        from langchain_core.messages import AIMessage as _AI

        import app.harness.hooks.reasoning_boost as rb
        from app.harness.agent_middleware import HarnessAgentMiddleware
        from app.harness.setup import setup_harness

        setup_harness()
        monkeypatch.setattr(mr, "current_tier", lambda: Tier.MAIN)
        # 关掉「主 loop 第一轮开 reasoning」——它同样经 model_override 落地，会盖住本例要断言的
        # 「预算 MAIN 档不碰模型」。两者的优先级协作另有专测（tests/test_reasoning_boost.py）。
        monkeypatch.setattr(rb, "BOOST_ENABLED", False)

        seen: list[Any] = []

        async def handler(req: Any) -> Any:
            seen.append(req.model)
            return _AI(content="ok")

        mw = HarnessAgentMiddleware(original_query="买包")
        await mw.awrap_model_call(self._fake_request(), handler)
        assert seen == ["原始模型"]


class TestTokenBudgetGate:
    @pytest.mark.asyncio
    async def test_main_tier_allows_expensive_tools(self, monkeypatch: Any) -> None:
        monkeypatch.setattr("app.harness.hooks.tool_gates.current_tier", lambda: Tier.MAIN)
        assert await check_token_budget({"tool_name": "item_search"}) is None

    @pytest.mark.asyncio
    async def test_lite_tier_still_allows_expensive_tools(self, monkeypatch: Any) -> None:
        """lite 只是换个便宜模型，不该影响任务能力——收权从 minimal 才开始。"""
        monkeypatch.setattr("app.harness.hooks.tool_gates.current_tier", lambda: Tier.LITE)
        assert await check_token_budget({"tool_name": "item_search"}) is None

    @pytest.mark.asyncio
    async def test_minimal_tier_blocks_cost_amplifiers(self, monkeypatch: Any) -> None:
        """从 minimal 就拦，而不是等撞线——撞线时连收尾用的 shopping_summary 都付不起了。"""
        monkeypatch.setattr("app.harness.hooks.tool_gates.current_tier", lambda: Tier.MINIMAL)
        with pytest.raises(HookRejectSignal):
            await check_token_budget({"tool_name": "item_search"})

    @pytest.mark.asyncio
    async def test_minimal_tier_keeps_terminal_tools(self, monkeypatch: Any) -> None:
        """收尾链必须留着，否则任务硬停、已收敛的候选全丢。"""
        monkeypatch.setattr("app.harness.hooks.tool_gates.current_tier", lambda: Tier.MINIMAL)
        assert await check_token_budget({"tool_name": "shopping_summary"}) is None
        assert await check_token_budget({"tool_name": "item_picker"}) is None
