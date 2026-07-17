"""只给主 loop 第一轮开 reasoning（其余轮次维持基座的快档）。

三条边界（每条都对应一个真会犯的错）：
1. 第一轮 override 成 reasoning 模型 —— 编排决策轮值得想清楚；
2. 第二轮起不 override —— 那几轮的决策空间被阶段机 + id 化夹死，thinking 买不到东西；
3. 子 loop（depth ≥ 1）不插手 —— 子只按 demands 搜一个平台，没有编排可言，恒为快档。
"""

from __future__ import annotations

from typing import Any

import pytest

import app.harness.hooks.reasoning_boost as rb


@pytest.fixture(autouse=True)
def _enable_boost(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rb, "BOOST_ENABLED", True)


def _ctx(round_number: int) -> dict[str, Any]:
    return {"round_number": round_number, "original_query": "买个耐用的旅行包"}


@pytest.mark.asyncio
async def test_first_round_gets_reasoning(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()
    monkeypatch.setattr(rb, "get_llm", lambda: sentinel)
    out = await rb.boost_first_round(_ctx(1))
    assert out is not None
    assert out["model_override"] is sentinel


@pytest.mark.asyncio
async def test_later_rounds_stay_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rb, "get_llm", lambda: object())
    for rnd in (2, 3, 7):
        assert await rb.boost_first_round(_ctx(rnd)) is None


@pytest.mark.asyncio
async def test_sub_agent_first_round_stays_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    """子 loop 也有自己的 round_number=1，但它没有编排决策——不给它开思考。"""
    monkeypatch.setattr(rb, "get_llm", lambda: object())
    monkeypatch.setattr(rb, "current_fork_depth", lambda: 1)
    assert await rb.boost_first_round(_ctx(1)) is None


@pytest.mark.asyncio
async def test_disabled_by_env_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """闸关掉 → 回到「全程零 reasoning」，用于 A/B 对照。"""
    monkeypatch.setattr(rb, "BOOST_ENABLED", False)
    monkeypatch.setattr(rb, "get_llm", lambda: object())
    assert await rb.boost_first_round(_ctx(1)) is None


@pytest.mark.asyncio
async def test_budget_router_wins_over_boost(monkeypatch: pytest.MonkeyPatch) -> None:
    """预算见底时，便宜档模型必须盖掉第一轮的 reasoning——钱不够就别想了，先跑完。

    靠 priority 排序保证：reasoning_boost(10) 先写，budget_router(20) 后写覆盖。这里直接按
    Pipeline 真实顺序跑一遍 pre_think，验证最终留在 context 里的是便宜档那个。
    """
    import app.agent.model_router as mr
    from app.harness.middleware import harness
    from app.harness.setup import setup_harness

    setup_harness()
    cheap = object()
    boosted = object()
    monkeypatch.setattr(rb, "get_llm", lambda: boosted)
    monkeypatch.setattr(mr, "current_tier", lambda: mr.Tier.LITE)
    monkeypatch.setattr(mr, "tier_model", lambda _tier: cheap)

    ctx = await harness.run("pre_think", {**_ctx(1), "messages": [], "system_message": None})

    assert ctx["model_override"] is cheap


@pytest.mark.asyncio
async def test_reuse_turn_skips_reasoning(monkeypatch: pytest.MonkeyPatch) -> None:
    """复用轮（planner 判 reuse）连第一轮也不开思考——plan 已写死「不检索，直接精挑」。

    第一轮之所以值得开 reasoning，是因为它要在「单平台直搜 / 跨平台 fork / 先查品类常识」之间做
    选择；reuse 轮这些分支一个都不在，那些 thinking token 买不到东西。
    """
    monkeypatch.setattr(rb, "get_llm", lambda: object())
    monkeypatch.setattr(rb, "get_retrieval_mode", lambda: "reuse")
    assert await rb.boost_first_round(_ctx(1)) is None


@pytest.mark.asyncio
async def test_search_turn_still_boosts(monkeypatch: pytest.MonkeyPatch) -> None:
    """护栏：只有 reuse 轮豁免。search / augment（要真检索、有编排可选）照常开思考。

    planner 预置降级时 get_retrieval_mode() 也返回默认的 search → 落在安全侧（照常开）。
    """
    sentinel = object()
    monkeypatch.setattr(rb, "get_llm", lambda: sentinel)
    for mode in ("search", "augment"):
        monkeypatch.setattr(rb, "get_retrieval_mode", lambda m=mode: m)
        out = await rb.boost_first_round(_ctx(1))
        assert out is not None and out["model_override"] is sentinel
