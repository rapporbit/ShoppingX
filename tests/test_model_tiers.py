"""档位策略表：哪一轮用哪一档，装配与换档读同一份取值。

这批测试替代原 ``test_reasoning_boost.py``。**断言的是档位标签，不是模型对象身份**——旧测试
断的是「第一轮 override 成 get_llm() 返回的那个 sentinel」，而线上恰恰是基座本来就是 get_llm()、
override 成同一个实例、什么都没发生（审查报告 P0-1）。sentinel 相等的断言在那种情况下照样绿，
所以那种写法根本测不出这个 bug。改成「基座档 ≠ 第一轮档」这条关系式，脱钩就会红。
"""

from __future__ import annotations

from typing import Any

import pytest

from app.agent import llm
from app.harness import adapter


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """三个档位键都清掉，让每个用例只受自己 setenv 的那条影响。"""
    for key in ("MAIN_LOOP_TIER_BASE", "MAIN_LOOP_TIER_FIRST", "WORKER_TIER"):
        monkeypatch.delenv(key, raising=False)


def _ctx(round_number: int = 1) -> dict[str, Any]:
    return {"round_number": round_number, "original_query": "买个耐用的旅行包"}


# ── 档位解析 ──


def test_defaults_are_all_fast_no_thinking() -> None:
    """默认口径：**全程零思考**——基座、第一轮、worker 三处都是快档。

    ``MAIN_LOOP_TIER_FIRST=same`` 是 2026-09-09 用户的决定：先把 thinking 整体固定为 off，
    再去动模型组合（变量隔离）。改这三个默认值等于改延迟基线，改之前先重跑基线。
    """
    assert llm.main_loop_tier_base() == "fast"
    assert llm.main_loop_tier_first() == "same"
    assert llm.worker_tier() == "fast"


def test_same_means_no_switch_at_all() -> None:
    """``same`` 下第一轮也不换档——全程就是基座那一档，一次 override 都不发生。"""
    assert adapter._first_round_tier(_ctx(1)) is None
    assert adapter._first_round_tier(_ctx(2)) is None


def test_tier_names_map_to_distinct_factories(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "get_llm", lambda: "REASONING")
    monkeypatch.setattr(llm, "get_fast_llm", lambda: "FAST")
    monkeypatch.setattr(llm, "get_lite_llm", lambda: "LITE")
    assert llm.get_tier_llm("reasoning") == "REASONING"
    assert llm.get_tier_llm("fast") == "FAST"
    assert llm.get_tier_llm("lite") == "LITE"
    assert llm.get_tier_llm("  FAST ") == "FAST"  # 大小写与空白容错


def test_unknown_tier_falls_back_to_reasoning(monkeypatch: pytest.MonkeyPatch) -> None:
    """拼错档位名不抛也不静默降智：回退到能力更强那档，代价是钱不是质量。"""
    monkeypatch.setattr(llm, "get_llm", lambda: "REASONING")
    assert llm.get_tier_llm("resoning") == "REASONING"


# ── 第一轮加档 ──
#
# 下面这批**必须显式 `MAIN_LOOP_TIER_FIRST=reasoning`**（`_boost_on` fixture）。默认已是
# `same`，不显式开的话 `_first_round_tier` 在第一个条件上就早退，每一条断言 `is None` 的用例
# 都会「因为加档整个关着」而通过——测的是关关着，不是豁免逻辑。这类假绿正是 P0-1 的病根。


@pytest.fixture
def _boost_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAIN_LOOP_TIER_FIRST", "reasoning")


def test_first_round_gets_boosted(_boost_on: None) -> None:
    assert adapter._first_round_tier(_ctx(1)) == "reasoning"


def test_later_rounds_stay_on_base(_boost_on: None) -> None:
    """第 2 轮起回基座档——这条一旦失守，主 loop 每轮都在付 thinking 解码。"""
    for rnd in (2, 3, 7):
        assert adapter._first_round_tier(_ctx(rnd)) is None


def test_worker_first_round_not_boosted(_boost_on: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """worker 也有自己的 round_number=1，但它只按 demands 搜一个平台，没有编排可言。"""
    monkeypatch.setattr("app.agent.fork_guard.current_fork_depth", lambda: 1)
    assert adapter._first_round_tier(_ctx(1)) is None


def test_reuse_turn_not_boosted(_boost_on: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """复用轮：plan 已写死「不检索，直接精挑」，第一轮值得开思考的分支一个都不在。"""
    monkeypatch.setattr("app.api.context.get_retrieval_mode", lambda: "reuse")
    assert adapter._first_round_tier(_ctx(1)) is None


def test_search_and_augment_turns_still_boosted(
    _boost_on: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """护栏：只有 reuse 豁免。planner 预置降级时读到默认 search → 照常加档（安全侧）。"""
    for mode in ("search", "augment"):
        monkeypatch.setattr("app.api.context.get_retrieval_mode", lambda m=mode: m)
        assert adapter._first_round_tier(_ctx(1)) == "reasoning"


def test_boost_is_noop_when_base_already_equals_first(
    _boost_on: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**P0-1 的回归**：基座被改成 reasoning 时，第一轮不该再「加」一次。

    历史上这正是 bug 的形态——基座 = reasoning、boost 也顶 reasoning，override 成同一个实例，
    看着在换档其实什么都没发生，还掩盖了「每轮都开思考」的口径倒退。
    """
    monkeypatch.setenv("MAIN_LOOP_TIER_BASE", "reasoning")
    assert adapter._first_round_tier(_ctx(1)) is None


def test_budget_downgrade_wins_over_boost(_boost_on: None) -> None:
    """预算见底时便宜档必须盖掉第一轮加档——钱不够就别想了，先跑完。

    协作靠 ``on_model_call`` 里的 ``ctx.get("model_tier") or _first_round_tier(ctx)``：Hook
    写过档就轮不到加档。这里直接按那条表达式验一遍。
    """
    ctx = {**_ctx(1), "model_tier": "lite"}
    assert (ctx.get("model_tier") or adapter._first_round_tier(ctx)) == "lite"
