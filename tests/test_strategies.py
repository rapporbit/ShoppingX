"""成功策略沉淀（批 4 / 18-4）：匹配 / 生命周期 / system prompt 注入位 / 收尾结账。

这里钉的都是**静默失效**类的不变式——策略机制坏掉时不会报错，只会「策略没生效」或者
「记在了错的策略头上」，两者都零日志、零红灯。所以每条用例对应一个具体的错法：

- 匹配用裸 ``in`` → ``bag`` 命中 ``baggage``，策略在错的轮次注入。
- 注入位允许改写整段 prompt → 有人顺手删掉 ``<termination>``，Agent 开始不收尾。
- 注入清单不每轮重置 → 第二轮拿上一轮的清单结账，账记到错的策略头上。
- 连续失败计数不清零 → 攒够三次偶发限流就把一条好策略淘汰了。
"""

from __future__ import annotations

import pytest

from app.memory.strategies import (
    MAX_HEALTH,
    Strategy,
    force_strategies,
    get_strategy_store,
    make_slug,
    match_strategies,
    render_strategy_block,
    strategies_for_query,
)


def _s(slug: str, *, keywords: list[str], category: str = "预算陷阱", health: int = 3) -> Strategy:
    return Strategy(
        category=category,
        slug=slug,
        trigger=f"触发-{slug}",
        trigger_keywords=keywords,
        actions=[f"动作-{slug}"],
        health=health,
    )


# ── 身份与渲染 ────────────────────────────────────────────────────────────────────


def test_dedup_key_is_derived_not_handwritten() -> None:
    """去重身份由 category + slug 派生；slug 缺失时从 trigger 兜底（不会出现空身份）。"""
    assert _s("landed_cost", keywords=["预算"]).dedup_key == "预算陷阱:landed_cost"
    bare = Strategy(category="c", trigger="先算到手价 Then Sort")
    assert bare.dedup_key == "c:then_sort"  # 中文被剥掉、首尾下划线被 strip
    assert bare.slug == ""  # 派生不回写字段——身份是算出来的，不是存两份


def test_make_slug_normalizes_llm_freeform() -> None:
    assert make_slug("Landed Cost First!") == "landed_cost_first"
    assert make_slug("  ") == "unnamed"  # 全被剥掉也要有身份，否则 dedup_key 变成 "c:"


def test_render_block_empty_gives_empty_string() -> None:
    """一条都没匹配上就不塞空占位——省 token，也不给模型「有个空区块」的噪声。"""
    assert render_strategy_block([]) == ""
    block = render_strategy_block([_s("a", keywords=["预算"])])
    assert "<learned_strategies>" in block and "动作-a" in block
    assert "低于" in block  # 必须写明优先级低于 <constraints>，否则策略会被拿去绕硬约束


# ── 匹配（确定性，零 LLM）────────────────────────────────────────────────────────


def test_match_uses_word_boundary_not_substring() -> None:
    """``bag`` 不该命中 ``baggage``——裸 ``in`` 会让策略在完全无关的轮次注入。"""
    s = _s("bagcare", keywords=["bag"])
    assert match_strategies("recommend a baggage allowance guide", [s]) == []
    assert match_strategies("recommend a bag", [s]) == [s]


def test_match_skips_retired_and_respects_limit() -> None:
    live = [_s(f"k{i}", keywords=["预算"]) for i in range(5)]
    dead = _s("dead", keywords=["预算"])
    dead.status = "retired"
    got = match_strategies("预算 300", [*live, dead], limit=2)
    assert len(got) == 2
    assert all(g.status == "active" for g in got)


def test_match_order_is_stable_across_pool_order() -> None:
    """同分策略的注入顺序必须与库的返回顺序无关——否则 system prompt 尾巴每轮都在抖，
    前缀缓存跟着失效（本模块 docstring 里那笔账的直接依赖）。"""
    pool = [_s("b", keywords=["预算"]), _s("a", keywords=["预算"]), _s("c", keywords=["预算"])]
    first = [s.dedup_key for s in match_strategies("预算 300", pool)]
    second = [s.dedup_key for s in match_strategies("预算 300", list(reversed(pool)))]
    assert first == second


def test_match_ranks_more_hits_then_healthier() -> None:
    two_hits = _s("two", keywords=["预算", "便宜"], health=1)
    one_hit_healthy = _s("one", keywords=["预算"], health=3)
    got = match_strategies("预算 300 要便宜的", [one_hit_healthy, two_hits])
    assert [s.slug for s in got] == ["two", "one"]  # 命中词数优先于血量


def test_empty_query_matches_nothing() -> None:
    assert match_strategies("", [_s("a", keywords=["预算"])]) == []


# ── 生命周期：命中回血 / 连续失败淘汰 ─────────────────────────────────────────────


@pytest.mark.anyio
async def test_store_roundtrip_and_active_filter() -> None:
    store = get_strategy_store()
    await store.upsert(_s("alive", keywords=["预算"]))
    dead = _s("dead", keywords=["预算"])
    await store.upsert(dead)
    await store.record_outcome([dead.dedup_key], success=False)
    await store.record_outcome([dead.dedup_key], success=False)
    await store.record_outcome([dead.dedup_key], success=False)

    active = {s.dedup_key for s in await store.read_active()}
    assert active == {"预算陷阱:alive"}
    assert len(await store.read_all()) == 2  # 退休不是删除，出处与账要留着


@pytest.mark.anyio
async def test_success_resets_consecutive_failures() -> None:
    """**连续**失败才淘汰：中间成功一次就清零。偶发限流 / 网关 5xx 攒够三次不该判死一条策略。"""
    store = get_strategy_store()
    s = _s("flaky", keywords=["预算"])
    await store.upsert(s)
    await store.record_outcome([s.dedup_key], success=False)
    await store.record_outcome([s.dedup_key], success=False)
    await store.record_outcome([s.dedup_key], success=True)  # 回血 + 清零
    retired = await store.record_outcome([s.dedup_key], success=False)
    assert retired == []
    row = next(x for x in await store.read_all() if x.dedup_key == s.dedup_key)
    assert row.status == "active" and row.consecutive_failures == 1
    assert row.hits == 4  # hits 记「上过几次场」，成败都算


@pytest.mark.anyio
async def test_health_is_capped_and_hits_accumulate() -> None:
    store = get_strategy_store()
    s = _s("cap", keywords=["预算"])
    await store.upsert(s)
    for _ in range(5):
        await store.record_outcome([s.dedup_key], success=True)
    row = next(x for x in await store.read_all() if x.dedup_key == s.dedup_key)
    assert row.health == MAX_HEALTH  # 回血不会把血量刷到天上去
    assert row.hits == 5


@pytest.mark.anyio
async def test_reupsert_revives_retired_strategy() -> None:
    """重新蒸馏出同一条 = 它刚又过了一次门禁，该复活；否则这套机制只会单向减少。"""
    store = get_strategy_store()
    s = _s("revive", keywords=["预算"])
    await store.upsert(s)
    for _ in range(3):
        await store.record_outcome([s.dedup_key], success=False)
    assert not await store.read_active()

    await store.upsert(_s("revive", keywords=["预算", "便宜"]))
    row = next(iter(await store.read_active()))
    assert row.status == "active" and row.health == MAX_HEALTH
    assert row.consecutive_failures == 0 and row.trigger_keywords == ["预算", "便宜"]


@pytest.mark.anyio
async def test_record_outcome_on_unknown_key_is_noop() -> None:
    assert await get_strategy_store().record_outcome(["不存在:x"], success=False) == []
    assert await get_strategy_store().record_outcome([], success=True) == []


# ── 注入位（on_system_prompt）与结账位（on_session_end）───────────────────────────


@pytest.mark.anyio
async def test_force_strategies_bypasses_store_and_resets() -> None:
    """门禁重放靠强制态注入未入库的候选；出了作用域必须回到读库，否则重放会污染线上路径。"""
    await get_strategy_store().upsert(_s("in_db", keywords=["预算"]))
    with force_strategies([_s("candidate", keywords=["预算"])]):
        assert [s.slug for s in await strategies_for_query("预算 300")] == ["candidate"]
    assert [s.slug for s in await strategies_for_query("预算 300")] == ["in_db"]


@pytest.mark.anyio
async def test_hook_appends_to_system_prompt_without_touching_prefix() -> None:
    """注入只许**追加**：原 prompt 逐字仍是新 prompt 的前缀。

    这既是缓存前缀那笔账的前提，也是安全线——钩子若能改写整段，谁都能悄悄删掉 ``<termination>``。
    """
    from app.agent.agents import _run_system_prompt_hooks
    from app.agent.prompts import get_system_prompt

    base = get_system_prompt()
    await get_strategy_store().upsert(_s("cheap_first", keywords=["预算"]))
    got = await _run_system_prompt_hooks(base, role="main", query="预算 300 买个包")
    assert got.startswith(base)
    assert "<learned_strategies>" in got and "动作-cheap_first" in got


@pytest.mark.anyio
async def test_hook_is_noop_for_workers_and_for_unmatched_query() -> None:
    from app.agent.agents import _run_system_prompt_hooks

    await get_strategy_store().upsert(_s("cheap_first", keywords=["预算"]))
    # worker 跑的是收窄后的子任务，主 loop 的打法塞进去只会稀释它那段专职 prompt
    assert await _run_system_prompt_hooks("BASE", role="search", query="预算 300") == "BASE"
    # 匹配不上就一个字都不加（不塞空区块）
    assert await _run_system_prompt_hooks("BASE", role="main", query="今天天气怎么样") == "BASE"


@pytest.mark.anyio
async def test_injected_list_is_rewritten_every_turn() -> None:
    """每轮必写注入清单，哪怕是空——否则第二轮会拿上一轮的清单去结账，账记到错的策略头上。"""
    from app.agent.agents import _run_system_prompt_hooks
    from app.harness.hooks.strategy_inject import injected_strategy_keys

    await get_strategy_store().upsert(_s("cheap_first", keywords=["预算"]))
    await _run_system_prompt_hooks("BASE", role="main", query="预算 300")
    assert injected_strategy_keys() == ("预算陷阱:cheap_first",)
    await _run_system_prompt_hooks("BASE", role="main", query="今天天气怎么样")
    assert injected_strategy_keys() == ()


@pytest.mark.anyio
async def test_session_end_settles_by_terminal_tool_and_final_text() -> None:
    from app.agent.agents import _run_system_prompt_hooks
    from app.harness.hooks.strategy_inject import settle_strategies

    store = get_strategy_store()
    s = _s("cheap_first", keywords=["预算"])
    await store.upsert(s)

    async def _turn(called: set[str], final: str) -> Strategy:
        await _run_system_prompt_hooks("BASE", role="main", query="预算 300")
        await settle_strategies({"called_tools": called, "final_answer": final})
        return next(x for x in await store.read_all() if x.dedup_key == s.dedup_key)

    # 调到终结工具 + 有回复 = 成功
    assert (await _turn({"planner", "shopping_summary"}, "给你三件")).consecutive_failures == 0
    # 跑满 max_iters 没收尾 = 失败（这正是策略最可能造成的坏）
    assert (await _turn({"planner", "item_search"}, "半截话")).consecutive_failures == 1
    # 收了尾但最终回复被输出审核清成空串 = 失败（所以本钩子的 priority 排在审核之后）
    assert (await _turn({"shopping_summary"}, "   ")).consecutive_failures == 2
    assert (await _turn({"shopping_summary"}, "补上了")).consecutive_failures == 0


@pytest.mark.anyio
async def test_session_end_is_noop_without_injection() -> None:
    """没注入过就没有账可结——别把一轮成败记到「刚好在库里」的策略头上。"""
    from app.harness.hooks.strategy_inject import settle_strategies

    store = get_strategy_store()
    s = _s("untouched", keywords=["预算"])
    await store.upsert(s)
    await settle_strategies({"called_tools": {"item_search"}, "final_answer": ""})
    row = next(x for x in await store.read_all() if x.dedup_key == s.dedup_key)
    assert row.hits == 0 and row.consecutive_failures == 0
