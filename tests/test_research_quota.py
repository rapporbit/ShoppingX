"""C3：research 的会话级独立配额，与 web_search 配额分账。

四组验收：① 按搜索条数预扣、超额回 ERROR 哨兵而非抛；② 与 ``WEB_SEARCH_TASK_QUOTA`` 互不透支；
③ 同轮 batch 并发两条 research 不会把上限撑爆；④ 闸算出的条数与工具真发出的条数一致。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import app.harness.hooks.budget as tg
from app.api.context import _SESSION_TASKS, set_session_tasks
from app.harness.middleware import HarnessMiddleware
from app.harness.retrieval_budget import (
    _STATE,
    RESEARCH_SEARCH_QUOTA,
    charge_research,
    note_item_search,
    note_web_search,
    research_remaining,
    web_search_allowed,
)
from app.harness.state import GuardState
from app.tools.research import normalize_targets
from app.utils.thread_ctx import thread_scope

SESSION_DIR = Path("/tmp/shoppingx-test-research-quota-session")


@pytest.fixture(autouse=True)
def _clean_tree() -> None:
    """每条测试独立一个会话：fixture 不在 thread_scope 内，只能直接清 ``_STATE`` 的键。"""
    _STATE.pop(str(SESSION_DIR), None)
    _SESSION_TASKS.pop(str(SESSION_DIR), None)
    yield
    _STATE.pop(str(SESSION_DIR), None)
    _SESSION_TASKS.pop(str(SESSION_DIR), None)


def _gate() -> HarnessMiddleware:
    mw = HarnessMiddleware()
    mw.register("pre_tool_call", "search_gate", tg.check_search, priority=45)
    return mw


async def _call(mw: HarnessMiddleware, targets: list[str]) -> dict:
    return await mw.run(
        "pre_tool_call",
        {"tool_name": "research", "tool_args": {"targets": targets}, "_guard": GuardState()},
    )


class TestQuotaAccounting:
    def test_charges_by_search_count_not_call_count(self) -> None:
        """配额单位是搜索条数：一次 3 target 的调用扣 3。"""
        with thread_scope("main", SESSION_DIR):
            assert charge_research(3) is True
            assert research_remaining() == RESEARCH_SEARCH_QUOTA - 3

    def test_denied_call_does_not_charge(self) -> None:
        """扣不动就一条都不扣——否则连续几次超额请求会把余额蚕食成 0，缩小 targets 也过不了。"""
        with thread_scope("main", SESSION_DIR):
            charge_research(RESEARCH_SEARCH_QUOTA - 1)
            assert charge_research(3) is False
            assert research_remaining() == 1
            assert charge_research(1) is True  # 缩小 targets 后仍能过

    def test_unscoped_never_blocks(self) -> None:
        """无 session 作用域（单测 / 离线直调）失效方向中性：不拦。"""
        assert charge_research(99) is True
        assert research_remaining() == RESEARCH_SEARCH_QUOTA


class TestSeparateFromWebSearch:
    def test_research_does_not_consume_web_search_quota(self) -> None:
        """research 搜 6 条后，evaluate 任务的 web_search 配额仍是满的。"""
        with thread_scope("main", SESSION_DIR):
            set_session_tasks(["evaluate"])
            note_item_search(3)  # 已有候选：位置门本来会拦，靠任务配额放行
            assert charge_research(RESEARCH_SEARCH_QUOTA) is True
            assert web_search_allowed() is True

    def test_web_search_does_not_consume_research_quota(self) -> None:
        """反向同理：web_search 用尽任务配额，research 额度不受影响。"""
        with thread_scope("main", SESSION_DIR):
            set_session_tasks(["evaluate"])
            note_item_search(3)
            for _ in range(5):
                note_web_search()
            assert web_search_allowed() is False  # 任务配额已尽
            assert research_remaining() == RESEARCH_SEARCH_QUOTA


class TestGate:
    @pytest.mark.asyncio
    async def test_over_quota_rejects_with_sentinel(self) -> None:
        """超额回 ERROR 哨兵（``_rejected`` + raw 文案），不抛异常。"""
        mw = _gate()
        with thread_scope("main", SESSION_DIR):
            out = await _call(mw, ["A", "B", "C"])
            assert not out.get("_rejected")
            out = await _call(mw, ["D", "E", "F"])
            assert not out.get("_rejected")  # 正好用满 6 条
            out = await _call(mw, ["G"])
            assert out.get("_rejected")
            assert out.get("_reject_raw")
            assert "[research 未执行]" in out["_reject_reason"]

    @pytest.mark.asyncio
    async def test_sentinel_tells_remaining(self) -> None:
        """额度不够但没用完时，哨兵要报出还剩几条——模型缩小 targets 就能过，这是真出路。"""
        mw = _gate()
        with thread_scope("main", SESSION_DIR):
            await _call(mw, ["A", "B", "C", "D"])  # 超 3 个被截断 → 实扣 3
            await _call(mw, ["E", "F"])  # 实扣 2，余 1
            out = await _call(mw, ["G", "H", "I"])
            assert out.get("_rejected")
            assert "只剩 1 条" in out["_reject_reason"]
            assert "保留最关键的 1 个对象" in out["_reject_reason"]
            out = await _call(mw, ["G"])  # 按哨兵说的缩到 1 个 → 过
            assert not out.get("_rejected")
            out = await _call(mw, ["H"])  # 额度真用完 → 换成「已用完」文案
            assert "已用完" in out["_reject_reason"]

    @pytest.mark.asyncio
    async def test_empty_targets_costs_nothing(self) -> None:
        """空 targets 不耗额度：交给工具自己回「没有给出研究对象」。"""
        mw = _gate()
        with thread_scope("main", SESSION_DIR):
            out = await _call(mw, ["", "  "])
            assert not out.get("_rejected")
            assert research_remaining() == RESEARCH_SEARCH_QUOTA

    @pytest.mark.asyncio
    async def test_same_turn_batch_cannot_overspend(self) -> None:
        """同轮 batch 并发两条 research：判与扣在同一同步段，第二条只能拿到剩下的额度。"""
        mw = _gate()

        async def _one() -> dict:
            with thread_scope("main", SESSION_DIR):
                return await _call(mw, ["A", "B", "C", "D"])

        outs = await asyncio.gather(_one(), _one(), _one())
        rejected = [o for o in outs if o.get("_rejected")]
        with thread_scope("main", SESSION_DIR):
            assert research_remaining() == 0
        assert len(rejected) == 1  # 3 次 × 3 条 = 9 > 6，必须挡下一条

    def test_gate_counts_match_tool_behavior(self) -> None:
        """闸预扣的条数 = 工具真发出的条数：两边共用 ``normalize_targets``，不各写一份。"""
        raw = ["A", "", "  ", "B", "C", "D"]  # 空串不算、超 3 个截断
        assert len(normalize_targets(raw)) == 3
        assert normalize_targets(None) == []
