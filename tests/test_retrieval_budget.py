"""检索预算：会话共享的召回信号 + 任务口径配额 + 检索总量越线。

隔离检索作用域（原定点调查用）已于 2026-09-16 删除，门控只剩会话共享语义。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.api.context import set_session_tasks
from app.api.run_state import reset_run_state
from app.harness.hooks.budget import charge_retrieval
from app.harness.middleware import HookRejectSignal
from app.harness.retrieval_budget import (
    WEB_SEARCH_TASK_QUOTA,
    note_item_search,
    note_web_search,
    web_search_allowed,
)
from app.harness.state import GuardState
from app.utils.thread_ctx import thread_scope

SESSION_DIR = Path("/tmp/shoppingx-test-retrieval-budget-session")


@pytest.fixture(autouse=True)
def _clean_tree() -> None:
    """每条测试独立一棵树：避免 session_dir 键跨测试串台。

    显式传 session_dir，不用无参形式——fixture 运行时不在任何 thread_scope 内
    （``get_session_dir()`` 返回 None），不传就清不到 SESSION_DIR 这一份。
    """
    reset_run_state(SESSION_DIR)
    yield
    reset_run_state(SESSION_DIR)


def test_unscoped_allows_before_any_search() -> None:
    with thread_scope("main", SESSION_DIR):
        assert web_search_allowed() is True  # 还没搜过 → 独立知识查询场景放行


def test_unscoped_tree_wide_blocks_once_anything_found() -> None:
    """未开隔离作用域：行为与现状一致——树上任一处搜到候选，全树都拦（回归保护）。"""
    with thread_scope("main", SESSION_DIR):
        note_item_search(3)  # 找到候选
        assert web_search_allowed() is False


def test_unscoped_tree_wide_allows_when_all_empty() -> None:
    with thread_scope("sub-platform-a", SESSION_DIR):
        note_item_search(0)
    with thread_scope("sub-platform-b", SESSION_DIR):
        # 全树目前为止都是空召回 → 兜底放行。
        assert web_search_allowed() is True


def test_no_session_scope_returns_true() -> None:
    """无 session 作用域（单测直调）：退化为放行，不报错。"""
    assert web_search_allowed() is True


def test_task_quota_allows_evaluate_with_candidates() -> None:
    """窄口径用途门：evaluate 任务在有候选后仍放行（配额内），不再只靠逃生门。"""
    with thread_scope("main", SESSION_DIR):
        note_item_search(total_recall=8)  # 有候选 → 原「位置门」会拦
        set_session_tasks(["evaluate", "landed_cost"])
        assert web_search_allowed() is True


def test_task_quota_exhausts_then_blocks() -> None:
    """配额用尽后落回位置门语义：有候选 → 拦（防挤气球）。"""
    with thread_scope("main", SESSION_DIR):
        note_item_search(total_recall=8)
        set_session_tasks(["evaluate"])
        for _ in range(WEB_SEARCH_TASK_QUOTA):
            assert web_search_allowed() is True
            note_web_search()
        assert web_search_allowed() is False


async def test_retrieval_cap_soft_then_hard() -> None:
    """检索总量只剩会话这一条计数路：cap 内放行 → cap+1 软收敛 → 再越硬挡。

    回退口径（GuardState.retrieval_count / DEFAULT_RETRIEVAL_CAP）已删，这条用例把原来只有
    回退路覆盖到的越线判据移到 thread_scope 里跑。
    """
    guard = GuardState(retrieval_cap=2)
    with thread_scope("main", SESSION_DIR):
        for _ in range(2):
            ctx: dict = {"tool_name": "item_search", "_guard": guard}
            assert await charge_retrieval(ctx) is None
            assert "converge_count" not in ctx

        ctx = {"tool_name": "item_search", "_guard": guard}
        assert await charge_retrieval(ctx) is ctx  # 软越线：照常执行 + 追加收敛指令
        assert ctx["converge_count"] == 3

        with pytest.raises(HookRejectSignal):
            await charge_retrieval({"tool_name": "item_search", "_guard": guard})


async def test_retrieval_not_counted_without_session_scope() -> None:
    """无 session 作用域（单测直调）：不计数也不拦——失效方向中性，与本模块其他闸一致。"""
    guard = GuardState(retrieval_cap=0)
    assert await charge_retrieval({"tool_name": "item_search", "_guard": guard}) is None


def test_task_quota_not_granted_to_recommend() -> None:
    """recommend 主链路不吃配额：有候选照旧拦死，延迟零回退。"""
    with thread_scope("main", SESSION_DIR):
        note_item_search(total_recall=8)
        set_session_tasks(["recommend", "landed_cost"])
        assert web_search_allowed() is False
