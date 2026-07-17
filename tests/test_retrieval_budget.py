"""检索预算的隔离检索作用域测试（isolated_retrieval_scope）。

覆盖：串行 dispatch_tool（定点商品调查等独立子任务）场景下，web_search 门控应只看
「本子任务自己的召回结果」，不受全树其它子任务（兄弟平台 / 兄弟商品）是否已找到候选影响；
非隔离场景（parallel_dispatch_tool 跨平台泛搜、主 loop 直调）维持现状的全树共享语义不变。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.agent.retrieval_budget import (
    _STATE,
    WEB_SEARCH_TASK_QUOTA,
    _isolated_var,
    isolated_retrieval_scope,
    note_item_search,
    note_web_search,
    web_search_allowed,
)
from app.api.context import _SESSION_TASKS, set_session_tasks
from app.utils.thread_ctx import thread_scope

SESSION_DIR = Path("/tmp/shoppingx-test-retrieval-budget-session")


@pytest.fixture(autouse=True)
def _clean_tree() -> None:
    """每条测试独立一棵树：避免 session_dir 键跨测试串台。

    直接清 ``_STATE`` 的字典键，不用 ``reset_tree()``——那个函数靠 ContextVar 读当前
    session_dir，fixture 运行时不在任何 thread_scope 内（``get_session_dir()`` 返回 None），
    调了也清不到 SESSION_DIR 这个键。
    """
    _STATE.pop(str(SESSION_DIR), None)
    _SESSION_TASKS.pop(str(SESSION_DIR), None)
    yield
    _STATE.pop(str(SESSION_DIR), None)
    _SESSION_TASKS.pop(str(SESSION_DIR), None)


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


def test_isolated_scope_ignores_sibling_success() -> None:
    """商品 A 搜到了，不该连带拦掉商品 B（B 自己没搜到）的 web_search 兜底——核心场景。"""
    with thread_scope("sub-item-a", SESSION_DIR), isolated_retrieval_scope():
        note_item_search(2)  # 商品 A 搜到了

    with thread_scope("sub-item-b", SESSION_DIR), isolated_retrieval_scope():
        note_item_search(0)  # 商品 B 自己没搜到
        # 非隔离语义下这里会被 A 的成功拦掉；隔离语义下只看自己，应该放行。
        assert web_search_allowed() is True


def test_isolated_scope_blocks_when_own_search_found_something() -> None:
    """隔离作用域不是「永远放行」——自己搜到了就该拦，跟非隔离语义的「找到就别再找」一致。"""
    with thread_scope("sub-item-c", SESSION_DIR), isolated_retrieval_scope():
        note_item_search(5)
        assert web_search_allowed() is False


def test_isolated_scope_does_not_affect_sibling_tree_wide_reading() -> None:
    """隔离子任务的召回不会污染全树共享计数——非隔离的兄弟仍看不到它，除非它自己也搜到过。"""
    with thread_scope("sub-item-isolated", SESSION_DIR), isolated_retrieval_scope():
        note_item_search(4)  # 隔离子任务搜到了

    with thread_scope("sub-platform-plain", SESSION_DIR):
        # 非隔离读取用的是 nonempty_item_search（隔离子任务的召回仍计入其中，因为它本质也是一次
        # 真实的 item_search）——这与「树上已有候选，非隔离子任务别再用 web_search 找更好」一致。
        assert web_search_allowed() is False


def test_isolated_retrieval_scope_resets_on_exception() -> None:
    assert _isolated_var.get() is False
    with pytest.raises(RuntimeError):
        with isolated_retrieval_scope():
            assert _isolated_var.get() is True
            raise RuntimeError("boom")
    assert _isolated_var.get() is False


def test_no_session_scope_returns_true() -> None:
    """无 session 作用域（单测直调）：两种语义都退化为放行，不报错。"""
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


def test_task_quota_not_granted_to_recommend() -> None:
    """recommend 主链路不吃配额：有候选照旧拦死，延迟零回退。"""
    with thread_scope("main", SESSION_DIR):
        note_item_search(total_recall=8)
        set_session_tasks(["recommend", "landed_cost"])
        assert web_search_allowed() is False
    with isolated_retrieval_scope():
        assert web_search_allowed() is True
