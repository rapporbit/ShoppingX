"""app/tools/ask_user 工具层测试：fork 拒绝 / 无会话 / 应答 / 超时 / 清理取消 / 任务取消。

与 test_clarification.py 的分工：那边测桥（Future 生命周期），这边测工具语义——
尤其是「用户在澄清等待中点取消」时任务必须真的死掉，不能被兜底文案吸收。
"""

import asyncio

import pytest

from app.agent.fork_guard import enter_fork
from app.api.clarification import cancel_pending, has_pending, resolve_pending
from app.tools.ask_user import ask_user
from app.utils.thread_ctx import thread_scope

TID = "askuser-test-thread"


@pytest.fixture(autouse=True)
def _clean():
    yield
    cancel_pending(TID)


async def _invoke() -> str:
    return await ask_user.ainvoke({"question": "要男款还是女款？"})


async def _wait_pending() -> None:
    for _ in range(50):
        if has_pending(TID):
            return
        await asyncio.sleep(0.01)
    raise AssertionError("ask_user 没有在期限内挂起 pending Future")


async def test_fork_depth_rejected(tmp_path):
    """子 Agent（fork depth>0）调用被拒，不产生 pending。"""
    with thread_scope(TID, tmp_path), enter_fork():
        result = await _invoke()
    assert "子任务无权" in result
    assert not has_pending(TID)


async def test_no_thread_id_skips():
    """无活跃会话（thread_id 未绑定）时直接返回跳过文案。"""
    result = await _invoke()
    assert "无活跃会话" in result


async def test_happy_path_returns_user_reply(tmp_path):
    """用户经 WS 回复后，工具返回原文并清掉 pending。"""
    with thread_scope(TID, tmp_path):
        task = asyncio.create_task(_invoke())
        await _wait_pending()
        assert resolve_pending(TID, "女款")
        assert await task == "女款"
    assert not has_pending(TID)


async def test_timeout_returns_fallback(tmp_path, monkeypatch):
    """超时未回复返回兜底文案，循环可继续。"""
    monkeypatch.setattr("app.tools.ask_user.ASK_USER_TIMEOUT_SEC", 0.05)
    with thread_scope(TID, tmp_path):
        result = await _invoke()
    assert "未在规定时间内回复" in result


async def test_cleanup_cancel_returns_fallback(tmp_path):
    """仅 Future 被清理取消（任务本身没被 cancel）→ 返回兜底文案而非炸掉。"""
    with thread_scope(TID, tmp_path):
        task = asyncio.create_task(_invoke())
        await _wait_pending()
        cancel_pending(TID)
        result = await task
    assert "未在规定时间内回复" in result


async def test_task_cancel_propagates(tmp_path):
    """复刻 cancel 端点顺序：cancel_pending + task.cancel() → 任务必须真的取消。

    若 ask_user 把任务级 CancelledError 当超时吞掉，Agent 会带着兜底文案继续跑，
    用户点了取消却取消不掉——这是本用例守住的行为。
    """
    async def _agent_like():
        with thread_scope(TID, tmp_path):
            await _invoke()
            await asyncio.sleep(30)  # 若取消被吞，会走到这里继续"干活"

    task = asyncio.create_task(_agent_like())
    # thread_scope 在任务内部绑定，等它真正挂起 pending
    for _ in range(50):
        if has_pending(TID):
            break
        await asyncio.sleep(0.01)
    assert has_pending(TID)

    cancel_pending(TID)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled()
