"""app/api/clarification 单元测试：Future 的创建 / 解决 / 取消 / 超时。"""

import asyncio

import pytest

from app.api.clarification import cancel_pending, create_pending, has_pending, resolve_pending


@pytest.fixture(autouse=True)
def _clean():
    """每个用例后清理残留的 pending Future。"""
    yield
    cancel_pending("test-thread")


async def test_create_and_resolve():
    fut = create_pending("test-thread")
    assert has_pending("test-thread")
    assert not fut.done()
    assert resolve_pending("test-thread", "女款")
    assert await fut == "女款"
    assert not has_pending("test-thread")


async def test_resolve_nonexistent_returns_false():
    assert not resolve_pending("no-such-thread", "hello")


async def test_cancel_pending():
    fut = create_pending("test-thread")
    cancel_pending("test-thread")
    assert not has_pending("test-thread")
    assert fut.cancelled()


async def test_cancel_nonexistent_is_noop():
    cancel_pending("no-such-thread")


async def test_timeout():
    fut = create_pending("test-thread")
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(fut, timeout=0.05)
    assert not has_pending("test-thread") or fut.done()


async def test_replace_existing():
    """create_pending 对同一 thread 再调会 cancel 旧的、返回新的。"""
    fut1 = create_pending("test-thread")
    fut2 = create_pending("test-thread")
    assert fut1.cancelled()
    assert not fut2.done()
    resolve_pending("test-thread", "ok")
    assert await fut2 == "ok"


async def test_resolve_after_cancel_returns_false():
    create_pending("test-thread")
    cancel_pending("test-thread")
    assert not resolve_pending("test-thread", "too late")
