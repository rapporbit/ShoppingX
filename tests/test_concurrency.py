"""并发准入的确定性单测：双池优先级队列 + 请求指纹去重 + fork 级并发 Semaphore。

三层各测各的语义：
- ``PriorityRequestQueue``（任务级）：分池、有界排队、队列满则拒、再平衡、取消不漏槽。
- ``dedup``（幂等第 3 层）：窗口内同 (user_id, query) 判重复，窗口外放行。
- ``fork_concurrency_scope``（fork 级）：超额**排队**，同一时刻并发子任务数不超上限。
"""

from __future__ import annotations

import asyncio

import pytest

from app.api import dedup
from app.api.concurrency import (
    PriorityRequestQueue,
    classify_request,
    estimated_wait_seconds,
)
from app.harness.budgets import fork_concurrency_scope, get_fork_semaphore


@pytest.fixture(autouse=True)
def _clear_dedup() -> None:
    dedup.reset()


# ---------- 分类器 ----------
def test_classify_short_thread_is_normal() -> None:
    assert classify_request(0) == "normal"
    assert classify_request(3) == "normal"


def test_classify_long_thread_is_heavy() -> None:
    assert classify_request(20) == "heavy"


# ---------- 任务级：PriorityRequestQueue ----------
def test_reserve_takes_slot_when_free() -> None:
    q = PriorityRequestQueue(normal_slots=2, heavy_slots=1)
    res = q.try_reserve("normal")
    assert res is not None and res.admitted is True
    assert q.active == 1


def test_pools_are_independent() -> None:
    """分池的全部意义：heavy 占满槽，normal 照常直接进——大请求堵不死小请求。"""
    q = PriorityRequestQueue(normal_slots=2, heavy_slots=1)
    heavy = q.try_reserve("heavy")
    assert heavy is not None and heavy.admitted

    # heavy 池已满，再来的 heavy 只能排队
    heavy2 = q.try_reserve("heavy")
    assert heavy2 is not None and heavy2.admitted is False

    # 但 normal 池毫发无伤，直接拿槽
    normal = q.try_reserve("normal")
    assert normal is not None and normal.admitted is True


def test_queue_when_slots_full() -> None:
    q = PriorityRequestQueue(normal_slots=1, heavy_slots=1)
    assert q.try_reserve("normal") is not None  # 占槽
    queued = q.try_reserve("normal")
    assert queued is not None
    assert queued.admitted is False
    assert queued.position == 1
    assert q.pending("normal") == 1


def test_queue_positions_increment_before_enqueue() -> None:
    """准入判定与真正入队之间隔着一次调度——在途请求也得算进排队位置，否则三个人都是「第 1 位」。"""
    q = PriorityRequestQueue(normal_slots=1, heavy_slots=1, queue_depth=5)
    q.try_reserve("normal")  # 占槽
    first = q.try_reserve("normal")
    second = q.try_reserve("normal")
    assert first is not None and second is not None
    assert (first.position, second.position) == (1, 2)


def test_rejects_when_queue_full() -> None:
    q = PriorityRequestQueue(normal_slots=1, heavy_slots=1, queue_depth=2)
    q.try_reserve("normal")  # 槽
    q.try_reserve("normal")  # 队列 1
    q.try_reserve("normal")  # 队列 2
    assert q.try_reserve("normal") is None  # 队列满 → 429


def test_release_frees_slot() -> None:
    q = PriorityRequestQueue(normal_slots=1, heavy_slots=1)
    res = q.try_reserve("normal")
    assert res is not None
    q.release(res)
    assert q.active == 0
    assert q.try_reserve("normal") is not None


def test_release_is_idempotent() -> None:
    q = PriorityRequestQueue(normal_slots=2, heavy_slots=1)
    res = q.try_reserve("normal")
    assert res is not None
    q.release(res)
    q.release(res)  # 重复释放不该把计数压成负数、也不该凭空多出槽位
    assert q.active == 0


def test_release_of_queued_reservation_does_not_free_a_slot() -> None:
    """排队中被取消的任务从没持过槽——归还就成了凭空多出一个槽。"""
    q = PriorityRequestQueue(normal_slots=1, heavy_slots=1)
    held = q.try_reserve("normal")
    queued = q.try_reserve("normal")
    assert held is not None and queued is not None
    q.release(queued)
    assert q.active == 1  # 仍是持槽那位占着
    assert q.pending("normal") == 0  # 在途名额已销账


def test_force_reserve_bypasses_limit() -> None:
    """强占仅在旧任务**持槽**时成立：它马上被 cancel 并还槽，真实并发不变（调用方须自行保证）。"""
    q = PriorityRequestQueue(normal_slots=1, heavy_slots=1)
    old = q.try_reserve("normal")
    assert old is not None and old.admitted
    forced = q.force_reserve("normal")  # 覆盖重发：不该被自己的旧任务挡在门外
    assert forced.admitted is True
    assert q.stats()["normal"]["active"] == 2
    q.release(old)  # 旧任务被 cancel，还回它的槽
    assert q.stats()["normal"]["active"] == 1  # 回落到上限之内


@pytest.mark.asyncio
async def test_cancelled_waiter_release_restores_heavy_capacity() -> None:
    """排队者被取消也要触发再平衡——否则 normal 队列排空了，heavy 还一直被压在下限。"""
    q = PriorityRequestQueue(normal_slots=1, heavy_slots=3, queue_depth=10)
    held = q.try_reserve("normal")
    queued = [q.try_reserve("normal") for _ in range(3)]
    assert q.stats()["heavy"]["capacity"] == 1  # 积压 → heavy 被压缩

    waiters = [asyncio.create_task(q.wait_turn(r)) for r in queued if r]
    await asyncio.sleep(0)
    for w in waiters:
        w.cancel()
    await asyncio.gather(*waiters, return_exceptions=True)
    for r in queued:  # _runner 的 finally 总会归还凭据
        assert r is not None
        q.release(r)

    assert q.pending("normal") == 0
    assert q.stats()["heavy"]["capacity"] == 3  # 队列排空 → heavy 恢复
    assert held is not None
    q.release(held)


@pytest.mark.asyncio
async def test_wait_turn_returns_immediately_when_admitted() -> None:
    q = PriorityRequestQueue(normal_slots=1, heavy_slots=1)
    res = q.try_reserve("normal")
    assert res is not None
    await asyncio.wait_for(q.wait_turn(res), timeout=0.5)


@pytest.mark.asyncio
async def test_queued_task_runs_after_release() -> None:
    q = PriorityRequestQueue(normal_slots=1, heavy_slots=1)
    held = q.try_reserve("normal")
    queued = q.try_reserve("normal")
    assert held is not None and queued is not None

    waiter = asyncio.create_task(q.wait_turn(queued))
    await asyncio.sleep(0)  # 让 waiter 真正进队列
    assert not waiter.done()

    q.release(held)
    await asyncio.wait_for(waiter, timeout=0.5)
    assert queued.admitted is True
    assert q.active == 1


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_leak_slot() -> None:
    """排队中被取消（用户点了取消）→ 队列摘除，不占槽也不漏槽。"""
    q = PriorityRequestQueue(normal_slots=1, heavy_slots=1)
    held = q.try_reserve("normal")
    queued = q.try_reserve("normal")
    assert held is not None and queued is not None

    waiter = asyncio.create_task(q.wait_turn(queued))
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    assert q.pending("normal") == 0
    q.release(held)
    assert q.active == 0  # 被取消的等待者没有偷偷占着槽


@pytest.mark.asyncio
async def test_cancel_after_wakeup_returns_slot() -> None:
    """唤醒方先占槽再 set_result；若等待者恰在此刻被取消，槽必须还回去。"""
    q = PriorityRequestQueue(normal_slots=1, heavy_slots=1)
    held = q.try_reserve("normal")
    queued = q.try_reserve("normal")
    assert held is not None and queued is not None

    waiter = asyncio.create_task(q.wait_turn(queued))
    await asyncio.sleep(0)
    q.release(held)  # 唤醒 waiter 并替它占槽（此时 waiter 还没被调度）
    waiter.cancel()  # 抢在它醒来之前取消
    with pytest.raises(asyncio.CancelledError):
        await waiter

    assert q.active == 0  # 槽被归还，没有泄漏


@pytest.mark.asyncio
async def test_no_queue_jumping() -> None:
    """有人在排队时，后来的请求不许直接占槽——否则队首可能永远等不到。"""
    q = PriorityRequestQueue(normal_slots=1, heavy_slots=1)
    held = q.try_reserve("normal")
    queued = q.try_reserve("normal")
    assert held is not None and queued is not None
    waiter = asyncio.create_task(q.wait_turn(queued))
    await asyncio.sleep(0)

    latecomer = q.try_reserve("normal")
    assert latecomer is not None
    assert latecomer.admitted is False  # 老实排队去

    q.release(held)
    await asyncio.wait_for(waiter, timeout=0.5)
    q.release(queued)
    q.release(latecomer)


def test_rebalance_shrinks_heavy_when_normal_backs_up() -> None:
    """normal 积压 → 压缩 heavy 容量，高峰期优先保短任务。"""
    q = PriorityRequestQueue(normal_slots=1, heavy_slots=3, queue_depth=10)
    assert q.stats()["heavy"]["capacity"] == 3
    q.try_reserve("normal")  # 占槽
    for _ in range(3):  # 积压 3 个
        q.try_reserve("normal")
    assert q.stats()["heavy"]["capacity"] == 1


def test_rebalance_restores_heavy_when_normal_drains() -> None:
    q = PriorityRequestQueue(normal_slots=1, heavy_slots=3, queue_depth=10)
    held = q.try_reserve("normal")
    queued = [q.try_reserve("normal") for _ in range(3)]
    assert q.stats()["heavy"]["capacity"] == 1

    for res in queued:
        assert res is not None
        q.release(res)  # 排队者陆续离开
    assert held is not None
    q.release(held)
    assert q.stats()["heavy"]["capacity"] == 3  # 恢复


def test_rebalance_never_preempts_running_heavy_tasks() -> None:
    """缩容不抢占已在跑的任务——Agent 任务跑到一半被掐断是不可接受的。"""
    q = PriorityRequestQueue(normal_slots=1, heavy_slots=3, queue_depth=10)
    heavies = [q.try_reserve("heavy") for _ in range(3)]
    assert all(r is not None and r.admitted for r in heavies)

    q.try_reserve("normal")
    for _ in range(3):
        q.try_reserve("normal")
    assert q.stats()["heavy"]["capacity"] == 1
    assert q.stats()["heavy"]["active"] == 3  # 三个都还在跑，一个都没被掐


def test_estimated_wait_divides_by_capacity() -> None:
    """池子有 5 个槽时，排第 3 位不必等 3 个任务跑完——第一批退出就轮到了。"""
    assert estimated_wait_seconds(0, 5) == 0
    assert estimated_wait_seconds(3, 5) == estimated_wait_seconds(1, 5)
    assert estimated_wait_seconds(6, 5) > estimated_wait_seconds(5, 5)


# ---------- 幂等第 3 层：请求指纹去重 ----------
def test_dedup_first_submit_passes() -> None:
    assert dedup.check_duplicate("alice", "买旅行三件套") is None


def test_dedup_catches_repeat_across_thread_ids() -> None:
    """前端刷新会换 thread_id——active_tasks 看不见这种重复，指纹能。"""
    dedup.remember("alice", "买旅行三件套", "thread-1")
    assert dedup.check_duplicate("alice", "买旅行三件套") == "thread-1"


def test_dedup_distinguishes_users() -> None:
    dedup.remember("alice", "买包", "thread-1")
    assert dedup.check_duplicate("bob", "买包") is None


def test_dedup_distinguishes_queries() -> None:
    dedup.remember("alice", "买包", "thread-1")
    assert dedup.check_duplicate("alice", "买鞋") is None


def test_dedup_remember_purges_expired_entries() -> None:
    """``check_duplicate`` 只在「调用方不自带 thread_id」时才被调到；只在它里面 purge 的话，
    走前端那条路（永远自带 thread_id）的部署里 ``_recent`` 会随任务数无界增长。"""
    import time as _time

    dedup.reset()
    dedup.remember("alice", "老 query", "t-old")
    assert len(dedup._recent) == 1

    real_monotonic = _time.monotonic
    dedup.time.monotonic = lambda: real_monotonic() + dedup.DEDUP_WINDOW_SEC + 1  # type: ignore[assignment]
    try:
        dedup.remember("bob", "新 query", "t-new")  # 只调 remember，不调 check_duplicate
        assert len(dedup._recent) == 1  # 过期的老条目被顺手清掉了
    finally:
        dedup.time.monotonic = real_monotonic  # type: ignore[assignment]


def test_dedup_check_does_not_register() -> None:
    """查询不登记——被 429 的请求不该留下指纹，否则退避重试会被当成重复提交再拒一次。"""
    dedup.reset()
    assert dedup.check_duplicate("alice", "买包") is None
    assert dedup.check_duplicate("alice", "买包") is None  # 仍未登记
    assert len(dedup._recent) == 0


def test_dedup_window_expires(monkeypatch: pytest.MonkeyPatch) -> None:
    dedup.remember("alice", "买包", "thread-1")
    # 把时钟推过窗口
    now = __import__("time").monotonic() + dedup.DEDUP_WINDOW_SEC + 1
    monkeypatch.setattr(dedup.time, "monotonic", lambda: now)
    assert dedup.check_duplicate("alice", "买包") is None


# ---------- fork 级：fork_concurrency_scope（超额排队，限并发峰值） ----------
async def test_fork_semaphore_none_outside_scope() -> None:
    assert get_fork_semaphore() is None  # 无作用域（单测/示例）→ 不限并发


async def test_fork_semaphore_caps_peak_concurrency() -> None:
    peak = 0
    current = 0

    async def worker() -> None:
        nonlocal peak, current
        sem = get_fork_semaphore()
        assert sem is not None  # gather 子任务继承 ContextVar 快照 → 拿到同一个 Semaphore
        async with sem:
            current += 1
            peak = max(peak, current)
            await asyncio.sleep(0.02)
            current -= 1

    with fork_concurrency_scope(2):
        await asyncio.gather(*(worker() for _ in range(6)))

    assert peak <= 2  # 6 个子任务争 2 个槽，同一时刻并发不超 2


async def test_fork_semaphore_scope_resets() -> None:
    with fork_concurrency_scope(3):
        assert get_fork_semaphore() is not None
    assert get_fork_semaphore() is None  # 离开作用域后还原
