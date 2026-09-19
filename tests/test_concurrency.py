"""并发准入的确定性单测：双池优先级队列 + 请求指纹去重。

两层各测各的语义：
- ``PriorityRequestQueue``（任务级）：分池、有界排队、队列满则拒、再平衡、取消不漏槽。
- ``dedup``（幂等第 3 层）：窗口内同 (user_id, query) 判重复，窗口外放行。
"""

from __future__ import annotations

import asyncio

import pytest
from conftest import FakeRedis  # tests/ 的 conftest（pytest 已把它放进 sys.path）

from app.api import dedup
from app.api.concurrency import (
    PriorityRequestQueue,
    classify_request,
    estimated_wait_seconds,
)


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


# ---------- 幂等第 3 层：请求指纹去重（阶段 1-2 起窗口在 Redis）----------
async def test_dedup_first_submit_passes() -> None:
    assert await dedup.check_duplicate("alice", "买旅行三件套", "thread-1") is None


async def test_dedup_catches_repeat_across_thread_ids() -> None:
    """脚本 / 压测器每次换 thread_id——第 1 层看不见这种重复，指纹能，且领回**原** thread。"""
    assert await dedup.check_duplicate("alice", "买旅行三件套", "thread-1") is None
    assert await dedup.check_duplicate("alice", "买旅行三件套", "thread-2") == "thread-1"


async def test_dedup_distinguishes_users() -> None:
    await dedup.check_duplicate("alice", "买包", "thread-1")
    assert await dedup.check_duplicate("bob", "买包", "thread-2") is None


async def test_dedup_distinguishes_queries() -> None:
    await dedup.check_duplicate("alice", "买包", "thread-1")
    assert await dedup.check_duplicate("alice", "买鞋", "thread-2") is None


async def test_dedup_check_registers_atomically(fake_redis: FakeRedis) -> None:
    """``SET NX`` 一条命令同时完成查与登记——这正是老实现（先查后登记）在并发下漏掉的那一步。"""
    assert await dedup.check_duplicate("alice", "买包", "thread-1") is None
    assert len(fake_redis.store) == 1


async def test_dedup_forget_lets_retry_through(fake_redis: FakeRedis) -> None:
    """被 429 / already_running 拒掉的请求必须撤销指纹，否则用户退避重试会被自己刚才那次挡住。"""
    assert await dedup.check_duplicate("alice", "买包", "thread-1") is None
    await dedup.forget("alice", "买包")
    assert fake_redis.store == {}
    assert await dedup.check_duplicate("alice", "买包", "thread-2") is None


async def test_dedup_window_expires(fake_redis: FakeRedis) -> None:
    await dedup.check_duplicate("alice", "买包", "thread-1")
    fake_redis.now += dedup.DEDUP_WINDOW_SEC + 1  # 把 Redis 的时钟推过窗口
    assert await dedup.check_duplicate("alice", "买包", "thread-2") is None


async def test_dedup_unavailable_is_raised_not_swallowed() -> None:
    """Redis 挂了要抛（调用方转 503），**不退回进程内**——那等于「看着在去重，其实每台各去各的」。"""

    class _Broken:
        async def set(self, *a: object, **kw: object) -> object:
            raise ConnectionError("redis down")

        async def get(self, key: str) -> object:
            raise ConnectionError("redis down")

        async def delete(self, *keys: str) -> object:
            raise ConnectionError("redis down")

    dedup.set_client(_Broken())
    with pytest.raises(dedup.DedupUnavailable):
        await dedup.check_duplicate("alice", "买包", "thread-1")


async def test_dedup_disabled_never_reports_duplicate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TASK_DEDUP_ENABLED", "false")
    await dedup.check_duplicate("alice", "买包", "thread-1")
    assert await dedup.check_duplicate("alice", "买包", "thread-2") is None
