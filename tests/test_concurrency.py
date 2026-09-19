"""请求分档与指纹去重的确定性单测。

- ``classify_request`` / ``estimated_wait_seconds``：档位判定与排队预估（进程内准入池随阶段 1
  条 7 删除，那批测试一并删掉——它测的东西已经不存在了）。
- ``dedup``（幂等第 3 层）：窗口内同 (user_id, query) 判重复，窗口外放行。
"""

from __future__ import annotations

import asyncio

import pytest
from conftest import FakeRedis  # tests/ 的 conftest（pytest 已把它放进 sys.path）

from app.api import dedup
from app.api.concurrency import classify_request, estimated_wait_seconds


# ---------- 分类器 ----------
def test_classify_short_thread_is_normal() -> None:
    assert classify_request(0) == "normal"
    assert classify_request(3) == "normal"


def test_classify_long_thread_is_heavy() -> None:
    assert classify_request(20) == "heavy"


def test_estimated_wait_divides_by_capacity() -> None:
    """有 5 个并行消费位时，排第 3 位不必等 3 个任务跑完——第一批退出就轮到了。"""
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
