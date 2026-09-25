"""背压闸（``TaskQueue.admit``）：判定与登记一步完成，突发下放行数不超上限；各条路径都还名额。

上半段走 HTTP + 进程内队列，钉 server 侧的接线（闸排第一、幂等命中还名额、入队后撤登记）；
下半段钉 Redis 那段 Lua，需要真 Redis（``ADMIT_TEST_REDIS_URL``，默认连压测台的
``globex-multi-redis`` 16379 的 9 号库），没有就 skip——手写 FakeRedis 只能把脚本语义再抄一遍，
验不到脚本本身。
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

import app.api.server as server
from app.api import dedup
from app.queue import InProcessQueue, set_task_queue
from app.queue import redis_stream as rs


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    async with AsyncClient(transport=ASGITransport(app=server.app), base_url="http://test") as c:
        yield c


@pytest.fixture
def queue(monkeypatch: pytest.MonkeyPatch) -> Iterator[InProcessQueue]:
    q = InProcessQueue()
    set_task_queue(q)
    monkeypatch.setattr(server, "QUEUE_POLL_SECONDS", 0.02)
    yield q
    for handle in list(server.active_tasks.values()):
        if not handle.task.done():
            handle.task.cancel()
    server.active_tasks.clear()
    dedup.reset()
    set_task_queue(None)


async def test_burst_never_admits_more_than_cap(
    client: AsyncClient, queue: InProcessQueue, monkeypatch: pytest.MonkeyPatch
) -> None:
    """20 个请求同时到、上限 3：放进来的恰好 3 条。闸后的库操作拖慢一点，把旧写法的竞态窗口撑开。"""
    monkeypatch.setattr(server, "QUEUE_MAX_DEPTH", 3)
    real = server._history_turns

    async def _slow(thread_id: str) -> int:
        await asyncio.sleep(0.02)
        return await real(thread_id)

    monkeypatch.setattr(server, "_history_turns", _slow)
    resps = await asyncio.gather(
        *(
            client.post("/api/task", json={"query": f"q{i}", "thread_id": f"b{i}"})
            for i in range(20)
        )
    )
    codes = [r.status_code for r in resps]
    assert codes.count(200) == 3 and codes.count(429) == 17
    await asyncio.sleep(0.05)  # 等影子协程入队
    assert await queue.depth() == 3
    assert queue._admitted == set()  # 入队后登记都撤了


async def test_rejected_request_skips_db_work(
    client: AsyncClient, queue: InProcessQueue, monkeypatch: pytest.MonkeyPatch
) -> None:
    """被拒的请求不碰归属登记与历史轮数——这是挪闸的目的。"""
    monkeypatch.setattr(server, "QUEUE_MAX_DEPTH", 0)
    touched: list[str] = []

    async def _spy(*args: Any, **_kw: Any) -> None:
        touched.append("claim")

    monkeypatch.setattr(server, "_claim_thread_if_needed", _spy)
    resp = await client.post("/api/task", json={"query": "买帐篷", "thread_id": "r1"})
    assert resp.status_code == 429
    assert touched == []


async def test_idempotent_hit_returns_the_slot(
    client: AsyncClient, queue: InProcessQueue, monkeypatch: pytest.MonkeyPatch
) -> None:
    """同 thread 同 query 再发一次 → already_running，没入队，名额当场还。"""
    monkeypatch.setattr(server, "QUEUE_MAX_DEPTH", 5)
    body = {"query": "买帐篷", "thread_id": "i1"}
    assert (await client.post("/api/task", json=body)).status_code == 200
    again = await client.post("/api/task", json=body)
    assert again.json()["status"] == "already_running"
    assert queue._admitted == set()


async def test_failure_after_gate_returns_the_slot(
    client: AsyncClient, queue: InProcessQueue, monkeypatch: pytest.MonkeyPatch
) -> None:
    """闸后任何一步抛错（这里是归属校验 403），名额都要还，否则每次失败永久吃掉一个名额。"""
    monkeypatch.setattr(server, "QUEUE_MAX_DEPTH", 5)

    async def _deny(*_a: Any, **_kw: Any) -> None:
        raise server.HTTPException(403, "无权访问该会话")

    # /api/task 走合并版 claim（perf/claim-merge），/api/task/async 仍走单独的归属登记
    monkeypatch.setattr(server, "_claim_thread_and_run_or_reject", _deny)
    monkeypatch.setattr(server, "_claim_thread_if_needed", _deny)
    for path in ("/api/task", "/api/task/async"):
        resp = await client.post(path, json={"query": "买帐篷", "thread_id": "f1"})
        assert resp.status_code == 403
    assert queue._admitted == set()


# ---------- Redis Lua：真 Redis ----------
REDIS_URL = os.environ.get("ADMIT_TEST_REDIS_URL", "redis://localhost:16379/9")


def _probe_redis() -> bool:
    async def _ping() -> bool:
        import redis.asyncio as aredis

        client = aredis.from_url(REDIS_URL, socket_connect_timeout=1, socket_timeout=1)
        try:
            return bool(await client.ping())
        except Exception:
            return False
        finally:
            await client.aclose()

    return asyncio.run(_ping())


requires_redis = pytest.mark.skipif(
    not _probe_redis(), reason=f"需要真 Redis（{REDIS_URL}）来验准入 Lua 的原子性"
)


@pytest.fixture
async def rq() -> AsyncIterator[rs.RedisStreamQueue]:
    import redis.asyncio as aredis

    client = aredis.from_url(REDIS_URL, decode_responses=True)
    await client.flushdb()
    queue = rs.RedisStreamQueue(client)
    await queue.ensure_group()
    yield queue
    await client.flushdb()
    await client.aclose()


@requires_redis
async def test_lua_concurrent_admits_stop_at_cap(rq: rs.RedisStreamQueue) -> None:
    results = await asyncio.gather(*(rq.admit(f"t{i}", 8) for i in range(50)))
    assert sum(r is not None for r in results) == 8


@requires_redis
async def test_lua_counts_undelivered_messages(rq: rs.RedisStreamQueue) -> None:
    for i in range(3):
        await rq.enqueue(rs.IntentTask.create(task_id=f"m{i}", thread_id=f"m{i}", query="x"))
    assert await rq.admit("a", 4) == 3  # 3 条未投递 + 0 条登记
    assert await rq.admit("b", 4) is None  # 3 + 1 已到上限
    await rq.release_admission("a")
    assert await rq.admit("b", 4) == 3


@requires_redis
async def test_lua_stale_ticket_expires(
    rq: rs.RedisStreamQueue, monkeypatch: pytest.MonkeyPatch
) -> None:
    """持有者崩在「准入 → 入队」之间：登记过期后名额自动回来。"""
    monkeypatch.setattr(rs, "_ADMIT_TTL_MS", 50)
    assert await rq.admit("dead", 1) == 0
    assert await rq.admit("next", 1) is None
    await asyncio.sleep(0.1)
    assert await rq.admit("next", 1) == 0
