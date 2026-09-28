"""``threads`` 条件更新作「同 thread 唯一真相」。

钉的是四件事：认领的三种结局（started / already_running / replaced）、释放按身份、过期行可被
接管、**两个并发认领只有一个能赢**——最后这条正是搬进 DB 的理由，进程内字典在多副本下给不出。
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from sqlalchemy import update

from app.db import runs
from app.db.models import Thread
from app.db.runs import RunClaim, active_run_id, claim_thread_and_run, release_thread_run
from app.db.session import session_factory


async def _make_thread(thread_id: str = "t-run", user_id: str = "u1") -> None:
    async with session_factory()() as db:
        db.add(Thread(id=thread_id, user_id=user_id, title="t"))
        await db.commit()


async def _claim(thread_id: str, run_id: str, query: str, user_id: str = "u1") -> RunClaim:
    return await claim_thread_and_run(thread_id, run_id, query, user_id=user_id, verify_user=False)


async def _age_thread(thread_id: str, seconds: int) -> None:
    """把 ``updated_at`` 往回推——模拟「进程被 kill -9，行停在 running」。"""
    async with session_factory()() as db:
        await db.execute(
            update(Thread)
            .where(Thread.id == thread_id)
            .values(updated_at=datetime.now(UTC) - timedelta(seconds=seconds))
        )
        await db.commit()


async def test_first_claim_starts() -> None:
    await _make_thread()
    claim = await _claim("t-run", "run-1", "买包")
    assert claim.outcome == "started"
    assert await active_run_id("t-run") == "run-1"


async def test_same_query_is_already_running() -> None:
    """刷新 / 双击：领回原任务，且带着**原 run_id**（取消口与前端都按它认人）。"""
    await _make_thread()
    await _claim("t-run", "run-1", "买包")
    claim = await _claim("t-run", "run-2", "买包")
    assert claim.outcome == "already_running"
    assert claim.previous_run_id == "run-1"
    assert claim.can_start is False
    assert await active_run_id("t-run") == "run-1"  # 旧 run 没被顶掉


async def test_different_query_replaces() -> None:
    """改主意重问一句 = 覆盖重发：接班，并把旧 run_id 交出来让调用方去取消它。"""
    await _make_thread()
    await _claim("t-run", "run-1", "买包")
    claim = await _claim("t-run", "run-2", "买鞋")
    assert claim.outcome == "replaced"
    assert claim.previous_run_id == "run-1"
    assert claim.can_start is True
    assert await active_run_id("t-run") == "run-2"


async def test_release_only_clears_my_own_run() -> None:
    """覆盖重发时旧 run 的 finally 晚几个 tick 才跑——它不能把刚接班的新 run 摘掉。"""
    await _make_thread()
    await _claim("t-run", "run-1", "买包")
    await _claim("t-run", "run-2", "买鞋")

    assert await release_thread_run("t-run", "run-1") is False  # 我已经被换下来了
    assert await active_run_id("t-run") == "run-2"
    assert await release_thread_run("t-run", "run-2") is True
    assert await active_run_id("t-run") is None


async def test_stale_run_can_be_taken_over() -> None:
    """进程被 kill -9 留下的 running 行不能把 thread 永久锁死：超过 TTL 即可接管。"""
    await _make_thread()
    await _claim("t-run", "run-dead", "买包")
    await _age_thread("t-run", runs.STALE_RUN_SEC + 60)

    assert await active_run_id("t-run") is None  # 读侧先把它当没在跑
    claim = await _claim("t-run", "run-new", "买包")  # 连同 query 都一样也能接管
    assert claim.outcome == "started"
    assert await active_run_id("t-run") == "run-new"


async def test_first_turn_registers_row_and_starts() -> None:
    """首轮：行不存在 → 插行（归属 + 标题）时就带着 running，一次提交。"""
    claim = await _claim("t-new", "run-1", "买一个通勤双肩包")
    assert claim.outcome == "started"
    assert await active_run_id("t-new") == "run-1"
    async with session_factory()() as db:
        row = await db.get(Thread, "t-new")
    assert row is not None
    assert (row.user_id, row.title) == ("u1", "买一个通勤双肩包")


async def test_concurrent_claims_have_exactly_one_winner() -> None:
    """**搬进 DB 的理由**：同一 thread 的两个请求同时到（两台副本），只有一个能起任务。

    判定与占位是同一条 UPDATE，所以不依赖「两步之间没有 await」——这里刻意用 gather 让两条并发。
    """
    await _make_thread()
    first, second = await asyncio.gather(
        _claim("t-run", "run-a", "买包"),
        _claim("t-run", "run-b", "买包"),
    )
    outcomes = sorted([first.outcome, second.outcome])
    assert outcomes == ["already_running", "started"]

    winner = "run-a" if first.outcome == "started" else "run-b"
    assert await active_run_id("t-run") == winner


async def test_long_query_is_truncated_not_rejected() -> None:
    """超长 query 存前 500 字（列宽），不该在写入时炸——判重按同一截断值比。"""
    await _make_thread()
    long_query = "买" * (runs.QUERY_MAX + 200)
    assert (await _claim("t-run", "run-1", long_query)).outcome == "started"
    assert (await _claim("t-run", "run-2", long_query)).outcome == "already_running"


async def test_foreign_thread_is_rejected_and_untouched() -> None:
    """拿别人的 thread_id 起任务：403，且行一个字段都不能动（不过户、不占位、不顶时间）。"""
    await _make_thread(user_id="owner")
    await _age_thread("t-run", 3600)
    async with session_factory()() as db:
        before = await db.get(Thread, "t-run")
        assert before is not None
        before_updated = before.updated_at

    try:
        await _claim("t-run", "run-x", "买包", user_id="intruder")
        raise AssertionError("应抛 PermissionError")
    except PermissionError:
        pass

    async with session_factory()() as db:
        row = await db.get(Thread, "t-run")
    assert row is not None
    assert row.user_id == "owner"
    assert row.active_run_id is None
    assert row.updated_at == before_updated


async def test_unknown_user_on_first_turn_is_lookup_error() -> None:
    """鉴权开着、token 的 sub 在 users 表查无此人 → LookupError（路由转 401），且不插行。"""
    try:
        await claim_thread_and_run("t-ghost", "run-1", "买包", user_id="nobody", verify_user=True)
        raise AssertionError("应抛 LookupError")
    except LookupError:
        pass
    async with session_factory()() as db:
        assert await db.get(Thread, "t-ghost") is None


async def test_concurrent_first_turns_have_exactly_one_winner() -> None:
    """首轮并发（行还不存在）：两条都走插行，主键挡下后到的那条，它按 already_running 领回。"""
    first, second = await asyncio.gather(
        _claim("t-race", "run-a", "买包"),
        _claim("t-race", "run-b", "买包"),
    )
    assert sorted([first.outcome, second.outcome]) == ["already_running", "started"]
    winner = "run-a" if first.outcome == "started" else "run-b"
    assert await active_run_id("t-race") == winner


async def test_already_running_still_bumps_updated_at() -> None:
    """领回原任务也要把会话顶到侧栏最前——原先单独那步归属登记每次都会顶。"""
    await _make_thread()
    await _claim("t-run", "run-1", "买包")
    await _age_thread("t-run", 60)  # 仍远小于 STALE_RUN_SEC，不会被当死行接管
    claim = await _claim("t-run", "run-2", "买包")
    assert claim.outcome == "already_running"
    async with session_factory()() as db:
        row = await db.get(Thread, "t-run")
    assert row is not None
    updated = row.updated_at if row.updated_at.tzinfo else row.updated_at.replace(tzinfo=UTC)
    assert datetime.now(UTC) - updated < timedelta(seconds=10)


async def test_existing_title_is_kept() -> None:
    """续聊不改标题（标题是首轮那句话）；只有空标题才补。"""
    await _make_thread()  # title="t"
    await _claim("t-run", "run-1", "买鞋")
    async with session_factory()() as db:
        row = await db.get(Thread, "t-run")
    assert row is not None and row.title == "t"
