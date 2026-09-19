"""同 thread 的**唯一真相**：谁在跑、跑的是哪句话（阶段 1-2）。

**它替掉的是什么。** 幂等第 1 层此前读 API 进程内的 ``active_tasks`` 字典：同一个 thread 的两个
请求打到两台副本，各自看见「本进程没人在跑」，于是各起一个 run——两轮事件往同一条 WS 上推，
额度也扣两份。真相搬进 DB 之后，判定与占位是**同一条 UPDATE**：

    UPDATE threads SET active_run_id=:new, run_status='running', active_query=:q
     WHERE id=:tid AND (active_run_id IS NULL OR run_status != 'running')

数据库保证同时到达的两条只有一条影响行数为 1。**原子性不再依赖「中间没有 await」**——那是 1-1
里把预扣硬挤到幂等判定之前的原因，这一层不再需要它（预扣的位置留待 1-4 一并收拾）。

**三种结局不合成一个布尔值**（调用方要按它们做完全不同的事）：

- ``started``：抢到了，照常起任务。
- ``already_running``：同一句话又发了一遍（刷新 / 双击）→ 领回原任务，不重跑。
- ``replaced``：同一个 thread 换了一句话 → 覆盖重发，得把取消送给**旧 run**（它可能在另一台
  worker 上，所以 :class:`RunClaim` 要把旧 run_id 带出来）。

**释放按身份，不盲清。** 覆盖重发时旧 run 的 finally 晚几个 tick 才跑，盲清会把刚接班的新 run
的登记摘掉，之后谁也认不出这个 thread 正忙。所以 :func:`release_thread_run` 带
``WHERE active_run_id = :mine``——与 ``active_tasks`` 那处 ``handle.task is current_task()``
同一手法。

**没有 TTL 扫表。** 进程被 kill -9 时行会停在 running，这个 thread 就再也发不出新任务。兜底不是
后台清理协程，而是 :data:`STALE_RUN_SEC`：``updated_at`` 超过它的 running 行视为死的，下一个请求
直接接管（值取得比 ``MAIN_AGENT_TIMEOUT_SEC`` 大，正在跑的 run 不会被误判——同 ``run_holds``
的 ``expires_at`` 一个思路）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal, cast

from sqlalchemy import CursorResult, Executable, or_, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Thread
from app.db.session import session_factory
from app.utils.env import env_int

logger = logging.getLogger("shoppingx.runs")

#: ``active_query`` 的列宽。比较也按截断后的值做——前 500 字完全相同的两句长 query 会被判成
#: 「同一句」，代价是领回原任务而不是重跑，可接受。
QUERY_MAX = 500

#: 多久没动静的 running 行算死的（进程被 kill -9 留下的）。必须 > 单轮最长耗时
#: （``MAIN_AGENT_TIMEOUT_SEC=300``），否则在跑的 run 会被后来者接管，同一个 thread 真的跑两个。
STALE_RUN_SEC = env_int("THREAD_STALE_RUN_SEC", 900)

Outcome = Literal["started", "already_running", "replaced"]


@dataclass(frozen=True)
class RunClaim:
    """一次认领的结果。``previous_run_id`` 只在两种非 started 的结局里有值。"""

    outcome: Outcome
    previous_run_id: str | None = None

    @property
    def can_start(self) -> bool:
        """能不能真的起任务（``replaced`` 也要起——只是得先把旧的掐掉）。"""
        return self.outcome != "already_running"


def _stale_before() -> datetime:
    return datetime.now(UTC) - timedelta(seconds=STALE_RUN_SEC)


async def _rowcount(db: AsyncSession, stmt: Executable) -> int:
    result = cast(CursorResult, await db.execute(stmt))
    return result.rowcount or 0


async def claim_thread_run(thread_id: str, run_id: str, query: str) -> RunClaim:
    """认领这个 thread 的「在跑」位置，返回三种结局之一（见模块 docstring）。

    **第一条 UPDATE 就是判定本身**：抢到（影响 1 行）= 没有别人在跑，或者在跑的那位已经过期。
    抢不到才回头读一行看是谁、跑的是哪句话。这个顺序不能倒过来——先读后判是「读后判」，两个并发
    请求会双双读到「空闲」再各自写入。

    **覆盖重发那条路也是条件更新**（``WHERE active_run_id = :旧``）：读到旧 run 与真正接班之间，
    旧 run 可能刚好结束、另一个请求已经接了班；带上旧 id 就只有一个人能接管，抢输的那个按
    ``already_running`` 领回——宁可让他去看已经在跑的那轮，也不能两轮一起跑。
    """
    text = query[:QUERY_MAX]
    async with session_factory()() as db:
        free = update(Thread).where(
            Thread.id == thread_id,
            or_(
                Thread.active_run_id.is_(None),
                Thread.run_status != "running",
                Thread.updated_at < _stale_before(),
            ),
        )
        taken = await _rowcount(
            db, free.values(active_run_id=run_id, run_status="running", active_query=text)
        )
        await db.commit()
        if taken:
            return RunClaim("started")

        row = await db.get(Thread, thread_id)
        if row is None:
            # 行不该不存在（起任务前一定先登记归属），但真不存在时不能把用户卡死：放行，
            # 这一轮退回「进程内 active_tasks 说了算」的老语义，并留一条日志说明真相层缺位。
            logger.warning("thread 未登记，幂等第 1 层退回进程内：thread_id=%s", thread_id)
            return RunClaim("started")

        previous = row.active_run_id
        if row.active_query == text:
            return RunClaim("already_running", previous)

        replaced = await _rowcount(
            db,
            update(Thread)
            .where(Thread.id == thread_id, Thread.active_run_id == previous)
            .values(active_run_id=run_id, run_status="running", active_query=text),
        )
        await db.commit()
        if replaced:
            return RunClaim("replaced", previous)
        return RunClaim("already_running", previous)


async def release_thread_run(thread_id: str, run_id: str) -> bool:
    """把这个 thread 标回空闲——**仅当**在跑的还是我。返回是否真的清掉了。

    盲清会摘掉刚接班的那位（覆盖重发时旧 run 的 finally 晚几个 tick 才跑），所以条件里带
    ``active_run_id = :mine``。返回 False 不是错误，是「我已经被换下来了」，调用方不必做任何事。
    """
    async with session_factory()() as db:
        cleared = await _rowcount(
            db,
            update(Thread)
            .where(Thread.id == thread_id, Thread.active_run_id == run_id)
            .values(active_run_id=None, run_status="idle", active_query=""),
        )
        await db.commit()
        return bool(cleared)


async def active_run_id(thread_id: str) -> str | None:
    """这个 thread 此刻在跑的 run（过期的当没在跑）。排障与测试用，主链路不依赖它。"""
    async with session_factory()() as db:
        row = await db.get(Thread, thread_id)
        if row is None or row.run_status != "running" or row.active_run_id is None:
            return None
        updated = row.updated_at
        if updated is not None:
            # SQLite 存的是不带时区的 UTC 串，读回来是 naive；MySQL / PG 读回来带 tz。两种都要能比。
            if updated.tzinfo is None:
                updated = updated.replace(tzinfo=UTC)
            if updated < _stale_before():
                return None
        return row.active_run_id
