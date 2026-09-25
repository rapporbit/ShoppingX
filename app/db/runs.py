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

from sqlalchemy import CursorResult, Executable, func, or_, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.accounts import TITLE_MAX
from app.db.models import Thread, User
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


async def claim_thread_and_run(
    thread_id: str,
    run_id: str,
    query: str,
    *,
    user_id: str,
    verify_user: bool,
    title_max: int = TITLE_MAX,
) -> RunClaim:
    """归属登记 + 抢「在跑」位置，**一个会话、一次 commit**（起任务的热路径）。

    原先是两步：``accounts.claim_thread``（读行 → 校验属主 → 插行或顶 ``updated_at`` → commit）
    再 ``claim_thread_run``（条件 UPDATE → commit），同一行写两遍、两个会话、两次提交。
    合并后续聊的常见路径只剩**一条** UPDATE：属主、空闲、顶时间戳、占位同在一个 WHERE 里，
    影响 1 行就全部成立，原子性仍由数据库的条件更新保证（见模块 docstring）。

    影响 0 行才回头读一行分辨是哪种：行不存在（首轮 → 插行，插时就带 running）、属主不是自己
    （``PermissionError``）、同一句话（``already_running``）、换了一句话（``replaced``）。
    异常语义与 ``claim_thread`` 一致：越权 ``PermissionError``，凭证用户不存在 ``LookupError``。
    """
    text = query[:QUERY_MAX]
    title = query[:title_max]
    now = datetime.now(UTC)
    running = {"active_run_id": run_id, "run_status": "running", "active_query": text}
    async with session_factory()() as db:
        # **本事务用 READ COMMITTED**：MySQL 默认 REPEATABLE READ 下，快路径那条 UPDATE 命中不存在的
        # 行（首轮）会加间隙锁，并发首轮各持一把再去 INSERT 就互相等成死锁（1213）——而且 id 不同、
        # 只是落在同一个间隙里的新会话也会互卡，500 路新会话突发必现。RC 不加间隙锁；条件更新的
        # 原子性不受影响（同一行上的两条 UPDATE 后到的等锁、拿到后按最新提交版本重判 WHERE）。
        if db.bind.dialect.name == "mysql":
            await db.connection(execution_options={"isolation_level": "READ COMMITTED"})
        fast = (
            update(Thread)
            .where(
                Thread.id == thread_id,
                Thread.user_id == user_id,
                or_(
                    Thread.active_run_id.is_(None),
                    Thread.run_status != "running",
                    Thread.updated_at < _stale_before(),
                ),
            )
            .values(
                **running,
                updated_at=now,
                title=func.coalesce(func.nullif(Thread.title, ""), title),
            )
        )
        if await _rowcount(db, fast):
            await db.commit()
            return RunClaim("started")

        row = await db.get(Thread, thread_id)
        if row is None:
            if verify_user and await db.get(User, user_id) is None:
                raise LookupError("凭证对应的用户不存在")
            db.add(Thread(id=thread_id, user_id=user_id, title=title, updated_at=now, **running))
            try:
                await db.commit()
                return RunClaim("started")
            except IntegrityError:
                # 并发首轮：另一个请求刚插了同一个 thread_id。回滚后按「行已存在」重新分辨。
                await db.rollback()
                row = await db.get(Thread, thread_id)
                if row is None:  # 插入冲突后又被删掉，极端罕见：让调用方当作库故障
                    raise
        if row.user_id != user_id:
            raise PermissionError("无权访问该会话")

        previous = row.active_run_id
        if row.active_query == text:
            # 领回原任务。仍把 updated_at 顶一下：原先的归属登记每次都会顶（侧栏按它排序）。
            await db.execute(update(Thread).where(Thread.id == thread_id).values(updated_at=now))
            await db.commit()
            return RunClaim("already_running", previous)

        replaced = await _rowcount(
            db,
            update(Thread)
            .where(Thread.id == thread_id, Thread.active_run_id == previous)
            .values(**running, updated_at=now),
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
