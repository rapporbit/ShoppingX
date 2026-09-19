"""多副本首启的迁移串行闸（阶段 1 · 条 9）。

要挡的事故：四个容器（2 API + 2 worker）同时首启、一起 ``upgrade head``。MySQL 的 DDL 不是事务性的，
两个进程同时建同一张表 → 一个撞 ``1050 Table already exists`` 半途退出，留一张建了一半的库，重跑
还撞同一个错。真 MySQL 不在单测里起（那是 docker-compose.multi.yml 的活），这里测三件能测的：

1. SQLite 直接放行（不绕锁，本地开发与 pytest fixture 一条都别受影响）；
2. 锁的 SQL 真按方言发出去（MySQL ``GET_LOCK`` / PostgreSQL ``pg_try_advisory_lock``），并成对释放；
3. **等锁超时后的两条分支**——库已到 head 就放行，没到就拒绝启动。这是最关键的一条：
   判反了要么把副本饿死在启动上，要么让它带着半截库开张。
"""

from __future__ import annotations

from typing import Any

import pytest

from app.db import session as db_session


class _FakeResult:
    def __init__(self, value: Any) -> None:
        self._value = value

    def scalar(self) -> Any:
        return self._value


class _FakeConn:
    """只记 SQL 与参数的假连接——锁的行为全在「发了什么语句」上，不需要真库。"""

    def __init__(self, replies: list[Any]) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def execute(self, stmt: Any, params: dict[str, Any] | None = None) -> _FakeResult:
        self.calls.append((str(stmt), params or {}))
        return _FakeResult(self.replies.pop(0) if self.replies else None)


MYSQL_DSN = "mysql+aiomysql://u:p@h:3306/globex?charset=utf8mb4"
PG_DSN = "postgresql+asyncpg://u:p@h:5432/globex"


@pytest.mark.asyncio
async def test_sqlite_skips_the_lock_entirely() -> None:
    """SQLite 是单文件、DDL 在事务里、失败自动回滚，重跑即可——不该为它开一条额外连接。"""
    async with db_session._migration_lock("sqlite+aiosqlite:///:memory:") as got:
        assert got is True


@pytest.mark.asyncio
async def test_mysql_named_lock_is_acquired_and_released() -> None:
    conn = _FakeConn([1])
    assert await db_session._acquire_lock(conn, MYSQL_DSN, 42) is True
    sql, params = conn.calls[0]
    assert "GET_LOCK" in sql
    assert params == {"name": db_session.MIGRATION_LOCK_NAME, "wait": 42}

    await db_session._release_lock(conn, MYSQL_DSN)
    assert "RELEASE_LOCK" in conn.calls[1][0]


@pytest.mark.asyncio
async def test_mysql_lock_timeout_reports_false() -> None:
    """GET_LOCK 等到超时返回 0（出错返回 NULL）——两种都不能当成拿到了。"""
    assert await db_session._acquire_lock(_FakeConn([0]), MYSQL_DSN, 1) is False
    assert await db_session._acquire_lock(_FakeConn([None]), MYSQL_DSN, 1) is False


@pytest.mark.asyncio
async def test_postgres_polls_try_advisory_lock() -> None:
    conn = _FakeConn([None, 1])  # 第一次没拿到，第二次拿到
    assert await db_session._acquire_lock(conn, PG_DSN, 10) is True
    assert len(conn.calls) == 2
    assert all("pg_try_advisory_lock" in sql for sql, _ in conn.calls)
    key = conn.calls[0][1]["key"]
    assert key == db_session._advisory_key(db_session.MIGRATION_LOCK_NAME)

    await db_session._release_lock(conn, PG_DSN)
    assert "pg_advisory_unlock" in conn.calls[2][0]


def test_advisory_key_is_stable_across_processes() -> None:
    """用 sha1 而不是内置 hash()：后者带进程级随机盐，两个副本会算出不同的 key，锁形同虚设。"""
    assert db_session._advisory_key("globex_alembic") == db_session._advisory_key("globex_alembic")
    assert db_session._advisory_key("a") != db_session._advisory_key("b")
    assert -(2**63) <= db_session._advisory_key("globex_alembic") < 2**63


def _async_value(value: Any) -> Any:
    async def _fn() -> Any:
        return value

    return _fn


# ---------- 等锁超时后的两条分支 ----------
def _lock_returning(got: bool) -> Any:
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _fake(_dsn: str):  # noqa: ANN202
        yield got

    return _fake


@pytest.mark.asyncio
async def test_timeout_passes_when_db_is_already_at_head(monkeypatch: pytest.MonkeyPatch) -> None:
    """等超时未必是坏事：先起的副本已经升完了，这时候还把自己饿死在启动上才是错的。"""
    monkeypatch.setattr(db_session, "_migration_lock", _lock_returning(False))
    monkeypatch.setattr(db_session, "_head_revision", lambda: "0014_thread_active_run")
    monkeypatch.setattr(db_session, "_current_revision", _async_value("0014_thread_active_run"))
    ran: list[bool] = []
    monkeypatch.setattr(db_session, "_migrate_sync", lambda legacy: ran.append(legacy))

    await db_session.init_db()
    assert ran == []  # 放行，但绝不能自己再跑一遍迁移


@pytest.mark.asyncio
async def test_timeout_refuses_to_start_when_db_is_behind(monkeypatch: pytest.MonkeyPatch) -> None:
    """版本没到 head 还拿不到锁 = 另一个副本卡在迁移里。带着半截库开张比起不来危险得多。"""
    monkeypatch.setattr(db_session, "_migration_lock", _lock_returning(False))
    monkeypatch.setattr(db_session, "_head_revision", lambda: "0014_thread_active_run")
    monkeypatch.setattr(db_session, "_current_revision", _async_value("0012_memory_facts"))

    with pytest.raises(RuntimeError, match="迁移锁"):
        await db_session.init_db()


@pytest.mark.asyncio
async def test_tables_are_probed_inside_the_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    """探库必须在锁**里面**：等锁那几十秒里别的副本正在建表，锁外探到的清单一拿到锁就过期，
    legacy 会判反（探时无 users、拿到锁时已经有了 → 照样 stamp → 版本号被按回基线）。"""
    order: list[str] = []
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _tracking(_dsn: str):  # noqa: ANN202
        order.append("lock")
        yield True
        order.append("unlock")

    async def _tables() -> set[str]:
        order.append("probe")
        return {"alembic_version", "users"}

    monkeypatch.setattr(db_session, "_migration_lock", _tracking)
    monkeypatch.setattr(db_session, "_existing_tables", _tables)
    monkeypatch.setattr(
        db_session, "_migrate_sync", lambda legacy: order.append(f"migrate:{legacy}")
    )

    await db_session.init_db()
    assert order == ["lock", "probe", "migrate:False", "unlock"]
