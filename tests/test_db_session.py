"""数据库引擎装配的单测（批2-5）：SQLite 的 WAL / busy_timeout，以及按驱动分支的池参数。

WAL 这条是**多进程写同一个库的前提**（批 2 起 API 与 worker 是两个进程），所以要真的开一个引擎、
真的读回 ``PRAGMA journal_mode`` 来断言——只断言「代码里写了这句 PRAGMA」等于没测。
MySQL / PostgreSQL 的池参数无法在没有真库的情况下起引擎，故只测 :func:`_engine_kwargs` 这个纯函数。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import text

from app.db.session import _engine_kwargs, make_engine


# ---------- SQLite：连接级 PRAGMA ----------
@pytest.mark.asyncio
async def test_sqlite_connection_is_wal_with_busy_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SQLITE_BUSY_TIMEOUT_MS", "7000")
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path / 'wal.db'}")
    try:
        async with engine.connect() as conn:
            assert (await conn.execute(text("PRAGMA journal_mode"))).scalar_one() == "wal"
            assert (await conn.execute(text("PRAGMA busy_timeout"))).scalar_one() == 7000
            # 外键强制是 M17 就有的，别在加 WAL 时把它挤掉（同一条 connect 事件里）。
            assert (await conn.execute(text("PRAGMA foreign_keys"))).scalar_one() == 1
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_sqlite_pragmas_apply_to_every_pooled_connection(tmp_path: Path) -> None:
    """PRAGMA 是**连接级**的，池里换一条就得重设一次——只在建库时设一次是经典坑。"""
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path / 'pool.db'}")
    try:
        for _ in range(3):  # 每次进出连接池都重新取一条连接
            async with engine.connect() as conn:
                assert (await conn.execute(text("PRAGMA foreign_keys"))).scalar_one() == 1
                assert (await conn.execute(text("PRAGMA busy_timeout"))).scalar_one() > 0
    finally:
        await engine.dispose()


# ---------- 按驱动分支的引擎参数 ----------
def test_sqlite_gets_no_pool_params() -> None:
    """本地文件库没有「连接被服务端掐断」这回事，池参数对它是纯噪声。"""
    kwargs = _engine_kwargs("sqlite+aiosqlite:///var/globex.db")
    assert kwargs == {"connect_args": {"check_same_thread": False}}


def test_mysql_gets_pool_and_utf8mb4(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DB_POOL_SIZE", "9")
    kwargs = _engine_kwargs("mysql+aiomysql://u:p@h/globex")
    assert kwargs["pool_size"] == 9
    assert kwargs["pool_pre_ping"] is True  # 连接在池里被服务端掐掉，是取出来用时才发现的
    assert kwargs["pool_recycle"] == 1800  # MySQL 的 wait_timeout 会单方面关空闲连接
    assert kwargs["connect_args"] == {"charset": "utf8mb4"}  # 3 字节的 "utf8" 装不下 emoji


def test_postgres_gets_pool_without_charset() -> None:
    kwargs = _engine_kwargs("postgresql+asyncpg://u:p@h/globex")
    assert kwargs["pool_pre_ping"] is True
    assert kwargs["pool_recycle"] == 3600  # 服务端不主动掐空闲连接，回收可以更宽松
    assert "connect_args" not in kwargs  # charset 是 MySQL 独有的坑，别抄给 PG


def test_pool_params_are_env_tunable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DB_MAX_OVERFLOW", "3")
    monkeypatch.setenv("DB_POOL_TIMEOUT", "11")
    monkeypatch.setenv("DB_POOL_RECYCLE", "60")
    kwargs = _engine_kwargs("postgresql+asyncpg://u:p@h/globex")
    assert (kwargs["max_overflow"], kwargs["pool_timeout"], kwargs["pool_recycle"]) == (3, 11, 60)
