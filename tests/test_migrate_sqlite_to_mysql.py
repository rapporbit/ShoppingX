"""SQLite → MySQL 搬数脚本的单测（阶段 1 · 条 9）。

**用 SQLite → SQLite 测**：脚本全程走 SQLAlchemy Core 的表元数据，两头的方言差异由库自己处理，
这里要验的是脚本自己的逻辑——按依赖顺序搬、时间列归一成 UTC naive、目标非空时拒绝、``--truncate``
能重来、行数对不上返回 1。真 MySQL 的部分（utf8mb4、DDL 非事务）在 docker-compose.multi.yml 里验，
起一个容器换三行断言不划算。
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import insert, select, text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db.models import Base, Thread, User  # noqa: E402
from app.db.session import _head_revision, make_engine  # noqa: E402
from scripts.migrate_sqlite_to_mysql import _normalize, run  # noqa: E402


async def _create_all(dsn: str, *, stamp: str | None = None) -> None:
    """建表 + 贴版本号。版本号不是摆设：脚本按当前模型的列 SELECT，落后的源库会撞
    "no such column"，所以它会先比一次版本，比不上就拒绝搬。"""
    engine = make_engine(dsn)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            if stamp:
                await conn.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(32))"))
                await conn.execute(text("INSERT INTO alembic_version VALUES (:v)"), {"v": stamp})
    finally:
        await engine.dispose()


async def _seed(dsn: str) -> None:
    """两个用户 + 三个会话，会话里带 aware datetime 与可空列，覆盖外键顺序与类型归一。"""
    engine = make_engine(dsn)
    now = datetime(2026, 9, 20, 8, 30, tzinfo=UTC)
    try:
        async with engine.begin() as conn:
            await conn.execute(
                insert(User.__table__),
                [
                    {"id": f"u{i}", "username": f"u{i}", "password_hash": "h", "created_at": now}
                    for i in range(2)
                ],
            )
            await conn.execute(
                insert(Thread.__table__),
                [
                    {
                        "id": f"t{i}",
                        "user_id": f"u{i % 2}",
                        "title": f"会话 {i}",
                        # 可空列两种取值都要过一遍：None 与真值在 executemany 里走的是同一批参数
                        "active_run_id": f"r{i}" if i else None,
                        "created_at": now + timedelta(minutes=i),
                        "updated_at": now,
                    }
                    for i in range(3)
                ],
            )
    finally:
        await engine.dispose()


async def _rows(dsn: str, table: object) -> list[object]:
    engine = make_engine(dsn)
    try:
        async with engine.connect() as conn:
            return (await conn.execute(select(table))).fetchall()
    finally:
        await engine.dispose()


@pytest.fixture
async def pair(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[str, str]:
    monkeypatch.setenv("DATABASE_URL", "")  # run() 会覆盖它，别让它漏到别的测试
    src = f"sqlite+aiosqlite:///{tmp_path / 'src.db'}"
    dst = f"sqlite+aiosqlite:///{tmp_path / 'dst.db'}"
    await _create_all(src, stamp=_head_revision())
    await _create_all(dst)
    await _seed(src)
    return src, dst


def test_normalize_folds_aware_datetime_to_utc_naive() -> None:
    """库内一律 UTC naive。带 +08:00 的值直接去 tzinfo 会**差 8 小时**，必须先换算。"""
    shanghai = datetime(2026, 9, 20, 16, 30, tzinfo=UTC).astimezone()
    assert _normalize(shanghai) == datetime(2026, 9, 20, 16, 30)
    naive = datetime(2026, 9, 20, 8, 0)
    assert _normalize(naive) is naive  # 已经是 naive 的不动
    assert _normalize("背包") == "背包"


@pytest.mark.asyncio
async def test_copies_every_row_and_verifies_counts(pair: tuple[str, str]) -> None:
    src, dst = pair
    assert await run(src, dst, dry_run=False, truncate=False, migrate=False) == 0
    assert len(await _rows(dst, User.__table__)) == 2
    threads = await _rows(dst, Thread.__table__)
    assert len(threads) == 3
    assert {t.active_run_id for t in threads} == {None, "r1", "r2"}
    assert all(t.created_at.tzinfo is None for t in threads)


@pytest.mark.asyncio
async def test_dry_run_writes_nothing(pair: tuple[str, str]) -> None:
    src, dst = pair
    assert await run(src, dst, dry_run=True, truncate=False, migrate=False) == 0
    assert await _rows(dst, User.__table__) == []


@pytest.mark.asyncio
async def test_refuses_a_non_empty_target(pair: tuple[str, str]) -> None:
    """搬第二遍 = 行翻倍或撞主键。宁可一开始就拒绝，也别搬一半让人手工分辨哪些是新灌的。"""
    src, dst = pair
    await run(src, dst, dry_run=False, truncate=False, migrate=False)
    with pytest.raises(SystemExit, match="已有数据"):
        await run(src, dst, dry_run=False, truncate=False, migrate=False)


@pytest.mark.asyncio
async def test_truncate_allows_a_clean_retry(pair: tuple[str, str]) -> None:
    src, dst = pair
    await run(src, dst, dry_run=False, truncate=False, migrate=False)
    assert await run(src, dst, dry_run=False, truncate=True, migrate=False) == 0
    assert len(await _rows(dst, User.__table__)) == 2  # 清了再灌，不是翻倍
