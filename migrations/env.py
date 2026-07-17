"""Alembic 运行时环境（M17）——把迁移接到本项目的 async 引擎与模型元数据上。

**三处与模板不同的地方，都是踩过的坑：**

1. **DSN 与引擎复用 app/db/session.py，不从 alembic.ini 读。** 事实来源只能有一个：DATABASE_URL。
   模板默认从 ini 拿 url，那样迁移可能跑到与应用不同的库上，而且改后端（SQLite→Postgres）要改两处。

2. **``render_as_batch=True``（SQLite 必须）。** SQLite 只支持极有限的 ALTER TABLE——改列类型、
   加约束、删列一律不支持。batch 模式会把这类变更翻译成「建新表 → 拷数据 → 删旧表 → 改名」。
   不开这个，将来第一条改字段的迁移在 SQLite 上直接报 "unsupported ALTER"。

3. **online 迁移走 async engine + ``run_sync``，且引擎现造现弃。** Alembic 的迁移上下文是同步 API，
   而我们的驱动是 aiosqlite/asyncpg——``run_sync`` 是官方给的桥，``asyncio.run`` 起本地循环。这里
   **不复用 app 那个共享引擎**：应用启动时是在 ``asyncio.to_thread`` 里调 Alembic 的，线程内的
   ``asyncio.run`` 是一个**新事件循环**，而 async 引擎的连接池绑定创建它的循环——跨循环借用是偶发
   故障的经典来源。故这里 ``make_engine()`` 造一个临时的，跑完 ``dispose``。
"""

from __future__ import annotations

import asyncio

from alembic import context
from sqlalchemy import Connection

from app.db.models import Base
from app.db.session import database_url, make_engine

# autogenerate 拿它跟真实库对比，生成 diff。模型是「目标态」的事实来源。
target_metadata = Base.metadata


def _configure(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        render_as_batch=True,  # ← SQLite 的 ALTER 短板，见模块 docstring
        compare_type=True,  # 列类型变更也要被 autogenerate 看见（默认竟是不比的）
    )


def run_migrations_offline() -> None:
    """``--sql`` 模式：不连库，把迁移打印成 SQL 脚本（DBA 审阅 / 生产手动执行时用）。"""
    context.configure(
        url=database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def _do_run(connection: Connection) -> None:
    _configure(connection)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    engine = make_engine()
    try:
        async with engine.connect() as connection:
            await connection.run_sync(_do_run)
    finally:
        await engine.dispose()  # 临时引擎，别把连接留给一个即将结束的事件循环


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
