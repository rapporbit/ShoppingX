"""数据库引擎与会话（M16 / M17）——建连接、跑迁移、给路由发 session。

**建表为什么从 ``create_all`` 换成 Alembic（M17）。** 起初两张表从零建，``create_all`` 最省事。
但它**只建不存在的表、不改已存在的表**：一旦开始往库里搬东西（收藏、跨会话历史……）就要动表结构，
而 ``create_all`` 对已有表的字段变更**静默不生效**——本地删库重建看着一切正常，线上那份老库悄悄少
一列，直到某个查询炸了才发现。趁库里只有两张表、结构还简单时换掉，代价最小。

**线上那份老库怎么接管（关键）。** VPS 上的 ``var/globex.db`` 是 ``create_all`` 建的，里面有真实
注册用户，却**没有 ``alembic_version`` 表**。直接 ``upgrade head`` 会去执行「建 users 表」→ 表已存在
→ 启动即崩。所以 :func:`init_db` 先探一眼：**有业务表却没有版本号** = create_all 时代的老库，其结构
与基线迁移等价，于是只 ``stamp`` 贴上版本号（不执行任何 DDL），再继续 upgrade 后续迁移。空库则直接
从头跑全部迁移。两条路都幂等，重启多少次都一样。

**驱动相关的调教全在 :func:`_engine_kwargs` 与 :func:`_tune_sqlite_connection` 两处**（逐条理由见
它们的 docstring）：SQLite 走「连接级 PRAGMA」（外键 / WAL / busy_timeout），MySQL / PostgreSQL 走
「连接池参数」（限池 / 预检 / 回收）。**WAL 是批 2 多进程的前提**——API 与 worker 拆成两个进程后，
默认的 rollback journal 会让它们的读写互相把对方挡在 "database is locked" 上。
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from typing import Any

from alembic import command
from alembic.config import Config
from sqlalchemy import event, inspect
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.utils.env import env_int
from app.utils.path_utils import PROJECT_ROOT

logger = logging.getLogger("shoppingx.db")

# 库文件默认落 var/（**不能放 data/**：生产 compose 里 data 是只读挂载 `:ro`，SQLite 要写就得
# 有个可写的持久目录，故单列 var/ 并在编排里挂命名卷）。整个 DSN 可由 DATABASE_URL 覆盖——
# 将来换 Postgres 只改这一个环境变量，模型与查询一行不动。
_DEFAULT_DB = PROJECT_ROOT / "var" / "globex.db"

#: 基线迁移的 revision id：它建的表结构 == create_all 时代的 users/threads。老库 stamp 到这里。
BASELINE_REVISION = "0001_accounts"


def database_url() -> str:
    """DSN 的唯一事实来源——应用、迁移（migrations/env.py）、脚本都向这里要，不各读各的。"""
    url = os.environ.get("DATABASE_URL", "").strip()
    if url:
        return url
    _DEFAULT_DB.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite+aiosqlite:///{_DEFAULT_DB}"


def _engine_kwargs(dsn: str) -> dict[str, Any]:
    """按驱动分支引擎参数——**池参数对 SQLite 与对网络库根本不是一回事**。

    - **SQLite**：库就是本地一个文件，"连接" 只是打开文件，没有握手 / 认证 / 空闲被服务端掐断
      这些问题，所以 ``pool_size`` / ``pool_recycle`` 之类**一个都不设**（设了也只是让 SQLAlchemy
      多绕一层）。它真正需要的是连接级 PRAGMA，见 :func:`_tune_sqlite_connection`。
    - **MySQL / PostgreSQL**：连接是网络资源，要限池、要预检、要定期回收。``pool_pre_ping`` 是
      这里最值钱的一条：连接在池里躺着时被服务端/中间件单方面关掉，是**取出来用的那一刻**才发现
      的——表现为随机一次请求 500，重试就好，最难查的一类。
    - ``pool_recycle`` 在 **MySQL 上必须给**：服务端 ``wait_timeout``（云厂商常调到几分钟）到点
      就掐空闲连接，池自己不知道。PostgreSQL 默认不掐空闲连接，给一个更宽松的值即可。
    """
    if dsn.startswith("sqlite"):
        # SQLite 默认禁止跨线程复用连接，而 async 引擎本就会在不同线程间调度，不关掉这条会随机
        # 报 "SQLite objects created in a thread can only be used in that same thread"。
        return {"connect_args": {"check_same_thread": False}}

    kwargs: dict[str, Any] = {
        "pool_size": env_int("DB_POOL_SIZE", 5),
        "max_overflow": env_int("DB_MAX_OVERFLOW", 10),
        "pool_timeout": env_int("DB_POOL_TIMEOUT", 30),
        "pool_pre_ping": True,
    }
    if "mysql" in dsn:
        kwargs["pool_recycle"] = env_int("DB_POOL_RECYCLE", 1800)
        # utf8mb4 而不是 MySQL 那个名不副实的 "utf8"（只有 3 字节，装不下 emoji 与部分生僻字）。
        # 商品标题里 emoji 遍地，编码不对不是乱码而是**插入直接报错**。
        kwargs["connect_args"] = {"charset": "utf8mb4"}
    else:
        kwargs["pool_recycle"] = env_int("DB_POOL_RECYCLE", 3600)
    return kwargs


def make_engine(url: str | None = None) -> AsyncEngine:
    """按 DSN 造一个 async 引擎，并挂上该驱动需要的调教。

    独立成函数是为了让**迁移能用自己的临时引擎**：Alembic 的入口是同步 API，应用启动时得在
    ``asyncio.to_thread`` 里调它，线程里会开一个**新的事件循环**——而 async 引擎的连接池绑定创建它的
    循环，把应用那个共享引擎拿去跨循环用是自找偶发故障。故迁移一律现造现弃（用完 ``dispose``）。
    """
    dsn = url or database_url()
    engine = create_async_engine(dsn, **_engine_kwargs(dsn))
    if dsn.startswith("sqlite"):
        event.listen(engine.sync_engine, "connect", _tune_sqlite_connection)
    return engine


def _tune_sqlite_connection(dbapi_conn: Any, _record: Any) -> None:
    """每条新连接都设一遍 PRAGMA——SQLite 的 PRAGMA 是**连接级**的（``journal_mode`` 除外，
    它写在库文件头里、一次生效永久有效），连接池里换一条连接就得重设，所以只能挂 connect 事件。

    三条各管一件事：

    - ``foreign_keys=ON``：**SQLite 默认不执行外键约束**（为兼容老库）。不显式打开的话，
      ``threads.user_id`` 指向一个不存在的用户也能插进去，外键形同虚设。
    - ``journal_mode=WAL``：**多进程写同一个库的前提**（批 2 起 API 与 worker 是两个进程）。
      默认的 rollback journal 下，写事务会拿排他锁**把读也挡住**，两个进程稍一并发就互相 "database
      is locked"；WAL 下读写不互斥（读旧快照、写追加到 -wal），只有写与写才排队。
    - ``busy_timeout``：写与写终究要排队，默认 busy handler 是**立刻**抛 "database is locked"——
      连 1 毫秒都不等。设成 5 秒 = 撞上并发写时先自旋等一会儿，绝大多数争用在这几毫秒里就化解了。

    WAL 设不上（内存库 / 只读挂载 / 网络文件系统上不支持）时**只告警不抛**：那种部署本来就不是
    多进程写，退回默认日志模式照样能跑，为它把整个应用起不来是本末倒置。
    """
    cur = dbapi_conn.cursor()
    try:
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute(f"PRAGMA busy_timeout={env_int('SQLITE_BUSY_TIMEOUT_MS', 5000)}")
        cur.execute("PRAGMA journal_mode=WAL")
        mode = (cur.fetchone() or [""])[0]
        if str(mode).lower() != "wal":
            logger.warning("SQLite 未能切到 WAL（当前 %s），多进程并发写会更容易撞锁", mode)
    except Exception:  # noqa: BLE001 —— 调教失败不该让应用起不来，见 docstring
        logger.warning("SQLite PRAGMA 调教失败，退回默认设置", exc_info=True)
    finally:
        cur.close()


_engine = make_engine()


def get_engine() -> AsyncEngine:
    """应用共享的那个引擎（请求 / 后台任务用）。迁移**不要**用它，见 :func:`make_engine`。"""
    return _engine


_Session = async_sessionmaker(_engine, expire_on_commit=False)


def _alembic_config() -> Config:
    cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
    # script_location 在 ini 里是相对路径，而进程的 cwd 未必是项目根（uvicorn 可以从任何地方起）。
    cfg.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
    return cfg


def _table_names(sync_conn: Connection) -> set[str]:
    return set(inspect(sync_conn).get_table_names())


async def _existing_tables() -> set[str]:
    engine = make_engine()
    try:
        async with engine.connect() as conn:
            return await conn.run_sync(_table_names)
    finally:
        await engine.dispose()


def _migrate_sync(stamp_baseline: bool) -> None:
    """在**独立线程**里跑 Alembic（同步 API）。migrations/env.py 会在这里开自己的事件循环。"""
    cfg = _alembic_config()
    if stamp_baseline:
        command.stamp(cfg, BASELINE_REVISION)
    command.upgrade(cfg, "head")


async def init_db() -> None:
    """把库升到最新版本（启动时调一次，幂等）。

    老库（create_all 建的、无 ``alembic_version``）先 stamp 到基线再 upgrade——见模块 docstring。
    ``to_thread``：Alembic 的 command API 是同步的，且 env.py 内部要 ``asyncio.run``，在已经跑着事件
    循环的 lifespan 里直接调会撞 "asyncio.run() cannot be called from a running event loop"。
    """
    tables = await _existing_tables()
    legacy = "alembic_version" not in tables and "users" in tables
    if legacy:
        logger.info(
            "检测到 create_all 时代的老库（%d 张表，无版本号）：stamp 到基线后再迁移", len(tables)
        )
    await asyncio.to_thread(_migrate_sync, legacy)


async def get_db() -> AsyncIterator[AsyncSession]:
    """FastAPI 依赖：给一个请求作用域的 session，请求结束自动关。"""
    async with _Session() as session:
        yield session


def session_factory() -> async_sessionmaker[AsyncSession]:
    """给非请求上下文（后台任务 / 脚本 / 测试）拿 session 用——它们没有 FastAPI 的依赖注入。"""
    return _Session
