"""部署形态闸：起服前确认「跨进程的真相」真的在（阶段 1 条 7）。

API 与 worker 两个进程入口共用这一份——两边各写一份校验，迟早漂成「API 起得来、worker 起不来」
那种只在滚动发布中途暴露的形态。放在 ``app/`` 顶层而不是 ``app/api/`` 里，是为了让 worker 不必为
一道校验把整个 FastAPI 模块（连同它 import 的 agent / tools 链）拖进来。
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress

from app.db.session import database_url
from app.queue import queue_redis_url
from app.utils.env import env_int

logger = logging.getLogger("shoppingx.deployment")


async def ping_redis(url: str) -> None:
    """连一次 Redis 并 ``PING``。连不上 / 超时都往外抛，由调用方决定怎么处理。"""
    import redis.asyncio as aredis

    client = aredis.from_url(url, socket_connect_timeout=2.0, socket_timeout=2.0)
    try:
        await asyncio.wait_for(client.ping(), timeout=3.0)
    finally:
        with suppress(Exception):
            await client.aclose()


async def assert_deployment_deps() -> None:
    """起服前的形态闸（阶段 1 条 7）：**跨进程的真相必须真的在**，否则拒绝启动。

    **为什么是 fail-fast 而不是降级。** 多副本是唯一形态：真相只在 MySQL 与 Redis，进程里没有真相。
    DATABASE_URL 指着 SQLite 时，每台副本各有一份自己的 users / threads / run_holds——配额、归属、
    「同 thread 只跑一个」三样全部各算各的，而且**没有任何报错**，要等用户投诉才看得出来。Redis
    不可达同理：任务入不了队，用户对着 running 等到超时。这两种坏法都是静默的，所以只能在启动时拦。

    本地想手工起后端：``docker compose -f docker/docker-compose.multi.yml up -d mysql redis``，再把
    DATABASE_URL / EVENT_REDIS_URL 指过去。单元测试不受影响——ASGITransport 不跑 lifespan。
    """
    dsn = database_url()
    if not dsn.startswith("mysql"):
        raise RuntimeError(
            f"DATABASE_URL 必须是 MySQL（当前：{dsn.split('://')[0]}）。"
            "多副本下 SQLite 让每台副本各有一份配额/归属真相，且不会报错——SQLite 现在只留作单元"
            "测试 fixture。搬数见 scripts/migrate_sqlite_to_mysql.py"
        )

    url = queue_redis_url()
    try:
        await ping_redis(url)
    except Exception as exc:
        raise RuntimeError(
            f"任务队列的 Redis 不可达（{url}）：{exc}。API 只负责入队，Redis 不在就没有队列可入"
        ) from exc

    # worker 的排空窗口必须够一轮任务跑完，否则滚动发布每次都在 interrupted 上收场（用户要重发）。
    grace = env_int("WORKER_GRACE_SECONDS", 330)
    turn_timeout = env_int("MAIN_AGENT_TIMEOUT_SEC", 300)
    if grace < turn_timeout:
        logger.warning(
            "WORKER_GRACE_SECONDS=%d < MAIN_AGENT_TIMEOUT_SEC=%d：发布时正在跑的任务会被掐成 "
            "interrupted。建议 grace ≥ 超时 + 收尾余量（约 %d）",
            grace,
            turn_timeout,
            turn_timeout + 30,
        )
