"""起服前的形态闸（阶段 1 条 7）：库不是 MySQL / 队列 Redis 不通 → 拒绝启动。

**为什么这道闸值得单独一个文件。** 它挡的两种坏法都是静默的：SQLite 下多副本各算各的配额与归属、
Redis 不通时任务入不了队，两者都不报错，只会让用户看到「额度对不上」「一直在转圈」。所以这里钉的
不是某段代码的行为，而是「这两种误配必须在启动时就变红」这条约定本身。

单元测试不受这道闸影响：ASGITransport 不跑 lifespan，所以整套 API 用例照旧在 SQLite 上跑。
"""

from __future__ import annotations

import logging

import pytest

from app import deployment


@pytest.fixture
def redis_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    """把 Redis 探活钉成成功——只验库那一半时不必真连一个实例。"""

    async def _ok(_url: str) -> None:
        return None

    monkeypatch.setattr(deployment, "ping_redis", _ok)


async def test_rejects_sqlite(monkeypatch: pytest.MonkeyPatch, redis_ok: None) -> None:
    monkeypatch.setattr(deployment, "database_url", lambda: "sqlite+aiosqlite:///./var/globex.db")
    with pytest.raises(RuntimeError, match="MySQL"):
        await deployment.assert_deployment_deps()


async def test_accepts_mysql(monkeypatch: pytest.MonkeyPatch, redis_ok: None) -> None:
    monkeypatch.setattr(deployment, "database_url", lambda: "mysql+aiomysql://root@db:3306/globex")
    await deployment.assert_deployment_deps()  # 不抛即通过


async def test_rejects_unreachable_redis(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(deployment, "database_url", lambda: "mysql+aiomysql://root@db:3306/globex")
    monkeypatch.setattr(deployment, "queue_redis_url", lambda: "redis://127.0.0.1:1/0")
    with pytest.raises(RuntimeError, match="Redis 不可达"):
        await deployment.assert_deployment_deps()


async def test_warns_when_grace_is_shorter_than_one_turn(
    monkeypatch: pytest.MonkeyPatch, redis_ok: None, caplog: pytest.LogCaptureFixture
) -> None:
    """grace < 单轮超时只警告不拒绝：它让发布期的任务被掐成 interrupted，是坏配置不是坏形态。"""
    monkeypatch.setattr(deployment, "database_url", lambda: "mysql+aiomysql://root@db:3306/globex")
    monkeypatch.setenv("WORKER_GRACE_SECONDS", "60")
    monkeypatch.setenv("MAIN_AGENT_TIMEOUT_SEC", "300")
    with caplog.at_level(logging.WARNING, logger="shoppingx.deployment"):
        await deployment.assert_deployment_deps()
    assert any("WORKER_GRACE_SECONDS" in r.message for r in caplog.records)
