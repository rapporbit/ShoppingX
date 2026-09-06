"""跨进程共享熔断的确定性单测（批2-5）。

核心是**双进程冒烟**：两个 ``CircuitBreaker`` 实例（模拟 A / B 两个副本，各自的本地计数互不相干）
共用一份假 Redis 状态，走完「A 打开熔断 → B 拒调 → 恢复窗口过 → B 探测成功记录 → A 侧关闭」。
真 Redis 不进单测（同 test_queue / test_backplane 的口径），假 Redis 只实现本模块用到的四个命令。

另外两组：**Redis 故障必须放行**（熔断是优化不是正确性前提，抖一下就全站不可用是自伤），
以及**默认关**时一次 Redis 都不碰。
"""

from __future__ import annotations

from typing import Any

import pytest

from app.utils import shared_breaker
from app.utils.circuit_breaker import CLOSED, OPEN, CircuitBreaker

pytestmark = pytest.mark.asyncio


class FakeRedis:
    """只实现 hgetall / hincrby / hset / expire / delete 五个命令的最小假 Redis。

    ``fail_on`` 里的命令一律抛——用来验「Redis 故障放行」这条降级方向。
    """

    def __init__(self, *, fail_on: set[str] | None = None) -> None:
        self.data: dict[str, dict[str, str]] = {}
        self.fail_on = fail_on or set()
        self.calls: list[str] = []

    def _guard(self, cmd: str) -> None:
        self.calls.append(cmd)
        if cmd in self.fail_on:
            raise ConnectionError(f"fake redis down on {cmd}")

    async def hgetall(self, key: str) -> dict[str, str]:
        self._guard("hgetall")
        return dict(self.data.get(key, {}))

    async def hincrby(self, key: str, field: str, amount: int) -> int:
        self._guard("hincrby")
        bucket = self.data.setdefault(key, {})
        value = int(bucket.get(field, "0")) + amount
        bucket[field] = str(value)
        return value

    async def hset(self, key: str, field: str, value: str) -> int:
        self._guard("hset")
        self.data.setdefault(key, {})[field] = value
        return 1

    async def expire(self, key: str, ttl: int) -> bool:
        self._guard("expire")
        return key in self.data

    async def delete(self, key: str) -> int:
        self._guard("delete")
        return 1 if self.data.pop(key, None) is not None else 0


@pytest.fixture
def shared(monkeypatch: pytest.MonkeyPatch) -> Any:
    """装一份共享状态并把两个时钟都捏在手里（恢复窗口靠它推进，不真 sleep）。

    **两个都要捏**：共享状态存的是 ``time.time()``（unix 秒，跨进程可比），进程内断路器用的是
    ``time.monotonic()``（各进程零点不同）。只捏一个，测出来的「窗口过没过」两边就对不上。
    """
    fake = FakeRedis()
    shared_breaker.set_shared_store(shared_breaker.SharedBreakerStore(fake))
    clock = {"t": 1_000_000.0}
    monkeypatch.setattr(shared_breaker.time, "time", lambda: clock["t"])
    monkeypatch.setattr("app.utils.circuit_breaker.time.monotonic", lambda: clock["t"])
    yield fake, clock
    shared_breaker.reset_shared_store()


def _breaker(name: str = "tool:item_search") -> CircuitBreaker:
    # 每个「副本」建自己的实例：进程内计数本就互不相干，共享的只有 Redis 那份。
    return CircuitBreaker(name, failure_threshold=3, recovery_timeout=60.0)


# ---------- 双进程冒烟：A 开 → B 拒 → B 成功 → A 关 ----------
async def test_open_in_a_rejects_in_b_and_success_in_b_closes_a(shared: Any) -> None:
    fake, clock = shared
    a, b = _breaker(), _breaker()

    # ① A 连续失败到阈值 → A 本地 OPEN，且共享状态里落了 opened_at。
    for _ in range(3):
        await shared_breaker.record_failure(a)
    assert a.state == OPEN
    assert float(fake.data["globex:breaker:tool:item_search"]["opened_at"]) > 0

    # ② B 的本地计数是干净的（fail_count=0、CLOSED），照样被远端裁决拒掉——这就是共享的意义。
    assert b.state == CLOSED
    assert await shared_breaker.allow(b) is False
    # 被远端拒掉时**不许动本地状态机**：否则留下一个没有成败记录的半开态。
    assert b.state == CLOSED

    # ③ 恢复窗口过 → B 放行一次探测（远端 half）。
    clock["t"] += 61
    assert await shared_breaker.allow(b) is True

    # ④ B 探测成功 → 清共享状态。
    await shared_breaker.record_success(b)
    assert "globex:breaker:tool:item_search" not in fake.data

    # ⑤ A 侧跟着关闭：下一次 allow 读到远端 clear，就地把本地也复位（不必各自再等一遍窗口）。
    assert await shared_breaker.allow(a) is True
    assert a.state == CLOSED


async def test_local_open_still_rejects_when_remote_is_clear(shared: Any) -> None:
    """两道闸**都同意才放行**：远端干净不等于本地可以放行（本副本自己刚踩满阈值）。

    这条同时是「远端写不进去时别把本地保护也关掉」的回归——把 :func:`allow` 里的复位条件放宽成
    「远端干净就复位」，本用例即红。
    """
    fake, _clock = shared
    a = _breaker()
    for _ in range(3):
        a.record_failure()  # 只打本地，不写远端（模拟远端已被别的副本清掉）
    assert a.state == OPEN
    assert await shared_breaker.allow(a) is False


async def test_failures_below_threshold_do_not_open(shared: Any) -> None:
    fake, _clock = shared
    a, b = _breaker(), _breaker()
    for _ in range(2):  # 未到阈值 3
        await shared_breaker.record_failure(a)
    assert fake.data["globex:breaker:tool:item_search"]["fails"] == "2"
    assert "opened_at" not in fake.data["globex:breaker:tool:item_search"]
    assert await shared_breaker.allow(b) is True


async def test_success_clears_shared_count(shared: Any) -> None:
    """成功即删键 —— 与进程内 ``_on_success`` 的「清零计数」逐条对齐，不是「减一」。"""
    fake, _clock = shared
    a = _breaker()
    for _ in range(2):
        await shared_breaker.record_failure(a)
    await shared_breaker.record_success(a)
    assert fake.data == {}


# ---------- Redis 故障：一律放行 ----------
async def test_redis_read_failure_falls_back_to_local(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeRedis(fail_on={"hgetall"})
    shared_breaker.set_shared_store(shared_breaker.SharedBreakerStore(fake))
    try:
        a = _breaker()
        assert await shared_breaker.allow(a) is True  # 问不到 → 交回本地那道闸，本地干净 → 放行
        for _ in range(3):
            await shared_breaker.record_failure(a)
        # 本地已 OPEN：远端问不到时**不能**变成无脑放行，本地口径仍然作数。
        assert await shared_breaker.allow(a) is False
    finally:
        shared_breaker.reset_shared_store()


async def test_redis_write_failure_keeps_local_counting() -> None:
    fake = FakeRedis(fail_on={"hincrby"})
    shared_breaker.set_shared_store(shared_breaker.SharedBreakerStore(fake))
    try:
        a = _breaker()
        for _ in range(3):
            await shared_breaker.record_failure(a)  # 写不进去也不抛
        assert a.state == OPEN  # 本地计数照常推进
    finally:
        shared_breaker.reset_shared_store()


async def test_malformed_opened_at_is_treated_as_clear(shared: Any) -> None:
    """脏数据（手工改键 / 版本不一致）不能把整条主链路打挂，按「没记录」处理。"""
    fake, _clock = shared
    fake.data["globex:breaker:tool:item_search"] = {"opened_at": "not-a-number"}
    assert await shared_breaker.allow(_breaker()) is True


# ---------- 默认关：一次 Redis 都不碰 ----------
async def test_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BREAKER_SHARED", raising=False)
    shared_breaker.reset_shared_store()
    assert shared_breaker.shared_enabled() is False
    assert shared_breaker.get_shared_store() is None

    a = _breaker()
    assert await shared_breaker.allow(a) is True
    for _ in range(3):
        await shared_breaker.record_failure(a)
    assert a.state == OPEN  # 纯进程内口径，与批 2 之前逐字一致
    assert await shared_breaker.allow(a) is False
    await shared_breaker.record_success(a)
    assert a.state == CLOSED


async def test_env_flag_turns_it_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BREAKER_SHARED", "1")
    shared_breaker.reset_shared_store()
    assert shared_breaker.shared_enabled() is True
    shared_breaker.reset_shared_store()
