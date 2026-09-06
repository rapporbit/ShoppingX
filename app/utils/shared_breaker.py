"""跨进程共享熔断（批 2）——把进程内断路器的状态多写一份到 Redis，让副本之间互相看得见。

**为什么要它。** :mod:`app.utils.circuit_breaker` 的计数住在进程里。单进程时这没问题；拆出
worker 副本之后，「某个工具彻底挂了」这件事**每个副本都要自己重新踩满 3 次**才知道——N 个副本
就是 3N 次白等超时，而且每个副本各自计时、各自恢复，谁也不知道别人已经熔断了。共享一份状态
后：任一副本踩到阈值，其余副本下一次调用就直接快速失败。

**口径与进程内那套逐条对齐，不另发明一套：**

- 判据仍是**连续失败计数**（低 QPS 下比滑动窗口失败率稳），阈值与恢复窗口沿用调用方那个
  ``CircuitBreaker`` 实例上的值，本模块不自带默认值。
- **「谁算失败」不在这里判**。参数校验类失败（``ValidationError``）不计入熔断——那是模型的锅、
  不是基础设施故障——这条豁免仍由 ``app/harness/adapter.py`` 的调用点把关（共享层根本收不到那次
  失败）。共享层只负责把**已经判定过**的成败多写一份，判据单一事实源不搬家。
- 两道闸**都同意才放行**：远端说 OPEN 就拒（哪怕本地干净），远端没记录时仍要过本地那道。

**降级方向：Redis 故障一律放行**（退回纯本地口径）。理由与队列相反、与事件背板相同：熔断是**优化**
（省掉白等超时），不是正确性前提。Redis 抖一下就把所有工具判成不可用，等于自己制造一次全站故障。
同理客户端设了**短 ``socket_timeout``**——熔断判定在主链路上，Redis 卡住时必须立刻退回本地，
而不是让每次工具调用先干等 5 秒（那正是它本该消除的那种等待）。这与背板 / 控制面「订阅端绝不设
``socket_timeout``」的取舍相反，两处别互相抄：那边是长连接监听，这边是一问一答。

**时钟口径必须换**：进程内用 ``time.monotonic()``（各进程零点不同，跨进程比较毫无意义），共享
状态一律存 ``time.time()`` 的 unix 秒。副本间时钟偏几秒只会让恢复窗口早/晚几秒到，无害。

``BREAKER_SHARED=0``（默认）时本模块是空操作：``get_shared_store()`` 返回 ``None``，三个入口
函数退化成直接调进程内断路器，一次 Redis 都不碰。
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Literal

from app.utils.circuit_breaker import CircuitBreaker
from app.utils.env import env_bool, env_int

logger = logging.getLogger("shoppingx.shared_breaker")

#: 共享状态的键前缀，一个断路器一个 hash：``fails``（连续失败数）/ ``opened_at``（unix 秒）。
KEY_PREFIX = "globex:breaker:"

#: 共享键的存活时长（秒）。进程内计数不需要它（进程重启即清零），跨进程的没有这个天然回收——
#: 一个再也不会被调用的工具留下的失败计数会永久占着键。取值要**远大于恢复窗口**，否则计数在
#: 熔断期间就自己过期了，等于把「连续失败」悄悄重置成 0。
_STATE_TTL = env_int("BREAKER_SHARED_TTL", 3600)

#: ``allow`` 的三种远端裁决 + 一种「问不到」。见 :meth:`SharedBreakerStore.verdict`。
Verdict = Literal["open", "half", "clear", "unknown"]


def shared_enabled() -> bool:
    """是否启用跨进程共享熔断。默认关——单进程部署下它只是一次多余的 Redis 往返。"""
    return env_bool("BREAKER_SHARED", False)


def _redis_url() -> str:
    """复用队列 / 控制面那个实例（键名带 ``globex:`` 前缀，同一个 db 里不会撞）。"""
    return (
        os.environ.get("BREAKER_REDIS_URL")
        or os.environ.get("QUEUE_REDIS_URL")
        or os.environ.get("EVENT_REDIS_URL", "redis://localhost:6379/2")
    )


class SharedBreakerStore:
    """Redis 上的一份熔断状态（一个断路器一个 hash）。方法**从不抛**：故障即 ``unknown`` / 静默。"""

    def __init__(self, client: Any, *, prefix: str = KEY_PREFIX, ttl: int = _STATE_TTL) -> None:
        self._client = client
        self._prefix = prefix
        self._ttl = ttl

    def _key(self, name: str) -> str:
        return f"{self._prefix}{name}"

    async def verdict(self, name: str, recovery_timeout: float) -> Verdict:
        """远端裁决：

        - ``open``：别的副本熔断了且恢复窗口未过 → 本次直接拒（不发起真实调用）。
        - ``half``：有失败记录但窗口已过 → 放行一次探测，**本地状态不动**。
        - ``clear``：远端干净（探测成功后被删掉 / 从没失败过）。
        - ``unknown``：Redis 问不到 —— 放行与否完全交回本地那道闸。
        """
        try:
            data = await self._client.hgetall(self._key(name))
        except Exception as exc:  # Redis 故障：放行（退回纯本地口径）
            logger.debug("共享熔断读失败（放行，退回本地口径）：%s", exc)
            return "unknown"
        if not data:
            return "clear"
        try:
            opened_at = float(data.get("opened_at") or 0.0)
        except (TypeError, ValueError):
            return "clear"
        if opened_at <= 0.0:
            return "half"  # 有失败计数但还没到阈值：不拦，也别让调用方 reset 本地
        return "open" if time.time() - opened_at < recovery_timeout else "half"

    async def record_failure(self, name: str, threshold: int, recovery_timeout: float) -> None:
        """记一次失败：累计计数，达到阈值就写 ``opened_at``（= 全体副本一起进 OPEN）。

        两个副本同时越过阈值时会各写一次 ``opened_at``，后写的赢——差几毫秒，无碍。
        """
        key = self._key(name)
        try:
            fails = int(await self._client.hincrby(key, "fails", 1))
            await self._client.expire(key, self._ttl)
            if fails >= threshold:
                await self._client.hset(key, "opened_at", str(time.time()))
                logger.warning(
                    "共享熔断 %s 转 OPEN（跨进程连续失败 %d 次），%.0fs 内各副本一律快速失败",
                    name,
                    fails,
                    recovery_timeout,
                )
        except Exception as exc:
            logger.debug("共享熔断写失败（本地计数仍生效）：%s", exc)

    async def record_success(self, name: str) -> None:
        """记一次成功：直接删键（= 计数清零 + 退出 OPEN），语义同进程内的 ``_on_success``。"""
        try:
            await self._client.delete(self._key(name))
        except Exception as exc:
            logger.debug("共享熔断清除失败（下次成功再试）：%s", exc)

    async def snapshot(self, name: str) -> dict[str, str]:
        """当前共享状态（观测 / 测试用）；读不到返回空 dict。"""
        try:
            data = await self._client.hgetall(self._key(name))
        except Exception:
            return {}
        return dict(data or {})


# ── 进程级单例 ───────────────────────────────────────────────────────────────
# 同 queue / backplane / control 的取舍：谁也不该自己 ``SharedBreakerStore(...)``，否则一个进程里
# 会攒出几个 Redis 连接；``_resolved`` 记「已经决定过了」，关着时不必反复读环境变量。
_store: SharedBreakerStore | None = None
_resolved = False


def get_shared_store() -> SharedBreakerStore | None:
    """返回进程级共享熔断状态；未启用 / 客户端建不起来时返回 ``None``（退化为纯进程内熔断）。"""
    global _store, _resolved
    if _resolved:
        return _store
    _resolved = True
    if not shared_enabled():
        return None
    try:
        import redis.asyncio as aredis  # 可选依赖，懒加载（与 queue / backplane / control 同源）

        client = aredis.from_url(
            _redis_url(),
            decode_responses=True,
            socket_connect_timeout=1.0,
            # 短超时是**有意的**：熔断判定在主链路上，Redis 卡住时必须立刻退回本地口径放行，
            # 而不是让每次工具调用先干等——那就成了熔断器自己在制造它本该消除的那种等待。
            socket_timeout=1.0,
        )
    except Exception as exc:
        logger.warning("共享熔断初始化失败，退回进程内熔断：%s", exc)
        return None
    _store = SharedBreakerStore(client)
    logger.info("跨进程共享熔断启用：%s（前缀 %s）", _redis_url(), KEY_PREFIX)
    return _store


def set_shared_store(store: SharedBreakerStore | None) -> None:
    """注入实例（测试 / 手工装配用）；传 ``None`` 即彻底关掉（不再按环境变量懒加载）。"""
    global _store, _resolved
    _store = store
    _resolved = True


def reset_shared_store() -> None:
    """复位为「按环境变量重新决定」（测试收尾用）。"""
    global _store, _resolved
    _store = None
    _resolved = False


# ── 调用方入口（工具熔断的三个动作，共享开着就多写/多读一份）─────────────────────
async def allow(breaker: CircuitBreaker) -> bool:
    """本次调用是否放行 —— **远端与本地都同意才放行**。

    先问远端再问本地，顺序不能反：``CircuitBreaker.allow()`` **有副作用**（可能把 OPEN 推进到
    HALF_OPEN 以放行一次探测）。先跑它、再被远端拒掉，本地就留下一个没有对应成败记录的半开态。

    远端 ``clear`` 时**只在本地恢复窗口也已过**的前提下就地复位本地：那一刻本地反正要放一次
    探测，而远端干净说明**已经有别的副本探测成功了**，于是把这次探测升格成直接恢复 CLOSED。
    窗口没过就不动它——远端「干净」也可能是**写不进去**（Redis 半死不活）造成的假象，凭它抹掉
    本地刚踩满的熔断，等于让一次 Redis 故障顺手把本地保护也关掉。
    """
    store = get_shared_store()
    if store is None:
        return breaker.allow()
    verdict = await store.verdict(breaker.name, breaker.recovery_timeout)
    if verdict == "open":
        logger.debug("共享熔断 %s 处于 OPEN（别的副本打开的），本进程快速失败", breaker.name)
        return False
    if verdict == "clear" and breaker.open_seconds >= breaker.recovery_timeout > 0:
        breaker.reset()
    return breaker.allow()


async def record_success(breaker: CircuitBreaker) -> None:
    """记一次成功：本地复位 + 清远端。放行后必调其一（与 :func:`record_failure` 成对）。"""
    breaker.record_success()
    store = get_shared_store()
    if store is not None:
        await store.record_success(breaker.name)


async def record_failure(breaker: CircuitBreaker) -> None:
    """记一次失败：本地累计 + 远端累计。

    **是否该记**由调用方判（``ValidationError`` 不记，见模块 docstring），本函数不做二次判断。
    """
    breaker.record_failure()
    store = get_shared_store()
    if store is not None:
        await store.record_failure(
            breaker.name, breaker.failure_threshold, breaker.recovery_timeout
        )
