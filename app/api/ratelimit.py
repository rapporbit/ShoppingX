"""认证端点的速率限制：**堵住「脚本刷号 → 每个号一份日额度 → 账单被打穿」这条路**。

M19 的 credit 配额是按人计的（每人每天 ``DAILY_QUOTA_USD``），可它默认「人」是稀缺的。开放注册一开，
人就不稀缺了：一个脚本一分钟能注册几百个号，每个号都合规地领一份日额度，加起来照样把账单烧穿。
配额闸和限流闸缺一不可——前者管「一个人能用多少」，后者管「能有多少个人」。

**两道闸，因为单靠任何一道都能被绕。**

1. **IP 滑动窗口**（本模块的 :class:`SlidingWindow`，住进程内存）：挡住「一台机器猛刷」这种最常见的
   情形。计数在内存里，进程一重启就清零——这是**明知的取舍**：为它引一个 Redis 硬依赖不值当，因为
   真正兜底的是第 2 道闸。攻击者就算掐着重启的点刷，也越不过每日总量。
2. **全局每日新增用户上限**（:func:`guard_daily_signups`，直接 ``COUNT`` users 表）：挡「换 IP 池
   刷号」。IP 能换，这个数换不掉——它查库里今天真多出几行，**跨重启、跨多 worker 都是同一个真源**，
   不需要任何额外存储。代价是它对正常用户也是硬顶：某天注册的人真超了上限，后来者会被挡在门外。
   对一个 demo 站点，「今天来晚了明天再来」远好过「账单被刷爆只能关站」。

登录口一并限流，但目的不同：那里防的是**撞库**，不是防刷号。
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from datetime import UTC, datetime

from fastapi import HTTPException, Request
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import User
from app.utils.env import env_bool, env_int


def rate_limit_enabled() -> bool:
    """总开关。本地开发默认**开着**：限流一旦只在生产开，就永远是没被测过的那段代码。
    额度都设得很宽，正常手工点击碰不到；测试里要关就显式设 ``RATE_LIMIT_ENABLED=false``。"""
    return env_bool("RATE_LIMIT_ENABLED", True)


def client_ip(request: Request) -> str:
    """取调用方 IP。**默认不信 ``X-Forwarded-For``**——它是一个纯粹由客户端写的 header，直接采信
    等于把限流的 key 交给攻击者自己填（每请求换一个伪造 IP，窗口永远撞不满）。

    只有当 ``TRUST_PROXY_HEADERS=true``（生产：Caddy 反代在前）才读它，且取**最后一个**元素：
    反代把它看到的真实对端 append 在末尾，前面那些可能是客户端自己伪造塞进来的。
    """
    if env_bool("TRUST_PROXY_HEADERS", False):
        xff = request.headers.get("x-forwarded-for", "")
        if xff:
            return xff.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


class SlidingWindow:
    """按 key 的滑动窗口计数器：``window_s`` 秒内最多 ``limit`` 次。

    用 :class:`deque` 存时间戳而非「固定窗口计数」：后者在窗口边界可以放进两倍流量（窗口末尾打满、
    跨过边界立刻再打满）。时间用 :func:`time.monotonic`——墙钟会被 NTP 往回拨，一拨窗口就乱。
    """

    def __init__(self, limit: int, window_s: float) -> None:
        self.limit = limit
        self.window_s = window_s
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    # 每个见过的 IP 都会在 _hits 里留一个 deque，不清理就是一条按「访问过的 IP 数」增长的内存泄漏
    # （公网上被扫尤其明显）。攒够这么多 key 就顺手扫掉空掉的那些——比给每个 key 挂定时器便宜。
    _SWEEP_AT = 4096

    def check(self, key: str) -> float | None:
        """记一次命中。放行返回 ``None``；超限返回**还要等几秒**（给调用方填 Retry-After）。"""
        now = time.monotonic()
        if len(self._hits) > self._SWEEP_AT:
            self._sweep(now)
        hits = self._hits[key]
        while hits and now - hits[0] >= self.window_s:
            hits.popleft()
        if len(hits) >= self.limit:
            return self.window_s - (now - hits[0])
        hits.append(now)
        return None

    def _sweep(self, now: float) -> None:
        """丢掉窗口已经整段过期的 key（它们的计数早已归零，留着只占内存）。"""
        stale = [k for k, h in self._hits.items() if not h or now - h[-1] >= self.window_s]
        for k in stale:
            del self._hits[k]

    def reset(self) -> None:
        """清空（测试用）。"""
        self._hits.clear()


# 额度取值：正常人注册一次、登录失败几次重来就够了，这些上限手工操作根本碰不到；脚本一上来就撞墙。
_register_by_ip = SlidingWindow(env_int("REGISTER_PER_IP_PER_HOUR", 5), 3600)
_login_by_ip = SlidingWindow(env_int("LOGIN_PER_IP_PER_15MIN", 20), 900)


def _enforce(window: SlidingWindow, key: str, what: str) -> None:
    if not rate_limit_enabled():
        return
    retry_after = window.check(key)
    if retry_after is not None:
        raise HTTPException(
            429,
            f"{what}过于频繁，请 {int(retry_after) + 1} 秒后再试",
            headers={"Retry-After": str(int(retry_after) + 1)},
        )


def guard_register_ip(request: Request) -> None:
    _enforce(_register_by_ip, client_ip(request), "注册")


def guard_login_ip(request: Request) -> None:
    _enforce(_login_by_ip, client_ip(request), "登录尝试")


def max_new_users_per_day() -> int:
    """全站每日新增用户上限；``<=0`` 表示不设闸。"""
    return env_int("MAX_NEW_USERS_PER_DAY", 50)


async def guard_daily_signups(db: AsyncSession) -> None:
    """第 2 道闸：今天全站已经新增了这么多用户，就不再放人进来（换 IP 也没用）。"""
    limit = max_new_users_per_day()
    if not rate_limit_enabled() or limit <= 0:
        return
    today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    count = await db.scalar(select(func.count()).select_from(User).where(User.created_at >= today))
    if (count or 0) >= limit:
        raise HTTPException(429, "今日注册名额已满，请明天再来")


def reset_all() -> None:
    """清空所有内存窗口（测试用：否则用例之间互相把对方的额度用光）。"""
    _register_by_ip.reset()
    _login_by_ip.reset()
