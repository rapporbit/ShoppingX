"""通用 async 断路器（B 块 · 韧性工程）——连续失败就「快速失败」，不再每次干等超时。

**解决什么问题。** 一个外呼（reranker / web_search…）一旦真的挂了，没有断路器时，**每一次**
调用都还要傻等到超时（reranker 10s、web_search 20s）才知道失败、才走降级。十几个并发请求就
是十几次 ×10s 的白等，把延迟和线程/连接全拖死。断路器记住「这家最近一直在挂」，在恢复窗口内
**立刻**抛 :class:`CircuitOpenError`，调用方直接走降级路径——把「每次都等超时」变成「等一次、
之后秒拒」。

**三态机。**
- ``CLOSED``（正常）：放行；连续失败累计到阈值 → 转 ``OPEN``。
- ``OPEN``（熔断）：直接抛 ``CircuitOpenError``，不发起真实调用；过了恢复窗口 → 转 ``HALF_OPEN``。
- ``HALF_OPEN``（探测）：放行（探测）一次真实调用，成功 → 回 ``CLOSED``，失败 → 退回 ``OPEN``。

**为什么用「连续失败计数」而不是文档草案里的「滑动窗口失败率」。** 这些外呼是**低频**的
（一次购物 query 也就调几次 rerank），滑动窗口里样本太少，失败率会被一两次抖动放大得失真。
连续失败计数对低 QPS 更稳：要连着失败 N 次才熔断，单次抖动不会误伤。这是「知道默认方案
（失败率）不匹配场景（低频）就别硬套」的又一例。

**并发口径（诚实标注）。** 状态读写都在同步段、``await`` 只发生在真实调用处，单线程 asyncio
下计数不会错。``HALF_OPEN`` 期未严格限制为「单探测」——并发下可能放过一两个探测请求，对这种
低频外呼无碍，不值得为它加锁换复杂度。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

logger = logging.getLogger("shoppingx.circuit")

T = TypeVar("T")

# 断路器状态。
CLOSED = "closed"
OPEN = "open"
HALF_OPEN = "half_open"

# 进程内断路器注册表（**按 name 索引**）——供可观测层（A 块 metrics）在 scrape 时枚举各外呼的
# 熔断状态。在这里登记（而非让 metrics 反向去各模块抓）保持依赖方向：observability → utils。
#
# 用 dict 而非 list 是有意为之：同名断路器**替换而非累积**。生产里断路器都是进程单例（reranker
# 走 lru_cache、web_search 模块级），name 唯一；但万一将来按平台/请求动态建同名断路器，list 会
# 只追加不删除地无界增长、且让 metrics 标签基数爆炸。按 name 去重从机制上堵掉这个隐患（也让测试
# 里反复 new 的同名断路器不累积）。
_BREAKERS: dict[str, CircuitBreaker] = {}


def all_breakers() -> list[CircuitBreaker]:
    """返回当前登记的所有断路器（按 name 去重后的快照，供 metrics 遍历上报状态）。"""
    return list(_BREAKERS.values())


class CircuitOpenError(Exception):
    """断路器处于 OPEN 态、本次调用被快速拒绝（未发起真实外呼）。

    调用方应捕获它、走降级路径——它在语义上等价于「外呼失败」，只是**没有真的等超时**。
    """


class CircuitBreaker:
    """一个外部依赖一个断路器实例（如 reranker / web_search 各一个）。

    用法：``await breaker.call(lambda: remote_coro())``。``call`` 内部：
    - OPEN 且未到恢复窗口 → 抛 ``CircuitOpenError``（调用方走降级）。
    - 否则发起真实调用：成功复位计数、失败累计；累计到阈值（或半开探测失败）即熔断。
    """

    __slots__ = (
        "name",
        "failure_threshold",
        "recovery_timeout",
        "_state",
        "_fail_count",
        "_opened_at",
    )

    def __init__(
        self, name: str, failure_threshold: int = 5, recovery_timeout: float = 30.0
    ) -> None:
        self.name = name
        # 连续失败几次转 OPEN。
        self.failure_threshold = failure_threshold
        # OPEN 后多久允许一次半开探测（秒）。
        self.recovery_timeout = recovery_timeout
        self._state = CLOSED
        self._fail_count = 0
        self._opened_at = 0.0
        _BREAKERS[name] = self  # 按 name 登记（同名替换，不累积），供 metrics 枚举状态

    @property
    def state(self) -> str:
        """当前状态（供 health / metrics / 测试观测）。读时不推进状态机。"""
        return self._state

    def allow(self) -> bool:
        """本次调用是否放行。OPEN 且恢复窗口已过 → 转 HALF_OPEN 并放行一次探测。

        **有副作用**（可能推进状态机到 HALF_OPEN），故调用方放行后必须成对调用
        :meth:`record_success` / :meth:`record_failure`，否则半开探测会悬空、永远回不到 CLOSED。

        gate 与 record 分开暴露，是为了让 Harness 的 Hook Pipeline 能把「熔断判定」放进
        pre_tool_call、把「熔断计数」放进 post_tool_call（refdocs 17-2 §2.2）——那里拿不到一个
        可以包起来 await 的 ``fn``，工具的执行由 Agent 框架负责。:meth:`call` 是它俩的组合糖。
        """
        if self._state != OPEN:
            return True
        if time.monotonic() - self._opened_at >= self.recovery_timeout:
            self._state = HALF_OPEN
            logger.info("断路器 %s 进入 HALF_OPEN，放行一次探测", self.name)
            return True
        return False

    def record_success(self) -> None:
        """记一次成功（放行后必调其一）。"""
        self._on_success()

    def record_failure(self) -> None:
        """记一次失败（放行后必调其一）。"""
        self._on_failure()

    async def call(self, fn: Callable[[], Awaitable[T]]) -> T:
        """经断路器执行一次异步调用 ``fn()``。OPEN 且未到恢复窗口时抛 ``CircuitOpenError``。

        = :meth:`allow` + 执行 + :meth:`record_success` / :meth:`record_failure` 的组合糖，
        供「调用方手里就有一个 async fn」的场景（reranker / web_search 等外呼）直接用。
        """
        if not self.allow():
            raise CircuitOpenError(f"断路器 {self.name} 处于 OPEN，快速失败")
        try:
            result = await fn()
        except Exception:
            self._on_failure()
            raise
        self._on_success()
        return result

    def _on_success(self) -> None:
        """调用成功：清零失败计数、回到 CLOSED（半开探测成功即恢复）。"""
        if self._state != CLOSED:
            logger.info("断路器 %s 探测成功，恢复 CLOSED", self.name)
        self._fail_count = 0
        self._state = CLOSED

    def _on_failure(self) -> None:
        """调用失败：累计失败；达到阈值或半开探测失败 → 转 OPEN 并记开闸时刻。"""
        self._fail_count += 1
        if self._state == HALF_OPEN or self._fail_count >= self.failure_threshold:
            if self._state != OPEN:
                logger.warning(
                    "断路器 %s 转 OPEN（连续失败 %d 次），%.0fs 内快速失败",
                    self.name,
                    self._fail_count,
                    self.recovery_timeout,
                )
            self._state = OPEN
            self._opened_at = time.monotonic()

    def reset(self) -> None:
        """手动复位到 CLOSED（测试 / 运维用）。"""
        self._state = CLOSED
        self._fail_count = 0
        self._opened_at = 0.0
