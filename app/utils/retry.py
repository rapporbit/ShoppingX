"""HTTP 外呼的退避重试（B 块 · 韧性工程）——只对「可重试」的失败退避重试。

**只重试该重试的。** 区分两类失败：
- **可重试**：网络超时、连接重置、5xx（服务端临时抽风）——再试一次大概率就好。
- **不可重试**：4xx（参数/鉴权错，重试还是错）、返回体格式错（数据问题，重试无意义）。

对不可重试的失败盲目重试，只会拖长「注定失败」的等待、放大下游压力。所以这里用 ``tenacity``
配一个**白名单判定**：只有 ``httpx.TimeoutException`` 和 5xx 的 ``HTTPStatusError`` 才退避重试，
其余立即抛出。退避带 **jitter**（随机抖动）——多个并发请求同时失败时，错开重试时刻，避免它们
「齐步走」再次同时打下游（惊群 / retry storm）。

**与断路器的配合。** 通常把重试包在断路器**内层**：``breaker.call(lambda: retried_remote())``。
这样断路器统计的是「重试都救不回的最终失败」——偶发抖动被重试消化、不算进熔断计数，真·持续
故障才推动熔断。见 :mod:`app.utils.circuit_breaker`。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

T = TypeVar("T")

# 可重试的连接 / 网络类瞬时故障：连接被拒 / DNS 抖动（ConnectError）、连接被对端重置或读写中断
# （ReadError/WriteError）、服务端过早关闭连接（RemoteProtocolError）。用**明确白名单**而非宽泛的
# httpx.TransportError——后者还含 ProxyError / UnsupportedProtocol 这类**配置错**，重试也是错。
_RETRYABLE_TRANSPORT = (
    httpx.ConnectError,
    httpx.ReadError,
    httpx.WriteError,
    httpx.RemoteProtocolError,
)


def is_retryable_http_error(exc: BaseException) -> bool:
    """判定一个异常是否「可重试」：超时 / 连接类瞬时故障 / 5xx。4xx / 格式错 / 配置错一律不重试。"""
    if isinstance(exc, httpx.TimeoutException):  # 各类超时（连接/读/写/连接池）
        return True
    if isinstance(exc, _RETRYABLE_TRANSPORT):  # 连接被拒 / 重置 / 对端过早断开
        return True
    if isinstance(exc, httpx.HTTPStatusError):  # 5xx 服务端临时抽风
        return exc.response.status_code >= 500
    return False


async def call_with_retry(
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    initial: float = 0.5,
    max_wait: float = 4.0,
) -> T:
    """执行一次异步外呼 ``fn()``，对可重试失败（超时 / 5xx）做指数退避 + jitter 重试。

    参数：
      - ``attempts``：最多尝试次数（含首次）。默认 3 = 首次 + 2 次重试。
      - ``initial`` / ``max_wait``：指数退避的初始与上限秒数（带 jitter）。

    不可重试的异常（4xx / 数据格式错）会**立即**原样抛出，不浪费时间重试。重试耗尽后抛出最后
    一次的异常，交给上层（断路器 / 各外呼自己的降级路径）处理。
    """
    retryer = AsyncRetrying(
        retry=retry_if_exception(is_retryable_http_error),
        stop=stop_after_attempt(attempts),
        wait=wait_exponential_jitter(initial=initial, max=max_wait),
        reraise=True,  # 重试耗尽抛原始异常，而非 tenacity 的 RetryError（上层好按类型分支）
    )
    return await retryer(fn)
