"""LLM 断路器（阶段 2 第 1 条）：按 ``(provider, model)` 一个，只对「对面挂了」计数。

**解决什么。** 2-1 之后模型出口可以跨供应商了，但一家挂掉时的表现还是老样子：每一次调用都要
耗到 ``LLM_REQUEST_TIMEOUT``（60s）才知道失败，并发在飞的请求全卡在那 60s 上，credit 扣了、
用户等着、队列堆着。断路器让这件事只发生前几次：连续 ``LLM_BREAKER_THRESHOLD`` 次真实故障之后
直接快速失败，``LLM_BREAKER_RECOVERY_SEC`` 后放一次探测。

**关键在「什么才算故障」——三档口径不能混：**

- **429 / 限流不计。** 它说明的是我们发太快，对面好得很。限流该由网关闸门
  （:class:`~app.agent.gateway.GatewayThrottle` 的 ``penalize``）降速处理；拿它熔断等于
  自己把自己关在门外。
- **总超时（60s）不计。** 一条流跑满 60s 说明它**慢**，不说明它**坏**——首 token 早就到了、
  内容也在吐。把慢流算成故障，长回答一多就会误熔断。
- **首 token 超时（15s）计。** 首 token 迟迟不来才是「这家没在干活」的信号。这一档还会**真的
  掐断请求**（见 :func:`first_token_timeout`），否则只是换个地方等满 60s，没有快速失败的收益。
- 5xx、连接失败计；4xx（BadRequest 等）不计——请求本身有问题，换谁都是 400。

**按进程，不共享。** 多副本下每个进程各学各的，N 个副本最多各白等 N×threshold 次。共享要走
Redis（``app.utils.shared_breaker`` 那套），对 LLM 这种高频出口不值得——它的收益在「第 6 次
之后不再等 60s」，而每进程前 5 次的代价本来就可接受。

**配了 ``LLM_FALLBACK_CHAIN`` 时它是「整条链都不行」的闸。** 键用的是模型句柄的 ref，而
litellm Router 的 fallback 发生在这层之下：链上任一家顶住了，这次调用就是成功。所以它熔断的
含义是「主 + 备全挂」，不是「主挂了」。不粉饰这一点——想要单家粒度得先把 fallback 从 Router
内部搬出来，那是另一笔账。
"""

import asyncio
import logging
import os
import re

from app.agent.transient import is_rate_limited
from app.utils.circuit_breaker import CircuitBreaker, CircuitOpenError

__all__ = [
    "CircuitOpenError",
    "FirstTokenTimeout",
    "breaker_enabled",
    "counts_as_failure",
    "first_token_timeout",
    "get_llm_breaker",
    "record_outcome",
    "reset_llm_breakers",
]

# 文本兜底：DashScope 等兼容口的 5xx 有时不落成 SDK 异常类型，而是把网关错误体透传成一条普通
# 异常（transient.py 开头那段说的同一件事）。只放明确指向**服务端**故障的词。
_SERVER_ERROR_MARKERS = (
    "internal server error",
    "bad gateway",
    "service unavailable",
    "gateway timeout",
)

# 状态码走词边界匹配，不用子串：``" 500"`` 这种写法会被 ``max_tokens 5000`` 命中，
# 白白熔断一个其实是参数错误的调用。
_SERVER_STATUS_RE = re.compile(r"\b5\d{2}\b")

_TIMEOUT_MARKERS = ("timed out", "timeout", "超时")

logger = logging.getLogger("shoppingx.llm.breaker")


class FirstTokenTimeout(asyncio.TimeoutError):
    """首 token 超过预算未到达，请求已被掐断。

    继承 ``asyncio.TimeoutError`` 是为了让 :func:`app.agent.transient.is_transient` 照旧判它
    为瞬时错误（换个模型 / 重来一轮是有意义的），不用在别处再加一条分支。
    """

    def __init__(self, model: str, seconds: float) -> None:
        super().__init__(f"模型 {model} 首 token 超过 {seconds:.1f}s 未到达，已掐断")


def _env(key: str) -> str:
    return (os.environ.get(key) or "").strip()


def _env_float(key: str, default: float) -> float:
    try:
        return float(_env(key) or default)
    except ValueError:
        return default


def breaker_enabled() -> bool:
    """总开关 ``LLM_BREAKER_ENABLED``，默认**开**。关掉即完全不 gate 不计数（本条的回滚开关）。"""
    raw = _env("LLM_BREAKER_ENABLED")
    if not raw:
        return True
    return raw.lower() in {"1", "true", "yes", "on"}


def first_token_timeout() -> float:
    """首 token 预算（秒），``LLM_FIRST_TOKEN_TIMEOUT``，默认 15；``<=0`` 关闭该档。

    每次调用现读，不缓存：后台热更新（``config_overrides``）改完不用重启进程。
    """
    return _env_float("LLM_FIRST_TOKEN_TIMEOUT", 15.0)


_llm_breakers: dict[str, CircuitBreaker] = {}


def get_llm_breaker(ref: str) -> CircuitBreaker | None:
    """取（或建）这个模型 ref 的断路器；开关关掉时返回 ``None``。

    ``ref`` 就是 ``provider/model``——2-1 之后 ``ThrottledChatModel.model`` 在直连与 Router
    两条路上都已经是这个形态，不用再解析一次。同 ref 的多个模型句柄（主档与 judge 档恰好同模型）
    共用一个断路器，这是对的：熔断的是**出口**，不是句柄。

    自己存一份注册表而不是复用 :mod:`app.utils.circuit_breaker` 的 ``_BREAKERS``——那张表按 name
    **替换**，重复构造会把状态洗掉，而这里是每次调用都要取。
    """
    if not breaker_enabled():
        return None
    breaker = _llm_breakers.get(ref)
    if breaker is None:
        breaker = CircuitBreaker(
            name=f"llm:{ref}",
            failure_threshold=int(_env_float("LLM_BREAKER_THRESHOLD", 5)),
            recovery_timeout=_env_float("LLM_BREAKER_RECOVERY_SEC", 30.0),
        )
        _llm_breakers[ref] = breaker
    return breaker


def reset_llm_breakers() -> None:
    """清空注册表（测试 / 运维用）。"""
    _llm_breakers.clear()


def _openai_types() -> tuple[type[Exception], ...] | None:
    """懒导入 openai 的异常类型；缺了就只靠文本判（与 transient.py 同口径）。

    返回 ``(APITimeoutError, APIConnectionError, APIStatusError)``，顺序即判据顺序。
    """
    try:
        import openai
    except ImportError:  # pragma: no cover - 环境里一定装了
        return None
    return (openai.APITimeoutError, openai.APIConnectionError, openai.APIStatusError)


def counts_as_failure(exc: BaseException) -> bool:
    """这个异常算不算「对面挂了」。判据顺序即优先级，改顺序会改语义。

    ``True`` = 累计失败；``False`` = 中立（既不累计也不清零，见
    :meth:`~app.utils.circuit_breaker.CircuitBreaker.record_neutral`）。
    """
    # 取消是我们自己发的（用户点停止 / 超时闸 / 上游 aclose），与对面无关。
    if isinstance(exc, (asyncio.CancelledError, GeneratorExit)):
        return False
    # 429 / 限流：我们发太快，不是它坏。降速由 GatewayThrottle.penalize 负责。
    if is_rate_limited(exc):
        return False
    # 首 token 超时：唯一一档「超时也算坏」，必须排在通用超时规则前面（它是其子类）。
    if isinstance(exc, FirstTokenTimeout):
        return True

    text = str(exc).lower()
    types = _openai_types()
    if types is not None:
        timeout_error, connection_error, status_error = types
        # 总超时：慢不等于坏。注意 APITimeoutError 是 APIConnectionError 的子类，先判它。
        if isinstance(exc, timeout_error):
            return False
        if isinstance(exc, connection_error):
            return True
        if isinstance(exc, status_error):
            return int(getattr(exc, "status_code", 0) or 0) >= 500
    if isinstance(exc, asyncio.TimeoutError) or any(m in text for m in _TIMEOUT_MARKERS):
        return False
    if any(m in text for m in _SERVER_ERROR_MARKERS) or _SERVER_STATUS_RE.search(text):
        return True
    # 认不出来的（我们自己的解析 bug、schema 校验…）一律中立：断路器只为依赖健康服务，
    # 拿本地 bug 去熔断供应商，只会把一个错误变成两个现象。
    return False


def record_outcome(breaker: CircuitBreaker | None, exc: BaseException | None) -> None:
    """放行之后必调一次：``exc=None`` 记成功，否则按 :func:`counts_as_failure` 记失败/中立。"""
    if breaker is None:
        return
    if exc is None:
        breaker.record_success()
        return
    if counts_as_failure(exc):
        logger.warning("断路器 %s 记一次失败：%s", breaker.name, exc)
        breaker.record_failure()
    else:
        breaker.record_neutral()
