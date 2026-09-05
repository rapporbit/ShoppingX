"""瞬时错误判据：哪些异常是「等一下再来就好」，哪些是「换个模型也没用」。

AgentScope 的 ``ChatModelBase`` 自带一份可重试异常表（``openai.RateLimitError`` 等），但那张表
只服务于**模型内部重试**。网关闸门（:mod:`app.agent.gateway`）要回答的是另外两个问题：

1. **要不要降速** —— 只有限流类错误才说明「是我发太快了」，超时 / 连接断不是，一起降速会把
   偶发抖动误判成拥塞，白白拖慢整条链路。
2. **要不要回退到备用模型** —— 瞬时错误重试完仍失败才值得换模型；``BadRequestError`` 这种
   请求本身有问题的，换个模型照样 400，回退只是把同一个错误多犯一遍、多烧一次钱。

**为什么不能只靠 SDK 异常类型**：本项目走 DashScope 的 OpenAI 兼容口，实测限流有时并不落成
``openai.RateLimitError``，而是包在一条普通异常的文本里（网关自己的错误体透传）。所以判据是
「SDK 类型 **或** 文本特征」两路取并，宁可多认几个瞬时错误（代价是多等一轮），也不要把限流
误判成永久失败（代价是整条任务白跑）。
"""

import asyncio

# 文本兜底特征（小写匹配）。只放**明确**指向限流/过载的词，别放 "error" 这种泛词。
_RATE_LIMIT_MARKERS = (
    "429",
    "rate limit",
    "ratelimit",
    "too many requests",
    "throttl",
    "requests per minute",
    "quota",
    "限流",
    "请求过于频繁",
)

_TRANSIENT_MARKERS = (
    "timeout",
    "timed out",
    "connection",
    "temporarily",
    "service unavailable",
    "bad gateway",
    "internal server error",
    "500",
    "502",
    "503",
    "504",
)


def _sdk_rate_limit_types() -> tuple[type[Exception], ...]:
    """懒导入 openai——SDK 是可选依赖，缺了也要能判文本。"""
    try:
        import openai
    except ImportError:  # pragma: no cover - 环境里一定装了，留个兜底
        return ()
    return (openai.RateLimitError,)


def _sdk_transient_types() -> tuple[type[Exception], ...]:
    try:
        import openai
    except ImportError:  # pragma: no cover
        return (asyncio.TimeoutError,)
    return (
        openai.APIConnectionError,
        openai.APITimeoutError,
        openai.RateLimitError,
        openai.InternalServerError,
        asyncio.TimeoutError,
    )


def is_rate_limited(exc: BaseException) -> bool:
    """是不是「发太快了」类错误——只有它该触发闸门降速。"""
    if isinstance(exc, _sdk_rate_limit_types()):
        return True
    text = str(exc).lower()
    return any(marker in text for marker in _RATE_LIMIT_MARKERS)


def is_transient(exc: BaseException) -> bool:
    """是不是「等一下再来就好」类错误——重试与回退备用模型的前提。

    ``CancelledError`` 明确排除：那是我们自己取消的（用户点了取消 / 超时闸），重试它等于
    无视取消信号。
    """
    if isinstance(exc, asyncio.CancelledError):
        return False
    if is_rate_limited(exc):
        return True
    if isinstance(exc, _sdk_transient_types()):
        return True
    text = str(exc).lower()
    return any(marker in text for marker in _TRANSIENT_MARKERS)
