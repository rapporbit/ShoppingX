"""走 LiteLLM Router 的模型实现（阶段 2 第 2 条）。

**落地形态：不翻译，只换出口。** ``OpenAIChatModel._call_api`` 里真正打网络的只有一行
（``self.client.chat.completions.create(**kwargs)``），前面是 kwargs 装配、后面是
``_parse_stream_response`` 解析。所以这里不重写 ``_call_api``、不把 chunk 翻成
``ChatResponse``——只把 ``self.client`` 换成一个**鸭子形状的垫片**，转调 ``Router.acompletion``。

这么做换来的是：cache_control 标记、``extra_body`` 的 enable_thinking、tools / tool_choice、
``stream_options``、usage 解析、thinking 字段……**一行都不用抄**，框架升级时也不会漂移。
spike 的 P1~P4 与 R1~R4 量的正是「同样的 body 经 Router 发出去还是不是同一份」。

**垫片要补的两处**（2026-09-20 实测，见 scratchpad 探针）：
1. litellm 的流对象（``FallbackStreamWrapper`` / ``CustomStreamWrapper``）**不支持
   ``async with``**，而解析段写的是 ``async with response as stream``。
2. 流式只有**最后一片**带 ``usage`` 属性，前面几片连这个属性都没有，而解析段是
   ``if chunk.usage:`` 硬取——不补就是 ``AttributeError``。

**异常不用管**：litellm 的异常是 openai 异常的真子类（实测 ``InternalServerError`` 的 mro 里
有 ``openai.APIStatusError``），所以 ``_get_retryable_exceptions()`` 与
``transient.is_rate_limited`` 照旧生效，一个字不用改。
"""

import logging
from typing import Any

from app.agent.capabilities import degraded_against
from app.agent.gateway import ThrottledChatModel
from app.agent.providers import (
    Endpoint,
    build_model_list,
    configure_litellm,
    resolve_endpoint,
)

logger = logging.getLogger("shoppingx.llm.router")


class _UsageDefault:
    """给缺 ``usage`` 属性的 chunk 兜底的薄代理（setattr 被拒时才用得上）。"""

    __slots__ = ("_inner",)

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        try:
            return getattr(self._inner, name)
        except AttributeError:
            if name == "usage":
                return None
            raise


class _StreamShim:
    """把 litellm 的流包成 ``openai.AsyncStream`` 的形状：支持 ``async with`` + 每片带 usage。"""

    def __init__(self, stream: Any, router: Any = None, on_served: Any = None) -> None:
        self._stream = stream
        self._router = router
        self._on_served = on_served

    async def __aenter__(self) -> "_StreamShim":
        return self

    async def __aexit__(self, *exc_info: Any) -> bool:
        # 上游取消（用户点停止 / 超时）时要把连接真关掉，否则 slot 还回去了、请求还在飞。
        aclose = getattr(self._stream, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except Exception:  # pragma: no cover - 关流失败不该盖过原异常
                logger.debug("关闭 litellm 流失败", exc_info=True)
        return False

    async def __aiter__(self) -> Any:
        first = True
        async for chunk in self._stream:
            if first:
                first = False
                # 首片就带 model_id（实测），所以切没切在第一片上就知道，不用等流走完。
                if self._on_served is not None:
                    ref = _served_ref(self._router, chunk)
                    if ref:
                        await self._on_served(ref)
            if not hasattr(chunk, "usage"):
                try:
                    chunk.usage = None
                except Exception:  # pragma: no cover - pydantic 拒绝写入时退代理
                    chunk = _UsageDefault(chunk)
            yield chunk


def _served_ref(router: Any, response: Any) -> str | None:
    """这次响应实际是哪条 deployment 给的（回我们自己的 ``provider/model`` 名）。

    **必须用 ``model_id`` 反查，不能看 ``response.model``**（2026-09-20 实测）：非流式时
    ``response.model`` 是上游原样回的名字，流式首片又变成 ``litellm_params.model`` 的后半段，
    两者都不是我们的 ref。``api_base`` 也不够——同一家的两个模型 base_url 相同，分不开。
    ``_hidden_params["model_id"]`` 是 Router 给每条 deployment 的稳定 id，反查 ``model_list``
    即得 ``model_name``，也就是 ``Endpoint.ref``。
    """
    hidden = getattr(response, "_hidden_params", None) or {}
    model_id = hidden.get("model_id")
    if not model_id:
        return None
    for deployment in getattr(router, "model_list", None) or []:
        if (deployment.get("model_info") or {}).get("id") == model_id:
            return deployment.get("model_name")
    return None


class _RouterCompletions:
    """``client.chat.completions`` 的替身。"""

    def __init__(self, router: Any, timeout: float | None, on_served: Any = None) -> None:
        self._router = router
        self._timeout = timeout
        self._on_served = on_served

    async def create(self, **kwargs: Any) -> Any:
        # 超时在直连那条路上是 ``AsyncClient(timeout=…)`` 的构造参数，Router 这条路上是
        # 每次调用的入参——同一个 ``LLM_REQUEST_TIMEOUT``，两条路语义对齐。
        if self._timeout is not None:
            kwargs.setdefault("timeout", self._timeout)
        response = await self._router.acompletion(**kwargs)
        if kwargs.get("stream"):
            return _StreamShim(response, self._router, self._on_served)
        await self._notify(response)
        return response

    async def _notify(self, response: Any) -> None:
        if self._on_served is None:
            return
        ref = _served_ref(self._router, response)
        if ref:
            await self._on_served(ref)


class _RouterChat:
    def __init__(self, completions: _RouterCompletions) -> None:
        self.completions = completions


class _RouterClient:
    """``openai.AsyncClient`` 的鸭子替身，只提供 ``.chat.completions.create``。"""

    def __init__(self, router: Any, timeout: float | None, on_served: Any = None) -> None:
        self.chat = _RouterChat(_RouterCompletions(router, timeout, on_served))


def build_router(primary: Endpoint, fallbacks: list[Endpoint]) -> Any:
    """按主出口 + fallback 链建一个 Router。

    ``num_retries=0`` 是定死的口径：重试只由 ``LLM_MAX_RETRIES`` 一处决定（它走的是
    ``ChatModelBase.__call__`` 那层）。让 Router 再叠一层，一个 429 就会被试 9 次。
    """
    configure_litellm()
    from litellm import Router

    chain = [ep.ref for ep in fallbacks if ep.ref != primary.ref]
    return Router(
        model_list=build_model_list(primary, fallbacks),
        fallbacks=[{primary.ref: chain}] if chain else [],
        num_retries=0,
    )


class RoutedChatModel(ThrottledChatModel):
    """``ThrottledChatModel`` + 出口换成 LiteLLM Router。

    继承而不是并列：闸门、限流退避、``model_fallback`` 上报、流式持 slot 那套逻辑一行不改，
    差别只有「请求从哪个 client 发出去」。

    **fallback 只兜首 token 之前**（spike R2 实测）：200 一回，Router 的 fallback 窗口就关了，
    流中途断连客户端一片都拿不到、直接抛 ``APIConnectionError``。那种情况靠的是
    ``LLM_MAX_RETRIES`` 重来一轮，不是这里的链。写文档时别把这句省掉。
    """

    def __init__(
        self,
        *args: Any,
        endpoint: Endpoint,
        fallback_endpoints: list[Endpoint] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.endpoint = endpoint
        self.fallback_endpoints = list(fallback_endpoints or [])
        self._router = build_router(endpoint, self.fallback_endpoints)
        self._served_reported: set[str] = set()
        # 覆盖父类建好的 openai.AsyncClient。它是在 ``OpenAIChatModel.__init__`` 里建的，
        # 此处才换得掉；那个实例没发过任何请求，丢掉无副作用。
        timeout = (self.client_kwargs or {}).get("timeout")
        self.client = _RouterClient(self._router, timeout, self._on_served)  # type: ignore[assignment]
        logger.info(
            "模型 %s 走 LiteLLM Router：主出口 %s，fallback %s",
            self.model,
            endpoint.provider,
            [ep.ref for ep in self.fallback_endpoints] or "无",
        )

    async def _on_served(self, ref: str) -> None:
        """Router 这条路上唯一能看见「真切了」的地方。

        父类的 ``_report_fallback_once`` 靠 ``role="fallback"`` 触发——那是老的
        ``ModelConfig.fallback_model`` 走法。Router 的 fallback 发生在 litellm 内部，
        角色始终是 main，所以**不补这一下，配了 ``LLM_FALLBACK_CHAIN`` 之后的切换在前端
        和日志里都是隐形的**：只表现为今天答得有点怪。

        每个目标只报一次（模型实例活在 ``lru_cache`` 里，一直报会刷屏）；上报链路的任何
        异常都不许冒泡进 AgentLoop。
        """
        if ref == self.endpoint.ref or ref in self._served_reported:
            return
        self._served_reported.add(ref)
        degraded = degraded_against(self.endpoint.ref, ref)
        logger.warning(
            "模型出口已切到 %s（主出口 %s），能力降级：%s",
            ref,
            self.endpoint.ref,
            degraded or "无",
        )
        try:
            from app.api import monitor

            await monitor.report_model_fallback(ref, degraded)
        except Exception:  # pragma: no cover - 观测是附属品
            logger.exception("model_fallback 事件上报失败")


def build_routed_model(ref: str, fallback_refs: list[str], **kwargs: Any) -> RoutedChatModel:
    """由模型名 + fallback 链名建模型。

    ``model`` 传的是 ``endpoint.ref``——Router 拿它选 deployment，必须与 ``model_list`` 里的
    ``model_name`` 逐字一致。``credential`` 也按主出口现给：父类构造时会拿它建一个
    ``openai.AsyncClient``（随后被垫片换掉），给对了至少不会在别处读到另一家的 key。
    """
    from agentscope.credential import OpenAICredential
    from pydantic import SecretStr

    primary = resolve_endpoint(ref)
    fallbacks = [resolve_endpoint(item) for item in fallback_refs]
    kwargs.setdefault(
        "credential",
        OpenAICredential(api_key=SecretStr(primary.api_key), base_url=primary.base_url),
    )
    kwargs["model"] = primary.ref
    return RoutedChatModel(endpoint=primary, fallback_endpoints=fallbacks, **kwargs)
