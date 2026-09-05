"""模型网关闸门：并发上限 + 请求起点间隔 + 限流退避 + 备用模型可观测。

**为什么要自己做这一层**：AgentScope 的 ``ChatModelBase`` 已经带了「单次调用失败重试」和
``ModelConfig.fallback_model``「主模型垮了换备用」，但它们都是**事后**补救。真正打死我们的是
事前那件事——同一轮里主 Agent 与多个 worker 同时发请求，网关按 RPM 掐你 429，重试再撞、再退避，
一条 query 的延迟就从 40s 抖到 3 分钟。所以闸门要管的是**发出去之前**：

- ``max_concurrency``：同时在飞的请求数上限（信号量）。
- ``min_interval``：两次请求**起点**之间的最小间隔。注意是起点到起点，不是「上一个结束后再等
  一会儿」——RPM 限的是单位时间内的请求条数，与每条跑多久无关。
- 撞到限流后临时把下一个可发时刻往后推（``penalize``），让闸门自己学会降速，而不是靠重试硬撞。

**流式的坑**：``stream=True`` 时 ``__call__`` 立刻返回一个 async generator，请求其实还在飞。
如果这时就把信号量还回去，并发上限形同虚设（N 个流可以同时挂着）。所以流式路径把 slot 一直
持有到**生成器耗尽或被关闭**为止，见 :meth:`ThrottledChatModel._stream_holding_slot`。
"""

import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from agentscope.model import OpenAIChatModel

from app.agent.transient import is_rate_limited

logger = logging.getLogger(__name__)


class GatewayThrottle:
    """并发信号量 + 起点间隔的组合闸门。一个进程一份，主 / 子 / 快档共享。"""

    def __init__(self, max_concurrency: int = 4, min_interval: float = 0.0) -> None:
        self.max_concurrency = max(1, int(max_concurrency))
        self.min_interval = max(0.0, float(min_interval))
        self._sem = asyncio.Semaphore(self.max_concurrency)
        self._pace_lock = asyncio.Lock()
        # 单调时钟：下一次「允许发出」的时刻。系统时间被改也不影响节流。
        self._next_allowed_at = 0.0

    async def _wait_for_pace(self) -> float:
        """在锁内预约本次请求的发出时刻，返回实际等待秒数。

        预约（把 ``_next_allowed_at`` 先推进再放锁）而不是「等完再改」：后者会让 N 个并发协程
        读到同一个旧时刻、一起冲出去，间隔闸等于没有。
        """
        async with self._pace_lock:
            now = time.monotonic()
            start_at = max(now, self._next_allowed_at)
            self._next_allowed_at = start_at + self.min_interval
        delay = start_at - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
            return delay
        return 0.0

    @asynccontextmanager
    async def slot(self) -> AsyncGenerator[None, None]:
        """占一个并发位并按起点间隔排队；退出时归还。"""
        await self._sem.acquire()
        try:
            waited = await self._wait_for_pace()
            if waited > 0:
                logger.debug("gateway 起点间隔等待 %.2fs", waited)
            yield
        finally:
            self._sem.release()

    def penalize(self, seconds: float) -> None:
        """撞到限流后把下一次可发时刻整体后推（同步方法，异常路径里也能安全调用）。"""
        if seconds <= 0:
            return
        self._next_allowed_at = max(self._next_allowed_at, time.monotonic()) + seconds
        logger.warning("gateway 撞到限流，后续请求推迟 %.1fs", seconds)


class ThrottledChatModel(OpenAIChatModel):
    """走闸门的 ``OpenAIChatModel``：并发/间隔受控，限流自动降速，备用模型上报 AGUI。

    ``role`` 只用于可观测：``"fallback"`` 的实例被调用，就意味着主模型已经垮到要换人了——
    这件事必须让前端看得见（一条 ``model_fallback`` 事件），否则「今天怎么变慢/变笨了」永远查不出。
    """

    def __init__(
        self,
        *args: Any,
        throttle: GatewayThrottle | None = None,
        rate_limit_backoff: float = 5.0,
        role: str = "main",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._throttle = throttle or GatewayThrottle()
        self._rate_limit_backoff = rate_limit_backoff
        self._role = role
        self._fallback_reported = False

    async def _report_fallback_once(self) -> None:
        """备用模型首次被调用时上报一次；上报链路的任何异常都不许冒泡进 AgentLoop。"""
        if self._role != "fallback" or self._fallback_reported:
            return
        self._fallback_reported = True
        try:
            from app.api import monitor

            await monitor.report_model_fallback(self.model)
        except Exception:  # pragma: no cover - 观测是附属品
            logger.exception("model_fallback 事件上报失败")

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        await self._report_fallback_once()
        cm = self._throttle.slot()
        await cm.__aenter__()
        try:
            res = await super().__call__(*args, **kwargs)
        except BaseException as exc:
            if is_rate_limited(exc):
                self._throttle.penalize(self._rate_limit_backoff)
            await cm.__aexit__(type(exc), exc, exc.__traceback__)
            raise
        if isinstance(res, AsyncGenerator):
            # 流式：请求还在飞，slot 交给包装生成器持有到耗尽/关闭。
            return self._stream_holding_slot(res, cm)
        await cm.__aexit__(None, None, None)
        return res

    async def _stream_holding_slot(
        self,
        stream: AsyncGenerator[Any, None],
        cm: Any,
    ) -> AsyncGenerator[Any, None]:
        """转发流式响应，把 slot 持有到生成器耗尽或被 ``aclose()``。"""
        try:
            async for chunk in stream:
                yield chunk
        except BaseException as exc:
            if is_rate_limited(exc):
                self._throttle.penalize(self._rate_limit_backoff)
            await cm.__aexit__(type(exc), exc, exc.__traceback__)
            raise
        else:
            await cm.__aexit__(None, None, None)
