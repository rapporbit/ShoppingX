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

**断路器挂在同一层**（阶段 2 第 1 条，口径见 :mod:`app.agent.llm_breaker`）：闸在取 slot 之前，
记账在调用/流结束之后，首 token 预算在第一片上。放这层是因为它和闸门问的是同一个问题的两半
——闸门管「发不发得出去」，断路器管「还值不值得发」。
"""

import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any

from agentscope.model import OpenAIChatModel

from app.agent.llm_breaker import (
    CircuitOpenError,
    FirstTokenTimeout,
    first_token_timeout,
    get_llm_breaker,
    record_outcome,
)
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


class _Attempt:
    """一次**尝试**的起点（不是一次调用的起点）。

    重试跑在 ``ChatModelBase.__call__`` 里面，我们在外面看不见。若首 token 预算从「取到 slot」
    起算，前两次失败尝试的耗时会被算进第三次的预算里——限流风暴时最后那次本来要成功的调用会被
    误掐、还记一次故障，正好把「429 不计」的设计绕过去。所以每进一次 ``_call_api`` 就重新打点。
    """

    __slots__ = ("started",)

    def __init__(self) -> None:
        self.started = time.monotonic()


# 只用于把 holder 从 ``__call__`` 递到 ``_call_api``（同一个 task 内），不跨协程共享：
# 并发调用各在各的 task，context 天然隔离。
_current_attempt: ContextVar[_Attempt | None] = ContextVar("llm_attempt", default=None)


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
        # 断路器的闸放在**取 slot 之前**：OPEN 态的意义就是不占资源直接拒，占了并发位再拒等于
        # 把快速失败的收益还回去。键用 ``self.model``——2-1 之后它在两条出口路上都是 provider/model。
        breaker = get_llm_breaker(self.model)
        if breaker is not None and not breaker.allow():
            raise CircuitOpenError(f"模型出口 {self.model} 断路器 OPEN，快速失败")
        cm = self._throttle.slot()
        await cm.__aenter__()
        # 计时从**拿到 slot 之后**开始：排队等并发位、等起点间隔是我们自己的节流，算进首 token
        # 预算里就会在高并发时集体误判对面挂了。真正的打点在 ``_call_api``（每次尝试一次）。
        attempt = _Attempt()
        token = _current_attempt.set(attempt)
        try:
            res = await super().__call__(*args, **kwargs)
        except BaseException as exc:
            if is_rate_limited(exc):
                self._throttle.penalize(self._rate_limit_backoff)
            record_outcome(breaker, exc)
            await cm.__aexit__(type(exc), exc, exc.__traceback__)
            raise
        finally:
            _current_attempt.reset(token)
        if isinstance(res, AsyncGenerator):
            # 流式：请求还在飞，slot 与断路器的记账都交给包装生成器。
            return self._stream_holding_slot(res, cm, breaker, attempt.started)
        record_outcome(breaker, None)
        await cm.__aexit__(None, None, None)
        return res

    async def _call_api(self, *args: Any, **kwargs: Any) -> Any:
        """每次**尝试**进来时重新打点首 token 预算的起点（重试的坑见 :class:`_Attempt`）。"""
        attempt = _current_attempt.get()
        if attempt is not None:
            attempt.started = time.monotonic()
        return await super()._call_api(*args, **kwargs)

    async def _first_chunk(self, agen: Any, started: float) -> Any:
        """取第一片，超预算就掐断并抛 :class:`FirstTokenTimeout`。

        预算是「从请求发出到首片到达」，所以要减掉 ``super().__call__`` 里已经花掉的那段
        （建连 + 拿响应头就发生在那里，流式下它先于任何一片返回）。

        掐断用的是 ``aclose()``：不关的话 slot 还回去了、HTTP 连接还挂着，对面真卡住时
        连接数会随请求一起涨——那正是我们要止的血。
        """
        budget = first_token_timeout()
        if budget <= 0:
            return await agen.__anext__()
        remaining = budget - (time.monotonic() - started)

        # **不能用 asyncio.wait_for**：它靠取消 ``__anext__`` 来实现超时，再拿任务的最终结果说话。
        # 而 AgentScope 的流包装 ``_stream()`` 把 ``CancelledError`` 吞了（catch 之后还会 yield
        # 一片累计结果），于是任务「成功」返回、wait_for 原样把那片空响应交回来——超时静默失效，
        # 表现是一次没内容的模型调用，不是报错。所以这里自己判定：只看时间到没到，不看任务结局。
        task = asyncio.ensure_future(agen.__anext__())
        done, _pending = await asyncio.wait({task}, timeout=max(remaining, 0.0))
        if not done:
            task.cancel()
            try:
                await task
            except BaseException:  # noqa: BLE001 - 取消后的任何结局都不改变「已超时」这个判定
                pass
            try:
                await agen.aclose()
            except Exception:  # pragma: no cover - 关流失败不该盖过超时本身
                logger.debug("首 token 超时后关流失败", exc_info=True)
            raise FirstTokenTimeout(self.model, budget)
        # 正常到片（或 ``StopAsyncIteration`` / 真实异常）都由这里原样抛出去。
        return task.result()

    async def _stream_holding_slot(
        self,
        stream: AsyncGenerator[Any, None],
        cm: Any,
        breaker: Any = None,
        started: float = 0.0,
    ) -> AsyncGenerator[Any, None]:
        """转发流式响应，把 slot 持有到生成器耗尽或被 ``aclose()``。

        断路器在这里才记账：流式下 ``__call__`` 返回时请求其实刚发出去，成败要等流走完才知道。
        """
        try:
            agen = stream.__aiter__()
            empty = False
            try:
                first = await self._first_chunk(agen, started)
            except StopAsyncIteration:  # 空流：对面直接结束，不是超时
                empty = True
            if not empty:
                yield first
                async for chunk in agen:
                    yield chunk
        except BaseException as exc:
            if is_rate_limited(exc):
                self._throttle.penalize(self._rate_limit_backoff)
            record_outcome(breaker, exc)
            await cm.__aexit__(type(exc), exc, exc.__traceback__)
            raise
        else:
            record_outcome(breaker, None)
            await cm.__aexit__(None, None, None)
