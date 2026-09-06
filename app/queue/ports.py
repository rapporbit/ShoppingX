"""任务队列端口（Protocol）+ 任务 / 状态值对象。

**为什么要队列。** 现状 ``POST /api/task`` 直接 ``asyncio.create_task`` 在 API 进程里跑
AgentLoop，并发上限由 :mod:`app.api.concurrency` 的双池准入守着。单进程下够用，但三件事做不到：
① API 进程重启（部署 / OOM）时在跑的任务全丢，用户那条 WS 永远停在 running；② 扩容只能整进程扩，
而「收请求」与「跑 Agent」的资源曲线完全不同；③ 峰值只能 429，削不了峰。队列把两件事拆成两个
进程：API 入队即返回，worker 按自己的并发度慢慢消费。

**队列不替代双池准入，两者管的是不同的事**：准入池管「本进程同时跑几个 loop」，队列管「任务在
哪个进程跑、掉不掉」。但 ``TASK_HEAVY_TURNS`` 这个阈值**必须两边共用同一份**（都走
:func:`app.api.concurrency.classify_request`）——各读各的会漂成「准入判 heavy、队列判 normal」，
长续聊照样堵在短任务前面，而且不报错。

**投递语义是 at-least-once。** Redis Stream 的 pending 重投、worker 崩溃重启都会让同一条任务被
消费两次，消费方必须自己幂等。本仓 API 侧的幂等第 1/3 层（``active_tasks`` 同 thread 去重、
:mod:`app.api.dedup` 指纹去重）挡不到这里——它们在入队之前。真正的风险面是写工具
（``create_order`` 重复消费 = 重复下单），靠 ``tools/_order_guard.py`` 的两段式确认卡兜：没出过卡
的 ``confirmed=True`` 会被退回成出卡，重投的那次只会再出一张卡，不会真下单。
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, Protocol, runtime_checkable

from app.api.concurrency import RequestClass, classify_request

TaskState = Literal["queued", "running", "done", "failed"]


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class IntentTask:
    """一条待消费的购物意图。字段与 :func:`app.agent.orchestrator.run_agent` 的入参一一对应。

    ``kind`` 就是准入池的 :data:`app.api.concurrency.RequestClass`（normal / heavy），队列按它分流到
    两条 Stream。**不存 ``history_turns`` 只存判完的 ``kind``**：轮数是入队瞬间的事实，任务在队列里
    躺几分钟后 worker 再去数一遍可能已经变了，同一条任务在两处被判成两个池就没法解释了。
    """

    task_id: str
    thread_id: str
    query: str
    user_id: str | None = None
    platforms: tuple[str, ...] = ()
    image_paths: tuple[str, ...] = ()
    kind: RequestClass = "normal"
    enqueued_at: str = field(default_factory=_now_iso)

    @classmethod
    def create(
        cls,
        *,
        task_id: str,
        thread_id: str,
        query: str,
        history_turns: int = 0,
        user_id: str | None = None,
        platforms: Sequence[str] | None = None,
        image_paths: Sequence[str] | None = None,
    ) -> IntentTask:
        """按历史轮数判池并构造任务——分流阈值的唯一入口，调用方不要自己拿轮数比大小。"""
        return cls(
            task_id=task_id,
            thread_id=thread_id,
            query=query,
            user_id=user_id,
            platforms=tuple(platforms or ()),
            image_paths=tuple(image_paths or ()),
            kind=classify_request(history_turns),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "thread_id": self.thread_id,
            "query": self.query,
            "user_id": self.user_id,
            "platforms": list(self.platforms),
            "image_paths": list(self.image_paths),
            "kind": self.kind,
            "enqueued_at": self.enqueued_at,
        }

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> IntentTask:
        """反序列化。**只有 task_id / thread_id / query 是必需的**——其余缺了就取默认值。

        队列里可能躺着上一个版本的进程写进去的消息（滚动更新期间新旧 worker 并存），对新增字段
        宽容能让升级不必清空队列；真正缺不得的那三个缺了就该进死信，故不给默认值。
        """
        kind = raw.get("kind")
        return IntentTask(
            task_id=raw["task_id"],
            thread_id=raw["thread_id"],
            query=raw["query"],
            user_id=raw.get("user_id"),
            platforms=tuple(raw.get("platforms") or ()),
            image_paths=tuple(raw.get("image_paths") or ()),
            kind="heavy" if kind == "heavy" else "normal",
            enqueued_at=raw.get("enqueued_at", ""),
        )


@dataclass(frozen=True)
class TaskStatus:
    """任务的最新状态。``GET /api/task/{id}`` 直接返回它（异步提交模式下前端轮询用）。

    ``queue_depth`` 是「队列里还有多少条没被领走」，**不是这条任务的精确排位**：Stream 不提供
    「某条消息排第几」的查询，真要算得 XRANGE 全流扫一遍。用户要的是量级（「前面还有几个」），
    为精度赔一次全流扫不值——排位反馈的精确版本在 WS 的 ``queue_status`` 事件里（那是准入池的
    位置，有真实 FIFO 队列可数）。
    """

    task_id: str
    state: TaskState
    thread_id: str = ""
    final_text: str = ""
    error: str = ""
    queue_depth: int = 0
    updated_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "state": self.state,
            "thread_id": self.thread_id,
            "final_text": self.final_text,
            "error": self.error,
            "queue_depth": self.queue_depth,
            "updated_at": self.updated_at,
        }

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> TaskStatus:
        state = raw.get("state", "queued")
        return TaskStatus(
            task_id=raw["task_id"],
            state=state if state in ("queued", "running", "done", "failed") else "failed",
            thread_id=raw.get("thread_id", ""),
            final_text=raw.get("final_text", ""),
            error=raw.get("error", ""),
            queue_depth=int(raw.get("queue_depth", 0) or 0),
            updated_at=raw.get("updated_at", ""),
        )


TaskHandler = Callable[[IntentTask], Awaitable[None]]


async def cancel_in_flight(tasks: set[asyncio.Task[None]]) -> None:
    """消费循环被取消时，连带掐掉它 ``create_task`` 出来的在途任务。两份实现共用。

    **不能指望取消会自动往下传**：``create_task`` 出来的是独立 task，取消父协程只会打断父协程当前
    那个 ``await``（``gather`` 是例外，它会把取消转给子任务）。少这一手，worker 优雅退出超时那条路
    上会留一批孤儿协程——进程都在退出了它们还在跑 LLM，消息既没 ack 也没人管。

    掐掉之后消息**留在 PEL 里没被 ack**，正是「超时转回 pending」要的效果。
    """
    if not tasks:
        return
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


@runtime_checkable
class TaskQueue(Protocol):
    """队列端口。Redis Stream 与进程内两份实现共用它，``server.py`` / ``worker.py`` 只认这个形状。

    用 ``Protocol`` 而非 ABC：两份实现之间没有共享逻辑可继承（一个说 Redis 协议、一个操作 deque），
    抽基类只会多一层空壳；结构化子类型也让测试里的假实现不必显式继承。
    """

    async def enqueue(self, task: IntentTask) -> None:
        """入队。**失败必须抛**——静默吞掉等于任务凭空消失，调用方要靠异常决定回落还是 5xx。"""
        ...

    async def set_status(self, status: TaskStatus) -> None: ...

    async def get_status(self, task_id: str) -> TaskStatus | None: ...

    async def depth(self) -> int:
        """待消费任务数（两条流之和）。取不到时返回 0——观测不该拖垮主链路。"""
        ...

    async def consume(
        self,
        consumer: str,
        handler: TaskHandler,
        should_stop: Callable[[], bool],
        concurrency: int = 1,
    ) -> None:
        """消费循环，跑到 ``should_stop()`` 为真且在途任务收干净才返回。"""
        ...

    async def close(self) -> None: ...
