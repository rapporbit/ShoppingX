"""进程内队列：``QUEUE_ENABLED=0``（默认）时的回落实现。

**为什么默认是它。** 本仓的单机部署（gcjp）只有一个后端容器，起 worker 进程与 Redis 消费者组纯属
给自己加运维面。回落实现让 ``server.py`` / ``worker.py`` 只认一个端口形状——开不开队列是一行环境
变量的事，代码没有 ``if queue_enabled`` 的分叉。

**它诚实地不做三件事**，因为在单进程里做了也是假的：
- **不重投**。消息和消费它的进程同生共死，进程没了 deque 也没了，「留着等下个 worker 捡」无对象。
- **不死信**。失败原因写进状态表就够，再单开一条内存死信流没人会去读。
- **不跨进程**。它是 :class:`app.queue.redis_stream.RedisStreamQueue` 的同形状替身，不是它的等价物；
  「重启不丢任务」这类能力只有开 Redis 才有——别在文档里把两者说成一回事。

保留的是**分级**：normal 先于 heavy 出队，与 Redis 侧的双流优先级语义一致，这样切换前后长续聊
与短对话的相对体验不变。
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict, deque
from collections.abc import Callable

from app.api.concurrency import RequestClass
from app.queue.ports import IntentTask, TaskHandler, TaskStatus, cancel_in_flight

logger = logging.getLogger("shoppingx.queue")

# 状态表条数上限：内存里躺着的是「跑完的任务最后长什么样」，只给轮询接口读几秒钟。不设上限的
# 话，一个长跑进程的字典会随任务数无界增长（每条几 KB 的 final_text）。
_STATUS_MAX = 1000
# 出队顺序即优先级，与 Redis 侧 STREAMS 的排列同义。
_ORDER: tuple[RequestClass, ...] = ("normal", "heavy")


class InProcessQueue:
    """:class:`app.queue.ports.TaskQueue` 的进程内实现（两条 deque + 内存状态表）。"""

    def __init__(self) -> None:
        self._pending: dict[RequestClass, deque[IntentTask]] = {
            "normal": deque(),
            "heavy": deque(),
        }
        self._status: OrderedDict[str, TaskStatus] = OrderedDict()
        # 有新任务时唤醒消费循环。不用 asyncio.Queue：要的是「两条队列按优先级取」，而 Queue 的
        # get() 只能挂在一条上，挂错那条就会在另一条有货时干等。
        self._arrival = asyncio.Event()

    async def enqueue(self, task: IntentTask) -> None:
        self._pending[task.kind].append(task)
        self._arrival.set()
        logger.info("入队 %s（进程内 %s，thread=%s）", task.task_id, task.kind, task.thread_id)

    async def set_status(self, status: TaskStatus) -> None:
        self._status[status.task_id] = status
        self._status.move_to_end(status.task_id)
        while len(self._status) > _STATUS_MAX:
            self._status.popitem(last=False)

    async def get_status(self, task_id: str) -> TaskStatus | None:
        return self._status.get(task_id)

    async def depth(self) -> int:
        return sum(len(q) for q in self._pending.values())

    async def close(self) -> None:
        self._arrival.set()  # 唤醒可能正挂着的消费循环，让它看见 should_stop

    def _pop(self) -> IntentTask | None:
        for kind in _ORDER:  # normal 优先，与 Redis 侧双流顺序一致
            queue = self._pending[kind]
            if queue:
                return queue.popleft()
        return None

    async def consume(
        self,
        consumer: str,
        handler: TaskHandler,
        should_stop: Callable[[], bool],
        concurrency: int = 1,
        *,
        poll_interval: float = 0.5,
    ) -> None:
        """消费循环，形状与 Redis 侧一致：跑到 ``should_stop()`` 为真且在途任务收干净才返回。"""
        limit = max(1, concurrency)
        in_flight: set[asyncio.Task[None]] = set()
        try:
            while not should_stop():
                in_flight = {t for t in in_flight if not t.done()}
                if len(in_flight) >= limit:
                    await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
                    continue
                task = self._pop()
                if task is None:
                    self._arrival.clear()
                    try:
                        await asyncio.wait_for(self._arrival.wait(), timeout=poll_interval)
                    except TimeoutError:
                        pass  # 超时是正常路径：借它回头看一眼 should_stop
                    continue
                in_flight.add(asyncio.create_task(self._handle_one(task, handler)))
            if in_flight:
                logger.info("停止领新任务，等 %d 个在途任务跑完", len(in_flight))
                await asyncio.gather(*in_flight, return_exceptions=True)
        except asyncio.CancelledError:
            # 与 Redis 侧同一手法：create_task 出来的在途任务不会随本协程一起被取消，得显式掐。
            await cancel_in_flight(in_flight)
            raise

    async def _handle_one(self, task: IntentTask, handler: TaskHandler) -> None:
        try:
            await handler(task)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # 不重投：单进程里失败多半是确定性的（坏 payload / 代码 bug），再跑一遍只是再错一次。
            logger.warning("任务失败：%s（%s）", task.task_id, exc)
            await self.set_status(
                TaskStatus(
                    task_id=task.task_id,
                    state="failed",
                    thread_id=task.thread_id,
                    error=str(exc),
                )
            )
