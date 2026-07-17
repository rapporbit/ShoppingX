"""任务准入：双池优先级队列（refdocs 16-5 §2）——大请求堵不死小请求。

**要解决的问题。** 全局只有一个任务槽池时，几个「30 轮续聊 + 跨平台 fork」的长任务能把槽全占满，
后面所有「帮我看下这个包多少钱」的短任务一律吃闭门羹。槽位是公平的，体验不是。

**做法：按重量分池，各池独立排队。**

    POST /api/task ──→ 分类器（看该 thread 的历史轮数）
                        ├─→ normal 池（5 槽 + 有界等待队列）  短对话
                        └─→ heavy  池（3 槽 + 有界等待队列）  长续聊
    两池都满 ──→ 429 + Retry-After

长任务最多占满 heavy 的 3 个槽，normal 的 5 个槽永远为短任务留着。这是分池的全部意义。

**对 refdocs 的两处主动更正：**

1. **队列有界，满了仍然 429。** refdocs 的 worker pool 模型隐含「无界排队」。Agent 任务动辄跑
   几十秒到几分钟，无界排队会堆出「看起来没满、其实全在等」的假象，用户对着进度条空等更久，
   还不如让他早点知道稍后再来。所以每池的等待队列有深度上限（``TASK_QUEUE_DEPTH``），
   **有界排队 + 溢出即拒**——排队换体验，上限守住背压。

2. **排队发生在后台 task 里，不在 HTTP 请求里。** ``POST /api/task`` 本就是「立即返回 thread_id，
   任务后台跑」（connect-first 协议）。所以「等槽」这段挪进后台协程 ``await wait_turn()``，
   HTTP 响应不受影响；用户在 WS 上收到 ``queue_status`` 事件看见自己排第几位。**唯一必须在
   endpoint 同步段做的是「占位判定」**（try_reserve）——它全同步无 ``await``，单线程下「检查 +
   占位」之间不会被别的请求插入，天然无竞态。这条约束和旧的 ``try_acquire`` 一模一样。

**动态再平衡。** normal 队列积压时把 heavy 的容量临时压到下限——高峰期优先保短任务。缩容不抢占
已在跑的任务（``active`` 允许暂时大于 ``capacity``，随任务退出自然回落），Agent 任务跑到一半被
掐断是不可接受的。

fork 级并发（同一任务内多个子 Agent）仍走 ``asyncio.Semaphore`` 的纯排队语义（见
:func:`app.harness.budgets.fork_concurrency_scope`）：子任务是任务自身的一部分，拒掉会丢平台覆盖。
两层各取最贴合语义的原语。
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Literal

from app.utils.env import env_int

logger = logging.getLogger("shoppingx.concurrency")

RequestClass = Literal["normal", "heavy"]

# 两池槽位。总和（默认 5+3=8）即全局最大并发主 AgentLoop 数——Agent 是成本放大器（fork + 多轮 +
# 工具链），无上限时并发一高就把下游（embedding / rerank / LLM 外呼）一起打爆。
NORMAL_SLOTS = env_int("TASK_NORMAL_SLOTS", 5)
HEAVY_SLOTS = env_int("TASK_HEAVY_SLOTS", 3)
# 再平衡时 heavy 池被压到的下限。不设 0：长任务也是用户，饿死它不是背压是拒绝服务。
HEAVY_SLOTS_MIN = env_int("TASK_HEAVY_SLOTS_MIN", 1)
# normal 池排队人数达到它 → 压缩 heavy 容量；回落到 0 → 恢复。
REBALANCE_PENDING = env_int("TASK_REBALANCE_PENDING", 3)
# 每池等待队列深度上限。超了才 429——排队换体验，上限守背压。
QUEUE_DEPTH = env_int("TASK_QUEUE_DEPTH", 8)
# 被拒时回给前端的建议重试间隔（秒）。
TASK_RETRY_AFTER_SEC = env_int("TASK_RETRY_AFTER_SEC", 10)
# 分类阈值：该 thread 已积累的历史轮数 ≥ 它即判为 heavy。续聊越长，上下文越大、跑得越久。
HEAVY_TURNS_THRESHOLD = env_int("TASK_HEAVY_TURNS", 8)
# 排队位置反馈里的「预估等待」= 位置 × 它。一个粗略常数即可——用户要的是「还要等很久吗」的量级，
# 不是秒级精度。真做加权移动平均反而会因为长尾任务把估计值拖得离谱。
AVG_TASK_SECONDS = env_int("TASK_AVG_SECONDS", 30)


class _Pool:
    """一个可调容量的槽位池：非阻塞占槽 + 有界 FIFO 等待队列。

    为什么不用 ``asyncio.Semaphore``：它的容量在构造后不可变，而再平衡要求运行时改容量；且它不
    暴露「当前有几个人在等」，而排队位置反馈正需要这个数。所以自己拿 Future 队列实现一个。

    **占槽发生在唤醒方**（``_wake`` 里先 ``active += 1`` 再 ``set_result``），不是等待方醒来后自己
    加。否则唤醒到执行之间的窗口里，别的协程能插队把槽抢走，容量约束就破了。

    **``_reserved`` 是什么。** 准入判定（同步，在 endpoint 里）与真正入队（``await acquire()``，在
    后台协程里）之间隔着一次调度。这中间的请求「已获准排队但还没进队列」，``len(_waiters)`` 看不见
    它们——不补偿的话，同时到达的三个请求会被告知「你排第 1 位」三次，队列深度上限也形同虚设。
    ``_reserved`` 就是这批在途请求的计数。
    """

    __slots__ = ("capacity", "_active", "_waiters", "_reserved")

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._active = 0
        self._waiters: deque[asyncio.Future[None]] = deque()
        self._reserved = 0

    @property
    def active(self) -> int:
        return self._active

    @property
    def pending(self) -> int:
        """排队等待的请求数：已在队列里的 + 已获准排队但还没走到 ``acquire`` 的。"""
        return len(self._waiters) + self._reserved

    @property
    def full(self) -> bool:
        return self._active >= self.capacity

    def try_acquire(self) -> bool:
        """非阻塞占槽。全同步无 ``await``——必须在 endpoint 同步段调用。

        **有人在排队时不放行**（``pending == 0`` 才占）：否则新来的请求会插队到等待者前面，
        FIFO 公平性直接失效，排在第一位的人可能永远等不到。
        """
        if self._active < self.capacity and self.pending == 0:
            self._active += 1
            return True
        return False

    def reserve_queue_slot(self) -> int:
        """登记一个「已获准排队、尚未入队」的在途请求，返回它的 1-based 排队位置。"""
        self._reserved += 1
        return self.pending

    def unreserve_queue_slot(self) -> None:
        """在途请求真正入队（或中途放弃）时销账。"""
        self._reserved = max(0, self._reserved - 1)

    def force_acquire(self) -> None:
        """无视容量与队列强占一槽——只用于「同 thread 覆盖重发」（旧任务马上释放它的槽）。"""
        self._active += 1

    async def acquire(self) -> None:
        """排队等一个槽。被取消时正确归还/摘除，不泄漏槽位。"""
        if self.try_acquire():
            return
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._waiters.append(fut)
        try:
            await fut
        except asyncio.CancelledError:
            if fut.done() and not fut.cancelled():
                # 已经被 _wake 唤醒并替我们占了槽，但等待者随即被取消 → 把槽还回去。
                self.release()
            else:
                # 还在队列里排着 → 直接摘掉自己。
                try:
                    self._waiters.remove(fut)
                except ValueError:  # pragma: no cover - 竞态兜底
                    pass
            raise

    def release(self) -> None:
        """释放一槽并唤醒队首。``max(0, ...)`` 防御多次释放把计数压成负数。"""
        self._active = max(0, self._active - 1)
        self._wake()

    def set_capacity(self, capacity: int) -> None:
        """运行时调容量。缩容**不抢占**已在跑的任务（active 可暂时 > capacity，随退出回落）。"""
        if capacity == self.capacity:
            return
        self.capacity = capacity
        self._wake()

    def _wake(self) -> None:
        while self._waiters and self._active < self.capacity:
            fut = self._waiters.popleft()
            if fut.cancelled():
                continue
            self._active += 1  # 占槽在唤醒方完成，杜绝唤醒窗口里被插队
            fut.set_result(None)


@dataclass
class Reservation:
    """一次任务准入的凭据：从 ``try_reserve`` 拿到，用完必须 ``release``。

    ``admitted`` 为真表示**已经持有槽位**（直接占到，或排队等到了）。它是 ``release`` 是否要归还
    槽位的唯一依据——排队途中被取消的任务从没持过槽，归还就成了凭空多出一个槽。

    ``counted`` 表示这张凭据还占着池子里一个「在途排队」名额（见 ``_Pool._reserved``），必须在
    入队时或放弃时销账，否则队列深度会被幽灵请求慢慢占满。
    """

    kind: RequestClass
    admitted: bool = False
    position: int = 0  # 入队时排在第几位（1-based）；直接占到槽则为 0
    counted: bool = field(default=False, repr=False)
    released: bool = field(default=False, repr=False)


class PriorityRequestQueue:
    """双池优先级队列：normal / heavy 各有独立槽位与有界等待队列。"""

    def __init__(
        self,
        normal_slots: int = NORMAL_SLOTS,
        heavy_slots: int = HEAVY_SLOTS,
        queue_depth: int = QUEUE_DEPTH,
    ) -> None:
        self._pools: dict[RequestClass, _Pool] = {
            "normal": _Pool(normal_slots),
            "heavy": _Pool(heavy_slots),
        }
        self._heavy_slots = heavy_slots
        self._queue_depth = queue_depth

    # ── 观测 ──

    @property
    def limit(self) -> int:
        """全局并发上限（两池容量之和）。"""
        return self._pools["normal"].capacity + self._pools["heavy"].capacity

    @property
    def active(self) -> int:
        return sum(p.active for p in self._pools.values())

    @property
    def full(self) -> bool:
        return all(p.full for p in self._pools.values())

    def pending(self, kind: RequestClass) -> int:
        return self._pools[kind].pending

    def stats(self) -> dict[str, dict[str, int]]:
        """供 /api/health 与 /metrics 读的快照。"""
        return {
            kind: {"active": p.active, "capacity": p.capacity, "pending": p.pending}
            for kind, p in self._pools.items()
        }

    # ── 准入 ──

    def try_reserve(self, kind: RequestClass) -> Reservation | None:
        """同步准入判定：占到槽 / 排进队 / 都不行返回 ``None``（调用方回 429）。

        全同步无 ``await``，必须在 endpoint 同步段调用（理由见模块 docstring）。
        """
        pool = self._pools[kind]
        if pool.try_acquire():
            self._rebalance()
            return Reservation(kind=kind, admitted=True)
        if pool.pending >= self._queue_depth:
            return None
        position = pool.reserve_queue_slot()
        res = Reservation(kind=kind, admitted=False, position=position, counted=True)
        self._rebalance()
        return res

    def force_reserve(self, kind: RequestClass) -> Reservation:
        """强占一槽（同 thread 覆盖重发专用）：用户在同一会话里改问，不该被自己的旧任务挡在门外。"""
        self._pools[kind].force_acquire()
        return Reservation(kind=kind, admitted=True)

    async def wait_turn(self, res: Reservation) -> None:
        """等到真正拿到槽为止。已直接占到槽的立即返回。

        在后台协程里 ``await``，不阻塞 HTTP 响应。被取消（用户点了取消 / 覆盖重发）时
        ``_Pool.acquire`` 负责不泄漏槽位。
        """
        if res.admitted:
            return
        pool = self._pools[res.kind]
        # 先销「在途」账再入队：这两步之间无 await，pending 的口径始终守恒（在途 → 队内）。
        if res.counted:
            pool.unreserve_queue_slot()
            res.counted = False
        await pool.acquire()
        res.admitted = True  # 到这里已持槽；acquire 返回与本行之间无 await，不会被 cancel 切进来

    def release(self, res: Reservation) -> None:
        """归还凭据。幂等（重复调用无副作用）；未持槽的凭据不归还槽位。

        三条退出路径都要正确销账：持槽的还槽；还在「在途」名额里的（任务在 ``wait_turn`` 之前就
        被取消）销在途账；已入队等待的由 ``_Pool.acquire`` 的取消分支自己摘除。

        **三条路径都要再平衡**：排队者被取消（既没持槽、也不在在途名额里）同样会让 normal 队列
        排空，此时若不重算，heavy 容量会一直被压在下限，直到下一次 reserve / release 才恢复。
        """
        if res.released:
            return
        res.released = True
        if res.admitted:
            self._pools[res.kind].release()
        elif res.counted:
            self._pools[res.kind].unreserve_queue_slot()
            res.counted = False
        self._rebalance()

    # ── 再平衡 ──

    def _rebalance(self) -> None:
        """normal 积压 → 压缩 heavy 容量；normal 清空 → 恢复。高峰期优先保短任务。"""
        normal_pending = self._pools["normal"].pending
        heavy = self._pools["heavy"]
        if normal_pending >= REBALANCE_PENDING and heavy.capacity > HEAVY_SLOTS_MIN:
            heavy.set_capacity(HEAVY_SLOTS_MIN)
            logger.info("normal 队列积压 %d，heavy 容量压到 %d", normal_pending, HEAVY_SLOTS_MIN)
        elif normal_pending == 0 and heavy.capacity < self._heavy_slots:
            heavy.set_capacity(self._heavy_slots)
            logger.info("normal 队列清空，heavy 容量恢复到 %d", self._heavy_slots)


def classify_request(history_turns: int) -> RequestClass:
    """按该 thread 已积累的历史轮数分类。

    轮数是最直接的重量代理：续聊越长，回喂的上下文越大、模型每轮解码越慢、越可能再触发 fork。
    比「按 query 长度」或「按用户等级」都更贴近真实成本，而且零成本可得（turns.json 已在磁盘上）。
    """
    return "heavy" if history_turns >= HEAVY_TURNS_THRESHOLD else "normal"


def estimated_wait_seconds(position: int, capacity: int) -> int:
    """排在第 ``position`` 位的粗略预估等待秒数（给用户看量级，不追求精度）。

    除以 ``capacity``：池子有 5 个槽时，排第 3 位并不需要等 3 个任务跑完——第一批退出就轮到了。
    不做加权移动平均：Agent 任务的耗时长尾很重（一次跨平台 fork 能到几分钟），均值会被拖得离谱，
    还不如一个诚实的粗略常数。
    """
    if position <= 0:
        return 0
    slots = max(1, capacity)
    rounds = -(-position // slots)  # 向上取整：等前面 ceil(position/slots) 批跑完
    return rounds * AVG_TASK_SECONDS


# 进程级单例：server.py 在 /api/task 入口用它做准入 + 排队。
task_queue = PriorityRequestQueue()
