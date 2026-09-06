"""独立 worker 进程：从削峰队列领意图任务，在本进程里跑主 AgentLoop。

``uv run python -m app.worker`` 起一个。它与 ``app.api.server`` 的分工是**收请求 / 跑 Agent 分家**：
API 进程只做鉴权、配额、幂等、入队，然后立刻回 ``thread_id``；真正烧 CPU 与 token 的那段在这里。
分家买到三件事——API 重启（部署 / OOM）时在跑的任务不丢（消息还在 Redis 的 PEL 里等重投）、两种
进程可以按各自的资源曲线独立扩容、峰值削得掉而不是只能 429。

**优雅退出是这个模块的主线，不是边角料。** K8s 滚动更新时 worker 一定会收到 SIGTERM，而一条购物
任务动辄跑 40s 到几分钟。收到信号后做三件事，顺序不能换：

1. **先停领新的**（``should_stop()`` 转真 → 消费循环不再 ``XREADGROUP``）。先停领再等在飞，反过来
   会边等边领，永远等不完。
2. **等在飞任务自然跑完**，最多等 ``WORKER_GRACE_SECONDS``。这段时间里任务照常上报 AGUI 事件、
   照常落盘，用户完全无感。
3. **超时就取消在飞任务**。被取消的那条消息**不 ack**，于是留在 PEL 里，由下一个 worker 的
   ``XAUTOCLAIM`` 领回重跑（``QUEUE_CLAIM_IDLE_MS`` 之后）。这就是「超时转回 pending」——宁可重跑
   一次（投递语义本就是 at-least-once，写工具那侧有两段式确认卡兜着），也不让任务凭空消失。

配套的 K8s 侧写法是 ``terminationGracePeriodSeconds`` 要**大于** ``WORKER_GRACE_SECONDS``，否则
kubelet 的 SIGKILL 会先到，第 2 步白设。那份 yaml 是批 2 后面一单的事。

**为什么状态表由 worker 写而不是 API 写。** ``GET /api/task/{id}`` 读的 ``TaskStatus`` 是「这条任务
现在跑到哪了」——只有真正在跑它的进程知道。API 侧写的话必然是猜的（入队即写 running，然后永远不会
变）。所以 running / done / failed 三个终态全在 :func:`handle_task` 里落。
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import socket
from contextlib import suppress
from typing import Any

from app.agent.orchestrator import run_agent
from app.api import clarification, control
from app.config import store as config_store
from app.db.session import init_db
from app.observability.logging import configure_logging
from app.queue import IntentTask, TaskQueue, TaskStatus, get_task_queue, queue_enabled
from app.utils.env import env_int
from app.utils.tokens import warm_tokenizer

logger = logging.getLogger("shoppingx.worker")

# 本进程同时跑几个 AgentLoop。默认 4 而不是 API 侧准入池的 8（5+3）：那 8 个槽是「一个进程既收请求
# 又跑 Agent」时的上限，拆开之后 worker 可以横向加副本，单副本压满反而让长尾更长。
WORKER_CONCURRENCY = env_int("WORKER_CONCURRENCY", 4)
# 收到 SIGTERM 后最多等在飞任务多久。要覆盖住绝大多数单轮耗时（本仓约 40s），又不能长到让滚动更新
# 卡住；超时的那些会转回 pending 重投，不是丢。
WORKER_GRACE_SECONDS = env_int("WORKER_GRACE_SECONDS", 120)


def consumer_name() -> str:
    """本 worker 在消费者组里的名字。

    带 pid：同一台机器 / 同一个镜像起多个副本时名字不能撞——``XREADGROUP`` 按 consumer 名维护各自的
    PEL，重名会让两个进程共用一份 pending 记录，谁 ack 了谁的都说不清。K8s 里用 ``WORKER_NAME``
    直接把 pod 名递进来更好读。
    """
    return os.environ.get("WORKER_NAME") or f"{socket.gethostname()}-{os.getpid()}"


async def handle_task(task: IntentTask, queue: TaskQueue | None = None) -> None:
    """消费一条任务：状态置 running → 跑 ``run_agent`` → 落 done / failed。

    **失败必须往外抛**：队列侧靠这个异常决定「留在 PEL 等重投」还是「重投超限进死信」。在这里吞掉
    等于任务默默消失，而状态表里还写着 running，谁都看不出发生了什么。写状态只是给轮询接口看的
    副产品，不是错误处理本身。

    **被取消时不写终态**：优雅退出把在飞任务掐掉时，这条消息没有被 ack，它会被下一个 worker 领回来
    重跑。此刻写 failed 会让轮询方以为已经有定论，而几秒后它又活了过来。

    **但「用户取消」是另一回事，两种取消必须分开**（批2-4）。它同样表现为 ``CancelledError``，
    可语义相反：任务不该被重投——用户要的就是它别再跑了，重投一遍等于取消按钮没用。判据是控制面
    的进程内标记（:func:`app.api.control.was_cancelled_locally`），只有它才知道这一刀是谁砍的。
    命中就吞掉取消、把消息 ack 掉、状态落 ``cancelled``；没命中照旧往外抛（= 交还队列）。
    """
    q = queue or get_task_queue()
    # 排队期间就被取消的：领到手先自查标记，一步都不用跑。这是「还在排队的任务也取消得掉」的落点
    # ——广播只能送到已经领走它的那个 worker，还没被领走的只能靠这张标记。
    if await control.consume_cancel_mark(task.task_id):
        logger.info("任务在排队期间已被取消，跳过：%s（thread=%s）", task.task_id, task.thread_id)
        await q.set_status(
            TaskStatus(task_id=task.task_id, state="cancelled", thread_id=task.thread_id)
        )
        return
    current = asyncio.current_task()
    if current is not None:
        control.register_inflight(task.task_id, task.thread_id, current)
    # 澄清的两件前置：本轮 turn_id 绑上下文（等待令牌要带它），并清掉上一轮崩溃留下的残留令牌
    # ——否则用户的回复会被转发给一个早已不存在的等待方，界面上表现为「答了没反应」。
    clarification.set_turn_id(task.task_id)
    await clarification.drop_stale_waiter(task.thread_id, turn_id=task.task_id)
    await q.set_status(TaskStatus(task_id=task.task_id, state="running", thread_id=task.thread_id))
    try:
        result = await run_agent(
            task.query,
            task.thread_id,
            user_id=task.user_id,
            # 空元组要还原成 None 而不是空列表：``platform_scope`` 把 None 解释为「用服务端默认」，
            # 把空列表解释为「一个平台都不启用」，两者差着一整轮空军。
            platforms=list(task.platforms) or None,
            image_paths=list(task.image_paths) or None,
        )
    except asyncio.CancelledError:
        if not control.was_cancelled_locally(task.task_id):
            raise  # 优雅退出：不 ack，留在 PEL 里等下一个 worker 领回重跑
        # 用户取消：run_agent 的 finally 已经上报 task_cancelled 并把这一轮的账记完（「取消即
        # 免单」的洞早堵住了，见 session_io.charge_quota）。这里只负责让消息被 ack 掉。
        if current is not None:
            current.uncancel()  # 取消已被消费，后面几个 await 才不会立刻再抛
        logger.info("任务被用户取消：%s（thread=%s）", task.task_id, task.thread_id)
        with suppress(Exception, asyncio.CancelledError):
            await q.set_status(
                TaskStatus(task_id=task.task_id, state="cancelled", thread_id=task.thread_id)
            )
            # 取消是这条消息的定论（它马上会被 ack），标记留着只是垃圾——取消常被连点好几次。
            await control.clear_cancel_mark(task.task_id)
        return
    except Exception as exc:
        logger.exception("任务失败：%s（thread=%s）", task.task_id, task.thread_id)
        await q.set_status(
            TaskStatus(
                task_id=task.task_id,
                state="failed",
                thread_id=task.thread_id,
                error=str(exc),
            )
        )
        raise
    finally:
        # 摘登记放 finally：任何收尾路径（正常 / 用户取消 / 优雅退出 / 异常）都不能把句柄留在表里
        # ——留着就是让下一次同 task_id 的取消去 cancel 一个早已结束的 task，静默无效。
        control.unregister_inflight(task.task_id)
    await q.set_status(
        TaskStatus(
            task_id=task.task_id,
            state="done",
            thread_id=task.thread_id,
            final_text=str(result.get("final_text") or ""),
        )
    )


async def _on_cancel(payload: dict[str, Any]) -> None:
    """控制面收到「取消」指令：掐掉本进程正在跑的那条任务（不在本进程就什么也不做）。

    **顺序与 API 单进程那条路逐字一致**：先 ``cancel_pending``（把 ``ask_user`` 从等待里放出来），
    再 cancel 任务本体。反过来的话，正挂在 Future 上的那个 await 会先吃到任务级取消，
    ``ask_user`` 里区分两种取消的那段判断（``cancelling() > 0``）就失去了它的前提。
    """
    thread_id = str(payload.get("thread_id") or "")
    task_id = str(payload.get("task_id") or "")
    if thread_id:
        clarification.cancel_pending(thread_id)
    if control.cancel_local(task_id=task_id or None, thread_id=thread_id or None):
        logger.info("按控制面指令取消任务：%s（thread=%s）", task_id, thread_id)


async def start_control_plane() -> control.ControlBus | None:
    """订阅控制面：取消 + 澄清回复两类指令。未启用（单进程部署）时返回 ``None``。

    两类指令共用一条订阅循环、一个 Redis 连接——它们的收件人都是「正在跑这一轮的那个进程」，
    分开订阅只会多一条要各自重连的长连接。
    """
    bus = control.get_control_bus()
    if bus is None:
        return None
    bus.on("cancel", _on_cancel)
    clarification.register_control_handlers()
    await bus.start()
    return bus


def install_signal_handlers(stop: asyncio.Event) -> None:
    """把 SIGTERM / SIGINT 接成「置一个 Event」，而不是直接掐进程。

    ``loop.add_signal_handler`` 而非 ``signal.signal``：后者在任意线程的任意指令间回调，改 asyncio
    对象不安全；前者由事件循环在自己的 tick 里回调。取不到（非 Unix / 非主线程）时退回 ``signal``
    并用 ``call_soon_threadsafe`` 递回循环——退化路径，本仓的部署形态走不到。
    """
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, stop.set)
        except (NotImplementedError, RuntimeError, ValueError):
            signal.signal(sig, lambda *_: loop.call_soon_threadsafe(stop.set))


async def run_worker(
    queue: TaskQueue | None = None,
    *,
    concurrency: int | None = None,
    grace_seconds: int | None = None,
    stop: asyncio.Event | None = None,
    install_signals: bool = True,
) -> None:
    """消费循环 + 优雅退出（三步见模块 docstring）。

    参数全可注入：测试据此不装信号处理器、直接推 ``stop`` Event。
    """
    q = queue or get_task_queue()
    stop_event = stop or asyncio.Event()
    grace = WORKER_GRACE_SECONDS if grace_seconds is None else grace_seconds
    if install_signals:
        install_signal_handlers(stop_event)

    limit = concurrency or WORKER_CONCURRENCY
    bus = await start_control_plane()
    logger.info("worker 启动：consumer=%s concurrency=%d", consumer_name(), limit)
    consume = asyncio.create_task(
        q.consume(consumer_name(), lambda t: handle_task(t, q), stop_event.is_set, limit)
    )
    stop_waiter = asyncio.create_task(stop_event.wait())
    try:
        await asyncio.wait({consume, stop_waiter}, return_when=asyncio.FIRST_COMPLETED)
        if consume.done():  # 消费循环自己退了（多半是抛了）——把异常带出去，别静默变成空跑
            await consume
            return
        logger.info("收到停止信号：不再领新任务，最多等 %ds 让在飞任务跑完", grace)
        _, pending = await asyncio.wait({consume}, timeout=grace)
        if pending:
            # 超时转回 pending：取消在飞任务 → 它们不 ack → 消息留在 PEL → 下一个 worker
            # XAUTOCLAIM 领回重跑。这是 at-least-once 的代价，也是「不丢任务」的兑现方式。
            logger.warning("在飞任务 %ds 内未跑完，取消并交还队列重投", grace)
            consume.cancel()
        with suppress(asyncio.CancelledError):
            await consume
    finally:
        stop_waiter.cancel()
        with suppress(asyncio.CancelledError):
            await stop_waiter
        if bus is not None:
            await bus.stop()
        await q.close()
        logger.info("worker 已退出")


async def bootstrap() -> None:
    """与 API 进程 ``lifespan`` 同一套启动前置：日志 / 建表 / 热更新参数 / 分词器预热。

    少任何一项都会在第一条任务上炸或跑出错值——后台管理页改过的参数存在库里，不 load 就是拿 ``.env``
    的旧值跑，而这种偏差不会报错（见记忆 ``structured-output-method-must-be-pinned``）。
    """
    configure_logging()
    await init_db()
    await config_store.load_into_memory()
    ok = await asyncio.to_thread(warm_tokenizer)
    logger.info("分词器预热%s", "成功" if ok else "降级为启发式")


async def amain() -> None:
    await bootstrap()
    await run_worker()


def main() -> None:
    """进程入口：``uv run python -m app.worker``。"""
    if not queue_enabled():
        # 队列关着时 get_task_queue() 给的是进程内实现——它的 deque 与 API 进程的 deque 是两个对象，
        # 这个 worker 会一条任务都收不到，还一声不吭地空转。宁可起不来。
        raise SystemExit(
            "QUEUE_ENABLED=0：worker 消费的进程内队列没有生产方，先把 QUEUE_ENABLED 打开"
        )
    asyncio.run(amain())


if __name__ == "__main__":
    main()
