"""请求分档：按 thread 的历史轮数把一次提交判成 normal / heavy。

**这个模块曾经是什么（阶段 1 条 7 之前）。** 一套进程内双池准入（normal 5 槽 + heavy 3 槽、有界
等待队列、动态再平衡、覆盖重发的 force_reserve）。它守的是「**本进程**同时跑几个 AgentLoop」。

**为什么整套删掉。** 形态收敛到多副本之后，AgentLoop 不在 API 进程里跑了：API 只负责入队，真正
跑 loop 的是 N 个 worker。此时进程内池守着的那个数既不是全局并发（每台副本各有一份自己的池），
也不再对应任何真实资源。并发上限改由三道**跨进程**的闸承担，各守各真正约束得住的那件事：

- 用户级并发 = `run_holds` 里 queued/running 的行数（`MAX_CONCURRENT_RUNS`，阶段 1 条 2，
  真相在 DB）；
- 全局背压 = 队列深度 `QUEUE_MAX_DEPTH`（超了 429 + Retry-After，见 `server._queue_depth_or_429`）；
- 实际并行度 = 每个 worker 的 `WORKER_CONCURRENCY` × 副本数。

留下来的只有「分档」这一件事：档位既决定 `run_holds` 预扣多少 credits（见 `app.db.holds`），也
决定任务进 Redis Stream 的哪条流（见 `app.queue.ports`）。两处必须读同一份分类函数，各读各的会
漂成「准入判 heavy、队列判 normal」。
"""

from __future__ import annotations

from typing import Literal

from app.utils.env import env_int

RequestClass = Literal["normal", "heavy"]

# 分类阈值：该 thread 已积累的历史轮数 ≥ 它即判为 heavy。续聊越长，上下文越大、跑得越久。
HEAVY_TURNS_THRESHOLD = env_int("TASK_HEAVY_TURNS", 8)
# 被拒时回给前端的建议重试间隔（秒）。
TASK_RETRY_AFTER_SEC = env_int("TASK_RETRY_AFTER_SEC", 10)
# 排队位置反馈里的「预估等待」= 位置 × 它。一个粗略常数即可——用户要的是「还要等很久吗」的量级，
# 不是秒级精度。真做加权移动平均反而会因为长尾任务把估计值拖得离谱。
AVG_TASK_SECONDS = env_int("TASK_AVG_SECONDS", 30)


def classify_request(history_turns: int) -> RequestClass:
    """按该 thread 已积累的历史轮数分类。

    轮数是最直接的重量代理：续聊越长，回喂的上下文越大、模型每轮解码越慢。比「按 query 长度」或
    「按用户等级」都更贴近真实成本，而且零成本可得（turns.json 已在磁盘上）。
    """
    return "heavy" if history_turns >= HEAVY_TURNS_THRESHOLD else "normal"


def estimated_wait_seconds(position: int, capacity: int) -> int:
    """排在第 ``position`` 位的粗略预估等待秒数（给用户看量级，不追求精度）。

    除以 ``capacity``：有 5 个并行消费位时，排第 3 位并不需要等 3 个任务跑完——第一批退出就轮到了。
    不做加权移动平均：Agent 任务的耗时长尾很重，均值会被拖得离谱，还不如一个诚实的粗略常数。
    """
    if position <= 0:
        return 0
    slots = max(1, capacity)
    rounds = -(-position // slots)  # 向上取整：等前面 ceil(position/slots) 批跑完
    return rounds * AVG_TASK_SECONDS
