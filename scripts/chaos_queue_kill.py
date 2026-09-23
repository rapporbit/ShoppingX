"""队列故障注入：kill -9 一个 worker，量「接管耗时 / 丢失数 / 误抢数」。

用法（需要一台**可清空**的真 Redis，默认 localhost:6380 的 db 15）::

    uv run python scripts/chaos_queue_kill.py --flush
    uv run python scripts/chaos_queue_kill.py --flush --heartbeat 0   # 对照组：关心跳

**测的是队列层，不是整条 Agent 链路。** worker 子进程跑的是真 :class:`RedisStreamQueue.consume`
（心跳、XAUTOCLAIM、终态去重都是生产代码），handler 换成桩：记一笔开始时刻 → sleep 随机时长 →
写 ``done`` 终态。这样一次实验几分钟、零 LLM 花费，量出来的就是调度层自己的数。

**四个数怎么算。**

- 接管耗时：被杀 worker 手上的每条任务，``kill 时刻 → 别的 worker 第一次开始跑它`` 的秒数。
- 丢失：入队了、实验结束时仍没有 ``done`` 终态的任务数（死信另计）。
- 误抢：**没被杀的** worker 正在跑的任务被别人重跑了（开始记录 > 1 且首个 worker 不是被杀的那个）。
  任务时长故意有一部分超过 claim 阈值，心跳失效的话这个数立刻不为 0。
- 重复完成：同一 task_id 写了两次 ``done``（终态去重失效的指纹）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

import redis.asyncio as aredis

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.queue.ports import IntentTask, TaskStatus  # noqa: E402
from app.queue.redis_stream import GROUP, STREAM_NORMAL, RedisStreamQueue  # noqa: E402

STARTS = "chaos:starts"
DONES = "chaos:dones"


# ── worker 子进程 ─────────────────────────────────────────────────────────────
async def run_worker(args: argparse.Namespace) -> None:
    client = aredis.from_url(args.redis_url)
    rq = RedisStreamQueue(client)

    async def _handler(task: IntentTask) -> None:
        await client.rpush(
            STARTS, json.dumps({"task": task.task_id, "worker": args.name, "t": time.time()})
        )
        await asyncio.sleep(float(task.query))  # query 里塞的是这条任务该跑多久
        await rq.set_status(
            TaskStatus(task_id=task.task_id, state="done", thread_id=task.thread_id)
        )
        await client.rpush(DONES, json.dumps({"task": task.task_id, "worker": args.name}))

    await rq.consume(
        args.name,
        _handler,
        lambda: False,
        args.concurrency,
        claim_idle_ms=args.claim_idle_ms,
        heartbeat_sec=args.heartbeat,
    )


# ── 编排进程 ─────────────────────────────────────────────────────────────────
def _spawn(args: argparse.Namespace, name: str) -> subprocess.Popen[bytes]:
    cmd = [sys.executable, __file__, "worker", "--name", name, "--redis-url", args.redis_url]
    cmd += ["--concurrency", str(args.concurrency), "--claim-idle-ms", str(args.claim_idle_ms)]
    cmd += ["--heartbeat", str(args.heartbeat)]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


async def _in_flight_of(client: aredis.Redis, consumer: str) -> list[str]:
    rows = await client.xpending_range(STREAM_NORMAL, GROUP, "-", "+", 1000, consumername=consumer)
    return [r["message_id"].decode() for r in rows]


async def orchestrate(args: argparse.Namespace) -> dict[str, object]:
    client = aredis.from_url(args.redis_url)
    if await client.dbsize() and not args.flush:
        raise SystemExit(f"{args.redis_url} 非空；确认可清空后加 --flush")
    await client.flushdb()
    rq = RedisStreamQueue(client)
    await rq.ensure_group()

    rng = random.Random(args.seed)
    durations = {f"t{i:03d}": rng.uniform(args.min_sec, args.max_sec) for i in range(args.tasks)}
    msg_to_task: dict[str, str] = {}
    for tid, sec in durations.items():
        await rq.enqueue(IntentTask.create(task_id=tid, thread_id=tid, query=f"{sec:.2f}"))
    for sid, fields in await client.xrange(STREAM_NORMAL):
        msg_to_task[sid.decode()] = json.loads(fields[b"payload"])["task_id"]

    names = [f"w{i}" for i in range(1, args.workers + 1)]
    procs = {n: _spawn(args, n) for n in names}
    await asyncio.sleep(args.kill_after)

    counts = {n: len(await _in_flight_of(client, n)) for n in names}
    victim = max(counts, key=lambda n: counts[n])
    orphans = [msg_to_task[m] for m in await _in_flight_of(client, victim)]
    os.kill(procs[victim].pid, signal.SIGKILL)
    kill_t = time.time()
    print(f"kill -9 {victim}（pid {procs[victim].pid}），手上 {len(orphans)} 条", flush=True)

    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        statuses = [await rq.get_status(t) for t in durations]
        if all(s is not None and s.state == "done" for s in statuses):
            break
        await asyncio.sleep(1)
    for n, p in procs.items():
        if n != victim:
            p.kill()

    starts: dict[str, list[dict[str, object]]] = {}
    for raw in await client.lrange(STARTS, 0, -1):
        row = json.loads(raw)
        starts.setdefault(row["task"], []).append(row)
    dones: dict[str, int] = {}
    for raw in await client.lrange(DONES, 0, -1):
        tid = json.loads(raw)["task"]
        dones[tid] = dones.get(tid, 0) + 1

    takeover = []
    for tid in orphans:
        later = [r["t"] for r in starts.get(tid, []) if r["worker"] != victim]
        if later:
            takeover.append(min(later) - kill_t)
    false_steals = [
        t for t, rows in starts.items() if len(rows) > 1 and rows[0]["worker"] != victim
    ]
    status_rows = [await rq.get_status(t) for t in durations]
    report = {
        "config": {k: v for k, v in vars(args).items() if k not in ("mode", "name")},
        "enqueued": len(durations),
        "done": sum(1 for s in status_rows if s is not None and s.state == "done"),
        "lost": [
            t for t, s in zip(durations, status_rows, strict=True) if s is None or s.state != "done"
        ],
        "dead_letters": await client.xlen("globex:intents:dead"),
        "longer_than_claim_idle": sum(
            1 for s in durations.values() if s * 1000 > args.claim_idle_ms
        ),
        "victim": victim,
        "orphans": len(orphans),
        "takeover_sec": {
            "n": len(takeover),
            "min": round(min(takeover), 2) if takeover else None,
            "median": round(statistics.median(takeover), 2) if takeover else None,
            "max": round(max(takeover), 2) if takeover else None,
        },
        "false_steals": false_steals,
        "double_done": [t for t, c in dones.items() if c > 1],
    }
    await client.aclose()
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", nargs="?", default="run", choices=["run", "worker"])
    ap.add_argument("--name", default="w1")
    ap.add_argument("--redis-url", default="redis://localhost:6380/15")
    ap.add_argument("--flush", action="store_true", help="确认该库可清空")
    ap.add_argument("--workers", type=int, default=3)
    # 默认让幸存者有空槽：它们只在「读不到新消息」时才去捡 PEL，槽位全满时量到的是排队时长，
    # 不是接管机制本身（这点另见报告里的说明）。
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--tasks", type=int, default=24)
    ap.add_argument("--min-sec", type=float, default=5.0)
    # 上限刻意超过 claim 阈值：有一批任务跑得比阈值还久，心跳不灵就会出现误抢。
    ap.add_argument("--max-sec", type=float, default=60.0)
    ap.add_argument("--kill-after", type=float, default=3.0)
    ap.add_argument("--heartbeat", type=float, default=5.0)
    ap.add_argument("--claim-idle-ms", type=int, default=30_000)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.mode == "worker":
        asyncio.run(run_worker(args))
        return
    report = asyncio.run(orchestrate(args))
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
