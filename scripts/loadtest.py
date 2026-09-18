"""阶梯压测：`--stages 2,5,10` 三档并发，每档出成功率 / P50 / P95 / 吞吐。

**它测的不是模型有多快，是这套排队机制在压力下会不会漏人。** 单看平均延迟没有意义——一条购物
任务本来就要几十秒；真正要盯的是：并发升上去之后，(1) 有没有请求被吃掉（提交成功却永远收不到
终态），(2) 429 是不是**该出现的时候才出现**（队列深度到顶是设计行为，静默超时不是），
(3) P95 与 P50 的差随并发怎么长——差距突然拉开，说明有请求在某道闸上排了整整一轮。

客户端模拟的是**真前端那条路**：先连 WS、等 `ws_ready`，再 POST /api/task（connect-first，
不这么做会丢掉任务开头那几个事件），然后一直读事件直到 `task_result` / `task_cancelled` /
`error`。所以这里量到的延迟是「用户从按下回车到看见结果」，不是「HTTP 响应回来」——后者在队列
模式下几十毫秒就返回了，拿它当延迟指标是自欺欺人。

跑法：

    # 对着本机双进程栈（API + worker + Redis），run_agent 打桩时秒级出表
    uv run python scripts/loadtest.py --base-url http://127.0.0.1:8199 --stages 2,5,10

    # 真实 LLM（会烧 token，慎跑）：把并发调小、超时调大
    uv run python scripts/loadtest.py --stages 2,5 --timeout 300

**跑之前先确认整轮缓存是关的**（`TURN_CACHE_ENABLED=0`，批2-5）：开着的话第二个用户开始全是
缓存命中，压出来的数好看得离谱且毫无意义。脚本会查一次 `/api/health` 并在开着时拒跑。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TERMINAL_EVENTS = {"task_result", "task_cancelled", "error"}


@dataclass
class Attempt:
    """一次「提交 → 拿到终态」的完整尝试。``ok`` 为假时 ``reason`` 说明卡在哪一步。"""

    ok: bool
    seconds: float
    reason: str = ""
    events: int = 0
    first_event: float | None = None  # 从连 WS 起到第一条 monitor_event 的秒数（用户按下回车到首次看到反馈）


@dataclass
class StageReport:
    """一档并发的汇总。``wall`` 是这一档从第一个请求发出到最后一个收尾的墙上时间。"""

    concurrency: int
    attempts: list[Attempt] = field(default_factory=list)
    wall: float = 0.0

    @property
    def total(self) -> int:
        return len(self.attempts)

    @property
    def succeeded(self) -> int:
        return sum(1 for a in self.attempts if a.ok)

    @property
    def success_rate(self) -> float:
        return self.succeeded / self.total if self.total else 0.0

    @property
    def throughput(self) -> float:
        """每秒完成几个**成功**的请求。失败的不算——把超时的也算进吞吐是自欺欺人。"""
        return self.succeeded / self.wall if self.wall > 0 else 0.0

    def latency(self, q: float) -> float:
        """成功请求的分位延迟。失败的不进统计：它们没有「延迟」只有「没回来」，
        混进去会让 P95 随失败率一起漂，看着像变快了。"""
        return percentile([a.seconds for a in self.attempts if a.ok], q)

    def first_event(self, q: float) -> float:
        """成功请求「连 WS → 第一条事件」的分位数：用户按下回车后多久看到反馈。"""
        return percentile([a.first_event for a in self.attempts if a.ok and a.first_event is not None], q)

    def rejected(self, q: float) -> float:
        """被 429 拒绝的请求「连 WS → 拿到 429」的分位数：背压拒得有多快。
        成功延迟量的是扛不扛得住，这个量的是拒不拒得干脆——过载时用户等 5s 才收到 429
        和秒拒是两种体验。"""
        return percentile([a.seconds for a in self.attempts if a.reason == "429 背压"], q)

    def failures(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for a in self.attempts:
            if not a.ok:
                out[a.reason] = out.get(a.reason, 0) + 1
        return out


def percentile(values: list[float], q: float) -> float:
    """最近秩（nearest-rank）分位数：第 ceil(q × n) 个样本。

    **刻意不用插值**：压测样本量小（每档十几个），插值出来的 P95 是两个真实样本之间一个谁也没
    经历过的数。最近秩给的一定是某次真实请求的耗时，讲给人听时不用附带一句「这个数是算出来的」。
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, min(len(ordered), math.ceil(q * len(ordered))))
    return ordered[rank - 1]


async def one_attempt(
    c: httpx.AsyncClient, base_url: str, query: str, budget_sec: float, token: str | None
) -> Attempt:
    """一个虚拟用户：连 WS → 等 ws_ready → POST /api/task → 读事件到终态。

    **每个虚拟用户用自己的 thread_id**：共用一个会走进幂等第 2 层（同 thread 换 query = 覆盖
    重发），后发的会把先发的掐掉，压出来的失败全是自己造的。

    **HTTP client 由外面传进来、整场共用**：构造一个 ``httpx.AsyncClient`` 要加载 SSL 证书链，
    本机实测约 9ms 纯 CPU；500 个虚拟用户各建一个就是 4.6s 串行在压测进程里，量到的 429 时延
    和首事件全被它抬高（2026-09-18 实测 500 并发 5.6s → 修后见表）。压测客户端先要自证不是瓶颈。
    """
    thread_id = f"lt-{uuid.uuid4().hex[:12]}"
    ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://")
    ws_url = f"{ws_url}/ws/{thread_id}" + (f"?token={token}" if token else "")
    t0 = time.perf_counter()
    seen = 0
    try:
        async with websockets.connect(ws_url, open_timeout=budget_sec) as ws:
            await asyncio.wait_for(ws.recv(), timeout=budget_sec)  # ws_ready（connect-first）
            resp = await c.post(
                "/api/task", json={"query": query, "thread_id": thread_id}, timeout=budget_sec
            )
            if resp.status_code == 429:
                # 429 是**设计行为**（队列深度到顶 / 等待队列满），与超时不是一回事，
                # 所以单独记一类原因。混在一起会让人以为系统崩了，其实是背压正常起效。
                return Attempt(False, time.perf_counter() - t0, "429 背压", seen)
            if resp.status_code >= 400:
                return Attempt(
                    False, time.perf_counter() - t0, f"HTTP {resp.status_code}", seen
                )
            first: float | None = None
            deadline = time.perf_counter() + budget_sec
            while time.perf_counter() < deadline:
                raw = await asyncio.wait_for(
                    ws.recv(), timeout=max(1.0, deadline - time.perf_counter())
                )
                msg = json.loads(raw)
                if msg.get("type") != "monitor_event":
                    continue
                seen += 1
                if first is None:
                    first = time.perf_counter() - t0
                event = msg.get("event")
                if event in TERMINAL_EVENTS:
                    ok = event == "task_result"
                    reason = "" if ok else f"终态 {event}"
                    return Attempt(ok, time.perf_counter() - t0, reason, seen, first)
        return Attempt(False, time.perf_counter() - t0, "超时未收到终态", seen)
    except TimeoutError:
        return Attempt(False, time.perf_counter() - t0, "超时未收到终态", seen)
    except Exception as exc:  # noqa: BLE001 —— 压测客户端的任何异常都只是一次失败，不该中断整场
        return Attempt(False, time.perf_counter() - t0, type(exc).__name__, seen)


async def run_stage(
    base_url: str,
    concurrency: int,
    requests: int,
    query: str,
    budget_sec: float,
    token: str | None,
) -> StageReport:
    """一档并发：``concurrency`` 个虚拟用户同时开跑，总共发 ``requests`` 个请求。

    用信号量控在场人数而不是分批发：分批（发一批、等一批）量到的是「批次时长」，任何一个慢请求
    都会把整批的吞吐拖下去，而真实流量不长那样——真实世界里有人跑完就立刻有新人进来。
    """
    sem = asyncio.Semaphore(concurrency)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    # 连接数上限放开：httpx 默认 100，500 并发时后 400 个会在客户端排队，量出来的又是假延迟。
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=concurrency)

    async def _one(c: httpx.AsyncClient, i: int) -> Attempt:
        async with sem:
            return await one_attempt(c, base_url, f"{query}（#{i}）", budget_sec, token)

    t0 = time.perf_counter()
    async with httpx.AsyncClient(base_url=base_url, headers=headers, limits=limits) as c:
        attempts = list(await asyncio.gather(*(_one(c, i) for i in range(requests))))
    return StageReport(concurrency=concurrency, attempts=attempts, wall=time.perf_counter() - t0)


def render_table(reports: list[StageReport]) -> str:
    """出一张能直接贴进报告的 Markdown 表。

    列的挑选是有取舍的：**没有平均值**。平均延迟在长尾分布上骗人——一个 200s 的超时能把十几个
    30s 的请求平均成 45s，看着「还行」，而实际上有一个用户已经放弃了。P50 说明典型体验，P95
    说明最差的那批人的体验，两个一起看才知道压力落在谁身上。
    """
    lines = [
        "| 并发 | 请求数 | 成功率 | P50 | P95 | 首事件 P50 | 首事件 P95 | 429 P50 | 429 P95 | 吞吐 (req/s) | 失败原因 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in reports:
        fails = "、".join(f"{k}×{v}" for k, v in sorted(r.failures().items())) or "—"
        lines.append(
            f"| {r.concurrency} | {r.total} | {r.succeeded}/{r.total}"
            f"（{r.success_rate:.0%}） | {r.latency(0.5):.2f}s | {r.latency(0.95):.2f}s "
            f"| {r.first_event(0.5):.3f}s | {r.first_event(0.95):.3f}s "
            f"| {r.rejected(0.5):.3f}s | {r.rejected(0.95):.3f}s "
            f"| {r.throughput:.2f} | {fails} |"
        )
    return "\n".join(lines)


async def assert_turn_cache_off(base_url: str) -> None:
    """整轮缓存开着就拒跑（与 ``scripts/eval/run_rubric.py`` 同一条口径，理由见批2-5）。

    探测失败只 warn 不拦：后端没起 / 版本老没这个字段时，拦下来只会让人以为压测脚本坏了。
    """
    try:
        async with httpx.AsyncClient(base_url=base_url, timeout=5) as c:
            data = (await c.get("/api/health")).json()
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠️  查不到 /api/health（{type(exc).__name__}），跳过整轮缓存检查", flush=True)
        return
    if bool((data.get("turn_cache") or {}).get("enabled")):
        raise SystemExit(
            "后端的整轮缓存（TURN_CACHE_ENABLED）开着：压出来的延迟会是缓存命中的假数，先关掉再跑"
        )


async def amain(args: argparse.Namespace) -> int:
    stages = [int(s) for s in args.stages.split(",") if s.strip()]
    await assert_turn_cache_off(args.base_url)
    print(f"\n阶梯压测 → {args.base_url}｜档位 {stages}｜每档 {args.requests} 个请求\n", flush=True)

    reports: list[StageReport] = []
    for c in stages:
        print(f"  ▶ 并发 {c} …", flush=True)
        report = await run_stage(
            args.base_url, c, args.requests, args.query, args.timeout, args.token
        )
        reports.append(report)
        print(
            f"    成功 {report.succeeded}/{report.total}"
            f"｜P50 {report.latency(0.5):.2f}s｜P95 {report.latency(0.95):.2f}s"
            f"｜吞吐 {report.throughput:.2f} req/s",
            flush=True,
        )
        # 档间留白：让上一档的在飞任务彻底收尾，否则下一档量到的是两档叠加的排队。
        await asyncio.sleep(args.cooldown)

    print("\n" + render_table(reports) + "\n", flush=True)
    if args.out:
        Path(args.out).write_text(render_table(reports) + "\n", encoding="utf-8")
        print(f"  表已写入 {args.out}\n", flush=True)
    failed = sum(r.total - r.succeeded for r in reports)
    print(f"  {'✅ 0 失败' if failed == 0 else f'❌ 共 {failed} 个失败请求'}\n", flush=True)
    return 0 if failed == 0 else 1


def main() -> int:
    p = argparse.ArgumentParser(description="globex 阶梯压测（httpx 提交 + WS 收事件）")
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--stages", default="2,5,10", help="并发档位，逗号分隔")
    p.add_argument("--requests", type=int, default=12, help="每档发多少个请求")
    p.add_argument("--query", default="买一个通勤双肩包，预算 300")
    p.add_argument("--timeout", type=float, default=120.0, help="单个请求等终态的上限（秒）")
    p.add_argument("--cooldown", type=float, default=2.0, help="档与档之间的留白（秒）")
    p.add_argument("--token", default=None, help="AUTH_ENABLED=true 时的 JWT")
    p.add_argument("--out", default=None, help="把 Markdown 表另存到这个文件")
    return asyncio.run(amain(p.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
