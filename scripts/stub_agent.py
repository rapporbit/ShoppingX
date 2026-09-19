"""打桩入口：把 ``run_agent`` 换成「session_created → 睡 N 秒 → task_result」，API 与 worker 共用。

多副本验收（阶段 1 的四个场景）要验的是**队列 / DB / Redis / WS 这条链路在两个副本下的正确性**，
不是 Agent 答得好不好。真跑 LLM 只会让每次验收慢几分钟、花钱，还引入「这次失败是模型抽风还是机制
错了」的噪声。打桩把那一段换成一个可控的 sleep，其余代码（准入、预扣、队列、事件、收尾）一律原样。

**打桩不进主链路代码**：没有 ``AGENT_STUB_SLEEP`` 这类 env 开关——生产误开就是全站假回答，而这种错
不会报任何异常。桩只活在这个脚本里，容器把 command 指过来即可，镜像与代码仍是同一份。

跑法::

    python -m scripts.stub_agent api      --sleep 3     # 起 FastAPI（打了桩）
    python -m scripts.stub_agent worker   --sleep 3     # 起 worker（打了桩）

``--sleep`` 也可用 ``STUB_SLEEP_SEC`` 给（compose 里用后者）。想验「关停掐断」就把它设得比
``WORKER_GRACE_SECONDS`` 长，任务自然会被掐在半路。
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _sleep_from_query(query: str, default: float) -> float:
    """从 query 里解析 ``sleep=<秒>``；没写或写坏了都用默认值（验收脚本传错不该让任务炸）。"""
    match = re.search(r"sleep=([0-9.]+)", query)
    if not match:
        return default
    try:
        return float(match.group(1))
    except ValueError:
        return default


def make_stub(sleep_s_default: float) -> Any:
    """造一个与 ``run_agent`` 同形的替身：同样进 ``thread_scope``、同样上报那两条事件。

    形状必须一致，否则验的就不是真链路了——``thread_scope`` 决定产物落在哪个 session_dir（1-4 的
    共享卷验收正靠它），两条事件决定前端与影子协程何时收尾。
    """
    from app.agent.session_io import release_hold
    from app.api import monitor
    from app.utils.path_utils import ensure_session_dir
    from app.utils.thread_ctx import thread_scope

    async def stub_run_agent(
        query: str, thread_id: str, user_id: str | None = None, run_id: str = "", **_: Any
    ) -> dict[str, Any]:
        t0 = time.monotonic()
        # query 里写 "sleep=40" 就睡 40 秒，覆盖默认值。场景脚本靠它精确控制「任务还在跑」这个
        # 窗口——杀 worker、掐关停都得在任务跑着的时候动手，固定 3s 抓不住。
        sleep_s = _sleep_from_query(query, default=sleep_s_default)
        session_dir = ensure_session_dir(thread_id)
        try:
            with thread_scope(thread_id, session_dir, user_id=user_id):
                await monitor.report_session_created(session_dir)
                # 产物落一个文件：多副本下这是「worker 写的东西 API 读不读得到」的唯一证据（1-4）。
                (session_dir / "summary.md").write_text(f"[stub] {query}\n", encoding="utf-8")
                await asyncio.sleep(sleep_s)
                await monitor.report_task_result(
                    f"[stub] {query}", items=[], elapsed_ms=int((time.monotonic() - t0) * 1000)
                )
        finally:
            # **桩必须自己还预扣**：真 run_agent 在 finally 里结算（有花销走 charge_quota、一分钱
            # 没花走 release_hold），而桩把那一整段换掉了。不补这一手，被掐的那一轮会留下一条
            # state=running 的 hold 占着额度到 TTL——那是桩的假象，会被误读成 1-3 漏了结算。
            # 用的就是 orchestrator 里那条分支的同一个函数，不是另写一份替身逻辑。
            if run_id:
                await release_hold(run_id)
        return {"thread_id": thread_id, "final_text": f"[stub] {query}", "items": []}

    return stub_run_agent


def run_api(sleep_s: float, host: str, port: int) -> None:
    import uvicorn

    from app.api import server

    server.run_agent = make_stub(sleep_s)  # type: ignore[attr-defined]
    uvicorn.run(server.app, host=host, port=port, log_level="warning")


def run_worker(sleep_s: float) -> None:
    from app import worker

    worker.run_agent = make_stub(sleep_s)  # type: ignore[attr-defined]

    async def _main() -> None:
        await worker.bootstrap()
        await worker.run_worker()

    asyncio.run(_main())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["api", "worker"])
    p.add_argument("--sleep", type=float, default=float(os.environ.get("STUB_SLEEP_SEC", "3")))
    p.add_argument("--host", default="0.0.0.0")  # noqa: S104 - 容器内监听；compose 不发布端口
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    if args.mode == "api":
        run_api(args.sleep, args.host, args.port)
    else:
        run_worker(args.sleep)


if __name__ == "__main__":
    main()
