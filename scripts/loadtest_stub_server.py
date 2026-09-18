"""压测用打桩服务：把 ``server.run_agent`` 换成「session_created → 睡 N 秒 → task_result」。

只量排队 / WS / 事件推送这条链路的开销，不碰 LLM。跑法（另一个终端跑 loadtest.py）：

    AUTH_ENABLED=false TURN_CACHE_ENABLED=0 TASK_NORMAL_SLOTS=20 TASK_QUEUE_DEPTH=20 \
      uv run python scripts/loadtest_stub_server.py --port 8199 --sleep 3
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Any

import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8199)
    p.add_argument("--sleep", type=float, default=3.0, help="模拟一轮 Agent 的耗时（秒）")
    args = p.parse_args()

    from app.api import monitor, server
    from app.utils.thread_ctx import thread_scope
    from app.utils.path_utils import ensure_session_dir

    async def stub_run_agent(query: str, thread_id: str, user_id: str | None = None, **_: Any) -> dict:
        t0 = time.monotonic()
        session_dir = ensure_session_dir(thread_id)
        with thread_scope(thread_id, session_dir, user_id=user_id):
            await monitor.report_session_created(session_dir)
            await asyncio.sleep(args.sleep)
            await monitor.report_task_result(
                f"[stub] {query}", items=[], elapsed_ms=int((time.monotonic() - t0) * 1000)
            )
        return {"thread_id": thread_id, "final_text": "[stub]", "items": []}

    server.run_agent = stub_run_agent  # type: ignore[attr-defined]
    uvicorn.run(server.app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
