"""压测用打桩服务：起一个打了桩的 API（桩本身在 :mod:`scripts.stub_agent`，两处共用一份）。

只量排队 / WS / 事件推送这条链路的开销，不碰 LLM。跑法（另一个终端跑 loadtest.k6.js）：

    AUTH_ENABLED=false TURN_CACHE_ENABLED=0 TASK_NORMAL_SLOTS=20 TASK_QUEUE_DEPTH=20 \
      uv run python scripts/loadtest_stub_server.py --port 8199 --sleep 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8199)
    p.add_argument("--sleep", type=float, default=3.0, help="模拟一轮 Agent 的耗时（秒）")
    args = p.parse_args()

    from scripts.stub_agent import run_api

    run_api(args.sleep, "127.0.0.1", args.port)


if __name__ == "__main__":
    main()
