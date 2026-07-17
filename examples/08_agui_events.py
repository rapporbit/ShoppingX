"""M8 示例：AGUI 事件协议 + WebSocket 实时推送（端到端真连接）。

这个脚本把 M8 的链路完整跑一遍，证明「后台任务每走一步，前端就实时收到一个事件」：

1. 起一个最小 FastAPI app，只挂一个 ``/ws/{thread_id}`` 端点——它用的就是
   :mod:`app.api.monitor` 共享的那个 ConnectionManager（生产服务在 M10 落地，端点逻辑一致）。
2. 用真正的 WebSocket 客户端（``websockets``，随 ``uvicorn[standard]`` 自带）连上去订阅。
3. 在同进程里模拟一次主 AgentLoop：在 ``thread_scope`` 里依次上报
   session_created → assistant_call → tool_start/tool_end → fork → … → task_result。
4. 客户端按序收到这些事件并打印。

这就是 ROADMAP M8 验收点「跑一条任务，事件按序经 WS 推出（脚本订阅验证）」的可执行版。

运行：uv run python examples/08_agui_events.py
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uvicorn  # noqa: E402
import websockets  # noqa: E402
from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # noqa: E402

from app.api import monitor  # noqa: E402
from app.utils.thread_ctx import thread_scope  # noqa: E402

HOST = "127.0.0.1"
PORT = 8077


def build_app() -> FastAPI:
    """最小 app：只有一个 WS 端点，复用 monitor 的全局 ConnectionManager。"""
    app = FastAPI()
    manager = monitor.get_connection_manager()

    @app.websocket("/ws/{thread_id}")
    async def ws_endpoint(websocket: WebSocket, thread_id: str) -> None:
        await manager.connect(websocket, thread_id)
        try:
            while True:  # 持续接收前端心跳，保持长连接；本例客户端不发，纯订阅。
                await websocket.receive_text()
        except WebSocketDisconnect:
            await manager.disconnect(websocket, thread_id)

    return app


async def fake_agent_task(thread_id: str) -> None:
    """模拟一次主 AgentLoop 的执行轨迹，每步上报一个 AGUI 事件。"""
    session_dir = Path("output") / thread_id
    with thread_scope(thread_id, session_dir):
        await monitor.report_session_created()
        await monitor.report_assistant_call(preview="拆解需求：跨平台搜旅行收纳袋")
        await monitor.report_tool_start("planner", intent="跨平台搜旅行收纳袋")
        await asyncio.sleep(0.05)
        await monitor.report_tool_end("planner", category="travel_organizer")
        # 主 loop 判定「能并行」→ fork 子 Agent（事件落在父 thread）。
        await monitor.report_fork("sub-ab12cd34-d1", "在 amazon 搜旅行收纳袋")
        await monitor.report_tool_start("item_search", query="旅行收纳袋", platform="amazon")
        await asyncio.sleep(0.05)
        await monitor.report_tool_end("item_search", platform="amazon", total_recall=20)
        await monitor.report_tool_start("price_compare", count=20, base="USD")
        await monitor.report_tool_end("price_compare", ranked=18, skipped=2)
        await monitor.report_task_result("为你精选 3 件旅行收纳袋……")


async def main() -> None:
    app = build_app()
    config = uvicorn.Config(app, host=HOST, port=PORT, log_level="warning")
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())
    try:
        while not server.started:  # 等服务真正起来  # noqa: ASYNC110
            await asyncio.sleep(0.05)

        thread_id = "demo-thread"
        manager = monitor.get_connection_manager()
        async with websockets.connect(f"ws://{HOST}:{PORT}/ws/{thread_id}") as ws:
            # 等 WS 端点把连接登记进 ConnectionManager，再开始上报。
            while not manager.is_connected(thread_id):  # noqa: ASYNC110
                await asyncio.sleep(0.02)

            print(f"== 已订阅 thread_id={thread_id}，开始跑模拟任务 ==\n")
            task = asyncio.create_task(fake_agent_task(thread_id))

            received: list[str] = []
            try:
                while True:
                    raw = await asyncio.wait_for(ws.recv(), timeout=3.0)
                    msg = json.loads(raw)
                    received.append(msg["event"])
                    data_brief = {k: v for k, v in msg["data"].items() if v is not None}
                    print(f"  [{msg['event']:<16}] {msg['message']}  {data_brief}")
                    if msg["event"] in {"task_result", "task_cancelled", "error"}:
                        break
            except TimeoutError:
                print("  (超时：未再收到事件)")
            await task

        print(f"\n== 共收到 {len(received)} 个事件，顺序：{' → '.join(received)} ==")
        assert received[0] == "session_created"
        assert received[-1] == "task_result"
        assert "fork" in received
        print("== 校验通过：首尾正确、含 fork 事件 ==")
    finally:
        server.should_exit = True
        await server_task


if __name__ == "__main__":
    asyncio.run(main())
