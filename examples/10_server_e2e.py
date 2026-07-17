"""M10 示例：FastAPI 前后端闭环的协议级端到端演示（真 uvicorn，零真实 LLM）。

这是 ROADMAP M10 验收点的可执行版——但**把真实主 loop 换成一个桩**（``_stub_run_agent``），
因为本 worktree 的数据/模型未必够跑一次真链路（见对话决策）。重点不是模型多聪明，而是
**前后端协议这条路通不通**：connect-first 不丢早期事件、事件按序到达、产物可下载、任务可取消。

跑一遍证明：

1. 起真正的 uvicorn（挂 ``app.api.server:app``）。
2. **connect-first**：客户端先本地生成 thread_id → 连 WS → 收到 ``ws_ready`` → 才 POST 起任务。
   这样任务上报的第一个事件（session_created）必落在已登记的连接上，不丢。
3. 桩任务依次上报 session_created → tool_start/end → fork → … → task_result（带商品卡 items），
   并把清单写进 ``output/<tid>/summary.md``。
4. 客户端按序收全事件 → ``GET /api/files/<tid>/summary.md`` 下载清单 → 校验。
5. 再起一个长任务演示 ``POST /api/task/<tid>/cancel`` → 收到 task_cancelled。

把真实 ``run_agent`` 接回来（拷好 .env + data/index）后，这个脚本一字不改就是真链路 demo。

运行：uv run python examples/10_server_e2e.py
"""

import asyncio
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uvicorn  # noqa: E402
import websockets  # noqa: E402

import app.api.server as server  # noqa: E402
from app.api import monitor  # noqa: E402
from app.utils.path_utils import ensure_session_dir  # noqa: E402
from app.utils.thread_ctx import thread_scope  # noqa: E402

HOST = "127.0.0.1"
PORT = 8078
BASE = f"http://{HOST}:{PORT}"


async def _stub_run_agent(query: str, thread_id: str, user_id: str | None = None) -> dict:
    """假主 loop：走一遍 run_agent 的「上报 + 落产物」职责，但不调真实模型。

    形状与真实 ``run_agent`` 一致（同签名、同事件序、同产物文件），所以接真模型时这个桩
    被替换掉即可，协议层不变。
    """
    session_dir = ensure_session_dir(thread_id)
    with thread_scope(thread_id, session_dir):
        await monitor.report_session_created(session_dir)
        await monitor.report_tool_start("planner", intent=query)
        await asyncio.sleep(0.05)
        await monitor.report_tool_end("planner", category="travel_set")
        # 「能并行」→ fork 跨平台子 Agent（事件落父 thread）。
        await monitor.report_fork("sub-amazon-d1", "在 amazon 搜旅行三件套")
        await monitor.report_fork("sub-shopee-d1", "在 shopee 搜旅行三件套")
        await monitor.report_tool_start("item_search", platform="amazon")
        await asyncio.sleep(0.05)
        await monitor.report_tool_end("item_search", platform="amazon", total_recall=20)
        await monitor.report_tool_start("shopping_summary", count=2)
        await monitor.report_tool_end("shopping_summary", items=2)

        items = [
            {
                "item_id": "A1",
                "platform": "amazon",
                "title": "MINIMAL VOYAGER 三件套",
                "landed_usd": 28.9,
                "reason": "防水尼龙非塑料；12 天到手",
            },
            {
                "item_id": "S2",
                "platform": "shopee",
                "title": "OUTBACK WAYFARER 三件套",
                "landed_usd": 27.3,
                "reason": "牛津布+帆布，零塑料件；9 天直邮免税",
            },
        ]
        summary_md = (
            "## 推荐 2 件\n"
            "1. MINIMAL VOYAGER（amazon）— 到手价 $28.9\n"
            "2. OUTBACK WAYFARER（shopee）— 到手价 $27.3\n"
        )
        (session_dir / "summary.md").write_text(summary_md, encoding="utf-8")
        await monitor.report_task_result(summary_md, items=items)
    return {"thread_id": thread_id, "final_text": summary_md, "items": items}


async def _slow_run_agent(query: str, thread_id: str, user_id: str | None = None) -> dict:
    """长任务桩：用于演示取消（卡在 await 点等被打断）。"""
    session_dir = ensure_session_dir(thread_id)
    with thread_scope(thread_id, session_dir):
        await monitor.report_session_created(session_dir)
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            await monitor.report_task_cancelled()
            raise
    return {"thread_id": thread_id}


async def _wait_ready(ws: object) -> None:
    """connect-first 的关键：收到 ws_ready 控制帧再回主流程（此时连接已登记）。"""
    while True:
        raw = await asyncio.wait_for(ws.recv(), timeout=3.0)  # type: ignore[attr-defined]
        if json.loads(raw).get("type") == "ws_ready":
            return


async def _run_happy_path(http: object) -> None:
    thread_id = uuid.uuid4().hex
    async with websockets.connect(f"ws://{HOST}:{PORT}/ws/{thread_id}") as ws:
        await _wait_ready(ws)  # 连接登记完成，再发任务——早期事件不丢
        resp = await http.post(  # type: ignore[attr-defined]
            f"{BASE}/api/task",
            json={"query": "便宜抗造旅行三件套，预算 300，不要塑料", "thread_id": thread_id},
        )
        assert resp.json()["thread_id"] == thread_id
        print(f"== [happy] 已 connect-first 起任务 thread_id={thread_id} ==\n")

        received: list[str] = []
        items_seen: list[dict] = []
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
            msg = json.loads(raw)
            if msg.get("type") == "ws_ready":
                continue
            received.append(msg["event"])
            brief = {k: v for k, v in msg["data"].items() if v is not None and k != "items"}
            print(f"  [{msg['event']:<16}] {msg['message']}  {brief}")
            if msg["event"] == "task_result":
                items_seen = msg["data"].get("items", [])
            if msg["event"] in {"task_result", "task_cancelled", "error"}:
                break

    print(f"\n== 共收到 {len(received)} 个事件：{' → '.join(received)} ==")
    assert received[0] == "session_created"
    assert received[-1] == "task_result"
    assert "fork" in received
    assert len(items_seen) == 2, "task_result 应带商品卡 items"
    print(f"== 商品卡 {len(items_seen)} 件随 task_result 下发 ✓ ==")

    # 下载产物清单。
    dl = await http.get(f"{BASE}/api/files/{thread_id}/summary.md")  # type: ignore[attr-defined]
    assert dl.status_code == 200 and "推荐" in dl.text
    print(f"== 下载 summary.md（{len(dl.text)} 字）✓ ==\n")


async def _run_cancel_path(http: object) -> None:
    server.run_agent = _slow_run_agent  # type: ignore[assignment]
    thread_id = uuid.uuid4().hex
    async with websockets.connect(f"ws://{HOST}:{PORT}/ws/{thread_id}") as ws:
        await _wait_ready(ws)
        await http.post(  # type: ignore[attr-defined]
            f"{BASE}/api/task", json={"query": "慢任务", "thread_id": thread_id}
        )
        # 等任务真正起来（收到 session_created）再取消。
        while True:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=3.0))
            if msg.get("event") == "session_created":
                break
        cancel = await http.post(f"{BASE}/api/task/{thread_id}/cancel")  # type: ignore[attr-defined]
        assert cancel.status_code == 200
        print(f"== [cancel] 已请求取消 thread_id={thread_id} ==")
        # 应收到 task_cancelled。
        seen_cancel = False
        try:
            while True:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=3.0))
                if msg.get("event") == "task_cancelled":
                    seen_cancel = True
                    break
        except TimeoutError:
            pass
    assert seen_cancel, "取消后应收到 task_cancelled 事件"
    print("== 收到 task_cancelled ✓ ==\n")


async def main() -> None:
    import httpx

    # 用桩替换真实主 loop（本 worktree 数据/模型未必够跑真链路；协议层不依赖模型）。
    server.run_agent = _stub_run_agent  # type: ignore[assignment]

    config = uvicorn.Config(server.app, host=HOST, port=PORT, log_level="warning")
    uv_server = uvicorn.Server(config)
    server_task = asyncio.create_task(uv_server.serve())
    try:
        while not uv_server.started:  # noqa: ASYNC110
            await asyncio.sleep(0.05)
        async with httpx.AsyncClient(timeout=10.0) as http:
            await _run_happy_path(http)
            await _run_cancel_path(http)
        print("== M10 协议级闭环全部通过：connect-first 不丢事件 / 商品卡 / 下载 / 取消 ==")
    finally:
        uv_server.should_exit = True
        await server_task


if __name__ == "__main__":
    asyncio.run(main())
