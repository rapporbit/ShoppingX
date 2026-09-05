"""S3 spike：写工具的确认挂起 → AgentState 落盘 → 换实例恢复 → 只执行一次。

手册 §6-L0 的 S3：一个 is_read_only=False 的工具在 DEFAULT 权限下触发
RequireUserConfirmEvent；把 AgentState 序列化落盘（模拟跨进程/重启），新建 Agent
传 state 后 reply(UserConfirmResultEvent)，断言工具执行且只执行一次。

这条决定批 1 的 TradeAgent 写工具确认走原生通路，还是保留本仓自建的 Future 桥接。

跑法：uv run python scripts/spikes/agentscope_s3_confirm_resume.py
"""

import asyncio
import json
import os
import tempfile
from pathlib import Path

from agentscope.agent import Agent
from agentscope.credential import OpenAICredential
from agentscope.event import ConfirmResult, RequireUserConfirmEvent, UserConfirmResultEvent
from agentscope.message import Msg, TextBlock
from agentscope.model import OpenAIChatModel
from agentscope.state import AgentState
from agentscope.tool import FunctionTool, Toolkit
from agentscope.tool._response import ToolChunk, ToolResultState
from dotenv import load_dotenv

load_dotenv()

EXEC_LOG: list[str] = []
SYS = "你是下单助手。用户已经确认购买，请直接调用 create_order_spike 工具下单，不要反问。"


def _msg(role: str, text: str) -> Msg:
    return Msg(name=role, role=role, content=[TextBlock(type="text", text=text)])


async def create_order_spike(item_id: str) -> ToolChunk:
    """下单（spike 用 mock）。何时调用：用户确认购买某件商品后。"""
    EXEC_LOG.append(item_id)
    return ToolChunk(
        content=[TextBlock(type="text", text=json.dumps({"order_id": "od_1", "item": item_id}))],
        state=ToolResultState.SUCCESS,
    )


def _model() -> OpenAIChatModel:
    return OpenAIChatModel(
        credential=OpenAICredential(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ["OPENAI_BASE_URL"],
        ),
        model=os.environ.get("LLM_MAIN", "deepseek-v4-flash"),
        stream=True,
        extra_body={"enable_thinking": False},
    )


async def _build_agent(state: AgentState | None = None) -> Agent:
    toolkit = Toolkit()
    await toolkit.add_tool(
        FunctionTool(create_order_spike, is_read_only=False),  # 写工具 → DEFAULT 下要确认
    )
    return Agent(
        name="trade_spike",
        system_prompt=SYS,
        model=_model(),
        toolkit=toolkit,
        state=state,
    )


async def main() -> None:
    result: dict = {}

    # ① 首次 reply：应挂起在 RequireUserConfirmEvent
    agent = await _build_agent()
    pending: RequireUserConfirmEvent | None = None
    events: list[str] = []
    async for ev in agent.reply_stream(_msg("user", "帮我下单 item_id=SKU-42")):
        events.append(type(ev).__name__)
        if isinstance(ev, RequireUserConfirmEvent):
            pending = ev
    result["phase1"] = {
        "paused": pending is not None,
        "exec_count": len(EXEC_LOG),
        "pending_tools": [tc.name for tc in pending.tool_calls] if pending else [],
        "event_types": sorted(set(events)),
    }
    if pending is None:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        raise SystemExit("S3 未挂起：DEFAULT 权限没有拦住写工具")

    # ② AgentState 落盘 + 新进程口径的反序列化（同进程模拟）
    path = Path(tempfile.gettempdir()) / "s3_agent_state.json"
    path.write_text(agent.state.model_dump_json())
    restored = AgentState.model_validate_json(path.read_text())
    result["state_roundtrip"] = {"bytes": path.stat().st_size, "session_id": restored.session_id}

    # ③ 换一个 Agent 实例，喂确认结果
    resumed = await _build_agent(state=restored)
    reply = await resumed.reply(
        UserConfirmResultEvent(
            reply_id=pending.reply_id,
            confirm_results=[
                ConfirmResult(confirmed=True, tool_call=tc) for tc in pending.tool_calls
            ],
        ),
    )
    text = "".join(b.text for b in reply.content if b.type == "text")
    result["phase2"] = {
        "exec_count": len(EXEC_LOG),
        "exec_log": EXEC_LOG,
        "executed_once": len(EXEC_LOG) == 1,
        "reply_tail": text[-120:],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
