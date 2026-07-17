"""M1 示例②：流式事件 —— 用 astream_events 实时观察循环过程。

目的：invoke 要等整条循环跑完才返回；真实任务可能十几秒，前端会以为卡死。
astream_events 在执行过程中逐步吐事件，让前端能实时显示「正在思考 / 正在调用 X」。
这些原生事件（on_chat_model_* / on_tool_*）就是 M8 AGUI 事件协议的数据源。

复用示例①的玩具工具，只把执行方式换成流式。
运行：uv run python examples/02_stream.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain.agents import create_agent  # noqa: E402

from app.agent.llm import get_llm  # noqa: E402
from examples.tools_toy import SYSTEM_PROMPT, item_search, planner, price_compare  # noqa: E402

# 关心的原生事件 → 对应 AgentLoop 阶段 / 未来 AGUI 事件的映射（教学用）。
EVENT_LABELS = {
    "on_chat_model_start": "Think  · 模型开始推理",
    "on_tool_start": "Act    · 工具开始执行",
    "on_tool_end": "Observe· 工具返回结果",
    "on_chat_model_end": "Reflect· 本轮模型推理完成",
}


async def main() -> None:
    agent = create_agent(
        model=get_llm(),
        tools=[planner, item_search, price_compare],
        system_prompt=SYSTEM_PROMPT,
    )

    query = "帮我搜旅行收纳袋，预算300、不要塑料，然后跨平台比个价给我推荐"
    print(f"[用户] {query}\n--- 事件流 ---")

    async for event in agent.astream_events({"messages": [("user", query)]}):
        label = EVENT_LABELS.get(event["event"])
        if label is None:
            continue  # 只展示关心的几类，过滤掉 chain/stream 等噪声事件
        name = event.get("name", "")
        detail = ""
        if event["event"] == "on_tool_start":
            detail = str(event["data"].get("input", ""))[:70]
        elif event["event"] == "on_tool_end":
            detail = str(event["data"].get("output", ""))[:70]
        print(f"  [{label}] {name} {detail}".rstrip())

    print("\n--- 事件流结束 ---")
    print("提示：这些原生事件就是 M8 AGUI 协议的数据源")
    print("  on_tool_start/end → tool_start/tool_end；on_chat_model_start → assistant_call")


if __name__ == "__main__":
    asyncio.run(main())
