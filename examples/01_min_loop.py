"""M1 示例①：最小 AgentLoop —— 跑通 Think → Act → Observe → Reflect。

目的：用最轻的玩具工具，演示「循环次数由模型自判，而非外部写死」。
- 工具按主线规范写（async + Pydantic 输出 + 给模型看的 docstring），见 examples/tools_toy.py。
- 用 LangChain V1 的 create_agent 承载循环；模型自己决定先 planner、再 item_search、
  再 price_compare，信息够了就用自然语言收尾、不再调工具。

运行：uv run python examples/01_min_loop.py
"""

import asyncio
import sys
from pathlib import Path

# 直接以脚本方式运行时，sys.path[0] 是 examples/ 而非项目根，需手动把根加入以 import app。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain.agents import create_agent  # noqa: E402
from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402

from app.agent.llm import get_llm  # noqa: E402
from examples.tools_toy import SYSTEM_PROMPT, item_search, planner, price_compare  # noqa: E402


def print_trajectory(messages: list) -> int:
    """打印消息轨迹，返回「工具调用轮数」（= tool_calls 总数）。"""
    rounds = 0
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                rounds += 1
                print(f"  [AI · Think→Act] 调用 {tc['name']}({tc['args']})")
        elif isinstance(msg, ToolMessage):
            print(f"  [Tool · Observe] {msg.name} -> {str(msg.content)[:90]}")
        elif isinstance(msg, AIMessage) and msg.content:
            print(f"  [AI · Reflect 收尾] {str(msg.content)[:200]}")
    return rounds


async def main() -> None:
    # LangChain V1 的 Agent 入口（旧 langgraph.prebuilt.create_react_agent 已废弃）。
    agent = create_agent(
        model=get_llm(),
        tools=[planner, item_search, price_compare],
        system_prompt=SYSTEM_PROMPT,
    )

    query = "帮我搜旅行收纳袋，预算300、不要塑料，然后跨平台比个价给我推荐"
    print(f"[用户] {query}\n")

    result = await agent.ainvoke({"messages": [("user", query)]})
    rounds = print_trajectory(result["messages"])

    last = result["messages"][-1]
    terminated_cleanly = isinstance(last, AIMessage) and not last.tool_calls
    print(f"\n循环轮数（模型自判）：{rounds} 次工具调用")
    print(f"是否干净收尾（最后一条是无工具调用的 AI 回复）：{terminated_cleanly}")


if __name__ == "__main__":
    asyncio.run(main())
