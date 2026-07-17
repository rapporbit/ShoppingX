"""M2 示例③：fork 三件事判断 + 安全四层自检。

上半场（真实 LLM）：主 Agent 面对「跨多个平台搜同一商品」——满足三件事里的「能并行」，
于是调 parallel_dispatch_tool 一次 fork 出多个同质子 Agent 并行检索，再合流。演示
fork 对主 loop 透明（就是「调了个工具拿到结果」）。

下半场（确定性，不调 LLM）：直接调安全四层的原语，打印它们确实拦得住——深度超限、
结果截断、循环检测。（完整断言见 tests/test_fork_safety.py）

注入玩具工具的方式：make_dispatch_tools(provider) 的 provider 返回玩具全集，于是子 Agent
拿到的是玩具工具——这就是「同质 fork」工具集 late-binding 的用法。

运行：uv run python examples/03_fork.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain.agents import create_agent  # noqa: E402
from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from app.agent.dispatch_tool import make_dispatch_tools  # noqa: E402
from app.agent.fork_guard import enter_fork  # noqa: E402
from app.agent.llm import get_llm  # noqa: E402
from app.harness.loop_detector import LoopDetector  # noqa: E402
from app.harness.truncation import truncate_tool_result  # noqa: E402

# --------------------------------------------------------------------------
# 玩具工具：带平台维度的检索，让「跨平台并行」有意义。
# --------------------------------------------------------------------------


class PlatformCandidate(BaseModel):
    name: str
    price_cny: float
    material: str


class PlatformSearchOutput(BaseModel):
    platform: str
    query: str
    candidates: list[PlatformCandidate]


@tool
async def platform_search(platform: str, query: str) -> PlatformSearchOutput:
    """在【单个】指定平台检索商品。跨多个平台时应并行 fork、每个子任务搜一个平台。"""
    await asyncio.sleep(0)
    fake = {
        "amazon": PlatformCandidate(name="硅胶分装套装", price_cny=89, material="硅胶"),
        "shopee": PlatformCandidate(name="帆布旅行收纳袋", price_cny=65, material="帆布"),
        "aliexpress": PlatformCandidate(name="尼龙折叠收纳包", price_cny=72, material="尼龙"),
    }
    cand = fake.get(platform, PlatformCandidate(name="通用收纳袋", price_cny=80, material="混纺"))
    return PlatformSearchOutput(platform=platform, query=query, candidates=[cand])


SYSTEM_PROMPT = """<role>你是 ShoppingX 购物助手（教学版），负责跨平台搜商品并对比。</role>
<tool_policy>
跨多个平台搜同一商品时，满足 fork 三件事的「能并行」——必须用 parallel_dispatch_tool
一次性把每个平台作为一个独立子任务并行 fork（每个 demands 形如「在 amazon 搜 X」），
不要自己串行一个个搜。被派到单个平台的子任务，直接用 platform_search 搜该平台即可。
</tool_policy>
<termination>结果都拿到后，直接用自然语言给出对比与推荐并结束，不要再调工具。</termination>"""

# late-binding：provider 返回玩具全集（含 dispatch 元工具自身）→ 子 Agent 同质且可递归。
# 注入玩具 SYSTEM_PROMPT，让子 Agent 与本示例的主 Agent 真正同质（否则子用全局 prompt 会跑偏）。
TOY_FULL_SET: list = []
toy_dispatch, toy_parallel_dispatch = make_dispatch_tools(
    lambda: TOY_FULL_SET, system_prompt=SYSTEM_PROMPT
)
TOY_FULL_SET.extend([platform_search, toy_dispatch, toy_parallel_dispatch])


def print_trajectory(messages: list) -> None:
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                fork_mark = " ⑂fork" if "dispatch" in tc["name"] else ""
                print(f"  [AI · Act{fork_mark}] {tc['name']}({tc['args']})")
        elif isinstance(msg, ToolMessage):
            print(f"  [Tool · Observe] {msg.name} -> {str(msg.content)[:100]}")
        elif isinstance(msg, AIMessage) and msg.content:
            print(f"  [AI · 收尾] {str(msg.content)[:220]}")


async def demo_three_things() -> None:
    print("=== 上半场：fork 三件事之「能并行」（真实 LLM）===")
    agent = create_agent(model=get_llm(), tools=TOY_FULL_SET, system_prompt=SYSTEM_PROMPT)
    query = "在 amazon、shopee、aliexpress 三个平台搜旅行收纳袋，对比后给我推荐"
    print(f"[用户] {query}\n")
    result = await agent.ainvoke({"messages": [("user", query)]}, config={"recursion_limit": 25})
    print_trajectory(result["messages"])


async def demo_safety_layers() -> None:
    print("\n=== 下半场：安全四层自检（确定性，不调 LLM）===")

    # ① 深度上限：处于最大深度时再 fork → 返回「拒绝」字符串，不崩。
    with enter_fork(), enter_fork():
        rejected = await toy_dispatch.ainvoke({"demands": "无限递归地再 fork 一层"})
    print(f"  ① 深度上限：{rejected}")

    # ③ 结果截断：超长结果尾部截断并留提示。
    long_text = "商品详情" * 5000
    truncated = truncate_tool_result(long_text)
    print(f"  ③ 结果截断：原 {len(long_text)} 字 → {len(truncated)} 字，尾部=…{truncated[-22:]}")

    # ④ 循环检测：同一工具刷屏到阈值即触发。
    det = LoopDetector(window=6, threshold=4)
    fired_at = next(i for i in range(1, 9) if det.record("platform_search"))
    print(f"  ④ 循环检测：第 {fired_at} 次重复调用 platform_search 触发提示")
    print("  （② 超时+迭代上限见 dispatch_tool 的 wait_for / recursion_limit）")


async def main() -> None:
    await demo_three_things()
    await demo_safety_layers()


if __name__ == "__main__":
    asyncio.run(main())
