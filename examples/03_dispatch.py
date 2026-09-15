"""M2 示例③：Supervisor-Workers 读写切分 + 派发安全四层自检。

**这个示例被重写过一次，原因值得记一笔**：早期版本演示的是「同质 fork」——子 Agent 是主 Agent
的完整克隆（同一份工具集、同一份 prompt），工具集靠 ``make_dispatch_tools(provider)`` 做
late-binding 注入。那套架构后来被推翻了：小事也要派一趟、子 Agent 拿着写工具却只是去搜个东西。
现在是 **Supervisor-Workers 按读写属性切分**，worker 的能力边界不再靠注入决定，而是结构性写死。

于是本示例也从「演示 fork 怎么注入工具」变成「**证明边界拦得住**」——全程确定性、不调 LLM、
不依赖索引，跑一遍就能看到边界确实在机制层。想看真实派发跑起来的样子，去 examples/09_main_agent.py。

演示四件事：
1. **读写边界是结构性的**：三个角色的 Toolkit 里到底有哪些工具（不是 prompt 劝退）。
2. **深度上限**：worker 手上根本没有 ``task_dispatch``，加上 ``enter_fork`` 的深度闸，双保险。
3. **结果截断**：超长工具结果尾部截断留提示。
4. **循环检测**：同一工具刷屏到阈值即触发。

运行：uv run python examples/03_dispatch.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.tool_registry import build_toolkit  # noqa: E402
from app.harness.fork_guard import (  # noqa: E402
    MAX_FORK_DEPTH,
    ForkLimitExceeded,
    enter_fork,
)
from app.harness.loop_detector import LoopDetector  # noqa: E402
from app.harness.truncation import truncate_tool_result  # noqa: E402

ROLES = ("main", "search", "trade")


async def demo_role_boundaries() -> None:
    """打印三个角色各自拿到的工具，边界一眼可见。"""
    print("=== ① 读写边界：Toolkit 发放范围（结构性，不靠 prompt）===")
    for role in ROLES:
        toolkit = await build_toolkit(role)
        tools = await toolkit.get_tool_schemas()
        names = sorted(t["function"]["name"] for t in tools)
        print(f"  [{role:6s}] {len(names)} 个工具：{', '.join(names)}")

    search_kit = await build_toolkit("search")
    search_names = {t["function"]["name"] for t in await search_kit.get_tool_schemas()}
    # 三条断言就是 §2.2 那三条硬规则的可执行版本。
    assert "create_order" not in search_names, "SearchAgent 不该拿到写工具"
    assert "task_dispatch" not in search_names, "worker 不该能再派发（深度上限的结构性保证）"
    trade_kit = await build_toolkit("trade")
    trade_names = {t["function"]["name"] for t in await trade_kit.get_tool_schemas()}
    assert "item_search" not in trade_names, "TradeAgent 不该拿到检索工具"
    print("  ✓ SearchAgent 无写工具 / 无 task_dispatch；TradeAgent 无检索工具")
    print("    → 「买第 2 个」的候选定位只能由主 Agent 在 demands 里给定 item_id")


async def demo_safety_layers() -> None:
    print("\n=== ② ~ ④ 派发安全四层自检（确定性，不调 LLM）===")

    # ② 深度上限：第一层保证是「worker 工具集里没有 task_dispatch」（上面已验证），
    #    enter_fork 的深度计数是第二层——万一有人绕过工具集直接调实现，这里还拦得住。
    with enter_fork() as depth:
        print(f"  ② 深度上限：进入第 {depth} 层派发 OK（上限 MAX_FORK_DEPTH={MAX_FORK_DEPTH}）")
        try:
            with enter_fork():
                print("     ✗ 不该到这里")
        except ForkLimitExceeded as exc:
            print(f"     ✓ 再派一层被拒绝：{exc}")

    # ③ 结果截断：超长结果尾部截断并留提示。
    long_text = "商品详情" * 5000
    truncated = truncate_tool_result(long_text)
    print(f"  ③ 结果截断：原 {len(long_text)} 字 → {len(truncated)} 字，尾部=…{truncated[-22:]}")

    # ④ 循环检测：同一工具刷屏到阈值即触发。
    det = LoopDetector(window=6, threshold=4)
    fired_at = next(i for i in range(1, 9) if det.record("item_search"))
    print(f"  ④ 循环检测：第 {fired_at} 次重复调用 item_search 触发提示")
    print("  （子 Agent 超时 + max_iters 见 dispatch_tool._run_worker 与 ReActConfig）")


async def main() -> None:
    await demo_role_boundaries()
    await demo_safety_layers()


if __name__ == "__main__":
    asyncio.run(main())
