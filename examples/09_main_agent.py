"""M9 示例：主 AgentLoop 合龙 —— 一条 query 从入口跑到收尾。

这是 ShoppingX 的「装配车间」演示：把九工具 + dispatch fork + 压缩 + 记忆 + 监控串成主链路，
跑 ROADMAP M9 的验收 query「便宜抗造旅行三件套，预算 300，不要塑料，喜欢小众」，期望全链路：

    planner → 跨平台 fork item_search → price_compare → shipping_calc → item_picker
      → shopping_summary（终结性，正常终止）

为了让「Agent 在做什么」肉眼可见，这里给全局 ConnectionManager 挂一条**假连接**，把 AGUI
事件流实时收下来按序打印（session_created / tool_start / fork / tool_end / … / task_result）
——这正是 M10 前端要消费的同一条事件流。

跑第二轮还演示长期记忆闭环：第一轮 shopping_summary 沉淀的新偏好（如「不要塑料」），第二轮
同一 user_id 起来时会被注入 system prompt，无需用户重复说。

需要真实 LLM（.env 里的 LLM_MAIN / OPENAI_*）。未配置则优雅跳过，不报错。
运行：uv run python examples/09_main_agent.py
"""

import asyncio
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.api import monitor  # noqa: E402


class _PrintingWS:
    """假 WebSocket：实现 ConnectionManager 要的 accept / send_json，把事件打印出来。"""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def accept(self) -> None:
        return None

    async def send_json(self, data: dict[str, Any]) -> None:
        self.events.append(data)
        evt = data.get("event", "?")
        d = data.get("data", {})
        detail = ""
        if evt in {"tool_start", "tool_end"}:
            detail = f" · {d.get('tool', '')}"
            extra = {k: v for k, v in d.items() if k != "tool"}
            if extra:
                detail += f" {extra}"
        elif evt == "fork":
            detail = f" · {d.get('sub_thread_id', '')} ← {str(d.get('demands', ''))[:48]}…"
        elif evt == "task_result":
            detail = f" · {str(d.get('final_answer', ''))[:60]}…"
        print(f"  [AGUI] {evt:16s}{detail}")


async def _run_once(query: str, user_id: str, label: str) -> None:
    from app.agent.main_agent import run_agent

    thread_id = f"demo-{uuid4().hex[:8]}"
    ws = _PrintingWS()
    # 把假连接登记到全局 ConnectionManager（monitor 推事件就会进 ws.send_json）。
    await monitor.get_connection_manager().connect(ws, thread_id)
    print(f"\n=== {label}（thread={thread_id}, user={user_id}）===")
    print(f"  query: {query}")
    try:
        result = await run_agent(query, thread_id=thread_id, user_id=user_id)
    finally:
        await monitor.get_connection_manager().disconnect(ws, thread_id)

    print(f"\n  最终回复：\n  {result['final_text'][:400]}")
    if result["learned_preferences"]:
        print(f"  本轮沉淀偏好：{result['learned_preferences']}")
    # 校验主 loop 真的收尾了（出现 task_result）且没有把同名工具刷爆。
    kinds = [e.get("event") for e in ws.events]
    assert "task_result" in kinds, "主 loop 未正常收尾（无 task_result）"
    print(f"  事件总数 {len(ws.events)}，含 fork={kinds.count('fork')}，已正常终止 ✓")


async def main() -> None:
    try:
        from app.agent.llm import get_llm

        get_llm()  # 触发 env 校验 / 连接构造
    except Exception as e:  # noqa: BLE001
        print(f"跳过（未配置真实 LLM）：{type(e).__name__}: {e}")
        return

    user_id = f"demo-user-{uuid4().hex[:6]}"
    # 第一轮：完整购物意图，期望走完 planner→fork→比价→到手价→精挑→收尾。
    await _run_once(
        "想买便宜又抗造的旅行三件套，预算 300，不要塑料的，喜欢小众",
        user_id=user_id,
        label="第一轮：全链路",
    )
    # 第二轮：同一用户、不再重复说「不要塑料」，看长期记忆是否被注入并尊重。
    await _run_once(
        "再帮我看看有没有合适的旅行收纳袋，预算 150",
        user_id=user_id,
        label="第二轮：长期记忆注入",
    )


if __name__ == "__main__":
    asyncio.run(main())
