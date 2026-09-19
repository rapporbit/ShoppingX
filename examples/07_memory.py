"""M7 示例：长期记忆的「跨会话记住偏好」闭环（离线，不调 LLM）。

复刻 refdocs/06 那个真实体验问题：
  会话 1：用户说「不要塑料」→ 写进记忆库（持久化）
  会话 1 结束：消息历史丢弃
  会话 2（新会话）：从库里读出「不要塑料」→ 注入本轮上下文 → Agent 记得

演示三件事：
  1) 写：``validate_fact``（三条写路径共用的单门，含 PII 过滤）→ ``upsert_facts``，同 key 覆盖。
  2) 注入：``select_tier_one_facts``（全部 constraint + 按新鲜度补位）→ ``render_memory_block``。
  3) 按需召回：``search_facts`` 按主题翻没进注入那批的旧事实（``recall_memories`` 工具走的就是它）。

**注入点不在 system prompt**：记忆每轮都可能变，混进 system 会打断跨轮稳定的 prompt cache
前缀。真实链路里它是 planner 之后的一条 system 消息（``harness.hooks.context_shaping``），
这里只演示那条消息的正文怎么来的。

运行：uv run python examples/07_memory.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db.session import init_db  # noqa: E402
from app.memory.fact_store import MemoryFactStore  # noqa: E402
from app.memory.facts import (  # noqa: E402
    MemoryWriteRejected,
    render_memory_block,
    select_tier_one_facts,
    validate_fact,
)

USER_ID = "user-abc123"


async def main() -> None:
    await init_db()
    store = MemoryFactStore()

    # ---- 会话 1：识别并沉淀事实（主链路里这步由 save_memory 工具或回合后抽取做）----
    print("=== 会话 1：用户搜「旅行收纳袋，不要塑料」===")
    facts = [
        validate_fact("material_avoid", "不接受塑料材质", "constraint", source_session="sess-1"),
        validate_fact("brand_taste", "偏好小众设计的品牌", "preference", source_session="sess-1"),
        validate_fact("budget_range", "单件预算 100-300 元", "preference", source_session="sess-1"),
    ]
    await store.upsert_facts(USER_ID, facts)
    print(f"已沉淀 {len(facts)} 条事实。\n")

    # 写入单门会挡下不该进长期记忆的东西，且**不回显被拒的内容**。
    try:
        validate_fact("card", "我的卡号 4111 1111 1111 1111")
    except MemoryWriteRejected as exc:
        print(f"写入过滤器拒绝了一条：{exc}\n")

    # ---- 会话 1 结束，消息历史丢弃。会话 2 是全新会话 ----
    print("=== 会话 2（新会话）：用户搜「洗漱包」===")
    tier_one = select_tier_one_facts(await store.get_facts(USER_ID))
    block = render_memory_block(tier_one)
    print("注入本轮上下文的记忆块：")
    print(block, "\n")
    assert "不接受塑料材质" in block, "constraint 必须每轮注入"
    print("✅ 用户没重复说「不要塑料」，但 Agent（通过注入的事实）记得。\n")

    # ---- 按需召回：没进注入那批的旧事实，按主题翻 ----
    print("=== search_facts：按主题翻旧记忆（recall_memories 工具走的就是它）===")
    await store.upsert_facts(
        USER_ID,
        [validate_fact("gift_recipient_mom", "给妈妈买过真丝围巾，她喜欢素色", "context")],
    )
    for fact in await store.search_facts(USER_ID, "妈妈"):
        print(f"  - [{fact.category.value}] {fact.key}: {fact.value}")

    # 同 key 覆盖 = 改主意（这也是「忘掉 X」的落地方式：写新值，不删）。
    await store.upsert_facts(USER_ID, [validate_fact("material_avoid", "塑料也可以", "constraint")])
    after = {f.key: f.value for f in await store.get_facts(USER_ID)}
    print(f"\n改主意后 material_avoid = {after['material_avoid']}（同 key 覆盖，不留两条矛盾事实）")


if __name__ == "__main__":
    asyncio.run(main())
