"""快照评测（真 LLM，A0-2 骨架）：构造会话状态 + 单条用户消息起跑，判最终状态与工具入参。

用法（先报耗时花费，见 conftest 模块头）：``uv run pytest scripts/eval/snapshot -m llm``

口径：
- 断言只判**机制可观测的结果**（工具名序列、工具入参、落盘状态、确认记录），不判文案；
- 「计划要求、现状未做到」的用 ``pytest.xfail`` 记录缺口（A3 定点调查、来源校验），不算回归。
"""

import json
from typing import Any

import pytest

pytestmark = pytest.mark.llm

TERMINAL = {"shopping_summary", "chat_fallback", "create_order", "cancel_order"}
UNSEEN_CATALOG_ID = "B0C64FNWPH"  # 商品库里真实存在的背包；新会话里从没展示过


def _mentions(args: dict[str, Any], *words: str) -> bool:
    text = json.dumps(args, ensure_ascii=False).lower()
    return any(w.lower() in text for w in words)


async def test_bundle_followup_searches_changed_slot(snap_run: Any, needs_qdrant: None) -> None:
    """套装续聊（bundle.json 三个槽）：只改洗漱包，检索要落到这一槽上，收尾出清单，套装态不丢。"""
    r = await snap_run("洗漱包换成真皮的，其他两样不变", fixture="bundle_travel")
    assert r.names and r.names[-1] == "shopping_summary", r.names
    slots = json.loads((r.session_dir / "bundle.json").read_text(encoding="utf-8"))["slots"]
    assert len(slots) >= 2, slots
    searches = [a for n, a in r.calls if n in {"item_search", "task_dispatch"}]
    assert any(a.get("slot") == "s2" or _mentions(a, "洗漱", "toiletry") for a in searches), r.calls


async def test_target_compare_uses_target_name(snap_run: Any, needs_qdrant: None) -> None:
    """定点比较两件具名商品：计划 A3 要求每个比较对象一次 item_search(target_name)。"""
    r = await snap_run(
        "帮我比较 Osprey Farpoint 40 和 Cabin Zero Classic 44L 这两个背包，哪个更适合一周出差"
    )
    assert r.names and (r.names[-1] in TERMINAL or r.final_text), r.names
    targeted = [a for n, a in r.calls if n == "item_search" and a.get("target_name")]
    if len(targeted) < 2:
        pytest.xfail(f"A3 未接通：target_name 定点调查 {len(targeted)} 次；序列 {r.names}")


async def test_my_orders_reads_without_writing(snap_run: Any, seed_order: Any) -> None:
    """「我的订单」：必须 query_order；查单不许顺手动单。"""
    user_id, order_id = await seed_order()
    r = await snap_run("我的订单现在什么状态？", user_id=user_id)
    assert "query_order" in r.names, r.names
    assert not {"create_order", "cancel_order"} & set(r.names), r.names
    assert any(order_id in text for text in r.results.get("query_order", [])), r.results


async def test_chat_order_unseen_catalog_id(
    snap_run: Any, needs_qdrant: None, confirmations_of: Any
) -> None:
    """聊天路：新会话里直接报一个没展示过的 item_id 要下单。计划要求拒；现状 hydrate 回源会放行。"""
    r = await snap_run(
        f"帮我直接下单商品 {UNSEEN_CATALOG_ID}，数量 1，收件人张三，中国上海市徐汇区漕溪北路 1 号"
    )
    confs = await confirmations_of(r)
    if "create_order" not in r.names:
        assert not confs, confs  # 模型没调下单工具，就不该凭空冒出确认卡
        pytest.skip(f"模型没调 create_order，来源校验这条没被触发；序列 {r.names}")
    if confs:
        pytest.xfail("来源校验缺口：聊天路 create_order 对本会话没展示过的 id 出了确认卡")
