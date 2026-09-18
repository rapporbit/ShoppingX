"""塑形模型看到的上下文：工具返回清理 / 偏好注入 / 成功策略注入与结账。

    on_system_prompt 50  system_prompt_append 装配期给主 Agent 追加两段：按用户原话匹配的在役策略
                                              <learned_strategies>、待决议确认卡 + 本会话订单
                                              <trade_state>（曾是两个 hook，2026-09-15 合一）
    pre_think        10  tool_result_pruner  工具返回总量超阈值 → 最旧的换占位，削到一半（D3）
    post_tool_call   50  preference_inject   planner 判出域后注入域内长期偏好
    on_session_end   90  strategy_feedback   给本轮注入过的策略结账：命中回血、连续失败淘汰

**两层压缩的分工**：这里做零成本的粗活（清最旧工具返回），框架的 ``compress_context`` 做花钱的
细活（超 ``trigger_ratio`` 时 LLM 写摘要进 ``state.summary``，随 session.json 持久化）。顺序上
粗活在前——能白削掉的体积，没必要先花一次 LLM 调用去总结它。cache_control 不在这里打，落在
formatter 层。
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import Any

from app.api.context import get_thread_id, get_user_id
from app.harness.budgets import (
    TERMINAL_TOOLS,
)
from app.harness.middleware import harness_hook
from app.harness.msgs import _attr
from app.memory.injector import PREF_EMPTY, build_preference_block
from app.memory.strategies import get_strategy_store, render_strategy_block, strategies_for_query
from app.trade.confirmations import trade_state
from app.trade.repository_sql import confirmation_repository, order_repository
from app.utils.env import env_int
from app.utils.tokens import count_tokens

logger = logging.getLogger("shoppingx.harness.context_shaping")


#: 上下文里工具返回的 token 总量超过它就开始清最旧的几条（0 = 关掉这道清理）。
#: 24k 是按「普通轮输入 22.5k」（round3 基线）定的：正常一轮碰不到，多轮续聊堆起来才会。
PRUNE_TOOL_RESULTS_TOKENS = env_int("PRUNE_TOOL_RESULTS_TOKENS", 24000)

#: 最近几条工具返回一律不动——模型当前这一步的推理就靠它们，清了等于让它凭空作答。
PRUNE_KEEP_RECENT = env_int("PRUNE_KEEP_RECENT", 6)

_PRUNED_PLACEHOLDER = "[早先的工具返回已清理以腾出上下文。还需要这段内容就重新调用对应工具。]"


def _set_block_output(block: Any, text: str) -> None:
    """block 可能是 pydantic 对象也可能是 dict（同 ``_attr`` 那两种来源），统一写回。"""
    if isinstance(block, dict):
        block["output"] = text
    else:
        block.output = text


@harness_hook("pre_think", name="tool_result_pruner", priority=10)
async def prune_old_tool_results(context: dict[str, Any]) -> dict[str, Any] | None:
    """工具返回式压缩：超阈值时把**最旧**的工具返回换成一行占位，直到总量减半。

    **为什么要自己做一层**：AgentScope 的 ``compress_context`` 只有「整体超 ``trigger_ratio``
    → LLM 写摘要」这一招，没有「清最旧工具返回」这种粗活。而多轮购物会话里吃掉上下文的正是
    工具返回——十几条 ``item_search`` 的候选 JSON，每条几千 token，其中早先那几轮的候选模型
    早就不看了（当前轮的候选登记表才是它的工作面）。先用零成本的替换削掉一半，再让框架决定要
    不要为剩下的花一次 LLM 摘要，顺序上这一层必须在前。

    **原地改 Msg 而不是只改视图**：``ctx["messages"]`` 里的 Msg 与 ``agent.state.context`` 是同一
    批对象（见 adapter 那段注释），原地改 block 等于同时改了持久化状态。这是有意的——只改视图
    的话每轮都要重清一次，而每轮清理的位置都不同，前缀缓存就会**轮轮都断**；改 state 则是断
    一次、之后形态稳定。代价是原文真的没了，所以占位文案要写明「重新调用对应工具」。

    保留最近 ``PRUNE_KEEP_RECENT`` 条不动，见该常量。
    """
    if PRUNE_TOOL_RESULTS_TOKENS <= 0:
        return None
    blocks: list[Any] = []
    for msg in context.get("messages") or []:
        for block in getattr(msg, "content", None) or []:
            if _attr(block, "type") == "tool_result":
                blocks.append(block)
    if len(blocks) <= PRUNE_KEEP_RECENT:
        return None

    sizes = [count_tokens(str(_attr(b, "output") or "")) for b in blocks]
    total = sum(sizes)
    if total <= PRUNE_TOOL_RESULTS_TOKENS:
        return None

    target = PRUNE_TOOL_RESULTS_TOKENS // 2
    pruned = 0
    for i in range(len(blocks) - PRUNE_KEEP_RECENT):
        if total <= target:
            break
        if sizes[i] <= len(_PRUNED_PLACEHOLDER):
            continue  # 已经是占位或本就极短，换了不省反亏
        _set_block_output(blocks[i], _PRUNED_PLACEHOLDER)
        total -= sizes[i] - count_tokens(_PRUNED_PLACEHOLDER)
        pruned += 1
    if pruned:
        logger.info("工具返回清理：%d 条 → 占位，总量 %d → %d token", pruned, sum(sizes), total)
    return None


@harness_hook("post_tool_call", name="preference_inject", priority=50)
async def inject_domain_preferences(context: dict[str, Any]) -> dict[str, Any] | None:
    """planner 返回后注入域内长期偏好。一轮至多注入一次——阶段机保证 planner 只成功跑一次
    （跑完即离开 PLANNING，而 planner 不在后续阶段的白名单里）。"""
    if context.get("tool_name") != "planner":
        return None

    user_id = get_user_id() or ""
    if not user_id:
        return None  # 匿名用户没有长期偏好

    block = await build_preference_block(user_id)
    if not block or block == PREF_EMPTY:
        return None  # 本轮域内没有任何偏好 → 不塞空占位（省 token，也不给模型噪声）

    logger.info("注入域内长期偏好（planner 后）")
    context.setdefault("inject_messages", []).append(
        {
            "role": "system",
            "content": (
                "<user_long_term_preferences>\n"
                f"{block}\n"
                "</user_long_term_preferences>\n"
                "以上是该用户与**本轮品类相关**的长期偏好，已由系统自动生效（检索词与精挑打分里\n"
                "都已并入，见 memory.assemble）——**不要**再把它们转述进任何工具参数，重复一遍不会\n"
                "让它们更生效，只会让你替用户做了他没授权的决定。它们在这里只为一件事：让你在向\n"
                "用户解释「为什么选这几件」时，说得出是哪条偏好起了作用。"
            ),
        }
    )
    return context


#: 本轮注入了哪几条策略（dedup_key）。装配期写、收尾时读。**每轮必写**（没匹配上就写空元组），
#: 否则同进程连跑两轮时，第二轮会拿上一轮的清单去结账——记在错的策略头上比不记更糟。
_injected: ContextVar[tuple[str, ...]] = ContextVar("injected_strategies", default=())


def injected_strategy_keys() -> tuple[str, ...]:
    """本轮注入过的策略 key（测试与结账钩子用）。"""
    return _injected.get()


async def _strategy_block(context: dict[str, Any]) -> str | None:
    """按本轮 query 匹配在役策略 → <learned_strategies> 块。每轮必写注入清单（哪怕空）。"""
    query = str(context.get("query") or "")
    matched = await strategies_for_query(query)
    _injected.set(tuple(s.dedup_key for s in matched))
    block = render_strategy_block(matched)
    if not block:
        return None  # 一条都没匹配上 → 不塞空占位（省 token，也别给模型噪声）
    logger.info("注入 %d 条成功策略：%s", len(matched), ", ".join(s.dedup_key for s in matched))
    return block


def render_trade_state_block(state: dict[str, Any]) -> str:
    """待决议确认卡 + 本会话订单 → ``<trade_state>`` 块。两边都空就返回空串（不塞空占位）。

    对齐参考项目 ``agent_state``：**不带地址、不带 hash**——模型只需要知道「有一张卡等着用户点」
    和「这轮已经落了哪些单」，好在用户问「我下单了吗」时不瞎答，也不重复出卡。
    """
    pending = state.get("pending_confirmations") or []
    orders = state.get("orders") or []
    if not pending and not orders:
        return ""
    lines = ["<trade_state>", "本会话的权威交易状态（服务端记录，以此为准，不要凭对话记忆猜）："]
    for c in pending:
        items = "、".join(f"{i['title']}×{i['quantity']}" for i in c.get("items", []))
        what = f"取消订单 {c.get('order_id')}" if c.get("action") == "cancel" else f"下单：{items}"
        lines.append(f"- 待用户在页面上点按钮的确认卡：{what}。用户口头说确认不算，别再出一张。")
    for o in orders:
        items = "、".join(f"{i['title']}×{i['quantity']}" for i in o.get("items", []))
        head = f"订单 {o['order_id']}（{o['status']}，{o['total']} {o['currency']}）"
        lines.append(f"- {head}：{items}")
    lines.append("</trade_state>")
    return "\n".join(lines)


async def _trade_state_block() -> str | None:
    """交易状态 → <trade_state> 块。未登录 / 无会话 / 库不可用时静默跳过——注入是锦上添花，
    不能让一次读库失败把整轮任务拖死。"""
    user_id, thread_id = get_user_id(), get_thread_id()
    if not user_id or not thread_id:
        return None
    try:
        state = await trade_state(
            confirmation_repository(), order_repository(), user_id=user_id, thread_id=thread_id
        )
    except Exception:  # noqa: BLE001
        logger.warning("读取交易状态失败，本轮不注入 <trade_state>", exc_info=True)
        return None
    return render_trade_state_block(state) or None


@harness_hook("on_system_prompt", name="system_prompt_append", priority=50)
async def append_system_prompt_blocks(context: dict[str, Any]) -> dict[str, Any] | None:
    """主 loop 装配期往 system prompt 末尾追加：先策略块、后交易状态块（顺序即渲染顺序）。"""
    if context.get("role") != "main":
        return None
    blocks = [b for b in (await _strategy_block(context), await _trade_state_block()) if b]
    if not blocks:
        return None
    context.setdefault("append", []).extend(blocks)
    return context


@harness_hook("on_session_end", name="strategy_feedback", priority=90)
async def settle_strategies(context: dict[str, Any]) -> dict[str, Any] | None:
    """给本轮注入过的策略结账。

    priority 90 排在输出审核（10）之后：审核可能把最终回复清成空串，那时这一轮该算失败——
    顺序反了就会把一次「输出全被判违规」记成成功。
    """
    keys = injected_strategy_keys()
    if not keys:
        return None
    called = context.get("called_tools") or set()
    final = context.get("final_answer")
    success = bool(set(called) & TERMINAL_TOOLS) and bool(isinstance(final, str) and final.strip())
    await get_strategy_store().record_outcome(keys, success=success)
    return None
