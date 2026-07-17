"""跨整棵 fork 树的 **token / 成本预算闸**（按 session_dir 聚合）—— 把「测」升级成「控」（F 块）。

**与 usage.py 的分工。** ``usage.py`` 是**事后测量**：一轮跑完，从 messages 把 token 用量聚合
出来发 Langfuse / 日志（carried / peak / cache_hit）。它只「看」，不「拦」。本模块是**运行时预算
闸**：每次模型调用一返回就把 token 换算成成本、累进**全树**计数；累计越线后，由 middleware 在请求
模型前把「成本放大器」工具（fork / 检索 / 品类洞察）从模型可见工具表里摘掉，逼 Agent 用现有
候选收尾——**机制兜底，不靠模型自觉**（与 retrieval_budget 次数闸、fork 安全四层同一套哲学）。

**为什么打在「全树」而非「单次调用」上。** Agent 是成本放大器：fork 出 N 个子 + 每子多轮工具链，
token 是乘法累积的。只盯单次调用拦不住「子任务们合起来烧爆预算」。所以复用 retrieval_budget 的
**session_dir 为键的模块级 dict**：主 loop 与所有子 Agent 按同一 key 累加（``thread_scope`` 让子
继承父 session_dir），才能真正全树归集。模块级 dict 由 ``run_agent`` 收尾调 :func:`reset_tree`。

**成本怎么算。** 价格表按「每百万 token」记 input / output / cache_read 三档（cache_read 享折扣）；
按模型名匹配，未知模型回退默认档。价格与预算上限全走 env，代码不写死费率。
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage

from app.api.context import get_session_dir
from app.utils.env import env_float

logger = logging.getLogger("shoppingx.token_budget")

# 每百万 token 的美元单价：(input, output, cache_read)。cache_read 命中享折扣（重发的历史前缀）。
#
# **默认档钉的是本项目实际在用的模型**：`deepseek-v4-flash`（经 DashScope / 百炼调用，官方声明与
# DeepSeek 官网同价）——input ¥1/M、output ¥2/M、cache hit ¥0.02/M，官方美元价 0.14 / 0.28 / 0.0028。
#
# 早先这里默认是 0.27 / 1.10 / 0.07（DeepSeek-V3 量级的估算档，从没核对过真实费率），把成本高估了
# 2~25 倍——**cache_read 那档偏得最狠（25×）**，而本项目多轮追问恰恰大量命中缓存前缀，等于把最该
# 便宜的那部分按贵价收。成本闸与 credit 配额都吃这张表，估错 = 闸提前夺权、用户额度平白缩水。
#
# 换模型 / 换 provider 时**必须回来改这里**（或经 env 覆盖），别让它继续沿用上一个模型的价。
_PriceTuple = tuple[float, float, float]


def _default_price() -> _PriceTuple:
    """默认价格档（每百万 token，美元），可经 env 覆盖。"""
    return (
        env_float("TOKEN_PRICE_INPUT", 0.14),
        env_float("TOKEN_PRICE_OUTPUT", 0.28),
        env_float("TOKEN_PRICE_CACHE_READ", 0.0028),
    )


# 按模型名子串匹配的价格表；命中不到回退默认档。
#
# 表为空是**成立的**，不是偷懒：进入本模块计费的只有主链路（主 loop + fork 子 Agent + 工具内部
# LLM），它们全走同一个 `LLM_MAIN`（快档只是关了 thinking，同模型同价）。judge 模型（qwen3.5-flash）
# 价格不同，但它只在**离线评测**里跑、不经过 run_agent，压根不进这张表。
# 真到了「一条链上跑两种不同价位的模型」那天（如主 loop 降级到 v4-pro），再在这里按名分价。
_PRICES: dict[str, _PriceTuple] = {}


def _price_for(model: str) -> _PriceTuple:
    """按模型名取价格档（子串匹配）；未知模型回退默认档（env 可调）。"""
    for key, price in _PRICES.items():
        if key in model:
            return price
    return _default_price()


def _budget_usd() -> float:
    """单次任务（一棵 fork 树）的成本上限（美元）。<=0 视为不设闸（关闭预算控制）。

    优先取本任务被压低过的 cap（:func:`set_task_cap`，即用户今日剩余额度），否则走 env 默认。
    """
    k = _key()
    if k is not None and k in _CAPS:
        return _CAPS[k]
    return env_float("TOKEN_BUDGET_USD", 0.50)


def _soft_ratio() -> float:
    """软线比例：累计成本达 上限×此值 即记 ``soft``（默认 0.8）。

    与 retrieval_budget 的软线不同：本块 ``soft`` 是**观测信号**（进 BUDGET_OUTCOME metric /
    收尾日志，用于看多少任务逼近预算），**不在循环内注入收敛提示**——成本是事后才测得的，软线
    那一刻当轮调用早已发生，临时插提示意义有限。真正的执行在 ``hard``（夺权摘工具）。
    """
    return env_float("TOKEN_BUDGET_SOFT_RATIO", 0.8)


@dataclass
class _TreeUsage:
    """一棵 fork 树（一次 run_agent）的累计用量与成本。"""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float = 0.0
    model_calls: int = 0
    # 已计费过的 AIMessage id，去重防重复计费（同一条 message 可能被多次 charge 看到）。
    _seen: set[str] = field(default_factory=set)


# session_dir(str) → 该任务一棵 fork 树的累计用量。主 / 各子 Agent 共享同一条目。
_STATE: dict[str, _TreeUsage] = {}
# session_dir(str) → 本次任务被压低后的成本上限（见 set_task_cap）。缺省即走 env 默认。
_CAPS: dict[str, float] = {}


def _key() -> str | None:
    sd = get_session_dir()
    return str(sd) if sd is not None else None


def _state(create: bool = True) -> _TreeUsage | None:
    """取当前 session 的用量状态；无 session 作用域（单测）返回 None。"""
    k = _key()
    if k is None:
        return None
    st = _STATE.get(k)
    if st is None and create:
        st = _TreeUsage()
        _STATE[k] = st
    return st


def _msg_cost(meta: dict, model: str) -> tuple[int, int, int, float]:
    """从一条 usage_metadata 算 (input, output, cache_read, cost_usd)。任一字段缺失按 0，绝不抛。"""
    inp = int(meta.get("input_tokens") or 0)
    out = int(meta.get("output_tokens") or 0)
    details = meta.get("input_token_details") or {}
    cache_read = int(details.get("cache_read") or 0)
    p_in, p_out, p_cache = _price_for(model)
    non_cached = max(inp - cache_read, 0)  # 命中缓存的部分单独按折扣价计
    cost = (non_cached * p_in + cache_read * p_cache + out * p_out) / 1_000_000
    return inp, out, cache_read, cost


def charge_tree_usage(messages: Sequence[BaseMessage]) -> float | None:
    """把本次模型调用返回的 AIMessage 用量计进全树，返回累计成本（美元）；无作用域返回 None。

    幂等：按 AIMessage id 去重——middleware 每次 ``awrap_model_call`` 只把**本次**新返回的消息
    传进来，但即便重复传同一条也不会重复计费。
    """
    st = _state()
    if st is None:
        return None
    for msg in messages:
        if not isinstance(msg, AIMessage):
            continue
        meta = getattr(msg, "usage_metadata", None)
        if not meta:
            continue
        mid = msg.id or ""
        if mid and mid in st._seen:
            continue
        if mid:
            st._seen.add(mid)
        model = (getattr(msg, "response_metadata", None) or {}).get("model_name", "") or ""
        inp, out, cache_read, cost = _msg_cost(meta, model)
        st.input_tokens += inp
        st.output_tokens += out
        st.cache_read_tokens += cache_read
        st.cost_usd += cost
        st.model_calls += 1
    return st.cost_usd


def charge_tool_llm_usage(usage_by_model: Mapping[str, Any]) -> None:
    """把**工具内部** LLM 调用的用量计进全树（planner / shopping_summary / chat_fallback）。

    这些调用不经过 agent middleware 的 ``awrap_model_call``——那里的 :func:`charge_tree_usage`
    只见主 loop 的模型调用，工具内的这几笔曾完全漏账（perf-audit-r5 实测：总账恰好只等于主 loop
    各次之和，成本与预算闸少算约 25% 时长对应的用量）。结构化输出（``with_structured_output``）
    会把 AIMessage 吞成 Pydantic 对象，callback 是拿到 usage 且不改工具语义的唯一口子——工具侧
    挂 ``UsageMetadataCallbackHandler`` 收集，收尾把 ``.usage_metadata`` 交给本函数入账。

    入参：``model_name → UsageMetadata`` 映射（handler 已按模型聚合，多次重试自动累加）。
    记账绝不反噬工具执行：任何异常吞掉记日志；无会话作用域（单测直调工具）静默跳过。
    不做去重——handler 是每次调用新建的局部对象，天然只含本次用量。
    """
    try:
        st = _state()
        if st is None or not usage_by_model:
            return
        for model, meta in usage_by_model.items():
            inp, out, cache_read, cost = _msg_cost(dict(meta), model)
            st.input_tokens += inp
            st.output_tokens += out
            st.cache_read_tokens += cache_read
            st.cost_usd += cost
            st.model_calls += 1
    except Exception:
        logger.debug("工具内部 LLM 记账失败，跳过（不反噬工具执行）", exc_info=True)


def peek_tree_cost() -> float | None:
    """只读当前全树累计成本（不计费），供「越线即夺权」在请求模型前判断。无作用域返回 None。"""
    st = _state(create=False)
    return st.cost_usd if st is not None else None


def budget_status() -> str:
    """当前预算档位：``"ok"`` / ``"soft"``（过软线，仅观测）/ ``"hard"``（过硬线，夺权收尾）。

    ``soft`` 只进 metric / 日志（见 :func:`_soft_ratio`），``hard`` 才触发 middleware 摘工具。
    无 session 作用域、或预算上限 <=0（不设闸）一律 ``"ok"``——不在单测 / 关闭场景平添门槛。
    """
    cap = _budget_usd()
    if cap <= 0:
        return "ok"
    cost = peek_tree_cost()
    if cost is None:
        return "ok"
    if cost >= cap:
        return "hard"
    if cost >= cap * _soft_ratio():
        return "soft"
    return "ok"


def budget_cap_usd() -> float:
    """本次任务的成本上限（美元）。``<=0`` 表示不设闸。"""
    return _budget_usd()


def remaining_ratio() -> float:
    """预算剩余比例（0.0 ~ 1.0）——模型路由降级的唯一输入（见 :mod:`app.agent.model_router`）。

    无 session 作用域（单测 / 离线脚本）或预算上限 <=0（不设闸）一律返回 ``1.0``——「没配预算」
    必须等价于「预算充裕」，否则降级会在所有单测里凭空触发。
    """
    cap = _budget_usd()
    if cap <= 0:
        return 1.0
    cost = peek_tree_cost()
    if cost is None:
        return 1.0
    return max(0.0, 1.0 - cost / cap)


def tree_snapshot() -> dict[str, float | int] | None:
    """当前全树用量快照（日志 / metrics 用）；无作用域返回 None。"""
    st = _state(create=False)
    if st is None:
        return None
    return {
        "input_tokens": st.input_tokens,
        "output_tokens": st.output_tokens,
        "cache_read_tokens": st.cache_read_tokens,
        "cost_usd": round(st.cost_usd, 6),
        "model_calls": st.model_calls,
    }


def set_task_cap(cap_usd: float) -> None:
    """把**本次任务**的成本上限压到 ``cap_usd``（不高于 env 的 ``TOKEN_BUDGET_USD``）。

    唯一调用方是 ``run_agent``（在 ``thread_scope`` 内——本函数按 session_dir 归集，没有作用域就
    无处可存）：用户的**今日剩余额度**（:mod:`app.db.quota`）在那里压进来。为什么需要它——入口的
    配额闸只能判「还有没有余额」，判完就放行；若剩余只够 $0.01 而单任务预算是 $0.50，这一次任务
    就能透支近半个额度。把 cap 压成 ``min(单任务预算, 今日剩余)`` 后，透支最多只到「额度刚好用尽」
    为止，超出部分由 hard 闸夺权收尾。

    与 ``_STATE`` 同样按 session_dir 归集（子 Agent 继承父 session_dir，故全树共用同一个 cap），
    同样由 :func:`reset_tree` 清掉。传 ``<=0`` 会被忽略——那等价于「不设闸」，而在这里它的语义恰恰
    相反（额度已耗尽），绝不能因此把闸门关掉。
    """
    k = _key()
    if k is None or cap_usd <= 0:
        return
    _CAPS[k] = min(cap_usd, env_float("TOKEN_BUDGET_USD", 0.50))


def reset_tree() -> None:
    """清掉本 session 的用量条目（任务收尾时调，防模块级 dict 无界增长）。"""
    k = _key()
    if k is not None:
        _STATE.pop(k, None)
        _CAPS.pop(k, None)
