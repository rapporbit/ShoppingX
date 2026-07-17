"""四档模型路由降级（refdocs 16-4 §3 / §6）——预算见底时，降档继续跑，而不是硬停。

**要解决的问题。** 仓库原先只有一个「硬线」：全树成本超上限 → 把成本放大器工具拦掉、逼模型收尾
（``tool_gates.token_budget_gate``）。这是**开关**，不是**梯度**——预算从 100% 花到 0% 的整个过程里
系统行为完全一致，直到最后一刻突然夺权。而 Agent 的成本是乘法累积的（fork × 多轮 × 工具链），
等撞线才反应往往已经晚了：这一轮的钱早花出去了。

**四档梯度，按剩余预算比例走：**

======================  =========  ==========================================================
剩余预算                 档位        行为
======================  =========  ==========================================================
> 50%                   main       主力模型，不限制
20% ~ 50%               lite       换便宜档模型（本项目=同模型关 reasoning），能力不变、少花钱
5% ~ 20%                minimal    lite 模型 + 注入简洁 hint + **机制上收走成本放大器工具**
< 5%                    fallback   **不调 LLM**，用已有候选拼一个诚实的回答
======================  =========  ==========================================================

**三处主动更正 refdocs：**

1. **lite 档不换弱模型，只关 reasoning。** refdocs 的 lite 是 Qwen3-35B → Qwen3-8B 这种降参数。
   本项目 M-perf 实测过：换更弱的小模型在 Agent 任务上**反而更慢更贵**——弱模型多绕几轮、自纠偏差，
   省下的单 token 成本被多出来的轮数吃光。正确做法是同一个 hybrid 模型只把 thinking 关掉：能力不降，
   省的是那段最贵的 reasoning 解码（详见 ``llm.get_fast_llm`` 的说明）。所以 lite / minimal 都复用
   ``get_fast_llm()``，除非显式配了 ``LLM_LITE``。

2. **minimal 档不只注入 hint，还真收走工具。** refdocs 只在 system prompt 末尾追加「不要再检索了」。
   本项目的一贯立场是**机制兜底优于提示词**（与 fork 安全四层、检索预算闸同源）：hint 照注入，但
   同时让 ``token_budget_gate`` 在 minimal 档就开始拦成本放大器工具。弱模型看不懂 hint 的时候，
   闸还在。

3. **fallback 不是错误，是「在剩下的预算里给出力所能及的最好回答」。** 它直接返回一条 AIMessage
   （无 tool_calls），AgentLoop 自然终止。它**绕过 post_reflect**——那里的终结纪律 Hook 会因为
   「没调终结工具就想收尾」而要求重发模型，可预算就是为此耗尽的，再重发一次纯属浪费。

**降档不可逆。** 成本只增不减，``remaining_ratio`` 单调下降，所以档位只会往下走。这意味着不必担心
在 lite / main 之间反复横跳导致 prompt cache 前缀失效——一次降档，一次缓存重建，仅此而已。
"""

from __future__ import annotations

import logging
import os
from enum import IntEnum
from functools import lru_cache
from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel

from app.agent.llm import get_fast_llm
from app.agent.token_budget import remaining_ratio
from app.utils.env import env_float

if TYPE_CHECKING:
    from app.tools.schemas import ItemCandidate

logger = logging.getLogger("shoppingx.agent.router")


class Tier(IntEnum):
    """预算档位。用 ``IntEnum`` 是为了能直接比大小（``tier >= Tier.MINIMAL``）—— 值越大越省。"""

    MAIN = 0
    LITE = 1
    MINIMAL = 2
    FALLBACK = 3

    @property
    def label(self) -> str:
        return self.name.lower()


def _thresholds() -> tuple[float, float, float]:
    """(lite, minimal, fallback) 三条剩余比例分界线，全走 env。"""
    return (
        env_float("BUDGET_TIER_LITE_RATIO", 0.5),
        env_float("BUDGET_TIER_MINIMAL_RATIO", 0.2),
        env_float("BUDGET_TIER_FALLBACK_RATIO", 0.05),
    )


def current_tier() -> Tier:
    """按全树剩余预算比例判当前档位。无预算闸 / 无 session 作用域一律 ``MAIN``。"""
    remaining = remaining_ratio()
    lite, minimal, fallback = _thresholds()
    if remaining <= fallback:
        return Tier.FALLBACK
    if remaining <= minimal:
        return Tier.MINIMAL
    if remaining <= lite:
        return Tier.LITE
    return Tier.MAIN


def tier_model(tier: Tier) -> BaseChatModel | None:
    """该档位该用哪个模型。``FALLBACK`` 返回 ``None``——那一档根本不调 LLM。

    ``MAIN`` 也返回 ``None``：表示「不覆盖，用 request 里原本那个」。这让适配器的逻辑退化成
    「拿到模型就 override，拿不到就不动」，不必区分「主力档」和「降级档」。
    """
    if tier in (Tier.FALLBACK, Tier.MAIN):
        return None
    return _lite_llm()


def _lite_llm() -> BaseChatModel:
    """便宜档模型。默认复用 ``get_fast_llm()``（同模型关 reasoning）；``LLM_LITE`` 可显式覆盖。"""
    name = os.environ.get("LLM_LITE")
    return _named_lite_llm(name) if name else get_fast_llm()


@lru_cache(maxsize=1)
def _named_lite_llm(model: str) -> BaseChatModel:
    """配了 ``LLM_LITE`` 时按名建模型（同 endpoint / 同温度）。缓存一份，复用连接池。"""
    from langchain.chat_models import init_chat_model

    return init_chat_model(
        model,
        model_provider="openai",
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
        temperature=env_float("LLM_TEMPERATURE", 0.3),
    )


# minimal 档注入的简洁 hint。用 ``[预算提醒]`` 前缀复用 ``session_hooks`` 已有的内部文案标记——
# 模型偶尔会把这段整个抄进给用户的回答里，输出审核靠这个前缀把它剔掉。
MINIMAL_HINT = (
    "[预算提醒] 本次任务的 token 预算已所剩不多（不足 20%）。从现在起：\n"
    "- 不要再发起任何新的检索（item_search / web_search / category_insight）或 fork 子任务；\n"
    "- Think 阶段不要展开长推理，直接给结论；\n"
    "- 基于已有的候选和观察结果，立刻走完剩余步骤并**调用终结性工具收尾**"
    "（有商品卡 → shopping_summary；纯文字 → chat_fallback）。"
)


def build_fallback_answer(user_query: str = "") -> str:
    """不调 LLM，用会话里已攒下的候选拼一个诚实的回答。

    **fallback 不是报错。** 用户的钱花完了，但我们手里往往已经有一批真实召回的候选——把它们如实
    列出来，比抛一个「预算超限」的红色错误有用得多。前端也不该按 error 样式渲染它（它是一条正常的
    task_result）。

    连候选都没有（预算在检索到任何东西之前就烧完了，通常意味着 fork 失控或模型死循环），就如实说
    信息不足并建议缩小范围——**绝不编造商品**（对齐 system prompt 的 P0 诚实红线）。
    """
    picks = _top_candidates(3)
    if not picks:
        hint = f"「{user_query[:40]}…」" if user_query else "本次请求"
        return (
            f"抱歉，{hint}的处理超出了本次任务的算力预算，尚未检索到可靠的商品候选，"
            "因此我不能给你一份推荐清单——编造商品不是选项。\n\n"
            "建议缩小范围后重新提问（明确品类 + 预算 + 一两个关键约束），会快很多。"
        )

    lines = ["由于本次任务的算力预算已用尽，以下是基于**已检索到的真实候选**为你整理的清单：\n"]
    for i, cand in enumerate(picks, 1):
        price = f"${cand.price_usd:.2f}" if cand.price_usd else "价格待确认"
        title = cand.title[:60]
        lines.append(f"{i}. **{title}**（{cand.platform}）— {price}")
    lines.append(
        "\n> 注：这份清单未经完整的比价与精挑流程，仅按召回顺序给出。"
        "如需更精准的推荐，请开启新对话并缩小检索范围。"
    )
    return "\n".join(lines)


def _top_candidates(n: int) -> list[ItemCandidate]:
    """从候选登记表取前 n 个候选。读不到（无会话作用域）返回空列表——绝不抛。"""
    try:
        from app.api.context import get_session_dir
        from app.tools._candidates import _REGISTRY

        sd = get_session_dir()
        if sd is None:
            return []
        return list(_REGISTRY.get(str(sd), {}).values())[:n]
    except Exception:
        logger.debug("读取候选失败，fallback 回答退化为无候选版本", exc_info=True)
        return []
