"""一次模型调用用哪一档：档位名 → 模型对象的解析，以及主 loop 第一轮加档判定。

这是「Hook 决策、适配器落地」里落地的那一半：Hook（``budget_router``）只给档位名，
模型对象在这里解析——档位 → 模型的解析**只此一处**。
"""

from __future__ import annotations

from typing import Any


def resolve_model_tier(tier: Any) -> Any | None:
    """档位名 → 本运行时的模型对象。空档位 = 不换模型（用装配期那个基座）。

    延迟导入 ``llm``：本模块在 Agent 装配前就被 import，模块级拉模型工厂会把 ``.env`` 的读取
    时机提前到 import 期，测试里 monkeypatch 环境变量就来不及了。
    """
    if not tier:
        return None
    from app.agent.llm import get_tier_llm

    return get_tier_llm(str(tier))


def first_round_tier(ctx: dict[str, Any]) -> str | None:
    """主 loop 第一轮该不该加档 —— 返回档位名，不加返回 ``None``。

    原为独立 Hook（``hooks/reasoning_boost``）。收进适配器侧是因为它与**装配期选的基座**是同一
    条口径的两半：拆成两处的那段时间里，基座被改成 reasoning 而 Hook 还在按「基座是快档」顶
    reasoning，override 成同一个实例，什么都没发生，也没有任何测试会红（审查报告 P0-1）。

    **预算降档优先**：调用方只在 ``model_tier`` 仍为空时才问本函数，所以 budget_router 写过
    lite 就是 lite —— 钱不够的时候，「想清楚」让位于「跑完」。（曾有「worker 不加档」一条豁免，
    A4 删子 Agent 后去掉。）
    """
    from app.agent.llm import main_loop_tier_base, main_loop_tier_first

    tier = main_loop_tier_first()
    if tier == "same" or ctx.get("round_number") != 1:
        return None
    return None if tier == main_loop_tier_base() else tier
