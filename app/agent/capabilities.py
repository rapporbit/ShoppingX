"""fallback 目标的能力门（阶段 2 第 3 条，**收窄版**）。

## 为什么不是计划原文那张 yaml 矩阵

计划写的是「每个 provider 一份 yaml（cache_control / enable_thinking / tool_choice / 流式
tool call / 存根），fallback 只允许跳到全绿的目标」。落地时收窄成了现在这样，三条理由：

1. **那张表判的是「没报 400」，判不了语义生效。** spike（`docs/plans/spike-litellm-2026-09-20.md`
   §2.3 标注 1）自己承认：cache_control 两家都「收下」，但 ``cached_tokens`` 全是 0——表里最贵
   的一列恰恰是这张表证明不了的。
2. **静态表会过期，过期的表比没有表更危险**：它给「已验证」的假象，而供应商改版不会通知你改 yaml。
3. **五列全绿是错的阈值。** cache_control / enable_thinking / 流式不支持的后果是变贵、变慢、
   变不稳——那是**可以接受的降级**，另一条路是整条任务挂掉。工具调用出不来才是「切了不如不切」：
   本仓的 Agent 没有工具就是个聊天框。所以门只判这一列。

## 判据：查得到才信，查不到不拦

情报源用 litellm 自己维护的模型表（``model_prices_and_context_window.json``，本地版 3916 条），
不自己攒——它是上游在维护的，我们攒的那份第二天就旧了。

**但它有个必须绕开的坑**（2026-09-20 实测）：``supports_function_calling`` 查不到的模型
**静默返回 False**，不抛异常（只往 stderr 打一行 "Provider List"）。本仓自己的模型名多半查
不到（``dashscope/qwen3.8-flash`` 查不到、``qwen-plus`` 查不到），拿它当放行门会把正在跑的
主模型判死。所以是**否决门**不是放行门：

- 查得到 且 ``supports_function_calling=False`` → 拒绝（有依据的否决）
- 查得到 且 True → 放行
- **查不到 → 放行**，只记一行 warning（表没收录 ≠ 不支持）

实测到的一条现成情报：``deepseek/deepseek-reasoner`` 在表里标 **False**，而
``deepseek/deepseek-chat`` / ``dashscope/deepseek-v4-flash`` 都是 True。备用选 reasoner 会
得到一个调不了工具的 Agent，这正是门要拦的那种错。

**查询键是 ``provider/model``（``Endpoint.ref``），不是寻址用的那个名字。** ``build_model_list``
里 deployment 写的是 ``openai/<model>``（告诉 litellm 按 OpenAI 兼容协议发），拿那个去查能力
一律查不到。同一个模型两个名字，别混。
"""

import logging

from app.agent.providers import configure_litellm

logger = logging.getLogger("shoppingx.llm.capability")

__all__ = ["degraded_against", "gate_fallback_refs", "supports_tool_calls"]

# 切换时要对给人看的那几项：表里的列名 → 事件里写的名字。工具调用不在这里——它是门的判据，
# 走到上报这一步说明已经放行过了。
_COMPARED = {
    "supports_prompt_cache_breakpoint": "prompt_cache_breakpoint",
    "supports_prompt_caching": "prompt_caching",
    "supports_reasoning": "reasoning",
    "supports_response_schema": "response_schema",
}


def supports_tool_calls(ref: str) -> bool | None:
    """这个 ``provider/model`` 支不支持工具调用。

    ``None`` = **表里没有这条**，不是「不支持」。调用方必须把 None 与 False 分开处理，
    合并就等于把没收录的模型全判死（模块 docstring 里那个坑）。
    """
    try:
        configure_litellm()
        import litellm
        from litellm.utils import supports_function_calling

        if ref not in litellm.model_cost:
            return None
        return bool(supports_function_calling(model=ref))
    except Exception:  # pragma: no cover - 情报源出问题不该拦住模型装配
        logger.debug("能力查询失败：%s", ref, exc_info=True)
        return None


def degraded_against(primary: str, target: str) -> list[str]:
    """从 ``primary`` 切到 ``target`` 少了哪些能力（供 ``model_fallback`` 事件写明降级项）。

    只报「主家有、备家没有」的**确定**差异：两边都得在表里查得到，有一边查不到就不报——
    宁可少说一句，也不要报一个查不到就当没有的假降级。这是 ``None``/``False`` 那条分界
    在上报侧的同一个规矩。
    """
    if primary == target:
        return []
    try:
        configure_litellm()
        import litellm
        from litellm import utils as lu

        if primary not in litellm.model_cost or target not in litellm.model_cost:
            return []
        out = []
        for fn_name, label in _COMPARED.items():
            fn = getattr(lu, fn_name, None)
            if fn is None:  # pragma: no cover - 上游改名时降级成不报，不炸
                continue
            if bool(fn(model=primary)) and not bool(fn(model=target)):
                out.append(label)
        return out
    except Exception:  # pragma: no cover - 上报是附属品
        logger.debug("能力差异比对失败：%s → %s", primary, target, exc_info=True)
        return []


def gate_fallback_refs(refs: list[str]) -> list[str]:
    """过滤 fallback 链，剔除「查得到且不支持工具调用」的目标，返回放行的那些。

    **拒绝为什么不是启动期直接炸**：炸了服务起不来，而 fallback 配错的代价是「没有备用」，
    不是「跑不了」。主出口照常工作时把整个服务拦在门外，是拿小故障换大故障。所以剔除 + 大声
    记日志；整条链全被拒时抬成 ``error``——那等于配了备用却一个都不能用，是要人看见的状态。
    """
    kept: list[str] = []
    for ref in refs:
        verdict = supports_tool_calls(ref)
        if verdict is False:
            logger.error(
                "fallback 目标 %s 被能力门拒绝：litellm 模型表标记它不支持工具调用，"
                "切过去会得到一个调不了工具的 Agent",
                ref,
            )
            continue
        if verdict is None:
            logger.warning(
                "fallback 目标 %s 不在 litellm 模型表里，能力未知，按放行处理（表没收录 ≠ 不支持）",
                ref,
            )
        kept.append(ref)
    if refs and not kept:
        logger.error("fallback 链里 %d 个目标全被能力门拒绝，本次等于没有备用出口", len(refs))
    return kept
