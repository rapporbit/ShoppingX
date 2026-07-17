"""Harness 的结构化信号源：从可靠数据结构读状态，不做字符串匹配猜测。

Hook 判断「有没有候选」「有没有踩黑名单」时，一律走这里，而不是去 grep 工具返回的文本——
工具返回格式会变，候选登记表和会话级 P_t 不会。

所有函数都是同步、零 IO、异常兜底返回中性值：Hook 里不能因为读信号失败而中断 Agent。
"""

from __future__ import annotations

import logging

logger = logging.getLogger("shoppingx.harness.signals")


def candidate_count() -> int:
    """当前会话候选登记表里的候选总数。读不到（无会话作用域等）返回 0。"""
    try:
        from app.api.context import get_session_dir
        from app.tools._candidates import _REGISTRY

        sd = get_session_dir()
        if sd is None:
            return 0
        return len(_REGISTRY.get(str(sd), {}))
    except Exception:
        logger.debug("candidate_count 读取失败", exc_info=True)
        return 0


def blacklist_terms() -> list[str]:
    """会话级 P_t 里的**硬** dislike 原子词（即黑名单）。

    只取 hard：soft dislike 在 item_picker 里是「减分」语义，出现在结果里不算违规。
    长期库的 dislike 需要 async + Store IO，不适合在每次工具返回后同步跑；会话级 P_t 已由
    curator 从长期偏好合流，覆盖绝大多数场景。
    """
    try:
        from app.api.context import get_session_pt

        pt = get_session_pt()
        if pt is None:
            return []
        return pt.dislike_terms()
    except Exception:
        logger.debug("blacklist_terms 读取失败", exc_info=True)
        return []


def blacklist_hits(text: str) -> list[str]:
    """``text`` 中命中的黑名单词（返回 P_t 原词，保序去重）。无黑名单或无命中返回 ``[]``。

    命中口径必须与执行层同一套：先 :func:`normalize_terms`（中↔英扩词）再 :func:`term_hits`
    （词边界 + 否定修饰）。初版是裸 ``t in lowered``，双向失效——P_t 的中文词（「塑料」）对
    英文结果永远匹不中（信号空转），英文词又无词边界与否定过滤（"plastic-free" 被数成违规）。
    检测层一旦弱于执行层（picker 走的是归一口径），这个兜底就永远兜不到执行层漏掉的东西。
    """
    if not text:
        return []
    terms = blacklist_terms()
    if not terms:
        return []
    lowered = text.lower()
    from app.utils.terms import normalize_terms, term_hits

    return [t for t in terms if t and any(term_hits(v, lowered) for v in normalize_terms([t]))]
