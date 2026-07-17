"""把「定位断点 → 压缩较旧区 → 打缓存标记」编排成一个纯函数。

由 ``app/harness/hooks/context_compress.py`` 的 pre_think Hook 在每次请求模型前调用。
"""

from __future__ import annotations

from langchain_core.messages import AnyMessage

from app.compress.breakpoint import DEFAULT_KEEP_RECENT, compute_breakpoint
from app.compress.compressor import (
    DEFAULT_MAX_TOOL_TOKENS,
    apply_cache_control,
    compress_after_breakpoint,
)


def post_step_compress(
    messages: list[AnyMessage],
    *,
    keep_recent: int = DEFAULT_KEEP_RECENT,
    max_tool_tokens: int = DEFAULT_MAX_TOOL_TOKENS,
    enable_cache_control: bool = False,
) -> list[AnyMessage]:
    """返回**模型应看到的**历史视图（不改 state 原文）。

    纯函数、确定性、幂等（同一份 messages 多次调用结果一致）——这是缓存前缀稳定的前提。
    压缩既省 token 又不丢历史：改的只是这一次请求送给模型的视图。
    """
    breakpoint_idx = compute_breakpoint(messages, keep_recent)
    compressed = compress_after_breakpoint(
        messages, breakpoint_idx, max_tool_tokens=max_tool_tokens
    )
    if enable_cache_control:
        compressed = apply_cache_control(compressed, breakpoint_idx)
    return compressed
