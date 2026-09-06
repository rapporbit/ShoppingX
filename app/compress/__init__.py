"""Cache Breakpoint 上下文压缩与缓存治理（L2 Cache-Aware 微压缩）。

一次 reply = 一条 assistant ``Msg``，轮内所有 tool_call / tool_result 都在它的 content 里，
所以断点与压缩都下沉到 **block 粒度**（见 :mod:`app.compress.blocks`）。把它挂进 Agent 生命周期
的是 ``app/harness/hooks/context_compress.py`` 的 pre_think Hook——控制面统一走 Hook Pipeline。

cache_control 标记不在这一层打（content block 是强类型的，塞不进未知字段），落在
:mod:`app.harness.formatter`——那一层的 dict 序列才是真正发出去的 payload。
"""

from app.compress.blocks import (
    DEFAULT_KEEP_RECENT,
    DEFAULT_MAX_TOOL_TOKENS,
    MAX_CACHE_MARKERS,
    MIN_CACHE_PREFIX_TOKENS,
    compress_blocks_before,
    compute_block_breakpoint,
    post_step_compress,
)

__all__ = [
    "DEFAULT_KEEP_RECENT",
    "DEFAULT_MAX_TOOL_TOKENS",
    "MAX_CACHE_MARKERS",
    "MIN_CACHE_PREFIX_TOKENS",
    "compress_blocks_before",
    "compute_block_breakpoint",
    "post_step_compress",
]
