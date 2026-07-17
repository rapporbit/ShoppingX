"""Cache Breakpoint 上下文压缩与缓存治理（L2 Cache-Aware 微压缩）。

公开三件套：定位断点（breakpoint）、压缩较旧区工具结果 + 打缓存标记（compressor）。
编排成纯函数的 ``post_step_compress`` 在 :mod:`app.compress.pipeline`；把它挂进 Agent 生命周期
的是 ``app/harness/hooks/context_compress.py`` 的 pre_think Hook——控制面统一走 Hook Pipeline。
"""

from app.compress.breakpoint import DEFAULT_KEEP_RECENT, compute_breakpoint
from app.compress.compressor import (
    DEFAULT_MAX_TOOL_TOKENS,
    MAX_CACHE_MARKERS,
    MIN_CACHE_PREFIX_TOKENS,
    apply_cache_control,
    compress_after_breakpoint,
    mark_system_cache,
)
from app.compress.pipeline import post_step_compress

__all__ = [
    "DEFAULT_KEEP_RECENT",
    "DEFAULT_MAX_TOOL_TOKENS",
    "MAX_CACHE_MARKERS",
    "MIN_CACHE_PREFIX_TOKENS",
    "apply_cache_control",
    "compress_after_breakpoint",
    "compute_breakpoint",
    "mark_system_cache",
    "post_step_compress",
]
