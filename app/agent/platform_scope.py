"""启用平台作用域：本轮任务**允许检索哪些平台**（默认只 amazon）。

为什么要有这层：语料 99.75% 是 amazon，此前主 loop 每次泛搜都无脑 fork 5 个平台，其中 4 个必然
空手而归——白烧 60% 的 token 与一整轮墙钟。fork 机制与教学架构原样保留，只是**不派注定空军的
子任务**：默认单平台（amazon）时主 loop 自己一次 item_search 收敛，用户在前端设置里勾上更多平台
才真的跨平台 fork 比价。

落地口径（与 memory / P_t 同一套路）：
- **不进 system prompt**：启用平台随用户设置而变，混进 system prompt 会打断本该跨轮/跨会话稳定的
  prompt cache 前缀（见 prompts.py 的说明）。它由 ``main_agent._inject_runtime_context`` 拼进当轮
  human message（缓存断点之后），并由本模块的 ContextVar 供工具层机制性执行。
- **prompt 只打动机、机制才是硬保证**（见 fork-guardrails-mechanism-not-prompt）：模型少列 / 多列
  平台都拦不住，故 ``dispatch_tool`` 补齐+丢弃、``item_search`` 的 Qdrant filter 一律以本模块的
  启用集合为准。
- fork 子 Agent 通过 asyncio Task 的 ContextVar 快照自动继承，无需显式传参。
"""

import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar

from app.utils.clean import PLATFORMS

# 兜底默认：只开 amazon（语料现实——其余平台在召回库里近乎空）。可用 env 覆盖（逗号分隔）。
_DEFAULT_ENV = "DEFAULT_ENABLED_PLATFORMS"
_FALLBACK: tuple[str, ...] = ("amazon",)

_enabled_var: ContextVar[tuple[str, ...] | None] = ContextVar(
    "shoppingx_enabled_platforms", default=None
)


def default_platforms() -> tuple[str, ...]:
    """进程级默认启用平台：env ``DEFAULT_ENABLED_PLATFORMS``（逗号分隔）→ 归一 → 兜底 amazon。"""
    return normalize_platforms(os.environ.get(_DEFAULT_ENV, "").split(","))


def normalize_platforms(raw: Sequence[str] | None) -> tuple[str, ...]:
    """把外部传入的平台列表归一：小写去空、只留 ``PLATFORMS`` 里认识的、去重保序；空则回落默认。

    未知平台名（拼错 / 库里没有的 ebay）**直接丢弃**而非报错——前端设置是可选项，不该因一个脏值让
    整个任务 400；丢干净后若一个都不剩，等价于「没选」，回落 amazon。
    """
    seen: list[str] = []
    for p in raw or ():
        key = p.strip().lower()
        if key in PLATFORMS and key not in seen:
            seen.append(key)
    return tuple(seen) or _FALLBACK


def get_enabled_platforms() -> tuple[str, ...]:
    """当前任务启用的平台；无上下文（离线脚本 / 单测直调工具）时取进程级默认。"""
    return _enabled_var.get() or default_platforms()


def is_multi_platform() -> bool:
    """本轮是否真的要跨平台（启用 ≥2 个平台）——单平台时不该 fork、不该跨平台比价。"""
    return len(get_enabled_platforms()) > 1


@contextmanager
def platform_scope(platforms: Sequence[str] | None) -> Iterator[tuple[str, ...]]:
    """在 ``run_agent`` 入口绑定本轮启用平台，离开自动还原（套路同 ``thread_scope``）。"""
    enabled = normalize_platforms(platforms) if platforms else default_platforms()
    token = _enabled_var.set(enabled)
    try:
        yield enabled
    finally:
        _enabled_var.reset(token)


def resolve_search_platforms(platform: str) -> tuple[str, ...]:
    """把 ``item_search(platform=...)`` 的入参**收口到启用集合**，返回实际要搜的平台元组。

    - ``"all"`` → 全部启用平台（**不是**全库：库里还有用户没勾的平台，不该被 all 捞进来）。
    - 具体平台且已启用 → 就搜它。
    - 具体平台但**未启用**（模型幻觉出一个没勾的平台 / 子 Agent 拿到脏 demands）→ 回落到启用集合，
      而不是照搜——用户明确没选它，搜了也是浪费+越权。
    """
    key = (platform or "all").strip().lower()
    enabled = get_enabled_platforms()
    if key != "all" and key in enabled:
        return (key,)
    return enabled
