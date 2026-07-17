"""安全四层之①：fork 深度上限。

同质子 Agent 的工具集里仍含 dispatch_tool（同质 fork 的递归能力保留），但**正常的单平台检索
子任务不满足 fork 三件事**（refdoc/11：子只做一次 item_search 就返回），不该再 fork 孙——
放任递归会资源指数爆炸。用 ContextVar 记录当前 fork 深度：主 loop 为 0，每 fork 一层 +1；
超过 ``MAX_FORK_DEPTH`` 时 ``enter_fork`` 抛 :class:`ForkLimitExceeded`，由 dispatch_tool
捕获并转成字符串错误回传主 loop（不让 Agent 崩）。

ContextVar 而非全局变量：每个 asyncio 子任务有独立深度副本，并行 fork 互不串扰。
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from app.observability.logging import bind_log_context, unbind_log_context

# 当前 fork 深度。主 loop=0，子 loop=1。
_fork_depth: ContextVar[int] = ContextVar("shoppingx_fork_depth", default=0)

# 深度上限：只允许一层 fork（主→子）。子任务是被收窄的单次检索（见 prompts.yml <fork_protocol>），
# 不该再 fork 孙；第二层被拦。需要更深的子链时再按 YAGNI 调大。
MAX_FORK_DEPTH = 1


class ForkLimitExceeded(Exception):
    """fork 深度超过 ``MAX_FORK_DEPTH`` 时抛出。"""


@contextmanager
def enter_fork() -> Iterator[int]:
    """进入一层 fork 作用域，返回进入后的深度；超限则抛 ForkLimitExceeded。

    离开作用域自动还原深度（即便子任务异常）。
    """
    cur = _fork_depth.get()
    if cur >= MAX_FORK_DEPTH:
        raise ForkLimitExceeded(f"fork 深度已达上限 {MAX_FORK_DEPTH}，拒绝再 fork")
    token = _fork_depth.set(cur + 1)
    # 把 fork_depth 绑进日志上下文（A 块）：子 Agent 内打的结构化日志自带「这是第几层 fork」。
    log_tokens = bind_log_context(fork_depth=cur + 1)
    try:
        yield cur + 1
    finally:
        _fork_depth.reset(token)
        unbind_log_context(log_tokens)


def current_fork_depth() -> int:
    """读取当前 fork 深度（主 loop 为 0）。"""
    return _fork_depth.get()
