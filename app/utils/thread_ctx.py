"""``thread_scope``：把 ContextVar 的 set/reset 封装成作用域，离开自动还原。

请求入口与 ``dispatch_tool`` fork 时都要写 thread_id / session_dir，手动 set+reset
重复且易漏 reset。用上下文管理器统一处理：

    async def run_agent(query: str, thread_id: str):
        session_dir = ensure_session_dir(thread_id)
        with thread_scope(thread_id, session_dir):
            await main_agent.ainvoke({"messages": [("user", query)]})

fork 子 Agent 时同样用它覆盖子 thread_id、但传入父 session_dir（产物归同一会话目录）。
"""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from app.api.context import _session_dir_var, _thread_id_var, _user_id_var
from app.observability.logging import bind_log_context, unbind_log_context


@contextmanager
def thread_scope(thread_id: str, session_dir: Path, user_id: str | None = None) -> Iterator[None]:
    """作用域内绑定 thread_id / session_dir（可选 user_id），离开自动还原到进入前的值。

    ``user_id`` 缺省（None）时**不动** user_id 上下文——fork 子 Agent 只覆盖 thread_id /
    session_dir，user_id 沿用父任务的绑定（子任务仍属同一用户，黑名单/偏好继续生效）。

    同一处还把 thread_id / user_id 绑进 structlog 的日志上下文（A 块）——「请求隔离」与「日志
    上下文传播」共用这一个入口，本作用域内打的结构化日志自动带上这些字段。
    """
    token_t = _thread_id_var.set(thread_id)
    token_s = _session_dir_var.set(session_dir)
    token_u = _user_id_var.set(user_id) if user_id is not None else None
    log_tokens = bind_log_context(thread_id=thread_id, user_id=user_id)
    try:
        yield
    finally:
        _thread_id_var.reset(token_t)
        _session_dir_var.reset(token_s)
        if token_u is not None:
            _user_id_var.reset(token_u)
        unbind_log_context(log_tokens)
