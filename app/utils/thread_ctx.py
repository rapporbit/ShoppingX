"""``thread_scope``：把 ContextVar 的 set/reset 封装成作用域，离开自动还原。

请求入口要写 thread_id / session_dir，手动 set+reset 重复且易漏 reset。用上下文管理器统一处理：

    async def run_agent(query: str, thread_id: str):
        session_dir = ensure_session_dir(thread_id)
        with thread_scope(thread_id, session_dir):
            await agent(Msg("user", query, "user"))

派 worker 时同样用它覆盖子 thread_id、但传入父 session_dir（产物归同一会话目录）。
"""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from app.api.context import (
    _request_id_var,
    _run_id_var,
    _session_dir_var,
    _thread_id_var,
    _user_id_var,
)
from app.observability.logging import bind_log_context, unbind_log_context


@contextmanager
def thread_scope(
    thread_id: str,
    session_dir: Path,
    user_id: str | None = None,
    run_id: str | None = None,
    request_id: str | None = None,
) -> Iterator[None]:
    """作用域内绑定 thread_id / session_dir（可选 user_id / run_id），离开自动还原到进入前的值。

    ``user_id`` 缺省（None）时**不动** user_id 上下文——fork 子 Agent 只覆盖 thread_id /
    session_dir，user_id 沿用父任务的绑定（子任务仍属同一用户，黑名单/偏好继续生效）。
    ``run_id`` 同理缺省不动：只有 ``run_agent`` 这一个入口知道本轮的 run_id。``request_id``
    则是从队列消息里读回来的（阶段 4-5）——worker 是另一个进程，ContextVar 传不过去，只能
    随消息带；带回来绑上，两个进程的日志才拼得成一条线。

    同一处还把 thread_id / user_id / request_id 绑进 structlog 的日志上下文（A 块）——「请求隔离」
    与「日志上下文传播」共用这一个入口，本作用域内打的结构化日志自动带上这些字段。
    """
    token_t = _thread_id_var.set(thread_id)
    token_s = _session_dir_var.set(session_dir)
    token_u = _user_id_var.set(user_id) if user_id is not None else None
    token_r = _run_id_var.set(run_id) if run_id is not None else None
    token_rq = _request_id_var.set(request_id) if request_id is not None else None
    # 空串不绑：bind_log_context 只跳过 None，绑个空串等于给每条日志加一列没用的空字段。
    log_tokens = bind_log_context(
        thread_id=thread_id, user_id=user_id, request_id=request_id or None
    )
    try:
        yield
    finally:
        _thread_id_var.reset(token_t)
        _session_dir_var.reset(token_s)
        if token_u is not None:
            _user_id_var.reset(token_u)
        if token_r is not None:
            _run_id_var.reset(token_r)
        if token_rq is not None:
            _request_id_var.reset(token_rq)
        unbind_log_context(log_tokens)
