"""结构化日志（A 块）——每条日志自动带 thread_id / user_id / fork_depth。

**为什么。** 一次 query 会 fork 多个子 Agent 并发跑，普通 ``print`` / stdlib logging 打出来的
日志混在一起、分不清哪条属于哪个任务 / 哪个子 Agent。structlog 把日志变成「带字段的事件」，并
通过 **contextvars** 自动给每条日志附上当前任务的身份——定位某个 thread 的问题时按字段一筛即可。

**复用 ContextVar 体系（一套机制两用）。** 不另搞一套传播：在 ``thread_scope`` 绑定 thread_id /
user_id 的同时，顺手用 structlog 的 contextvars 绑定同样的字段（fork_depth 在 ``enter_fork`` 绑）。
于是「请求上下文隔离」和「日志上下文传播」走的是同一处入口——这正是 A 块的核心叙事：
**ContextVar 既做 thread 隔离、又做日志上下文传播。**

**渐进迁移（诚实标注）。** 本块把 structlog 基础设施搭好、并让新代码 ``get_logger`` 即享上下文；
存量的少数 stdlib ``logging.getLogger`` 调用未强制全量替换——它们照常工作，只是暂不带结构化字段，
后续逐步迁移即可（不为一次性重写徒增改动面与风险）。
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import structlog

from app.security.log_sanitizer import sanitize_log_processor
from app.utils.env import env_bool

_configured = False


def configure_logging() -> None:
    """配置 structlog（幂等）。在服务启动（lifespan）时调一次。

    ``LOG_JSON=true`` 走 JSON 渲染（生产 / 给日志系统采集）；否则走带颜色的 console 渲染
    （开发可读）。``merge_contextvars`` 把作用域绑定的字段（thread_id 等）并进每条日志。
    """
    global _configured
    if _configured:
        return
    json_logs = env_bool("LOG_JSON", False)
    renderer: Any = (
        structlog.processors.JSONRenderer() if json_logs else structlog.dev.ConsoleRenderer()
    )
    # 脱敏排在 merge_contextvars 之后：contextvars 注入的 user_id 也得一起哈希，否则每条日志的
    # 上下文字段里都躺着明文 user_id（refdocs 16-6 §5）。开发时可设 LOG_SANITIZE=false 看原文。
    processors: list[Any] = [structlog.contextvars.merge_contextvars]
    if env_bool("LOG_SANITIZE", True):
        processors.append(sanitize_log_processor)
    structlog.configure(
        processors=[
            *processors,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str | None = None) -> Any:
    """取一个 structlog logger；通过它打的日志自动带当前任务上下文字段。"""
    return structlog.get_logger(name)


def bind_log_context(**fields: Any) -> Mapping[str, Any]:
    """把字段绑定进 structlog 的 contextvars（None 值跳过），返回 tokens 供还原。

    在 ``thread_scope`` / ``enter_fork`` 这类作用域入口调用，离开时用
    :func:`unbind_log_context` 还原。"""
    clean = {k: v for k, v in fields.items() if v is not None}
    return structlog.contextvars.bind_contextvars(**clean)


def unbind_log_context(tokens: Mapping[str, Any]) -> None:
    """还原 :func:`bind_log_context` 绑定的字段（作用域离开时调）。"""
    structlog.contextvars.reset_contextvars(**tokens)
