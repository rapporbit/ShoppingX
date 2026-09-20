"""交易确认记录（对齐参考项目 globex-agent 的 ``TradeConfirmation``）。

**一张确认卡 = 库里一条记录**，不再是一次性事件。模型工具（``create_order`` / ``cancel_order``）
和前端表单（``POST /api/threads/{id}/confirmations/orders``）产出的是**同一种记录**；决议（同意 /
拒绝）只走 HTTP、由用户在页面上点按钮，模型说「用户确认了」不算数——这就是把「两段式」从
会话级指纹（旧 ``_order_guard``）升级成持久化状态机的理由：刷新页面卡还在、重启进程闸还在、
快照 hash 保证点下去的和看到的是同一张。

状态机：``pending → approved | rejected``，另有派生态 ``expired``（pending 且过了 ``expires_at``，
不落库、按时钟算）。approved 才真调 ``place_order`` / ``cancel_order`` 落订单。
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

CONFIRMATION_TTL = timedelta(minutes=30)

Action = Literal["create", "cancel"]
Status = Literal["pending", "approved", "rejected"]


class DuplicateRequestError(Exception):
    """确认记录撞上唯一键（``request_key`` / ``operation_id``）。

    仓储层把数据库的 ``IntegrityError`` 翻成这个领域异常，用例层才不用 import SQLAlchemy
    （见 :mod:`app.trade.ports` 的 docstring）。它不是错误路径而是**并发路径**：先到的那行
    就是权威，用例接住它重读一次即可。
    """


class ConfirmationError(ValueError):
    """确认记录层的业务错误（找不到 / 已决议 / 过期 / hash 不符）。``code`` 给 HTTP 映射状态码。"""

    def __init__(self, message: str, code: str = "invalid") -> None:
        super().__init__(message)
        self.code = code


def _now() -> datetime:
    return datetime.now(UTC)


def new_confirmation_id() -> str:
    return f"cfm-{uuid.uuid4().hex[:16]}"


def new_operation_id() -> str:
    return f"operation-{uuid.uuid4().hex}"


def request_key(run_id: str, action: str, snapshot: str) -> str | None:
    """写工具的请求级幂等键：``{run_id}:{action}:{快照 hash 前 32 位}``。

    锚点为什么不是计划里写的 ``tool_call_id``：AgentScope 的 ``ToolMiddlewareBase`` 不把
    tool_call 的 id 传给中间件（``harness/adapter.py`` 里那行 ``ctx["tool_call_id"] = ""``
    就是它拿不到的证据），工具层根本没有这个值。同轮同载荷的调用本来就该收敛成同一张卡，
    ``snapshot_hash`` 是能拿到的、等价的锚。

    ``run_id`` 为空（HTTP 表单入口 / 离线脚本 / 单测）返回 ``None``——不参与唯一约束，
    行为与加这道之前一致。
    """
    if not run_id:
        return None
    return f"{run_id}:{action}:{snapshot[:32]}"


def snapshot_hash(action: str, payload: dict[str, Any]) -> str:
    """载荷指纹：前端 resolve 时带回来，服务端比对，保证「点的就是看到的那张」。"""
    canon = json.dumps({"action": action, "payload": payload}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class Confirmation:
    """确认记录。``payload`` 结构：

    - action=create：``items``（platform / item_id / title / unit_price_minor / currency /
      quantity / landed_usd）、``shipping_address``（recipient_name / country / state / city /
      address_line / postal_code / phone）、``total_amount_minor``、``currency``、
      ``amount_scope="merchandise_only"``、``order_kind="purchase_intent"``。
    - action=cancel：``order_id``、``reason``、``items``（同上，取自订单行）、
      ``total_amount_minor``、``currency``。

    ``result`` 只在 approved 后有：``order_id`` / ``status`` / ``total_amount_minor`` /
    ``currency``。
    """

    confirmation_id: str
    operation_id: str
    user_id: str
    thread_id: str
    action: Action
    payload: dict[str, Any]
    snapshot_hash: str
    expires_at: datetime
    status: Status = "pending"
    result: dict[str, Any] | None = None
    created_at: datetime = field(default_factory=_now)
    resolved_at: datetime | None = None
    #: 请求级幂等键（见 :func:`request_key`）。None = 不参与唯一约束的那条老路。
    request_key: str | None = None

    def expired(self, now: datetime | None = None) -> bool:
        return self.status == "pending" and (now or _now()) >= self.expires_at

    def envelope(self) -> dict[str, Any]:
        """给前端 / 事件 / 工具结果的纯数据视图。字段名与参考项目一致，前端校验器照它写。"""
        return {
            "confirmation_id": self.confirmation_id,
            "operation_id": self.operation_id,
            "buyer_id": self.user_id,
            "session_id": self.thread_id,
            "action": self.action,
            "status": self.status,
            "payload": self.payload,
            "snapshot_hash": self.snapshot_hash,
            "expires_at": self.expires_at.isoformat(),
            "expired": self.expired(),
            "result": self.result,
            "created_at": self.created_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
        }
