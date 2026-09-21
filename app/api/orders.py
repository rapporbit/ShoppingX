"""交易域接口：订单查询与取消 / 确认卡（下单、决议）/ 商品对比。从 ``server.py`` 拆出。"""

from __future__ import annotations

from typing import Any

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
)
from pydantic import BaseModel

from app.api import (
    monitor,
)
from app.api.auth import (
    get_current_user_id,
)
from app.api.guards import guard_thread, safe_session_dir
from app.tools._candidates import hydrate
from app.tools.present_comparison import compare_items
from app.trade.confirmation import ConfirmationError
from app.trade.confirmations import (
    list_confirmations,
    prepare_cancel_confirmation,
    prepare_order_confirmation,
    resolve_confirmation,
)
from app.trade.order import OrderStateError
from app.trade.repository_sql import confirmation_repository, order_repository
from app.trade.usecases import LineRequest, NoCandidateError, OrderNotFoundError, query_orders
from app.utils.path_utils import (
    OUTPUT_ROOT,
)
from app.utils.thread_ctx import thread_scope

router = APIRouter()


def _require_login(auth_uid: str | None) -> str:
    """订单接口一律要求登录——订单是**归属**数据，没有「匿名的订单」这回事。

    与偏好接口的 ``_assert_own`` 口径不同：那边关掉鉴权后退回「任意读」，因为偏好在关掉鉴权的
    本地开发里还得能看；订单不行——鉴权一关就人人可读所有订单，那不是开发便利，是洞。
    """
    if not auth_uid:
        raise HTTPException(401, "请先登录后查看订单")
    return auth_uid


@router.get("/api/orders")
async def list_orders(
    limit: int = 20, auth_uid: str | None = Depends(get_current_user_id)
) -> dict[str, Any]:
    """当前用户的订单列表（侧栏「我的订单」）。只列自己的——user_id 取自 token，不从查询参数收。"""
    uid = _require_login(auth_uid)
    orders = await query_orders(order_repository(), user_id=uid, limit=limit)
    return {"orders": [o.snapshot() for o in orders], "count": len(orders)}


@router.get("/api/orders/{order_id}")
async def get_order(
    order_id: str, auth_uid: str | None = Depends(get_current_user_id)
) -> dict[str, Any]:
    """单张订单详情。别人的单与不存在的单**回同一个 404**（理由见 usecases._load_owned）。"""
    uid = _require_login(auth_uid)
    try:
        found = await query_orders(order_repository(), user_id=uid, order_id=order_id)
    except OrderNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    return found[0].snapshot()


class CancelOrderBody(BaseModel):
    reason: str
    thread_id: str


@router.post("/api/orders/{order_id}/cancel")
async def cancel_order_endpoint(
    order_id: str,
    body: CancelOrderBody,
    auth_uid: str | None = Depends(get_current_user_id),
) -> dict[str, Any]:
    """从前端为一张订单**生成取消确认卡**（不经 Agent，也不直接取消）。

    取消和下单一样要先出卡、用户再点「确认取消」。这条路**没有**「先 query_order」
    的顺序闸——那道闸拦的是模型编订单号，而前端的取消按钮长在订单卡片上，订单号来自刚渲染的
    那张卡。归属与状态机仍照常校验。
    """
    uid = _require_login(auth_uid)
    await guard_thread(body.thread_id, auth_uid)
    try:
        conf = await prepare_cancel_confirmation(
            confirmation_repository(),
            order_repository(),
            user_id=uid,
            thread_id=body.thread_id,
            order_id=order_id,
            reason=body.reason,
        )
    except OrderNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    except OrderStateError as e:
        raise HTTPException(409, str(e)) from e
    except ConfirmationError as e:
        raise _confirmation_http_error(e) from e
    env = conf.envelope()
    await monitor.report_confirmation("required", env, thread_id=body.thread_id)
    return env


# --- 交易确认卡----------------------------

_CONFIRMATION_STATUS = {
    "unauthorized": 401,
    "not_found": 404,
    "conflict": 409,
    "expired": 410,
    "invalid": 400,
}


def _confirmation_http_error(e: ConfirmationError) -> HTTPException:
    return HTTPException(_CONFIRMATION_STATUS.get(e.code, 400), str(e))


class OrderItemBody(BaseModel):
    item_id: str
    quantity: int = 1


class PrepareOrderBody(BaseModel):
    items: list[OrderItemBody]
    shipping_address: dict[str, Any]


class ResolveConfirmationBody(BaseModel):
    snapshot_hash: str
    approved: bool


def _hydrate_for_thread(thread_id: str, uid: str) -> Any:
    """给 HTTP 入口用的候选 hydrate：进该 thread 的作用域再按 id 取。

    表单点「生成确认单」时任务早已结束、登记表只活一轮（候选不落盘），这里靠 :func:`hydrate`
    自带的「登记表未命中 → 按 id 回源 Qdrant」取回商品与价格。"""
    session_dir = safe_session_dir(OUTPUT_ROOT, thread_id)

    def _hydrate(ids: list[str]) -> list[Any]:
        with thread_scope(thread_id, session_dir, uid):
            return hydrate(ids)

    return _hydrate


@router.post("/api/threads/{thread_id}/confirmations/orders")
async def prepare_order_endpoint(
    thread_id: str,
    body: PrepareOrderBody,
    auth_uid: str | None = Depends(get_current_user_id),
) -> dict[str, Any]:
    """下单意向表单 → 服务端生成确认卡（不经模型、不下单）。商品与价格按 item_id 从本会话候选取。"""
    uid = _require_login(auth_uid)
    await guard_thread(thread_id, auth_uid)
    lines = [LineRequest(item_id=i.item_id, quantity=i.quantity) for i in body.items]
    try:
        conf = await prepare_order_confirmation(
            confirmation_repository(),
            user_id=uid,
            thread_id=thread_id,
            lines=lines,
            shipping_address=body.shipping_address,
            hydrate=_hydrate_for_thread(thread_id, uid),
        )
    except ConfirmationError as e:
        raise _confirmation_http_error(e) from e
    except (NoCandidateError, ValueError) as e:
        raise HTTPException(400, str(e)) from e
    env = conf.envelope()
    await monitor.report_confirmation("required", env, thread_id=thread_id)
    return env


class CompareBody(BaseModel):
    item_ids: list[str]


@router.post("/api/threads/{thread_id}/compare")
async def compare_endpoint(
    thread_id: str,
    body: CompareBody,
    auth_uid: str | None = Depends(get_current_user_id),
) -> dict[str, Any]:
    """对比栏「让 Agent 帮我比一比」：几件商品的逐件优劣 + 推荐哪件，**不走 AgentLoop**。

    **为什么是 REST 而不是发一句话给 Agent**：用户已经亲手勾了这几件并点了按钮，意图百分之百
    确定——再让主环规划一遍，换来的是几十秒往返和「模型可能回一段纯文字、对比表还是填不满」的
    不确定性。这里一次 fast 模型调用就出结构化结果，对比表按 item_id 逐列填。与 ``/api/similar``
    同一个取舍（那条是 0 次 LLM，这条是 1 次）。

    ``present_comparison`` 工具仍在工具面上：用户在对话里说「这几个哪个好」时由模型调，两条入口
    共用 :func:`compare_items`。

    **代价（明确记着）**：这条路的结论不进会话历史，Agent 后续不知道用户看过对比。当前是可接受
    的——对比是「看一眼就决定」的动作，不是需要被后续推理引用的事实；真要接回去，应该由前端把
    结论作为用户消息回发，而不是在这里偷偷写 messages。
    """
    await guard_thread(thread_id, auth_uid)
    session_dir = safe_session_dir(OUTPUT_ROOT, thread_id)
    with thread_scope(thread_id, session_dir, auth_uid):
        out = await compare_items(body.item_ids)
    return out.model_dump()


@router.get("/api/threads/{thread_id}/confirmations")
async def list_confirmations_endpoint(
    thread_id: str,
    limit: int = 20,
    auth_uid: str | None = Depends(get_current_user_id),
) -> dict[str, Any]:
    """本会话的确认记录（真源）。前端打开 / 刷新会话时拉一次，与事件流合并。"""
    uid = _require_login(auth_uid)
    await guard_thread(thread_id, auth_uid)
    try:
        confs = await list_confirmations(
            confirmation_repository(), user_id=uid, thread_id=thread_id, limit=limit
        )
    except ConfirmationError as e:
        raise _confirmation_http_error(e) from e
    return {"confirmations": [c.envelope() for c in confs]}


@router.post("/api/threads/{thread_id}/confirmations/{confirmation_id}/resolve")
async def resolve_confirmation_endpoint(
    thread_id: str,
    confirmation_id: str,
    body: ResolveConfirmationBody,
    auth_uid: str | None = Depends(get_current_user_id),
) -> dict[str, Any]:
    """用户在确认卡上点「确认 / 拒绝」。**唯一**能真正下单 / 取消的入口，模型没有对应工具。"""
    uid = _require_login(auth_uid)
    await guard_thread(thread_id, auth_uid)
    try:
        conf = await resolve_confirmation(
            confirmation_repository(),
            order_repository(),
            user_id=uid,
            thread_id=thread_id,
            confirmation_id=confirmation_id,
            snapshot_hash=body.snapshot_hash,
            approved=body.approved,
        )
    except ConfirmationError as e:
        raise _confirmation_http_error(e) from e
    except OrderNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    except OrderStateError as e:
        raise HTTPException(409, str(e)) from e
    env = conf.envelope()
    await monitor.report_confirmation("resolved", env, thread_id=thread_id)
    return env
