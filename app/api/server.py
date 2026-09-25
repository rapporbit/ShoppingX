"""FastAPI 服务 —— 把主 AgentLoop 暴露给浏览器，落地 M10 前后端闭环。

orchestrator 已经把 ``run_agent(query, thread_id, user_id)`` 跑通——本模块只补一层「对外
接口」，让用户在浏览器里发起任务、实时看事件流、取消、下载产物。

本模块只留**任务 / WS / 取消 / 澄清 / 历史 / 健康检查**这条主线；其余按领域拆成 router：
``files``（下载 / 上传）、``preferences``（长期记忆 / 会话约束 / 收藏 / 找相似）、``orders``
（订单 / 确认卡 / 对比）、``accounts``、``admin``、``skills``；共用守卫在 ``guards``。主要口子：

==============================  ============================================
接口                            解决什么
==============================  ============================================
``POST /api/task``              启动一次主 AgentLoop（后台跑），立即返回 thread_id
``POST /api/task/async``        异步提交：入队即返回 task_id，无 WS 的调用方用
``GET  /api/task/{task_id}``    查异步任务的状态 / 结果（轮询）
``WS   /ws/{thread_id}``        订阅该 thread 的 AGUI 事件流（长连接）
``POST /api/task/{tid}/cancel`` 用户主动取消长任务
``GET  /api/files/{tid}/{name}``下载本次会话产物（在 ``files`` router）
``POST /api/upload``            上传参考图到本次会话目录（在 ``files`` router）
``GET  /api/preferences/{uid}`` 读用户长期偏好（在 ``preferences`` router）
``GET  /api/history/{tid}``     读该 thread 的逐轮对话（前端回看 / 续聊，同 thread 复用即接上文）
==============================  ============================================

**事件不丢的关键约定（对原方案的主动更正）：**
原方案的流程是「POST 起任务 → 返回 tid → 前端再连 WS」，但 ``run_agent`` 一上来就上报
``session_created`` 等早期事件——任务已经在 ``create_task`` 里开跑，而 WS 还没连上，这些
早期事件会因「该 thread 无连接」被丢掉（只剩日志）。本实现改成 **connect-first**：前端先
本地生成 ``thread_id`` → 连 WS → 收到 ``ws_ready`` 确认连接已登记 → 才 POST 起任务。
``TaskRequest.thread_id`` 支持客户端指定，正是为此。这样 0 缓冲、0 改 M8 的 ConnectionManager
就把竞态关死，比在连接层堆事件缓冲更简单可靠。

**安全上做了什么**（公网上线后逐项补上的）：``safe_join``
防路径穿越、上传大小上限 + magic bytes 类型白名单、user_id 文件名净化、JWT 鉴权（I 块）+ thread
维度归属校验（M16）、认证限流（:mod:`app.api.ratelimit`）、CORS 白名单。仍未做：多租户数据面隔离。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field, replace
from typing import Any

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.agent.tracing import enqueue_span, parse_traceparent
from app.api import (
    accounts,
    admin,
    backplane,
    clarification,
    control,
    dedup,
    event_log,
    files,
    monitor,
    orders,
    preferences,
    skills,
)
from app.api.admin import dev_admin_username
from app.api.auth import (
    auth_enabled,
    create_access_token,
    decode_token,
    dev_token_enabled,
    get_current_user_id,
    resolve_identity,
    validate_auth_config,
)
from app.api.concurrency import (
    TASK_RETRY_AFTER_SEC,
    classify_request,
    estimated_wait_seconds,
)
from app.api.context import _request_id_var, new_request_id
from app.api.guards import guard_thread, safe_session_dir
from app.config import store as config_store
from app.db.accounts import MIN_PASSWORD_LEN, assert_owner, claim_thread, ensure_dev_admin
from app.db.holds import REASON_CONCURRENCY, HoldResult, acquire_hold, release
from app.db.quota import disabled_status as _disabled_quota
from app.db.quota import get_quota, quota_enabled
from app.db.runs import RunClaim, claim_thread_and_run, release_thread_run
from app.db.session import init_db, session_factory
from app.deployment import assert_deployment_deps
from app.memory.history import count_turns, read_turns
from app.observability import alerts, metrics
from app.observability.logging import bind_log_context, configure_logging, unbind_log_context
from app.queue import (
    TERMINAL_STATES,
    IntentTask,
    TaskQueue,
    TaskStatus,
    get_task_queue,
)
from app.recall.semantic_cache import turn_cache_status
from app.utils.env import env_int
from app.utils.path_utils import (
    OUTPUT_ROOT,
    safe_join,
)
from app.utils.tokens import warm_tokenizer
from app.worker import WORKER_CONCURRENCY

logger = logging.getLogger("shoppingx.server")

# ``run_agent`` 在模块级 import 进来（而不是每次调用现取）：测试大量
# ``monkeypatch.setattr(server, "run_agent", …)`` 靠的就是「它是本模块的一个名字」这点。

# ── 队列的三个常数 ──
#
# 准入池的 429 守的是「本进程同时跑几个 AgentLoop」。任务交给 worker 之后本进程一个 loop 都不跑，
# 那道闸就失效了——**必须换一道**，否则「削峰」会悄悄退化成无界堆积：队列看着能收，用户却在等一个
# 永远排不到的位置。所以队列模式下改用队列深度做背压，超了照样 429 + Retry-After。
QUEUE_MAX_DEPTH = env_int("QUEUE_MAX_DEPTH", 200)
# API 侧「等结果」协程的轮询间隔（秒）。它只是为了让 /inflight、取消口、幂等第 1 层的形状不变，
# 真正的实时性走 WS 事件，不需要秒级轮询。
QUEUE_POLL_SECONDS = env_int("QUEUE_POLL_MS", 1000) / 1000
# 等结果的上限。worker 整批挂掉时，API 侧的 waiter 不能就这么挂着——active_tasks 里留一条永不退休的
# 记录，会让同 thread 同 query 永远被幂等第 1 层判成 already_running。
QUEUE_WAIT_TIMEOUT_SEC = env_int("QUEUE_WAIT_TIMEOUT_SEC", 1800)
# 等「开始跑」的上限（阶段 4-1）。上面那条守的是「跑起来了但永远不收尾」，这条守的是**根本没人领**
# ——worker 整批挂了、或队列深度远超消费能力。两者的处置必须不同：那条只能报错认栽（任务可能真在跑，
# 作废它就是双跑），这条能连消息一起作废，因为「一直是 queued」本身就说明没有任何 worker 碰过它。
#
# 60s 不是拍的：正常排队等的是前面几条任务，estimated_wait_seconds 按 WORKER_CONCURRENCY 摊完通常
# 在几十秒内；真等过一分钟还没人领，多半不是忙而是没人在了。设 0 关掉这道闸。
QUEUE_START_TIMEOUT_SEC = env_int("QUEUE_START_TIMEOUT_SEC", 60)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """启动时预热 Qwen 本地分词器：首次加载约 3.3s，吃在 boot 而非用户请求里（见 utils/tokens.py）。

    用 ``to_thread`` 不阻塞事件循环；失败（未装 tokenizer extra）会降级为 CJK 启发式，不影响起服。

    并起 RT 告警的后台轮询 task——``/metrics`` 是被动的（Prometheus 不来拉就没人知道 P95 涨了），
    告警器是主动的那一半。shutdown 时 cancel 并等它退出，不留悬挂 task。
    """
    configure_logging()  # A 块：启用 structlog（带 thread_id/user_id 上下文）
    validate_auth_config()  # 开了鉴权却没配密钥 → 启动即 fail-fast，不拖到每请求 500
    await assert_deployment_deps()  # 阶段 1 条 7：库不是 MySQL / Redis 不通 → 起服即拒
    await init_db()  # M16：建 users / threads 两张表（幂等，已存在则跳过）
    # 后台管理页面改过的参数：库 → env → 各模块 _load_params()。必须在预热与建 agent 之前，
    # 否则本次启动的第一批任务会用着旧值跑（脏数据不会让它抛，见 store.load_into_memory）。
    await config_store.load_into_memory()
    if auth_enabled():
        logger.info("JWT 鉴权已开启：user_id 一律取 token 的 sub，忽略前端传入")
        # 本地调试默认管理员（DEV_ADMIN_USERNAME/PASSWORD 都配才建）：省掉每次重建库都要注册一遍。
        if dev_admin := dev_admin_username():
            password = os.getenv("DEV_ADMIN_PASSWORD", "")
            if len(password) < MIN_PASSWORD_LEN:
                logger.warning("DEV_ADMIN_PASSWORD 短于 %d 位，跳过建默认管理员", MIN_PASSWORD_LEN)
            else:
                async with session_factory()() as db:
                    created = await ensure_dev_admin(db, dev_admin, password)
                logger.warning(
                    "本地默认管理员 %s %s——线上 .env 务必不配 DEV_ADMIN_*",
                    dev_admin,
                    "已新建" if created else "已存在",
                )
        if dev_token_enabled():
            logger.warning("开发态发证口 /api/auth/token 已开启（AUTH_DEV_TOKEN）——生产务必关闭")
    else:
        logger.warning("JWT 鉴权未开启（AUTH_ENABLED=false）：user_id 信前端传入，存在越权读风险")
    ok = await asyncio.to_thread(warm_tokenizer)
    logger.info(
        "分词器预热%s（%s）",
        "成功" if ok else "降级为启发式",
        "Qwen 本地" if ok else "无 tokenizer",
    )

    # 参数覆盖对账（阶段 1 条 8）：库是唯一真相，本进程每 30s 跟进一次。worker 那边起的是同一个
    # 循环——后台改参数只打在 API 进程上，不对账的话 AgentLoop 所在的 worker 永远用着旧值。
    config_sync_task = asyncio.create_task(config_store.sync_loop())

    alert_task: asyncio.Task[None] | None = None
    if alerts.alerts_enabled():
        alert_task = asyncio.create_task(alerts.alert_loop())
        logger.info("工具 RT 告警轮询已启动")

    # 事件背板：订阅 Redis Pub/Sub，把**别的进程**（worker）发的 AGUI 事件转发给挂在本进程的
    # WebSocket。不订阅的话前端一条实时事件都收不到——任务在 worker 里跑，事件也发在那边。
    event_backplane = await backplane.start_forwarding(monitor.get_connection_manager())

    try:
        yield
    finally:
        if event_backplane is not None:
            await event_backplane.stop()
        await control.close_control_bus()  # 只关已经建出来的那个（发布端是懒加载的）
        config_sync_task.cancel()
        with suppress(asyncio.CancelledError):
            await config_sync_task
        if alert_task is not None:
            alert_task.cancel()
            with suppress(asyncio.CancelledError):
                await alert_task


app = FastAPI(title="ShoppingX Agent API", lifespan=lifespan)

# CORS 白名单。开发期前端（Vite :5173）与后端（:8000）不同源，故默认放开本机那几个源；生产用
# ``ALLOWED_ORIGINS=https://shopx.oiuu.de`` 收敛到自己的域名。
#
# **为什么不能继续留 `*`。** token 在 header 而非 cookie，`*` 不至于让别人直接冒用身份（这也是它
# 之前没出事的原因），但它等于公开授权任何网站的 JS 拿着**用户自己的 token** 调这套 API——别人做个
# 页面挂起来，用户的 credit 额度、LLM 账单都替他烧。配额是按人限的，人被借用了，闸就白设。
_ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=False,  # 身份走 Authorization header，不用 cookie
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _bind_request_id(request: Request, call_next: Any) -> Any:
    """给每个 HTTP 请求绑一个 request_id，并在响应头回显（阶段 4-5）。

    入口在中间件而不是各 endpoint 里：``/api/task`` 的三条幂等分支（already_running /
    duplicate / 429）各自提前返回，逐个手绑迟早漏一条——而恰恰是这几条最需要在日志里被找到
    （「用户点了三次，为什么只跑了一次」）。

    ``X-Request-Id`` 有就沿用：前面挡着 nginx 时那才是真正的请求入口，自己再生成一个等于在
    链路中间断一次。回显是给排障的人用的——出问题时截图里就带着这个 id。
    """
    incoming = (request.headers.get("X-Request-Id") or "").strip()[:64]
    request_id = incoming or new_request_id()
    token = _request_id_var.set(request_id)
    log_tokens = bind_log_context(request_id=request_id)
    try:
        response = await call_next(request)
    finally:
        unbind_log_context(log_tokens)
        _request_id_var.reset(token)
    response.headers["X-Request-Id"] = request_id
    return response


app.include_router(accounts.router)  # M16：注册 / 登录 / 我是谁 / 我的会话清单
app.include_router(admin.router)  # 后台管理：热更新模型档位 / 检索 / 展示参数
app.include_router(skills.router)  # 买家个人 Skill CRUD + 目录
app.include_router(files.router)  # 产物下载 / 参考图上传
app.include_router(preferences.router)  # 长期记忆 / 会话约束 / 收藏 / 找相似
app.include_router(orders.router)  # 订单 / 确认卡 / 商品对比


@dataclass
class TaskHandle:
    """一个活跃后台任务的句柄：``task``（本进程的影子协程）+ 发起它的 ``query`` 原文。

    比单存 ``asyncio.Task`` 多记一个 query，是为了 ``/inflight`` 能在前端刷新 / 切回对话、本地
    已无该轮上下文时，仍把「正在跑的那一轮」的提问原文回吐给前端重建（query 不落任何持久层，
    只随这个进程内句柄活着——任务一结束句柄即摘除，自然回收）。

    **它不再是「谁在跑」的答案**（阶段 1-2 起真相在 ``threads`` 表，1 条 7 起 loop 只在 worker 里
    跑）：这里的 ``task`` 是 API 侧等结果的影子协程，只服务取消口、``/inflight`` 与事件转发。
    """

    task: asyncio.Task[Any]
    query: str
    # 本轮参考图的文件名。和 query 一样是「正在跑那一轮」的提问内容，故一并随句柄活着：
    # 少了它，用户传图后一刷新，图会先消失、等任务收尾落库才又冒出来——一次没必要的闪烁。
    images: list[str] = field(default_factory=list)
    # 队列模式下这一轮在队列里的 id（单进程模式恒为 None）。取消口要靠它把指令精确送到 worker：
    # 按 thread 打取消标记会误伤覆盖重发时紧接着入队的**新**任务，见 app/api/control.py 的第 2 点。
    task_id: str | None = None


# thread_id → 正在跑的后台任务句柄。用于取消 / 防重 / 续看（同 thread 只留一个活跃任务）。
active_tasks: dict[str, TaskHandle] = {}

# 终结类事件：流里出现它即表示「这一轮已收尾」。/inflight 据此把「task_result 已进流但 _runner
# 的 finally 还没把任务摘出 active_tasks」的瞬时窗口判为「已结束」，避免与刚落盘的历史轮重出一份。
_TERMINAL_EVENTS = {"task_result", "task_cancelled", "error"}


class TaskRequest(BaseModel):
    """``POST /api/task`` 的请求体。

    ``thread_id`` 可由客户端预先生成并先连 WS（connect-first，见模块 docstring），
    不传则服务端兜底生成一个。
    """

    query: str
    thread_id: str | None = None
    user_id: str | None = None
    # 本轮启用的平台（前端设置面板勾选）。不传 / 空 → 服务端默认（amazon 单平台，见
    # agent.platform_scope）：语料 99.75% 是 amazon，默认不派注定空军的跨平台 fork；用户主动勾多个
    # 平台才真的跨平台比价。未知平台名在 normalize_platforms 里静默丢弃，不 400。
    platforms: list[str] | None = None
    # 本轮参考图的文件名（M20 图搜）：先 POST /api/upload 拿到 filename，再随任务带上来。
    # 只传文件名不传内容——图已在服务端 uploaded/<thread_id>/ 下，image_understand 工具自己去读。
    image_paths: list[str] | None = None
    # 输入框 ``/`` 显式选中的 skill 目录名（``my/<name>`` 个人 / 内置名）。服务端校验归属后把正文
    # 拼进本轮用户消息；找不到直接报错，不静默降级成普通搜索（见 orchestrator）。
    skill: str | None = None


class TokenRequest(BaseModel):
    """``POST /api/auth/token``（开发态发证）请求体：只认 user_id，不验密码。"""

    user_id: str


# --- 鉴权：开发态发证口（I 块）----------------------------------------------


@app.post("/api/auth/token")
async def issue_token(req: TokenRequest) -> dict[str, str]:
    """签发一个 ``sub=user_id`` 的 JWT，供前端 / 测试拿去当 ``Authorization: Bearer``。

    **demo 边界（诚实标注）：** 这是**开发态发证**——只认 user_id、**不验密码**，本质是个「冒名
    工厂」，仅为让鉴权链路能端到端被验证。真实密码登录 / OAuth / 刷新令牌不在范围内。**它单独由 ``AUTH_DEV_TOKEN`` 把守**（不与 ``AUTH_ENABLED`` 共开
    关），否则「开了鉴权」会反手暴露这个工厂、把刚堵的越权洞捅开。未开发证口（默认）一律 404。
    """
    if not (auth_enabled() and dev_token_enabled()):
        raise HTTPException(404, "发证口未开启（需 AUTH_ENABLED=true 且 AUTH_DEV_TOKEN=true）")
    if not req.user_id.strip():
        raise HTTPException(400, "user_id 不能为空")
    token = create_access_token(req.user_id.strip())
    return {"access_token": token, "token_type": "bearer"}


# --- 启动任务 ---------------------------------------------------------------


async def _enforce_quota(user_id: str | None) -> None:
    """credit 配额闸（M18）：今日额度用尽 → 402，连任务都不给起。

    放在最前（早于归属登记 / 占槽 / 指纹 / 入队）：额度不够的人不该在系统里留下任何足迹——不该认领
    thread、不该占并发槽、更不该在队列里排。402 Payment Required 是这里语义最准的码：不是没权限
    （403，他登录了且这是他自己的会话），也不是限速（429，等一会儿并不会好转），而是「这个周期的
    额度已经花完了」。detail 里带上限 / 已用 / 重置时刻，前端直接拿去展示。
    """
    if not (quota_enabled() and user_id):
        return
    async with session_factory()() as db:
        quota = await get_quota(db, user_id)
    if quota.exhausted:
        metrics.record_task_rejected("quota_exhausted")
        logger.info("配额耗尽，拒绝任务：user=%s period=%s", user_id, quota.period)
        raise HTTPException(402, detail={"error": "quota_exhausted", **quota.as_dict()})


async def _release_run_shielded(thread_id: str, run_id: str) -> None:
    """在收尾 ``finally`` 里清 ``threads`` 上的「在跑」标记，**取消也要清干净**。

    直接 ``await`` 不行：协程正在被取消时，finally 里的第一个挂起点会再吃一次 ``CancelledError``，
    那条 UPDATE 就发不出去，thread 会一直「正忙」到 ``THREAD_STALE_RUN_SEC`` 过期。shield 一层让
    它跑完（同 ``_report_cancel_if_queued`` / ``session_io.charge_quota`` 的手法）；这里吞掉的是
    shield 自己抛回来的那次取消，不影响正在传播的那个。
    """
    with suppress(asyncio.CancelledError, Exception):
        await asyncio.shield(asyncio.create_task(release_thread_run(thread_id, run_id)))


async def _rollback_claim(
    run_id: str, thread_id: str, *, user_id: str | None = None, query: str | None = None
) -> None:
    """任务最终没起来时，把进门占下的东西**原路还回去**：预扣的额度、``threads`` 上的「在跑」
    位置、（给了 query 时）Redis 里的那枚指纹。

    三样都得还，且顺序无关紧要——它们互不依赖。漏还任何一样的症状都是「用户被自己刚才那次失败
    挡住」：额度白占到 TTL、thread 再也发不出新任务、同一句话 5 秒内重试被判重复。
    """
    await release(run_id)
    await release_thread_run(thread_id, run_id)
    if query is not None:
        await dedup.forget(user_id, query)


async def _acquire_hold_or_reject(
    *, run_id: str, user_id: str | None, thread_id: str, kind: str
) -> HoldResult:
    """credit 预授权 + 用户级并发上限（阶段 1-1）：过了才准进门，见 :mod:`app.db.holds`。

    **为什么它不能并进上面那道 ``_enforce_quota``。** 那道闸读的是**事后账本**，同一个人并发发 20
    条时每条都读到「还剩很多」，全部放行。这里在进门时就把「打算花的」占住，后到的请求看见的余额
    已经扣过还在跑的那些——两道闸挡的是不同的东西，前者挡「今天花完了」，后者挡「同时开太多」。

    **两种拒绝码不一样**：额度耗尽 402（等一会儿也不会好转，要等日切），并发超限 429 +
    ``Retry-After``（前面那几个跑完就能进，重试是对的）。
    """
    result = await acquire_hold(run_id=run_id, user_id=user_id, thread_id=thread_id, kind=kind)
    if result.ok:
        return result
    metrics.record_task_rejected(result.reason)
    logger.info(
        "预授权拒绝：user=%s reason=%s active=%d", user_id, result.reason, result.active_runs
    )
    if result.reason == REASON_CONCURRENCY:
        raise HTTPException(
            429,
            detail=result.as_dict(),
            headers={"Retry-After": str(TASK_RETRY_AFTER_SEC)},
        )
    raise HTTPException(402, detail=result.as_dict())


async def _claim_thread_if_needed(thread_id: str, user_id: str | None, query: str) -> None:
    """归属登记（M16）：首轮把 thread 记到本人名下，后续轮顶新 updated_at（侧栏据此排序）。

    ``claim_thread`` 自带属主校验——拿别人的 thread_id 发消息会被它拒，否则「用他的 tid 说句话」
    就成了把他的会话过户到自己名下。

    **鉴权关闭时也登记一行**（阶段 1-2，归属写空身份、不查 users 表）：这行是幂等第 1 层的载体
    （:mod:`app.db.runs` 的条件更新落在它上面），没有行就没有真相。名字里的 ``if_needed`` 现在
    只剩「按需校验归属」这层意思。
    """
    authed = auth_enabled() and bool(user_id)
    async with session_factory()() as db:
        try:
            await claim_thread(db, thread_id, user_id or "", query, verify_user=authed)
        except PermissionError as exc:
            raise HTTPException(403, "无权访问该会话") from exc
        except LookupError as exc:  # token 合法但用户已不存在 → 让他重新登录
            raise HTTPException(401, "凭证已失效，请重新登录") from exc


async def _claim_thread_and_run_or_reject(
    thread_id: str, run_id: str, user_id: str | None, query: str
) -> RunClaim:
    """起任务的归属登记 + 幂等第 1 层（合并版，见 :func:`app.db.runs.claim_thread_and_run`）。

    它排在预扣之后（预扣必须在幂等判定之前，理由见 create_task），所以越权 / 凭证失效时要先把
    刚占的那笔还掉再报错，否则那笔额度要挂到 TTL。
    """
    authed = auth_enabled() and bool(user_id)
    try:
        return await claim_thread_and_run(
            thread_id, run_id, query, user_id=user_id or "", verify_user=authed
        )
    except PermissionError as exc:
        await release(run_id)
        raise HTTPException(403, "无权访问该会话") from exc
    except LookupError as exc:  # token 合法但用户已不存在 → 让他重新登录
        await release(run_id)
        raise HTTPException(401, "凭证已失效，请重新登录") from exc


async def _history_turns(thread_id: str) -> int:
    """该 thread 已积累的历史轮数（分类器的输入）。读不到一律按 0 算 → normal 池。

    正文进库后这里是一次 DB 查询（原先是同步读 turns.json），**必须在 endpoint 的无 await 区间
    之前就取好值**——``try_reserve`` 那段的原子性靠「一个 await 都没有」保证，把 await 挪进去就
    等于给准入判定开了个竞态窗口。轮数只是分池的输入（normal / heavy），早读一步不影响正确性。
    """
    try:
        return await count_turns(thread_id, safe_join(OUTPUT_ROOT, thread_id))
    except Exception:
        return 0


async def _admit_or_429(run_id: str) -> int:
    """队列模式的背压闸：待消费 + 已准入未入队到 :data:`QUEUE_MAX_DEPTH` 就 429（理由见该常数）。

    **放在 handler 最前**：被拒的请求不该先付归属登记、读历史轮数这些库操作（挪前之前 100 并发下
    429 P95 541ms，与放行的请求一样慢）。判定与登记是一步（``TaskQueue.admit``），所以挪前不会
    放大超卖。通过后由调用方负责 ``release_admission``：入队后撤，或没走到入队时撤。

    Redis 不可达时放行（口径同 ``depth()`` 取不到按 0）：真不可达的话后面的入队会失败并如实报错。
    """
    try:
        depth = await get_task_queue().admit(run_id, QUEUE_MAX_DEPTH)
    except Exception:
        logger.warning("背压闸读取失败，本次放行", exc_info=True)
        return 0
    if depth is None:
        metrics.record_task_rejected("queue_full")
        raise HTTPException(
            429,
            f"服务繁忙：队列已满（上限 {QUEUE_MAX_DEPTH}），请 {TASK_RETRY_AFTER_SEC}s 后重试",
            headers={"Retry-After": str(TASK_RETRY_AFTER_SEC)},
        )
    return depth


async def _release_admission(run_id: str) -> None:
    """撤准入登记；失败只记日志——登记自带过期（``QUEUE_ADMIT_TTL_MS``），漏撤只是晚一会儿还名额。"""
    try:
        await get_task_queue().release_admission(run_id)
    except Exception:
        logger.warning("撤准入登记失败：%s", run_id, exc_info=True)


async def _enqueue_intent(intent: IntentTask, depth: int) -> IntentTask:
    """先落 ``queued`` 状态、再入队；返回带上 traceparent 的那份任务。

    顺序反过来会**把状态倒退**：worker 可能已经领走并置成 running，我们随后写的 queued 会盖掉它，
    轮询方看见任务从「跑着」变回「排队」。先写则 worker 只会把状态往前推。

    整段包在入队 span 里（trace 的根），traceparent 随消息走，worker 那边接上。这段期间的日志绑
    trace_id——与 worker 侧同一个值，按它一筛就是两个进程的整条线。
    """
    queue = get_task_queue()
    with enqueue_span(
        task_id=intent.task_id,
        session_id=intent.thread_id,
        user_id=intent.user_id,
        kind=intent.kind,
    ) as parent:
        intent = replace(intent, traceparent=parent.header())
        log_tokens = bind_log_context(trace_id=parent.trace_id)
        try:
            await queue.set_status(
                TaskStatus(
                    task_id=intent.task_id,
                    state="queued",
                    thread_id=intent.thread_id,
                    queue_depth=depth,
                )
            )
            await queue.enqueue(intent)
        finally:
            unbind_log_context(log_tokens)
    return intent


async def _report_cancel_if_queued(task_id: str, thread_id: str) -> None:
    """任务被取消时，若它**还没被 worker 领走**就由 API 侧补一条 ``task_cancelled``。

    判据是状态表里仍写着 ``queued``。已经在跑的那些由 worker 进程的 ``run_agent`` 发（那条路还
    连着记账与产物清理），两边都发就成了重复事件。极窄的竞态（worker 刚领走、``running`` 还没落
    库）下会多发一条——前端按 thread 收尾，重复一条是可接受的，漏发一条是永远转圈。
    """
    status = await get_task_queue().get_status(task_id)
    if status is None or status.state == "queued":
        await monitor.report_task_cancelled(thread_id=thread_id)


async def _abandon_queued(intent: IntentTask, queue: TaskQueue) -> bool:
    """排队等太久还没人领：作废这条任务，把进门占下的东西原路还回。返回「是否真的作废了」。

    **先落标记、再复查状态**，顺序不能反。反过来有个窗口：复查时还是 ``queued``，落标记之前 worker
    把它领走并跑过了自查，标记就成了没人看的废纸——而我们这边已经按作废收了尾（用户看见超时、预扣
    还了），worker 那头照跑不误，一轮 run 白烧还没人认账。先落标记则 worker 只剩两种下场：自查在
    标记之后 → 看见标记、跳过；自查在标记之前 → 它必然已把状态推过 ``queued``，复查读得到，我们撤
    回标记继续等。

    剩下的窄窗口是「worker 已过自查、``running`` 还没落库」那几个 await。这一刻两边都以为自己说了
    算，用户会看到超时而任务其实在跑。收窄到这就够了——代价是一条多余的 ``queue_timeout`` 事件，
    不是重复下单（写边界另有确认卡与 run_id 幂等键守着）。

    没有控制面（单进程模式 / Redis 不可用）时**不作废**：标记落不下去就没人拦得住 worker 将来领走
    它，此时报超时等于骗用户——他看到失败去重发，而十分钟后那条老消息又跑了一遍。
    """
    if not await control.mark_cancel_only(intent.task_id):
        return False
    status = await queue.get_status(intent.task_id)
    if status is not None and status.state != "queued":
        await control.clear_cancel_mark(intent.task_id)
        return False
    logger.warning(
        "排队超时未开跑（%ds），已作废：task=%s thread=%s",
        QUEUE_START_TIMEOUT_SEC,
        intent.task_id,
        intent.thread_id,
    )
    metrics.record_task_rejected("queue_start_timeout")
    # 顺序同 worker 的 _finalize_interrupted：事件 → 还预扣/占位/指纹 → **最后**写终态。占位还挂着
    # 时就写终态，用户看到失败立刻重发，会被自己刚作废的这一轮以 already_running 挡在门外。
    await monitor.report_error(
        "queue_timeout", "排队超时，任务未能开始执行，请重发", thread_id=intent.thread_id
    )
    await _rollback_claim(
        intent.task_id, intent.thread_id, user_id=intent.user_id, query=intent.query
    )
    # 终态用 failed 而不是 cancelled：``cancelled`` 的约定是「用户自己掐的，别再重试」，而这里任务
    # 本身没毛病，重发就能好（口径见 queue/ports.py 的 TaskState 注释）。
    await queue.set_status(
        TaskStatus(
            task_id=intent.task_id,
            state="failed",
            thread_id=intent.thread_id,
            error="排队超时，任务未开始执行，请重发",
        )
    )
    return True


async def _queued_runner(intent: IntentTask, position: int) -> None:
    """队列模式下 API 侧的影子协程：入队 → 报排位 → 等 worker 跑完。

    **它不跑 Agent，存在的理由只有一个**：``active_tasks`` 是 ``/inflight``、幂等第 1 层、取消口
    三样共用的账本。队列模式下若不在 API 侧留一条记录，这三样会一起失效——而「前端零改动」的前提
    正是它们的行为不变。所以这里用一个廉价的轮询协程占住那个位置，跑完就摘。

    **取消**（批2-4 起）：cancel 这个 waiter 的同时，取消口会经控制面把指令送到 worker，那边照常
    上报 ``task_cancelled``。但任务**还在队列里没人领**时没有任何 worker 会为它发事件，前端就停在
    转圈上——所以这里补一条，且只在状态仍是 ``queued`` 时补（已经在跑的那些由 worker 发，避免两份）。
    """
    thread_id = intent.thread_id
    queue = get_task_queue()
    deadline = asyncio.get_running_loop().time() + QUEUE_WAIT_TIMEOUT_SEC
    try:
        try:
            intent = await _enqueue_intent(intent, position - 1)
        except Exception as exc:
            # 入队失败必须让用户看见。走 error 事件而不是 HTTP 5xx：响应早就返回了，而前端本就按
            # error 事件收尾（run_agent 抛异常时也是这条路），故零改动即可显示。
            logger.exception("入队失败：thread_id=%s", thread_id)
            await monitor.report_error(
                "enqueue_failed", f"任务入队失败：{exc}", thread_id=thread_id
            )
            return
        finally:
            # 入队成功后它已计在待消费数里，失败则名额该还——两种都撤（登记见 _admit_or_429）。
            await _release_admission(intent.task_id)
        # 本协程余下的日志（排队超时 / 作废）也带 trace_id。它是独立 task，绑定不外溢，不必还原。
        parent = parse_traceparent(intent.traceparent)
        bind_log_context(trace_id=parent.trace_id if parent else None)
        if position > 1:
            metrics.record_task_queued(intent.kind)
            await monitor.report_queue_status(
                thread_id,
                position,
                # 传「前面有几个人」而不是 position 本身——与准入池那边差一个身位是有原因的：在那边
                # 「进了等待队列」本身就意味着槽全满、你必须等一轮；这边队列里有人不等于 worker 忙，
                # 排第 1 位就是没人挡着你，说 30s 是凭空吓人。
                estimated_wait_seconds(position - 1, WORKER_CONCURRENCY),
                intent.kind,
            )
        loop = asyncio.get_running_loop()
        start_deadline = (
            loop.time() + QUEUE_START_TIMEOUT_SEC if QUEUE_START_TIMEOUT_SEC > 0 else None
        )
        while loop.time() < deadline:
            await asyncio.sleep(QUEUE_POLL_SECONDS)
            status = await queue.get_status(intent.task_id)
            if status is not None and status.state in TERMINAL_STATES:
                return
            still_queued = status is None or status.state == "queued"
            if start_deadline is not None and still_queued and loop.time() >= start_deadline:
                # 作废失败（没控制面 / worker 刚领走）就把这道闸关掉，继续等到 wait 超时。不重试：
                # 重试每轮都要打一次 Redis，而作废不掉的两个原因都不会在秒级内变。
                if await _abandon_queued(intent, queue):
                    return
                start_deadline = None
        logger.warning(
            "等结果超时（%ds）：task=%s thread=%s",
            QUEUE_WAIT_TIMEOUT_SEC,
            intent.task_id,
            thread_id,
        )
        await monitor.report_error("queue_timeout", "任务等待超时，请稍后重试", thread_id=thread_id)
    except asyncio.CancelledError:
        # shield：本协程正在被取消，直接 await 会在第一个挂起点再吃一次 CancelledError，事件就发
        # 不出去了（同 session_io.charge_quota 的手法）。发完再把取消原样往上抛。
        with suppress(asyncio.CancelledError, Exception):
            await asyncio.shield(
                asyncio.create_task(_report_cancel_if_queued(intent.task_id, thread_id))
            )
        raise
    finally:
        # 与 _runner 同一手法：按身份摘除，不按 key 盲删——覆盖重发时旧 waiter 的 finally 会晚几个
        # tick 才跑，盲删会把已登记的新任务摘掉。
        handle = active_tasks.get(thread_id)
        if handle is not None and handle.task is asyncio.current_task():
            active_tasks.pop(thread_id, None)
        # DB 侧真相按身份清（run_id == task_id）。队列模式下真正跑任务的是 worker 进程，但这条
        # 影子协程与那边同生共死（它轮到终态才返回），清在这里够用。worker 关停中断那条路上占位
        # 已经由 worker 先清过一遍（它写终态前清，好让用户能立刻重发），这里再清一次是空转，无害。
        await _release_run_shielded(thread_id, intent.task_id)


def _start_queued(
    req: TaskRequest,
    thread_id: str,
    user_id: str | None,
    turn_count: int,
    depth: int,
    run_id: str,
) -> dict[str, Any]:
    """登记影子协程并返回响应体。

    **全同步**：它处在 endpoint 的无 ``await`` 区间里（原子性理由见 create_task）。

    ``task_id`` 直接取准入时那笔预扣的 ``run_id``——两者是同一个东西：结算与消费侧去重都按它认人，
    各生成一个只会让「这条消息对应哪笔预扣」再也查不出来。
    """
    intent = IntentTask.create(
        task_id=run_id,
        thread_id=thread_id,
        query=req.query,
        history_turns=turn_count,
        user_id=user_id,
        platforms=req.platforms,
        image_paths=req.image_paths,
        skill=req.skill,
        # 带上本次 HTTP 请求的 id：worker 是另一个进程，ContextVar 过不去，只能随消息走。
        request_id=_request_id_var.get(),
    )
    position = depth + 1
    task = asyncio.create_task(_queued_runner(intent, position))
    active_tasks[thread_id] = TaskHandle(
        task=task,
        query=req.query,
        images=list(req.image_paths or ()),
        task_id=intent.task_id,
    )
    # 指纹已在 create_task 的 check_duplicate 里登记（``SET NX`` 查与登记同一步，1-2 起）。
    queued = position > 1
    return {
        "status": "queued" if queued else "started",
        "thread_id": thread_id,
        "queue_position": position if queued else 0,
        "task_id": intent.task_id,  # 契约是加法：老前端忽略这个键，脚本可拿它去 GET /api/task/{id}
    }


@app.post("/api/task")
async def create_task(
    req: TaskRequest, auth_uid: str | None = Depends(get_current_user_id)
) -> dict[str, Any]:
    """启动一次主 AgentLoop 后台任务，立即返回 thread_id（不等结果，避免前端傻等）。

    **鉴权（I 块）：** 开启 ``AUTH_ENABLED`` 后，跑任务用的 user_id **一律取 token 的 sub**
    （``resolve_identity`` 忽略 ``req.user_id``）——杜绝冒名把偏好写进他人名下。关闭时退回现状。

    **幂等三层——三层挡三种不同的重复：**

    1. **同 thread + 同 query 且还在跑** → ``already_running``，直接把用户领回原任务，不重跑。
       这是刷新页面 / 双击提交的典型形态。
    2. **同 thread + 不同 query** → 覆盖重发：cancel 旧任务、强占槽位起新的。这是用户在同一会话里
       改主意重新提问，**不是重复**，必须放行（也不该被自己的旧任务挡在门外）。
    3. **不自带 thread_id + 同 (user_id, query) 且在窗口内** → ``duplicate``，返回原任务的
       thread_id。只对脚本 / 裸 API 调用生效，前端不受影响（理由见下方那段注释）。

    原方案的「Checkpoint 防重跑」那一层本项目不做（无 checkpointer，见 dedup 模块 docstring）。

    **任务一律入队（阶段 1 条 7 起没有第二条路）：** 本进程只登记一个等结果的影子协程，AgentLoop
    跑在 worker 里。原先那套进程内准入池（normal/heavy 双池 + 排队 + 再平衡）随之删除——它守的是
    「本进程同时跑几个 loop」，而本进程一个 loop 都不跑了，留着只会把削峰上限按回单进程那 8 个数。
    背压改由三道跨进程的闸承担：用户级并发（``run_holds``）、队列深度（``QUEUE_MAX_DEPTH``，超了
    429 + Retry-After）、worker 的 ``WORKER_CONCURRENCY``。细节见 ``_start_queued``。
    """
    user_id = resolve_identity(auth_uid, req.user_id)
    thread_id = req.thread_id or uuid.uuid4().hex
    run_id = uuid.uuid4().hex
    # 背压闸排第一（理由见 _admit_or_429）：连配额都排在它后面——过载时额度用尽的人先拿到 429
    # 而非 402，重试时再见 402，换来的是拒绝不必先查库。
    queue_depth = await _admit_or_429(run_id)
    try:
        body, handed_off = await _submit_admitted(req, user_id, thread_id, run_id, queue_depth)
    except BaseException:
        await _release_admission(run_id)
        raise
    if not handed_off:  # 幂等命中 / 去重：没入队，名额当场还
        await _release_admission(run_id)
    return body


async def _submit_admitted(
    req: TaskRequest, user_id: str | None, thread_id: str, run_id: str, queue_depth: int
) -> tuple[dict[str, Any], bool]:
    """``create_task`` 过了背压闸之后的部分。返回 ``(响应体, 是否已交给入队协程)``。"""
    # 配额闸在最前。归属登记不再单独一步：它与幂等第 1 层并成下面那一次 claim_thread_and_run
    # （一个会话一次 commit），越权 / 凭证失效在那里报，报之前把预扣还掉。以上都在本 handler
    # 「无 await 区间」的前半段：下面「幂等判定 → 占槽 / 入队登记 → 写 active_tasks」那一整段
    # 仍然一个 await 都没有，原子性不受影响（单线程事件循环里，没有 await 就不会被别的请求插进来）。
    await _enforce_quota(user_id)

    # 分档的输入（历史轮数）在这里就取好：它要查库（await），而下面从幂等判定到入队登记那一整段
    # 必须一个 await 都没有——原子性全靠这个，见 _history_turns 的 docstring。
    turn_count = await _history_turns(thread_id)
    kind = classify_request(turn_count)

    # ── 预授权：占住额度 + 数在飞数。**必须在幂等判定之前** ──
    #
    # 位置是被无 await 区间逼出来的：下面从幂等第 1 层到占槽 / 入队那一整段的原子性全靠「一个
    # await 都没有」（同 thread 同 query 的两个请求若在中间被切开，会双双通过第 1 层各起一个 run）。
    # 代价是幂等命中的请求也先占一笔——那几条路各自在 return 前把它还掉，下面三处 release。
    await _acquire_hold_or_reject(run_id=run_id, user_id=user_id, thread_id=thread_id, kind=kind)

    # ── 幂等第 1 层：同 thread 上一个任务还活着。**真相在 DB，不在本进程**（阶段 1-2）──
    #
    # 判定与占位是同一条条件 UPDATE（见 app.db.runs），所以同一个 thread 打到两台副本时，只有
    # 一台的影响行数是 1，另一台按 already_running 把用户领回去。``active_tasks`` 降级为本进程
    # 缓存：它还管着取消口、/inflight 与影子协程的身份校验，但不再是「谁在跑」的答案。
    #
    # 归属登记（M16）也在这一步：属主校验、首轮插行、顶 updated_at 与占位同在一次提交里。
    claim = await _claim_thread_and_run_or_reject(thread_id, run_id, user_id, req.query)
    old = active_tasks.get(thread_id)
    if not claim.can_start:
        # 同一句话又发了一遍 → 领回原任务，不重跑、不占新槽、不动旧任务。
        metrics.record_task_rejected("already_running")
        logger.info("幂等命中（同 thread 同 query）：thread_id=%s", thread_id)
        await release(run_id)  # 没起新任务 → 那笔预扣当场还掉
        return {"status": "already_running", "thread_id": thread_id}, False
    # 同 thread 但换了 query → 覆盖重发。旧 run 可能跑在**另一个进程**里（队列模式 / 另一台副本），
    # 所以取消要按 claim 带回来的旧 run_id 送（run_id == task_id，1-1 起两者同一个东西）；本进程
    # 恰好也有影子协程时再顺手 cancel 一下，让 /inflight 立刻干净。
    is_replace = claim.outcome == "replaced"
    previous_run_id = claim.previous_run_id

    # ── 幂等第 3 层：跨 thread 的指纹去重。**只对不自带 thread_id 的客户端生效** ──
    #
    # 为什么加这个条件（对原方案的主动更正）：本项目是 connect-first——前端先本地生成 thread_id、
    # 连上 WS、才 POST。如果这里把它并进另一个 thread，事件全推给原 thread，前端连的那条 WS 一个
    # 字都收不到，界面永远卡在 running。而且前端的 thread_id 存在 localStorage、刷新不换，原方案
    # 设想的「刷新 → 换 thread_id → 重复提交」在本前端根本不会发生（那条路径由第 1 层完整覆盖）；
    # 真能触发的只剩「用户主动新建对话、再问一遍同一句话」——那是他的真实意图，去重反而是错的。
    #
    # 所以分工按「谁管 thread_id」划：自带 thread_id 的客户端（前端）由第 1 层管；不管 thread_id 的
    # 客户端（脚本 / 裸 API 调用，服务端兜底生成 tid）才走指纹去重——它们没有 WS 订阅要接，拿回
    # 原 thread_id 正好可以去 /inflight 续看。
    #
    # **窗口在 Redis（阶段 1-2）**，查与登记是同一条 ``SET NX EX``：多副本下才真的只跑一遍，也不
    # 再依赖「查到登记之间没有 await」。判重时要把刚占下的两样都还掉——预扣的额度，以及上面那条
    # 条件更新占下的「在跑」位置（这条路新生成过 thread_id，位置一定是自己抢到的）。
    if not is_replace and req.thread_id is None:
        try:
            dup_thread = await dedup.check_duplicate(user_id, req.query, thread_id)
        except dedup.DedupUnavailable as exc:
            await _rollback_claim(run_id, thread_id)
            logger.warning("去重窗口不可用，拒绝本次提交：%s", exc)
            raise HTTPException(503, "服务暂时不可用（去重窗口离线），请稍后重试") from exc
        if dup_thread is not None:
            metrics.record_task_rejected("duplicate")
            logger.info("幂等命中（指纹去重）：原 thread_id=%s", dup_thread)
            await _rollback_claim(run_id, thread_id)
            return {"status": "duplicate", "thread_id": dup_thread}, False

    # ── 交给 worker ──
    if is_replace and previous_run_id is not None:
        # 覆盖重发 = **换一轮**，不是多跑一轮（批2-4 补齐）：除了掐掉 API 侧的影子协程，还要把取消
        # 送到真正在跑它的 worker，否则用户改主意重问一句，旧问题仍在后台烧着 token，两轮的事件
        # 还会同时往同一条 WS 上推。
        #
        # 按 DB 里读到的**旧 run_id** 送（1-2 起）：旧 run 可能根本不在本进程的 active_tasks 里
        # （它是另一台副本收的），那种情况下按 old.task_id 送就是送了个空。
        #
        # 用 nowait：这里处在 endpoint 的**无 await 区间**里（幂等判定的原子性靠它，见本函数
        # docstring）。本地那一半是同步的、当场生效；剩下的 Redis 往返丢进后台任务，且它打的标记
        # 按**旧** run_id，不会误伤下面马上要入队的这条新任务。
        control.request_cancel_nowait(thread_id, previous_run_id)
        if old is not None:
            old.task.cancel()
    return _start_queued(req, thread_id, user_id, turn_count, queue_depth, run_id), True


@app.post("/api/task/async")
async def create_task_async(
    req: TaskRequest, auth_uid: str | None = Depends(get_current_user_id)
) -> dict[str, Any]:
    """异步提交：入队即返回 ``task_id``，API 侧不留任何等待协程。

    **与 ``POST /api/task`` 的分工按「调用方有没有 WS」划**。前端是 connect-first、事件走 WS，所以
    它用 ``/api/task``（那条路会在 API 侧留一个影子协程占住 ``active_tasks``，好让 ``/inflight``、
    取消口、幂等第 1 层照旧工作）。脚本 / 批量灌入没有 WS 可订阅，留影子协程纯属浪费——它们用这个
    口子拿 ``task_id``，再去 ``GET /api/task/{id}`` 轮询。

    配额 / 归属两道闸与主路径完全一致。
    """
    user_id = resolve_identity(auth_uid, req.user_id)
    thread_id = req.thread_id or uuid.uuid4().hex
    run_id = uuid.uuid4().hex
    depth = await _admit_or_429(run_id)  # 排第一，理由同 create_task
    try:
        await _enforce_quota(user_id)
        await _claim_thread_if_needed(thread_id, user_id, req.query)
        turn_count = await _history_turns(thread_id)
        kind = classify_request(turn_count)
        # 预授权与主路径同一道闸：脚本批量灌入正是并发透支最容易发生的地方，放过它等于把闸开在
        # 用不着的那一边。这条路没有幂等三层，拿到就直接入队，不需要任何 release 分支。
        await _acquire_hold_or_reject(
            run_id=run_id, user_id=user_id, thread_id=thread_id, kind=kind
        )
        intent = IntentTask.create(
            task_id=run_id,
            thread_id=thread_id,
            query=req.query,
            history_turns=turn_count,
            user_id=user_id,
            platforms=req.platforms,
            image_paths=req.image_paths,
            skill=req.skill,
            request_id=_request_id_var.get(),
        )
        try:
            await _enqueue_intent(intent, depth)
        except Exception as exc:
            # 队列的降级口径：入队失败必抛（见 app/queue/ports.py）。这里是「调用方决定」的那一半——
            # 同步提交口有 WS 可以补一条 error 事件，这个口子只有 HTTP 响应，故 503 说清楚。
            logger.exception("异步提交入队失败：thread_id=%s", thread_id)
            raise HTTPException(503, "任务入队失败，请稍后重试") from exc
        position = depth + 1
        return {
            "task_id": intent.task_id,
            "thread_id": thread_id,
            "status": "queued",
            "queue_position": position,
            # 同 _queued_runner：预估等待按「前面有几个人」算，排第 1 位就是 0（口径见那里的注释）。
            "estimated_wait_seconds": estimated_wait_seconds(depth, WORKER_CONCURRENCY),
        }
    finally:
        # 入队成功后由待消费数计它，失败 / 被别的闸拒则名额该还——两种都撤。
        await _release_admission(run_id)


@app.get("/api/task/{task_id}")
async def get_task_state(
    task_id: str, auth_uid: str | None = Depends(get_current_user_id)
) -> dict[str, Any]:
    """查一条异步任务的状态 / 结果。

    **属主校验只能在读到状态之后做**——``task_id`` 本身不带身份，得先拿它换出 ``thread_id`` 才知道
    该问谁。状态键有 TTL（``QUEUE_STATUS_TTL``），过期即 404：它是给一次提交轮询几分钟用的，不是
    历史存储，真·历史在 ``GET /api/history/{thread_id}``。
    """
    status = await get_task_queue().get_status(task_id)
    if status is None:
        raise HTTPException(404, "任务不存在或状态已过期")
    await guard_thread(status.thread_id, auth_uid)
    body = status.to_dict()
    # queue_depth 是队列深度不是精确排位（Stream 没有「排第几」的查询），故预估等待也只给量级。
    body["estimated_wait_seconds"] = (
        estimated_wait_seconds(status.queue_depth, WORKER_CONCURRENCY)
        if status.state == "queued"
        else 0
    )
    return body


# --- WebSocket 订阅 ---------------------------------------------------------


async def _ws_authorized(websocket: WebSocket, thread_id: str, token: str | None) -> bool:
    """WS 的属主校验。不通过就关连接（1008 = policy violation）并返回 False。

    **为什么 token 走 query 而不是 Authorization 头**：浏览器原生的 WebSocket API 压根不让设
    自定义请求头（只有 HTTP 请求能设），所以业界通行做法就是把它挂 query 上。代价要说清楚：
    URL 比头更容易被记进访问日志 / 代理日志，所以这枚 token 是有过期时间的短期凭证，不是长期
    密钥。真要更干净，得用一次性的 ticket 换连接，那是后话。
    """
    if not auth_enabled():
        return True
    try:
        uid = decode_token(token) if token else None
    except HTTPException:
        uid = None
    if uid is None:
        await websocket.close(code=1008)
        return False
    async with session_factory()() as db:
        try:
            await assert_owner(db, thread_id, uid)
        except PermissionError:
            await websocket.close(code=1008)
            return False
    return True


@app.websocket("/ws/{thread_id}")
async def ws_endpoint(
    websocket: WebSocket,
    thread_id: str,
    last_event_id: str | None = None,
    token: str | None = None,
) -> None:
    """前端订阅 thread_id 对应的 AGUI 事件流。

    connect-first 协议：登记连接后立即回一条 ``ws_ready`` 控制帧，前端**收到它再** POST 起
    任务，确保任务上报第一个事件时连接已在 ConnectionManager 里（早期事件不丢）。``ws_ready``
    用独立 ``type`` 与 monitor_event 区分，不污染事件流。

    **断线重连补发（D 块）：** 前端重连时带 ``?last_event_id=<上次收到的最后一个事件 id>``，
    登记连接后先从该 thread 的 Redis Stream 补发 last_event_id 之后的缺口事件，再转直播。补发与
    新直播事件可能有重叠，但事件都带单调递增的 stream id，前端按 id 去重即可。Redis 降级 / 无
    last_event_id 时补发为空，退回纯直播（现状）。

    **属主校验（M16）：** 握手前先验 ``?token=``——事件流是实时的对话内容，不校验等于把别人的
    整场对话开着直播。校验必须在 ``accept()`` 之前，否则连接已经建立，再关就是「先放进门再赶出去」。
    """
    if not await _ws_authorized(websocket, thread_id, token):
        return
    manager = monitor.get_connection_manager()
    await manager.connect(websocket, thread_id)
    try:
        await websocket.send_json({"type": "ws_ready", "thread_id": thread_id})
        # 先登记连接（上面）再补发：登记后的新事件走直播，补发的是断开窗口的历史，前端按 id 去重。
        if last_event_id:
            for payload in await event_log.replay_after(thread_id, last_event_id):
                await websocket.send_json(payload)
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
                continue
            try:
                msg = json.loads(data)
            except (json.JSONDecodeError, ValueError):
                continue
            if msg.get("type") == "clarification_response":
                # 队列模式下等着这条回复的 Future 在 worker 进程里，故走 deliver_reply（本地
                # 命中就地 resolve，落空再经控制面转发）。前端契约一字未动。
                from app.api.clarification import deliver_reply

                route = await deliver_reply(thread_id, msg.get("text", ""))
                if route not in ("local", "forwarded"):
                    logger.info("澄清回复无人接收：thread_id=%s（%s）", thread_id, route)
    except WebSocketDisconnect:
        pass
    finally:
        # 放 finally：不止正常断开，receive 抛其他异常（ASGI 状态错乱 / 连接重置）时也要注销，
        # 否则死连接残留在表里。注销按对象身份（ConnectionManager 内部用 `is` 校验），不误删重连。
        await manager.disconnect(websocket, thread_id)


# --- 取消任务 ---------------------------------------------------------------


@app.post("/api/task/{thread_id}/cancel")
async def cancel_task(
    thread_id: str, auth_uid: str | None = Depends(get_current_user_id)
) -> dict[str, str]:
    """取消某个正在跑的长任务。``task.cancel()`` 向协程注入 CancelledError，
    run_agent 任一 await 点被打断 → 上报 task_cancelled。

    属主校验（M16）：不然任何人拿到 thread_id 就能掐断别人正在跑的任务。

    **任务在 worker 进程里跑（批2-4 补齐）**：所以除了掐掉 API 侧那个等
    结果的影子协程，还要经控制面把取消送过去——落一个按 ``task_id`` 的标记（管住「还在队列里排队、
    没人领」的那些）+ 发一条广播（管住「已经被某个 worker 领走、正在跑」的那些）。worker 侧收到后
    先放开 ``ask_user`` 的等待再 cancel 任务本体，``run_agent`` 的 finally 照常上报
    ``task_cancelled`` 并把这一轮的账记完——「取消即免单」的口径一个字没变。
    """
    await guard_thread(thread_id, auth_uid)
    handle = active_tasks.get(thread_id)
    if not handle or handle.task.done():
        raise HTTPException(404, f"任务 {thread_id} 不存在或已结束")
    from app.api.clarification import cancel_pending

    cancel_pending(thread_id)
    # 先送远端再掐本地：影子协程一被 cancel，它的 finally 就把 active_tasks 摘了，那之后再想拿
    # task_id 就没处拿。顺序反过来在真实链路上是「偶尔取消不掉」，且只在竞态窗口里复现。
    await control.request_cancel(thread_id, handle.task_id)
    handle.task.cancel()
    return {"status": "cancelling", "thread_id": thread_id}


class ClarifyRequest(BaseModel):
    """``POST /api/clarify/{thread_id}`` 的请求体（回复文本与 WS 那条通路逐字同义）。"""

    text: str = ""


@app.post("/api/clarify/{thread_id}")
async def submit_clarification(
    thread_id: str, req: ClarifyRequest, auth_uid: str | None = Depends(get_current_user_id)
) -> dict[str, Any]:
    """回答 Agent 通过 ``ask_user`` 提出的澄清问题（HTTP 版）。

    **前端不用它**——浏览器那条路仍然是 WS 上的 ``clarification_response`` 帧，契约一字未动。这个
    口子是给没有 WS 的调用方（脚本 / 压测 / 端到端冒烟）准备的：批2-4 之前它们根本无法回答提问，
    只能干等到 120s 超时。两条路进的是同一个 :func:`app.api.clarification.deliver_reply`，所以
    「本地就有人等 → 就地 resolve；否则查令牌 → 经控制面转发给 worker」的判断只有一份。

    ``404`` 表示**没人在等这条 thread 的回复**（从没提问 / 已经超时 / 令牌过期）——令牌过期按取消
    处理，迟到的回复一律不投递，绝不能塞给下一个问题。
    """
    await guard_thread(thread_id, auth_uid)
    route = await clarification.deliver_reply(thread_id, req.text)
    if route == "publish_failed":
        raise HTTPException(503, "澄清回复转发失败（控制面不可用），请稍后重试")
    if route not in ("local", "forwarded"):
        raise HTTPException(404, f"会话 {thread_id} 当前没有等待回答的问题")
    return {"status": "delivered", "thread_id": thread_id, "route": route}


@app.get("/api/task/{thread_id}/inflight")
async def task_inflight(
    thread_id: str, auth_uid: str | None = Depends(get_current_user_id)
) -> dict[str, Any]:
    """该 thread 是否仍有任务在后台跑；在跑则连同「正在跑那一轮」的提问与已发生事件一并回吐。

    **它撑起「刷新 / 切回对话不打断、自动续看」：** 任务与 WS 本就解耦——刷新关页面、切到别的
    对话都只是断了**订阅**，后台任务照跑（除非用户点「取消」）。前端重新进入某对话时调本接口：
    - ``running=false``：该轮要么没跑过、要么已收尾（结论已落 turns.json，由 ``/api/history``
      回看）。
    - ``running=true``：回吐 ``query`` + ``events``（最后一个 session_created 起的「当前这轮」
      事件），前端据此重建在跑轮，再带 ``last_event_id`` 重连 WS 补缺口 + 续直播，无缝接上。

    **去重保护：** ``task_result`` 已进事件流、但 ``_runner`` 的 finally 还没把任务摘出
    ``active_tasks`` 的那个瞬时窗口里，单看 ``task.done()`` 仍是「在跑」，会和刚落盘的历史轮重出
    一份。故这里以「流里最后一条已是终结类事件」为准判其已结束（Redis 降级取不到事件时退回
    ``task.done()``，此窗口极短、可接受）。
    """
    await guard_thread(thread_id, auth_uid)  # M16：别人的 thread 不给回吐提问原文与事件流
    handle = active_tasks.get(thread_id)
    if handle is None or handle.task.done():
        return {"running": False, "query": None, "images": [], "events": []}
    events = await event_log.replay_current_run(thread_id)
    if events and events[-1].get("event") in _TERMINAL_EVENTS:
        return {"running": False, "query": None, "images": [], "events": []}
    return {"running": True, "query": handle.query, "images": handle.images, "events": events}


@app.get("/api/quota")
async def get_quota_status(auth_uid: str | None = Depends(get_current_user_id)) -> dict[str, Any]:
    """当前登录用户的 credit 余额（前端顶栏余额条 + 额度耗尽提示用）。

    **身份只认 token**，不接受任何查询参数——「查谁的余额」由凭证决定，否则改个 URL 就能窥探别人
    烧了多少。未开鉴权 / 未设配额时返回 ``enabled=false``，前端据此整块隐藏余额条（demo 模式下
    没有可信身份，本来就不设闸，见 :mod:`app.db.quota`）。
    """
    if not quota_enabled() or not auth_uid:
        return _disabled_quota().as_dict()
    async with session_factory()() as db:
        status = await get_quota(db, auth_uid)
    return status.as_dict()


@app.get("/api/history/{thread_id}")
async def get_history(
    thread_id: str, auth_uid: str | None = Depends(get_current_user_id)
) -> dict[str, Any]:
    """读取某段会话的逐轮对话（前端「回看 / 续聊」面板用）。

    返回 ``messages`` 表里累加的 ``user → assistant`` 对。thread 从未跑过或暂无历史时返回**空列表**
    而非 404——对前端是「新会话」而非「出错」，渲染空更自然。仍传一份会话目录进去：库上线前的老
    会话，正文还在它的 turns.json 里，第一次被点开时惰性迁进库（``thread_id`` 用户可控，故经
    ``_safe_session_dir`` 防路径穿越）。

    属主校验（M16）：这是最要紧的一个口——对话正文全在这里，不校验就等于谁拿到 thread_id
    谁就能读别人聊过什么。
    """
    await guard_thread(thread_id, auth_uid)
    session_dir = safe_session_dir(OUTPUT_ROOT, thread_id)
    return {"thread_id": thread_id, "turns": await read_turns(thread_id, session_dir)}


@app.get("/api/health")
async def health() -> dict[str, Any]:
    """探活 + 本进程影子协程数 + 队列积压（人读的概览；机器看板走 /metrics）。

    ``active_tasks`` 是**本副本**在等结果的轮数，不是全局在跑数（真相在 ``run_holds``）。
    队列深度读不到时回 ``None`` 而不是 500：探活不该因为观测项失败就把这个副本判死。

    ``turn_cache`` 挂在这里是给**评测脚本**看的：整轮缓存开着时跑 Rubric，分数会变成上一次那份
    的复读且全程零报错，所以 ``run_rubric.py`` 开跑前要能查到它、查到开着就拒跑。
    """
    try:
        depth: int | None = await get_task_queue().depth()
    except Exception:
        depth = None
    return {
        "status": "ok",
        "active_tasks": len(active_tasks),
        "queue": {"depth": depth, "max_depth": QUEUE_MAX_DEPTH},
        "turn_cache": turn_cache_status(),
    }


@app.get("/metrics")
async def metrics_endpoint() -> Response:
    """Prometheus 抓取端点（A 块）。被 scrape 时即时刷新「当前值」类 gauge——活跃任务 / 任务槽 /
    排队深度 / 断路器状态都是此刻读最准，不必实时维护；计数与耗时类指标则在各打点处实时累积。"""
    metrics.set_active_tasks(len(active_tasks))
    # 槽位口径随阶段 1 条 7 变了：本进程不跑 loop，"active" 是在等结果的影子协程数，"limit" 是队列
    # 深度上限（真正约束并发的是 run_holds 的用户级上限与各 worker 的 WORKER_CONCURRENCY）。
    metrics.set_task_slots(len(active_tasks), QUEUE_MAX_DEPTH)
    try:
        metrics.set_queue_pending("all", await get_task_queue().depth())
    except Exception:  # 观测失败不拖垮 scrape
        pass
    metrics.refresh_circuit_breakers()
    body, content_type = metrics.render()
    return Response(content=body, media_type=content_type)
