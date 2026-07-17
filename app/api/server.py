"""FastAPI 服务 —— 把主 AgentLoop 暴露给浏览器，落地 M10 前后端闭环。

第 14 章（M9）已经把 ``run_agent(query, thread_id, user_id)`` 跑通——本模块只补一层「对外
接口」，让用户在浏览器里发起任务、实时看事件流、取消、下载产物。六个口子：

==============================  ============================================
接口                            解决什么
==============================  ============================================
``POST /api/task``              启动一次主 AgentLoop（后台跑），立即返回 thread_id
``WS   /ws/{thread_id}``        订阅该 thread 的 AGUI 事件流（长连接）
``POST /api/task/{tid}/cancel`` 用户主动取消长任务
``GET  /api/files/{tid}/{name}``下载本次会话产物（summary.md / result.json）
``POST /api/upload``            上传参考图到本次会话目录
``GET  /api/preferences/{uid}`` 读用户长期偏好（前端偏好面板，refdoc 五接口外的补充）
``GET  /api/history/{tid}``     读该 thread 的逐轮对话（前端回看 / 续聊，同 thread 复用即接上文）
==============================  ============================================

**事件不丢的关键约定（对 refdoc 的主动更正）：**
refdoc 的流程是「POST 起任务 → 返回 tid → 前端再连 WS」，但 ``run_agent`` 一上来就上报
``session_created`` 等早期事件——任务已经在 ``create_task`` 里开跑，而 WS 还没连上，这些
早期事件会因「该 thread 无连接」被丢掉（只剩日志）。本实现改成 **connect-first**：前端先
本地生成 ``thread_id`` → 连 WS → 收到 ``ws_ready`` 确认连接已登记 → 才 POST 起任务。
``TaskRequest.thread_id`` 支持客户端指定，正是为此。这样 0 缓冲、0 改 M8 的 ConnectionManager
就把竞态关死，比在连接层堆事件缓冲更简单可靠。

**安全上做了什么**（refdoc 把这些划为「生产化留作业」，公网上线后逐项补上了）：``safe_join``
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Response,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.agent.main_agent import run_agent
from app.api import accounts, admin, dedup, event_log, monitor
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
    Reservation,
    classify_request,
    estimated_wait_seconds,
    task_queue,
)
from app.config import store as config_store
from app.db.accounts import assert_owner, claim_thread
from app.db.quota import disabled_status as _disabled_quota
from app.db.quota import get_quota, quota_enabled
from app.db.session import init_db, session_factory
from app.memory.assemble import blocking_exclude_terms
from app.memory.history import read_turns
from app.memory.injector import persist_new_preferences
from app.memory.parser import UserPrefDraft, parse_user_preference
from app.memory.session_state import load_pt, save_pt
from app.memory.store import FavoriteItem, PreferenceEntry, get_store
from app.observability import alerts, metrics
from app.observability.logging import configure_logging
from app.recall import get_recall_client
from app.recall.geo import SUPPORTED_COUNTRIES
from app.tools.image_understand import sniff_image_mime
from app.utils.env import env_int
from app.utils.path_utils import (
    OUTPUT_ROOT,
    UPLOAD_ROOT,
    safe_join,
)
from app.utils.terms import term_hits
from app.utils.tokens import warm_tokenizer

logger = logging.getLogger("shoppingx.server")

# 上传文件大小上限（参考图通常是截图；防一把超大文件打爆磁盘/内存）。
# **与 image_understand 读同一个 env**：两处各写一个数字的话，中间地带的图会「传得上去却看不了」——
# 上传口放行 9MB，工具侧按 8MB 判超限降级，用户只看到「传成功了但 Agent 说没看到图」。
MAX_UPLOAD_BYTES = env_int("UPLOAD_MAX_IMAGE_MB", 8) * 1024 * 1024


def _safe_session_dir(root: Path, thread_id: str) -> Path:
    """把 ``root/<thread_id>`` 经 ``safe_join`` 校验后返回——**thread_id 也是用户可控输入**。

    download 的 ``thread_id`` 来自 URL 段、upload 的来自表单，二者都可能塞 ``..``（如编码的
    ``%2e%2e`` 或表单里直接写 ``../../etc``）。若像最初那样 ``root / thread_id`` 直接拼，会在
    ``safe_join(filename)`` 之前就已逃出 root——文件名那道 safe_join 守的是错的那半截路径。
    这里对 thread_id 也走 safe_join，逃逸即 400（对齐 CONVENTIONS「文件路径一律 safe_join」）。
    """
    try:
        return safe_join(root, thread_id)
    except ValueError as exc:
        raise HTTPException(400, "非法会话标识") from exc


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """启动时预热 Qwen 本地分词器：首次加载约 3.3s，吃在 boot 而非用户请求里（见 utils/tokens.py）。

    用 ``to_thread`` 不阻塞事件循环；失败（未装 tokenizer extra）会降级为 CJK 启发式，不影响起服。

    并起 RT 告警的后台轮询 task——``/metrics`` 是被动的（Prometheus 不来拉就没人知道 P95 涨了），
    告警器是主动的那一半。shutdown 时 cancel 并等它退出，不留悬挂 task。
    """
    configure_logging()  # A 块：启用 structlog（带 thread_id/user_id/fork_depth 上下文）
    validate_auth_config()  # 开了鉴权却没配密钥 → 启动即 fail-fast，不拖到每请求 500
    await init_db()  # M16：建 users / threads 两张表（幂等，已存在则跳过）
    # 后台管理页面改过的参数：库 → env → 各模块 _load_params()。必须在预热与建 agent 之前，
    # 否则本次启动的第一批任务会用着旧值跑（脏数据不会让它抛，见 store.load_into_memory）。
    await config_store.load_into_memory()
    if auth_enabled():
        logger.info("JWT 鉴权已开启：user_id 一律取 token 的 sub，忽略前端传入")
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

    alert_task: asyncio.Task[None] | None = None
    if alerts.alerts_enabled():
        alert_task = asyncio.create_task(alerts.alert_loop())
        logger.info("工具 RT 告警轮询已启动")
    try:
        yield
    finally:
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


app.include_router(accounts.router)  # M16：注册 / 登录 / 我是谁 / 我的会话清单
app.include_router(admin.router)  # 后台管理：热更新模型档位 / 检索 / 展示参数


# --- 会话归属（M16：堵 thread 维度越权）------------------------------------


async def _guard_thread(thread_id: str, auth_uid: str | None) -> None:
    """校验当前用户有权访问这个 thread，否则 403。

    **这是 auth.py 当初点名却没做的那半边洞。** 原先所有 thread 接口（history / files / ws /
    cancel / upload）只按 thread_id 寻址、不问归属：thread_id 会出现在 URL 和事件流里，谁拿到
    就能读别人的对话历史、下载他的产物、连他的实时事件流。有了归属表，这里一句校验就封死。

    鉴权关闭时直接放行——demo 模式下没人认领会话，校验无从谈起（见 accounts.assert_owner）。
    """
    if not auth_enabled() or auth_uid is None:
        return
    async with session_factory()() as db:
        try:
            await assert_owner(db, thread_id, auth_uid)
        except PermissionError as exc:
            raise HTTPException(403, "无权访问该会话") from exc


@dataclass
class TaskHandle:
    """一个活跃后台任务的句柄：``task`` 本体 + 发起它的 ``query`` 原文 + 它的准入凭据。

    比单存 ``asyncio.Task`` 多记一个 query，是为了 ``/inflight`` 能在前端刷新 / 切回对话、本地
    已无该轮上下文时，仍把「正在跑的那一轮」的提问原文回吐给前端重建（query 不落任何持久层，
    只随这个进程内句柄活着——任务一结束句柄即摘除，自然回收）。

    多记一个 ``reservation``，是为了覆盖重发时能知道**旧任务到底有没有占着槽**：只有占着槽的旧
    任务才会在被 cancel 后还回一个槽，新任务才配 ``force_reserve`` 顶上去。旧任务若还在排队
    （没持槽），强占就是凭空多出一个并发——并发上限被悄悄绕过。
    """

    task: asyncio.Task[Any]
    query: str
    reservation: Reservation
    # 本轮参考图的文件名。和 query 一样是「正在跑那一轮」的提问内容，故一并随句柄活着：
    # 少了它，用户传图后一刷新，图会先消失、等任务收尾落库才又冒出来——一次没必要的闪烁。
    images: list[str] = field(default_factory=list)


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


class TokenRequest(BaseModel):
    """``POST /api/auth/token``（开发态发证）请求体：只认 user_id，不验密码。"""

    user_id: str


# --- 鉴权：开发态发证口（I 块）----------------------------------------------


@app.post("/api/auth/token")
async def issue_token(req: TokenRequest) -> dict[str, str]:
    """签发一个 ``sub=user_id`` 的 JWT，供前端 / 测试拿去当 ``Authorization: Bearer``。

    **demo 边界（诚实标注）：** 这是**开发态发证**——只认 user_id、**不验密码**，本质是个「冒名
    工厂」，仅为让鉴权链路能端到端被验证。真实密码登录 / OAuth / 刷新令牌留作业（与 refdocs「真实
    平台 OAuth 不覆盖」一致）。**它单独由 ``AUTH_DEV_TOKEN`` 把守**（不与 ``AUTH_ENABLED`` 共开
    关），否则「开了鉴权」会反手暴露这个工厂、把刚堵的越权洞捅开。未开发证口（默认）一律 404。
    """
    if not (auth_enabled() and dev_token_enabled()):
        raise HTTPException(404, "发证口未开启（需 AUTH_ENABLED=true 且 AUTH_DEV_TOKEN=true）")
    if not req.user_id.strip():
        raise HTTPException(400, "user_id 不能为空")
    token = create_access_token(req.user_id.strip())
    return {"access_token": token, "token_type": "bearer"}


# --- 启动任务 ---------------------------------------------------------------


async def _history_turns(thread_id: str) -> int:
    """该 thread 已积累的历史轮数（分类器的输入）。读不到一律按 0 算 → normal 池。

    正文进库后这里是一次 DB 查询（原先是同步读 turns.json），**必须在 endpoint 的无 await 区间
    之前就取好值**——``try_reserve`` 那段的原子性靠「一个 await 都没有」保证，把 await 挪进去就
    等于给准入判定开了个竞态窗口。轮数只是分池的输入（normal / heavy），早读一步不影响正确性。
    """
    try:
        return len(await read_turns(thread_id, safe_join(OUTPUT_ROOT, thread_id)))
    except Exception:
        return 0


@app.post("/api/task")
async def create_task(
    req: TaskRequest, auth_uid: str | None = Depends(get_current_user_id)
) -> dict[str, Any]:
    """启动一次主 AgentLoop 后台任务，立即返回 thread_id（不等结果，避免前端傻等）。

    **鉴权（I 块）：** 开启 ``AUTH_ENABLED`` 后，跑任务用的 user_id **一律取 token 的 sub**
    （``resolve_identity`` 忽略 ``req.user_id``）——杜绝冒名把偏好写进他人名下。关闭时退回现状。

    **幂等三层（refdocs 16-5 §3）——三层挡三种不同的重复：**

    1. **同 thread + 同 query 且还在跑** → ``already_running``，直接把用户领回原任务，不重跑。
       这是刷新页面 / 双击提交的典型形态。
    2. **同 thread + 不同 query** → 覆盖重发：cancel 旧任务、强占槽位起新的。这是用户在同一会话里
       改主意重新提问，**不是重复**，必须放行（也不该被自己的旧任务挡在门外）。
    3. **不自带 thread_id + 同 (user_id, query) 且在窗口内** → ``duplicate``，返回原任务的
       thread_id。只对脚本 / 裸 API 调用生效，前端不受影响（理由见下方那段注释）。

    refdocs 的「Checkpoint 防重跑」那一层本项目不做（无 checkpointer，见 dedup 模块 docstring）。

    **准入与排队（refdocs 16-5 §2）：** 按历史轮数分 normal / heavy 两池，长任务最多占满 heavy 的
    槽，短任务的槽永远留着。池满则进该池的有界等待队列（用户在 WS 上收到 ``queue_status`` 看排位），
    队列也满才 ``429 + Retry-After``。**准入判定全同步无 await**，单线程下无竞态；真正的「等」发生
    在后台协程里，不阻塞本响应。
    """
    user_id = resolve_identity(auth_uid, req.user_id)
    thread_id = req.thread_id or uuid.uuid4().hex

    # ── credit 配额闸（M18）：今日额度用尽 → 402，连任务都不给起 ──
    #
    # 放在最前（早于归属登记 / 占槽 / 指纹）：额度不够的人不该在系统里留下任何足迹——不该认领
    # thread、不该占并发槽、更不该在队列里排。402 Payment Required 是这里语义最准的码：不是没
    # 权限（403，他登录了且这是他自己的会话），也不是限速（429，等一会儿并不会好转），而是
    # 「这个周期的额度已经花完了」。detail 里带上限 / 已用 / 重置时刻，前端直接拿去展示。
    if quota_enabled() and user_id:
        async with session_factory()() as db:
            quota = await get_quota(db, user_id)
        if quota.exhausted:
            metrics.record_task_rejected("quota_exhausted")
            logger.info("配额耗尽，拒绝任务：user=%s period=%s", user_id, quota.period)
            raise HTTPException(402, detail={"error": "quota_exhausted", **quota.as_dict()})

    # ── 归属登记（M16）：首轮把 thread 记到本人名下，后续轮顶新 updated_at（侧栏据此排序）──
    #
    # 放在最前、且是本 handler 里**唯一**的 await 点之前半段：下面「幂等判定 → 占槽 → 写
    # active_tasks」那一整段仍然一个 await 都没有，原子性不受影响（单线程事件循环里，没有 await
    # 就不会被别的请求插进来）。claim_thread 自带属主校验——拿别人的 thread_id 发消息会被它拒，
    # 否则「用他的 tid 说句话」就成了把他的会话过户到自己名下。
    if auth_enabled() and user_id:
        async with session_factory()() as db:
            try:
                await claim_thread(db, thread_id, user_id, req.query)
            except PermissionError as exc:
                raise HTTPException(403, "无权访问该会话") from exc
            except LookupError as exc:  # token 合法但用户已不存在 → 让他重新登录
                raise HTTPException(401, "凭证已失效，请重新登录") from exc

    # 分池的输入（历史轮数）在这里就取好：它要查库（await），而下面从幂等判定到占槽那一整段
    # 必须一个 await 都没有——原子性全靠这个，见 _history_turns 的 docstring。
    turn_count = await _history_turns(thread_id)

    # ── 幂等第 1 层：同 thread 上一个任务还活着 ──
    old = active_tasks.get(thread_id)
    is_alive = bool(old and not old.task.done())
    if old is not None and is_alive and old.query == req.query:
        # 同一句话又发了一遍 → 领回原任务，不重跑、不占新槽、不动旧任务。
        metrics.record_task_rejected("already_running")
        logger.info("幂等命中（同 thread 同 query）：thread_id=%s", thread_id)
        return {"status": "already_running", "thread_id": thread_id}
    is_replace = is_alive  # 同 thread 但换了 query → 覆盖重发

    # ── 幂等第 3 层：跨 thread 的指纹去重。**只对不自带 thread_id 的客户端生效** ──
    #
    # 为什么加这个条件（对 refdocs 的主动更正）：本项目是 connect-first——前端先本地生成 thread_id、
    # 连上 WS、才 POST。如果这里把它并进另一个 thread，事件全推给原 thread，前端连的那条 WS 一个
    # 字都收不到，界面永远卡在 running。而且前端的 thread_id 存在 localStorage、刷新不换，refdocs
    # 设想的「刷新 → 换 thread_id → 重复提交」在本前端根本不会发生（那条路径由第 1 层完整覆盖）；
    # 真能触发的只剩「用户主动新建对话、再问一遍同一句话」——那是他的真实意图，去重反而是错的。
    #
    # 所以分工按「谁管 thread_id」划：自带 thread_id 的客户端（前端）由第 1 层管；不管 thread_id 的
    # 客户端（脚本 / 裸 API 调用，服务端兜底生成 tid）才走指纹去重——它们没有 WS 订阅要接，拿回
    # 原 thread_id 正好可以去 /inflight 续看。
    if not is_replace and req.thread_id is None:
        dup_thread = dedup.check_duplicate(user_id, req.query)
        if dup_thread is not None:
            metrics.record_task_rejected("duplicate")
            logger.info("幂等命中（指纹去重）：原 thread_id=%s", dup_thread)
            return {"status": "duplicate", "thread_id": dup_thread}

    # ── 准入：分池 + 占槽 or 排队 or 429。以下到 create_task 之间不得出现 await ──
    #
    # 强占（force_reserve）只在**旧任务真的占着槽**时才成立——它马上会被 cancel 并把槽还回来，
    # 新任务顶上去，真实并发不变。旧任务若还在排队（没持槽），强占就是凭空多出一个并发，把并发
    # 上限悄悄绕过；那种情况新任务老老实实走一遍准入（多半是排到旧任务腾出的那个队列位上）。
    kind = classify_request(turn_count)
    if is_replace and old is not None and old.reservation.admitted:
        reservation = task_queue.force_reserve(kind)
    else:
        maybe = task_queue.try_reserve(kind)
        if maybe is None:
            metrics.record_task_rejected("queue_full")
            raise HTTPException(
                429,
                f"服务繁忙：{kind} 队列已满（并发上限 {task_queue.limit}），"
                f"请 {TASK_RETRY_AFTER_SEC}s 后重试",
                headers={"Retry-After": str(TASK_RETRY_AFTER_SEC)},
            )
        reservation = maybe

    if old is not None and is_replace:  # old is not None 让 mypy 收窄类型
        old.task.cancel()

    async def _runner(res: Reservation) -> None:
        # run_agent 内部已对 CancelledError / 其他异常做了 AGUI 上报（task_cancelled / error）
        # 并重抛——这里不重复上报，只负责把任务从 active_tasks 摘掉，避免双份 error 事件。
        try:
            if not res.admitted:
                # 槽位被占满，本任务在队列里等。先告诉用户排在第几位——排队反馈的价值就在于
                # 用户不必对着空白屏幕猜，所以它必须早于 session_created 推出去。
                metrics.record_task_queued(res.kind)
                await monitor.report_queue_status(
                    thread_id,
                    res.position,
                    estimated_wait_seconds(res.position, task_queue.stats()[res.kind]["capacity"]),
                    res.kind,
                )
                await task_queue.wait_turn(res)
            await run_agent(
                req.query,
                thread_id,
                user_id=user_id,
                platforms=req.platforms,
                image_paths=req.image_paths,
            )
        except asyncio.CancelledError:
            logger.info("task cancelled: thread_id=%s", thread_id)
            raise
        except Exception:
            logger.exception("task failed: thread_id=%s", thread_id)
        finally:
            from app.api.clarification import cancel_pending

            cancel_pending(thread_id)
            # 凭据归还：持槽的还槽、排队中被取消的销在途账。放 finally 确保任何收尾路径
            # （正常 / 取消 / 异常 / 排队中被取消）都不漏放，否则槽会泄漏到「看似没满其实全占着」。
            task_queue.release(res)
            # 按**身份**摘除，不按 key 盲删：同 thread_id 重发时旧任务被 cancel，其 finally 会晚
            # 几个 tick 才跑——那时新任务已登记进同一 key，盲删会把活着的新任务摘掉（cancel/health
            # 就找不到它了）。与 ConnectionManager.disconnect 的 `is` 校验同一手法。
            handle = active_tasks.get(thread_id)
            if handle is not None and handle.task is asyncio.current_task():
                active_tasks.pop(thread_id, None)

    # asyncio.create_task 会复制当前 ContextVar 快照；run_agent 内部用 thread_scope 自己
    # 绑定 thread_id/session_dir，故这里无需预先 set_thread_context。
    task = asyncio.create_task(_runner(reservation))
    active_tasks[thread_id] = TaskHandle(
        task=task,
        query=req.query,
        reservation=reservation,
        images=list(req.image_paths or ()),
    )
    # 只有真正启动的任务才登记指纹：被 429 / already_running 的请求留下指纹的话，用户被拒之后的
    # 重试会被当成「重复提交」再拒一次，陷入死循环。
    dedup.remember(user_id, req.query, thread_id)
    status = "queued" if not reservation.admitted else "started"
    return {"status": status, "thread_id": thread_id, "queue_position": reservation.position}


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
                from app.api.clarification import resolve_pending

                resolve_pending(thread_id, msg.get("text", ""))
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
    """
    await _guard_thread(thread_id, auth_uid)
    handle = active_tasks.get(thread_id)
    if not handle or handle.task.done():
        raise HTTPException(404, f"任务 {thread_id} 不存在或已结束")
    from app.api.clarification import cancel_pending

    cancel_pending(thread_id)
    handle.task.cancel()
    return {"status": "cancelling", "thread_id": thread_id}


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
    await _guard_thread(thread_id, auth_uid)  # M16：别人的 thread 不给回吐提问原文与事件流
    handle = active_tasks.get(thread_id)
    if handle is None or handle.task.done():
        return {"running": False, "query": None, "images": [], "events": []}
    events = await event_log.replay_current_run(thread_id)
    if events and events[-1].get("event") in _TERMINAL_EVENTS:
        return {"running": False, "query": None, "images": [], "events": []}
    return {"running": True, "query": handle.query, "images": handle.images, "events": events}


# --- 文件接口 ---------------------------------------------------------------


@app.get("/api/files/{thread_id}/{filename:path}")
async def download_file(
    thread_id: str, filename: str, auth_uid: str | None = Depends(get_current_user_id)
) -> FileResponse:
    """下载某次会话产物（summary.md / result.json）。

    ``filename`` 用 ``:path`` 转换器（允许子目录形式的名字），**正因如此** ``safe_join`` 才是
    真正起作用的防线：``../../`` 这类越权拼接会被它拦下返回 400，而不是靠路由「不匹配斜杠」
    侥幸挡住。

    ``safe_join`` 挡的是「越出目录」，属主校验（M16）挡的是「合法路径但不是你的会话」——
    两道防线管的是两件事，缺一不可。
    """
    await _guard_thread(thread_id, auth_uid)
    session_dir = _safe_session_dir(OUTPUT_ROOT, thread_id)
    if not session_dir.exists():
        raise HTTPException(404, "会话不存在")
    try:
        target = safe_join(session_dir, filename)
    except ValueError as exc:  # 路径穿越企图：当 400 拒绝，不暴露内部路径
        raise HTTPException(400, "非法文件名") from exc
    if not target.is_file():
        raise HTTPException(404, f"文件不存在：{filename}")
    return FileResponse(target, filename=target.name)


@app.post("/api/upload")
async def upload_file(
    thread_id: str = Form(...),
    file: UploadFile = File(...),
    auth_uid: str | None = Depends(get_current_user_id),
) -> dict[str, str]:
    """上传参考图（如复刻款截图）到本次会话目录 ``uploaded/<thread_id>/``。

    两道最小防护：``safe_join`` 净化文件名（恶意 ``../../etc/passwd`` 落不出上传目录）+ 大小
    上限（超限不落盘）。**诚实标注**：这里先整文件读进内存再校验大小，挡的是「写爆磁盘」，
    挡不住「读爆内存」——真要防大文件得在读之前看 Content-Length / 流式分块校验，那属生产化
    硬化（限流 / 类型白名单同级），不在本课程主线。Starlette 的 UploadFile 超阈值会自动落临时
    文件而非全驻内存，已缓解大半。

    属主校验（M16）先于读文件：别人的会话目录不给写（否则可以往他的会话里塞图）。

    类型白名单（M20）：上传的图会被 image_understand 转 base64 送进 VL 模型，所以在**入口**就按
    magic bytes 认图——不认扩展名（改个名就绕过），不认 Content-Type（客户端随便填）。挡在这里，
    而不是等 provider 回一个 400 才知道用户传了个 PDF。"""
    await _guard_thread(thread_id, auth_uid)
    # thread_id 来自表单、完全可控：**先**校验路径合法（否则 ../ 会建到 root 外），再读文件——
    # 路径都非法了就不必把请求体读进内存，且「非法会话标识」的返回码不会被后面的类型校验掩盖成 415。
    upload_dir = _safe_session_dir(UPLOAD_ROOT, thread_id)
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"文件过大（上限 {MAX_UPLOAD_BYTES // 1024 // 1024}MB）")
    if not sniff_image_mime(raw):
        raise HTTPException(415, "只支持图片（jpg / png / webp / gif / bmp）")
    upload_dir.mkdir(parents=True, exist_ok=True)
    try:
        target = safe_join(upload_dir, file.filename or "upload.bin")
    except ValueError as exc:
        raise HTTPException(400, "非法文件名") from exc
    # 落盘是阻塞 IO，挪到线程池，别卡住事件循环（同 loop 还在推其他任务的事件 / 跑 agent）。
    await asyncio.to_thread(target.write_bytes, raw)
    return {"status": "ok", "filename": target.name}


@app.get("/api/uploads/{thread_id}/{filename:path}")
async def download_upload(
    thread_id: str, filename: str, auth_uid: str | None = Depends(get_current_user_id)
) -> FileResponse:
    """取回本会话上传的参考图，供前端在对话气泡里回显。

    与 ``/api/files`` 同构、但**根目录不同**（``uploaded/`` 而非 ``output/``）：那个口服务的是
    Agent 产出的结论文件，这个口服务的是用户传上来的输入。两道防线照旧——``safe_join`` 挡路径
    穿越，``_guard_thread`` 挡「路径合法但不是你的会话」（否则换个 thread_id 就能翻别人上传的图，
    而图往往比文字更私人）。

    为什么回看必须回服务端取、而不是前端缓一份 blob：blob URL 活不过一次刷新，而「我当时发的
    那张图」是对话的一部分——用户点回一段旧会话，图该还在。
    """
    await _guard_thread(thread_id, auth_uid)
    upload_dir = _safe_session_dir(UPLOAD_ROOT, thread_id)
    if not upload_dir.exists():
        raise HTTPException(404, "会话不存在")
    try:
        target = safe_join(upload_dir, filename)
    except ValueError as exc:
        raise HTTPException(400, "非法文件名") from exc
    if not target.is_file():
        raise HTTPException(404, f"图片不存在：{filename}")
    return FileResponse(target, filename=target.name)


# --- 长期偏好（前端偏好面板：读 / 手填 / 删 / 我的资料）----------------------


def _pref_json(entry: PreferenceEntry) -> dict[str, Any]:
    """一条偏好的**完整**结构化 JSON——偏好不是一句话，是一个实体，各字段驱动不同的检索行为。

    ``dedup_key`` 是删改的 handle；``keywords``/``blocking``/``domain`` 必须回吐给前端，因为它们
    才是真正决定行为的字段（blocking 的 keywords 进 item_picker 硬过滤黑名单、非 blocking 走
    Attenuator 减分、domain 限定生效范围）。只给用户看 content 一句话，等于把最该透明的部分藏了。
    ``slug`` 是去重身份的原子标识，一并给出——用户改字段时前端要拿它拼新 key。

    ``last_confirmed_at`` 给前端显示「这条 N 个月没用过了」——它**不参与任何打分**（半衰期衰减
    已删）。与其让系统按一个没人能解释的指数函数把偏好偷偷打七折，不如把「久未复现」摆到用户
    眼前，由他自己决定删不删。
    """
    return {
        "dedup_key": entry.dedup_key,
        "content": entry.content,
        "category": entry.category,
        "polarity": entry.polarity,
        "blocking": entry.blocking,
        "domain": entry.domain,
        "slug": entry.slug,
        "keywords": list(entry.keywords),
        "source": entry.source,  # agent=Agent 学到的 / user=你手填的（页面上出徽标）
        "created_at": entry.created_at.isoformat(),
        "last_confirmed_at": entry.last_confirmed_at.isoformat(),
    }


def _assert_own(user_id: str, auth_uid: str | None) -> None:
    """开启鉴权后只能读写**自己**的偏好（同 GET 的口径，写口尤其不能漏）。"""
    if auth_enabled() and auth_uid != user_id:
        raise HTTPException(403, "无权访问他人偏好")


class PreferenceParse(BaseModel):
    """``POST /parse`` 的请求体：一句自然语言，**只解析不落库**。

    拆成独立一步，是因为偏好是结构化实体：LLM 猜出来的 polarity / strength / keywords 直接写库，
    用户既不知道它写了什么，也没机会纠正「这条其实是软倾向」。先回吐草稿给前端渲染成可编辑卡，
    用户确认（或改完）再 POST 落库。
    """

    text: str


class PreferenceCreate(BaseModel):
    """``POST`` 的请求体：**结构化**条目（通常是 /parse 的草稿经用户确认 / 修改后回传）。"""

    entries: list[UserPrefDraft]


class ProfileUpdate(BaseModel):
    """「我的资料」表单：硬事实，用户显式设定，不靠 LLM 从聊天里猜。

    这两项和口味偏好（材质 / 风格）的失败模式不同——猜错收货国会把关税直接算错，
    猜错预算会把候选全卡掉，后果是「结果明显不对」而非「推荐差一点」，所以值得让用户显式设定。
    两个字段都可选：只传哪个就只更新哪个（``null`` = 不动）。
    """

    dest_country: str | None = None  # ISO 码，如 JP；空串 = 清除
    budget_max_usd: float | None = None  # 上限 USD；<=0 = 清除


@app.get("/api/preferences/{user_id}")
async def get_preferences(
    user_id: str, auth_uid: str | None = Depends(get_current_user_id)
) -> dict[str, Any]:
    """读取某用户沉淀的长期偏好，供前端「偏好面板」展示。

    **鉴权（I 块，堵越权读）：** 开启 ``AUTH_ENABLED`` 后，只能读**自己**的偏好——token 的 sub
    与 URL 段 user_id 不一致即 403。这正是本块要堵的洞：原来谁都能改 URL 读他人 Store。关闭时
    退回现状（任意读）。

    refdoc 五接口里没有这个——但 ROADMAP 的偏好面板需要一个读口。走 ``get_store()``（后端是
    SQLite），返回扁平 JSON。
    """
    _assert_own(user_id, auth_uid)
    entries = await get_store().read(user_id)
    # 最近被确认过的排前面，面板一眼看到最活跃的偏好。按时间排序而非按「衰减权重」——后者已删，
    # 而它做的事本来也就是这个：把久未复现的往后排。区别只在于现在它不再偷偷影响检索。
    entries.sort(key=lambda e: e.last_confirmed_at, reverse=True)
    return {"user_id": user_id, "preferences": [_pref_json(e) for e in entries]}


@app.post("/api/preferences/{user_id}/parse")
async def parse_preference(
    user_id: str,
    body: PreferenceParse,
    auth_uid: str | None = Depends(get_current_user_id),
) -> dict[str, Any]:
    """一句自然语言 → 结构化偏好草稿（**只解析，不落库**）。

    用户不可能自己填 ``slug`` / ``keywords``，但也不该被 LLM 猜的 ``polarity`` / ``strength``
    蒙在鼓里——这两步之间的解法就是这个端点：LLM 拆完先回吐草稿，前端渲染成可编辑的结构化卡，
    用户改完 / 确认后再 POST 落库。解析不出（用户写了句「你好」）→ 400。
    """
    _assert_own(user_id, auth_uid)
    drafts = await parse_user_preference(body.text)
    if not drafts:
        raise HTTPException(400, "没能从这句话里读出偏好，换个说法试试（如「不要塑料的」）")
    return {"drafts": [d.model_dump() for d in drafts]}


@app.post("/api/preferences/{user_id}")
async def add_preference(
    user_id: str,
    body: PreferenceCreate,
    auth_uid: str | None = Depends(get_current_user_id),
) -> dict[str, Any]:
    """落库若干条**结构化**偏好（``source="user"``）——通常是 /parse 的草稿经用户确认 / 修改后回传。

    走 ``persist_new_preferences``（与 curator 同一个唯一落库口）。``source="user"`` 让这些条目
    此后不被 curator 覆盖、也不随时间衰减（见 ``store._merge_on_collision`` / ``recency_weight``）。
    """
    _assert_own(user_id, auth_uid)
    if not user_id:
        raise HTTPException(400, "匿名用户无法沉淀偏好")
    entries = [e for e in body.entries if e.slug.strip() and e.content.strip()]
    if not entries:
        raise HTTPException(400, "偏好内容与 slug 不能为空")
    # 显式传 store（而非让 persist 自己 get_store()）：读写走同一个实例，测试里替换一处即可。
    written = await persist_new_preferences(
        user_id, entries, source_session="", source="user", store=get_store()
    )
    return {"added": [_pref_json(e) for e in written]}


@app.put("/api/preferences/{user_id}/entry/{dedup_key:path}")
async def update_preference(
    user_id: str,
    dedup_key: str,
    body: UserPrefDraft,
    auth_uid: str | None = Depends(get_current_user_id),
) -> dict[str, Any]:
    """就地修改一条偏好的结构化字段（改极性 / 硬软 / 品类 / 关键词 / 生效域）。

    ``dedup_key`` 由 ``polarity:category:domain:slug`` 派生，所以改这几个字段会**换一把钥匙**——
    实现就是「删旧 + 写新」，而不是原地改。这也是为什么不需要引入 uuid 主键：URL 里传的是**旧**
    key（前端本来就有），新 key 由新字段确定性算出。字段没变的情况下删的和写的是同一把 key，
    结果等价于覆盖，无副作用。

    改完一律 ``source="user"``——用户亲手改过的条目，就归他管（curator 此后不得覆盖）。
    """
    _assert_own(user_id, auth_uid)
    if not body.slug.strip() or not body.content.strip():
        raise HTTPException(400, "偏好内容与 slug 不能为空")
    store = get_store()
    await store.delete(user_id, dedup_key)  # 旧钥匙作废（字段没变时新旧同 key，等价于覆盖）
    written = await persist_new_preferences(
        user_id, [body], source_session="", source="user", store=store
    )
    return {"updated": [_pref_json(e) for e in written]}


@app.get("/api/session/{thread_id}/constraints")
async def get_session_constraints(
    thread_id: str, auth_uid: str | None = Depends(get_current_user_id)
) -> dict[str, Any]:
    """读本次会话累积的 P_t 约束集（偏好面板「本次会话」区；打开面板 / 断线重连时主动拉）。

    可见可纠（P_t 重构步骤三①）：约束录入过 LLM 的手（极性判反 / keywords 抽漏照样进 P_t 且无
    自愈性），抽错时唯一的兜底是用户看得见、点得掉。每条带 ``id``（删除按它打 DELETE）与
    ``source_quote``（让用户看懂这是自己哪句话）。会话无 pt.json / 已过期 → 空列表（同 load_pt
    容错口径）。
    """
    await _guard_thread(thread_id, auth_uid)
    pt = load_pt(_safe_session_dir(OUTPUT_ROOT, thread_id))
    return {
        "thread_id": thread_id,
        "epoch": pt.epoch,
        "budget_usd": pt.budget_usd,
        "category": pt.category,
        "constraints": [
            {
                "id": c.id,
                "content": c.content,
                "source_quote": c.source_quote,
                "polarity": c.polarity,
                "blocking": c.blocking,
            }
            for c in pt.constraints
        ],
    }


@app.delete("/api/session/{thread_id}/constraints/{constraint_id}")
async def delete_session_constraint(
    thread_id: str, constraint_id: str, auth_uid: str | None = Depends(get_current_user_id)
) -> dict[str, str]:
    """从本次会话的 P_t 里删一条约束（面板每行的 ×）——抽取出错时的人纠错入口。

    **不走撤回词面核验**：那道闸挡的是 LLM 幻觉 / 抄错 id，用户亲手点的就是那一条，他的删除
    是最高权威（识别 / 授权分离里的「授权」端）。直接按 id 从 active 集移除并落盘；下一轮
    run_agent 开局 load_pt 读回的就是删除后的状态。不存在的 id 静默成功（幂等，连点两次不报错）。
    删完把新快照推给该 thread 的 WS 连接，面板不必自己再拉一次。
    """
    await _guard_thread(thread_id, auth_uid)
    session_dir = _safe_session_dir(OUTPUT_ROOT, thread_id)
    pt = load_pt(session_dir)
    kept = [c for c in pt.constraints if c.id != constraint_id]
    if len(kept) != len(pt.constraints):
        pt.constraints = kept
        save_pt(session_dir, pt)
        await monitor.report_session_constraints(pt, thread_id=thread_id)
    return {"status": "ok"}


@app.delete("/api/preferences/{user_id}/{dedup_key:path}")
async def delete_preference(
    user_id: str,
    dedup_key: str,
    auth_uid: str | None = Depends(get_current_user_id),
) -> dict[str, str]:
    """删除一条偏好（页面上每行的 ×，以及回复下方「记住了 …」的撤销）。

    ``:path`` 转换器是因为 dedup_key 形如 ``dislike:material:global:plastic``，domain 段是自由
    文本、理论上可能含 ``/``。不存在的 key 静默成功（Store.delete 本身幂等）——用户连点两次
    不该看到报错。

    **删了会不会被 Agent 学回来？** 会，但只在用户重新提起同一件事时——那时他本来就是又说了
    一遍。为此加一张 tombstone 表（删除记录 + TTL + 写入前查禁）不值，真被抱怨了再加。
    """
    _assert_own(user_id, auth_uid)
    await get_store().delete(user_id, dedup_key)
    return {"status": "ok"}


@app.put("/api/preferences/{user_id}/profile")
async def update_profile(
    user_id: str,
    body: ProfileUpdate,
    auth_uid: str | None = Depends(get_current_user_id),
) -> dict[str, Any]:
    """更新「我的资料」——常用收货地 / 预算上限这类**硬事实**，用户显式设定，零 LLM。

    这两项走固定 ``slug``（``ship_to`` / ``budget_max``），于是 dedup_key 恒定，改值即覆盖——
    天然是单值语义，不会像口味偏好那样越攒越多。写成普通 ``PreferenceEntry`` 而非另起一张表：
    收货地本来就已经被 ``planner.resolve_dest_country_layered`` 的第 3 层（``category="location"``
    的长期偏好）消费，塞进同一个 Store 就直接生效，不用改检索链路一行代码。

    传空串 / 非正数 = 清除该项。
    """
    _assert_own(user_id, auth_uid)
    if not user_id:
        raise HTTPException(400, "匿名用户无法沉淀偏好")
    store = get_store()

    if body.dest_country is not None:
        code = body.dest_country.strip().upper()
        if not code:
            await store.delete(user_id, "like:location:global:ship_to")
        elif code not in SUPPORTED_COUNTRIES:
            raise HTTPException(400, f"不支持的收货国：{code}")
        else:
            # content 里放**大写 ISO 码**，因为 geo.resolve_dest_country 正是从 content 文本里
            # 正则解析国家的——这里写的格式必须是那边认得出的，否则资料填了却不生效。
            await store.write(
                user_id,
                PreferenceEntry(
                    slug="ship_to",
                    content=f"常用收货地：{code}",
                    category="location",
                    # 收货地是**跨品类**的用户事实（买鞋买沙发都寄同一个地方）→ global。
                    # 它也正好对上下面 delete 用的 "like:location:global:ship_to"。
                    domain="global",
                    polarity="like",
                    keywords=[code],
                    source="user",
                ),
            )

    if body.budget_max_usd is not None:
        if body.budget_max_usd <= 0:
            await store.delete(user_id, "like:budget:global:budget_max")
        else:
            amount = round(body.budget_max_usd, 2)
            await store.write(
                user_id,
                PreferenceEntry(
                    slug="budget_max",
                    content=f"预算上限约 {amount:g} 美元",
                    category="budget",
                    domain="global",  # 「我的资料」里的总预算档位，跨品类生效
                    polarity="like",
                    source="user",
                ),
            )

    entries = await store.read(user_id)
    return {"preferences": [_pref_json(e) for e in entries]}


@app.get("/api/favorites/{user_id}")
async def get_favorites(
    user_id: str, auth_uid: str | None = Depends(get_current_user_id)
) -> dict[str, Any]:
    """读取某用户收藏（♡）的商品，供前端「收藏抽屉」展示。新→旧。

    **收藏是纯展示数据**：它不注入 prompt、不进长期偏好库、不影响检索与精挑——刻意如此。
    收藏一件商品并不能可靠地推出任何偏好（可能只是想再比比价），拿它去改 Agent 行为是过度解读。
    这跟同在 Store 里的偏好 / 行为历史是两码事，那两个都会被喂进上下文。
    """
    _assert_own(user_id, auth_uid)
    return {
        "user_id": user_id,
        "favorites": [i.model_dump() for i in await get_store().read_favorites(user_id)],
    }


@app.post("/api/favorites/{user_id}")
async def add_favorite(
    user_id: str,
    body: FavoriteItem,
    auth_uid: str | None = Depends(get_current_user_id),
) -> dict[str, Any]:
    """收藏一件商品（点 ♡）。同 ``item_id`` 覆盖 → 重复点幂等。

    存的是**商品快照**而非只存 id：收藏跨会话长期留着，而候选登记表（``tools._candidates``）
    随会话清理，换个会话按 id 早捞不回商品了。前端点 ♡ 时手上正好有整张卡的数据，直接送来。
    """
    _assert_own(user_id, auth_uid)
    await get_store().write_favorite(user_id, body)
    return {"user_id": user_id, "item_id": body.item_id, "status": "ok"}


@app.delete("/api/favorites/{user_id}/{item_id}")
async def remove_favorite(
    user_id: str, item_id: str, auth_uid: str | None = Depends(get_current_user_id)
) -> dict[str, Any]:
    """取消收藏。``item_id`` 不存在则静默成功（幂等）。"""
    _assert_own(user_id, auth_uid)
    await get_store().delete_favorite(user_id, item_id)
    return {"user_id": user_id, "item_id": item_id, "status": "ok"}


@app.get("/api/similar/{item_id}")
async def get_similar(
    item_id: str,
    top_k: int = 8,
    auth_uid: str | None = Depends(get_current_user_id),
) -> dict[str, Any]:
    """「搜同款」：拿这件商品的向量在全库找近邻，同步返回一组相似商品。

    **刻意不走 AgentLoop**：这是一次纯向量检索（0 次 LLM 调用、亚秒级），塞进 Agent 只会换来
    几十秒的规划-工具-收尾开销，换不到任何东西。故它不进 ``FULL_TOOL_SET``，就是个 REST 端点。

    **但不豁免 blocking 黑名单**：「绝不推荐」的语义是「这件商品用户永远不该看到」，对任何展示
    通路都成立——不走 AgentLoop 省掉的是规划开销，不是用户授权的硬排除。这里没有会话品类域
    （planner 没跑），故用 :func:`blocking_exclude_terms`（不做域过滤，只收用户亲手勾的条目），
    命中判定与 item_picker / item_search 同一套 ``term_hits``。有排除词时多取一个 buffer 补位，
    别让黑名单用户拿到残缺的同款列表。匿名用户无黑名单，原样直通。

    返回形状直接对齐前端 ``ProductItem``：只有货价（``price_usd``，建库时预折算），**没有到手价**
    ——那要跑 ``shipping_calc``，不是这条通路该做的事，前端照实标「货价」即可。
    """
    top_k = max(1, min(top_k, 24))
    exclude = await blocking_exclude_terms(auth_uid or "")
    fetch_k = top_k + (8 if exclude else 0)
    cands = await asyncio.to_thread(get_recall_client().similar, item_id, fetch_k)
    if exclude:
        cands = [
            c
            for c in cands
            if not any(term_hits(kw, f"{c.title} {c.brand} {c.category}".lower()) for kw in exclude)
        ][:top_k]
    return {
        "item_id": item_id,
        "items": [
            {
                "item_id": c.item_id,
                "platform": c.platform,
                "title": c.title,
                "price_usd": c.price_usd,
                "image_url": c.image_url,
                "url": c.url,
                "score": round(c.score, 4),
            }
            for c in cands
        ],
    }


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
    await _guard_thread(thread_id, auth_uid)
    session_dir = _safe_session_dir(OUTPUT_ROOT, thread_id)
    return {"thread_id": thread_id, "turns": await read_turns(thread_id, session_dir)}


@app.get("/api/health")
async def health() -> dict[str, Any]:
    """探活 + 当前活跃任务数 + 双池槽位/排队用量（人读的概览；机器看板走 /metrics）。"""
    return {
        "status": "ok",
        "active_tasks": len(active_tasks),
        "task_slots": {
            "active": task_queue.active,
            "limit": task_queue.limit,
            "full": task_queue.full,
        },
        "pools": task_queue.stats(),
    }


@app.get("/metrics")
async def metrics_endpoint() -> Response:
    """Prometheus 抓取端点（A 块）。被 scrape 时即时刷新「当前值」类 gauge——活跃任务 / 任务槽 /
    排队深度 / 断路器状态都是此刻读最准，不必实时维护；计数与耗时类指标则在各打点处实时累积。"""
    metrics.set_active_tasks(len(active_tasks))
    metrics.set_task_slots(task_queue.active, task_queue.limit)
    for kind, pool in task_queue.stats().items():
        metrics.set_queue_pending(kind, pool["pending"])
    metrics.refresh_circuit_breakers()
    body, content_type = metrics.render()
    return Response(content=body, media_type=content_type)
