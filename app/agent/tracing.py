"""Langfuse 在线观测接入（v4 / OpenTelemetry）——主链路调试用。

**范围**：只 trace 主对话链路（``run_agent`` 主 loop），不碰 ``eval/`` 与
judge LLM——评测链路进了 trace 只会把线上数据搅浑。

**零胶水的由来**：Langfuse v4 本身就是 OTel SDK 的包装，client 初始化时会把自己的
TracerProvider 设成全局；而 AgentScope 原生的 ``TracingMiddleware`` 打的是标准 ``gen_ai.*``
语义属性，正好落进 Langfuse 的 span 过滤器（``is_genai_span``）放行的那一类。所以主链路的观测
= 装配时挂一个框架自带的中间件，**不需要自建 exporter，也不需要手工传 trace_id**：OTel 上下文
本身就是 ContextVar，同轮并发的工具子任务的 span 天然挂在父 span 下。

**一轮 = 一条 trace**：:func:`turn_span` 在 ``run_agent`` 入口开一个根 span，本轮所有模型 /
工具 span 都挂在它下面；多轮再靠 ``session_id``（= thread_id）在 UI 的 Sessions 视图聚成一次会话。
根 span **必须用 langfuse 自己的 tracer 起**（``start_as_current_observation``）——用 AgentScope
的 tracer 起会被上面那个过滤器静默丢掉，症状是子 span 全在、trace 却没有根。

**安静降级铁律**：观测是调试附属品，绝不能反噬主链路。没装 langfuse 包 / ``LANGFUSE_ENABLED``
非真 / 缺 PUBLIC|SECRET key / client 初始化失败，一律降级成「没有观测」且**绝不抛**。

**host 坑**：SDK 不设 host 会**静默发到 EU 默认区**。这里显式把 ``LANGFUSE_BASE_URL``
（缺省退 ``LANGFUSE_HOST``）喂给 client，``.env.example`` 用的是 ``https://us.cloud.langfuse.com``。
"""

from __future__ import annotations

import logging
import os
import re
import secrets
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from app.utils.env import env_bool

if TYPE_CHECKING:  # 只为类型标注；运行时不 import，避免 agent → eval 的反向依赖
    from app.eval.rubric import RubricResult

logger = logging.getLogger("shoppingx.tracing")


# 本轮（一次 run_agent）的 trace_id：主 loop 在 root 处生成并写入。ContextVar 天然按 async
# 上下文隔离（多用户并发各有各的），且新建的 asyncio.Task（如同轮 batch 工具的 gather）建时即
# 拷贝当前上下文 → 子 task 能读到父设的值。每轮 run_agent 都重新生成，不跨轮复用、无需手动 reset。
_current_trace_id: ContextVar[str | None] = ContextVar("langfuse_trace_id", default=None)

# score comment 的截断：单条 rationale 与整段 comment 各设上限，防 judge 长篇大论灌爆 UI 那一栏。
_RATIONALE_CLIP = 200
_COMMENT_CLIP = 1500

# W3C Trace Context 的 traceparent：``00-<32 hex trace_id>-<16 hex span_id>-<2 hex flags>``。
# 只认 version 00；全零 id 按规范是非法值（「没有 trace」），一并拒掉。
_TRACEPARENT_RE = re.compile(r"^00-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$")


@dataclass(frozen=True)
class TraceParent:
    """跨进程传的那一截 trace 上下文（W3C traceparent 的三段）。

    **为什么手写而不用 OTel propagator 的 extract**：extract 出来的是 OTel Context，接续时要 attach
    进当前上下文；而接收方的根 span 由 Langfuse 起，官方接续口是 ``trace_context={trace_id,
    parent_span_id}``——它会把远端父 span 强制标成 sampled。走 attach 那条路，API 侧没开观测时
    发来的 flags=00 会让 ParentBased 采样器把 worker 整轮 span 丢掉，
    症状是「开着观测却一条都没有」。
    """

    trace_id: str
    span_id: str
    sampled: bool

    def header(self) -> str:
        return f"00-{self.trace_id}-{self.span_id}-{'01' if self.sampled else '00'}"


def parse_traceparent(value: str | None) -> TraceParent | None:
    """解析 traceparent；空串 / 格式不对 / 全零 id 返回 None（老消息没有这个字段，属正常）。"""
    m = _TRACEPARENT_RE.match((value or "").strip().lower())
    if m is None:
        return None
    trace_id, span_id, flags = m.groups()
    if int(trace_id, 16) == 0 or int(span_id, 16) == 0:
        return None
    return TraceParent(trace_id, span_id, sampled=bool(int(flags, 16) & 0x01))


def new_traceparent() -> TraceParent:
    """观测没开时也造一个 traceparent：trace_id 照样是两个进程日志的关联键。

    flags 记 00（未采样）——这一截确实没有被记录，接收方据此不去挂一个不存在的父 span。
    """
    return TraceParent(secrets.token_hex(16), secrets.token_hex(8), sampled=False)


def _current_otel_traceparent() -> TraceParent | None:
    """当前 OTel 上下文里的活动 span → traceparent；没有有效 span 返回 None。"""
    try:
        from opentelemetry import trace as otel_trace

        ctx = otel_trace.get_current_span().get_span_context()
    except Exception:
        return None
    if not ctx.is_valid:
        return None
    return TraceParent(
        f"{ctx.trace_id:032x}", f"{ctx.span_id:016x}", sampled=bool(ctx.trace_flags.sampled)
    )


def langfuse_host() -> str:
    """Langfuse 站点地址（``LANGFUSE_BASE_URL`` 优先，退 ``LANGFUSE_HOST``），未配则空串。"""
    return (os.environ.get("LANGFUSE_BASE_URL") or os.environ.get("LANGFUSE_HOST") or "").rstrip(
        "/"
    )


def trace_url(trace_id: str | None) -> str | None:
    """把 trace_id 拼成可点开的 Langfuse 链接；缺 trace_id / 未配 host 则 None。

    两个消费者：RT 告警消息里「最慢那次调用」的链接（``observability/alerts.py``），飞轮里
    「这条规则是哪条 bad case 生出来的」的证据链接（``scripts/eval/evolve_p0.py``）。各写一份
    会在改配置键时悄悄漂移，故收在这里。

    配了 ``LANGFUSE_PROJECT_ID`` 给直达 URL；否则走通用路径 ``/trace/<id>``——实测返回 307，
    重定向到 ``/project/<pid>/traces/<id>``，照样点得开，只是多一跳。
    """
    if not trace_id:
        return None
    host = langfuse_host()
    if not host:
        return None
    project = os.environ.get("LANGFUSE_PROJECT_ID")
    return f"{host}/project/{project}/traces/{trace_id}" if project else f"{host}/trace/{trace_id}"


@lru_cache(maxsize=1)
def _get_client() -> Any | None:
    """构造并缓存 Langfuse client 单例；任何不就绪条件 → ``None``（安静降级）。

    缓存的是**重的那个**——client 内含 OTEL exporter + 后台 flush 线程，全进程建一次即可，
    全进程共用。轻量的 ``CallbackHandler`` 则**每次 invoke 现建**（见
    :func:`apply_tracing`）：handler 持有 per-run 状态（``_runs`` 等），共享一个反而会在并发
    invoke 间串状态，故按 langfuse 官方口径「一次请求一个 handler」。
    """
    if not env_bool("LANGFUSE_ENABLED", False):
        return None
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        logger.info("LANGFUSE_ENABLED 为真但缺 PUBLIC/SECRET key，跳过观测（降级无 trace）")
        return None
    try:
        from langfuse import Langfuse

        # host 取 LANGFUSE_BASE_URL，缺省退 LANGFUSE_HOST；两者都不设 SDK 会静默发到 EU 默认区。
        return Langfuse(public_key=public_key, secret_key=secret_key, host=langfuse_host() or None)
    except Exception:
        # 缺包（ImportError）/ 鉴权 / 网络任何故障都吞掉，降级为无观测，绝不拖垮主链路。
        logger.warning("Langfuse 初始化失败，降级为无观测", exc_info=True)
        return None


def current_trace_id() -> str | None:
    """本轮（一次 ``run_agent``）的 trace_id；未启用观测 / 未 ``apply_tracing(root=True)`` 则 None。

    给 ``run_agent`` 收尾放进返回值用——**下游（评测）不该自己读 ContextVar**：``run_agent``
    在 API 层跑在 ``asyncio.create_task`` 里，子 task 的 ``set()`` 不回传父上下文，外面读到的
    永远是 None。显式返回是唯一稳的传法，理由详见 :func:`record_rubric_scores`。
    """
    return _current_trace_id.get()


def _rubric_comment(result: RubricResult) -> str:
    """把评测结论压成一段 comment，让人在 Langfuse 的 score 列表里**一眼看出为什么扣分**。

    这是原方案「5 分钟定位 badcase」的 Step 2 所依赖的那一栏——只写
    ``P0=x P1=y`` 这类计数没有信息量，真正要的是「哪个维度破了」+「judge 给的判定依据」。
    故这里带上失败维度名与对应 rationale（截断防灌爆 UI）。
    """
    lines = [
        f"total={result.total:.1f}/100  pass={result.overall_pass}  p2_avg={result.p2_avg:.2f}/5"
    ]
    if result.p0_failures:
        lines.append("P0 破: " + " | ".join(result.p0_failures))
    if result.p1_violations:
        lines.append("P1 违规: " + " | ".join(result.p1_violations))
    if not result.p0_failures and not result.p1_violations:
        lines.append("P0/P1 全过（若分低则是 P2 质量分不高）")

    # 只摘失败项的判定依据——通过项的 rationale 是噪声，占满 comment 反而看不见重点。
    for s in result.scores:
        if s.tier in ("P0", "P1") and s.passed is False and s.rationale:
            lines.append(f"— [{s.id}] {s.dimension}: {s.rationale[:_RATIONALE_CLIP]}")

    comment = "\n".join(lines)
    return comment[:_COMMENT_CLIP]


def record_rubric_scores(trace_id: str | None, result: RubricResult) -> None:
    """把 Rubric 评测结论作为 score 挂回**被评的那条 trace**。

    **不违反本模块的范围约定**（「只 trace 主对话链路，不碰 eval/ 与 judge LLM」）：这里注入的是
    对一条**已存在** trace 的标注，judge 自己的 LLM 调用不会因此变成 span。评测链路依旧不进 trace。

    **trace_id 必须由调用方显式传入，不读 ContextVar**：评测在 ``asyncio.gather`` 建的 task 里跑，
    读 ContextVar 眼下碰巧能拿到值，但只要哪天 ``run_agent`` 被多包一层 task，就会静默取到 None、
    分数**全部被丢弃且不报错**——最难查的那类故障。显式传参把它变成编译期可见的数据流。

    三个 score：``rubric_total``（0-1，UI 里按它筛低分 trace）、``rubric_pass``（BOOLEAN，P0 闸）、
    ``rubric_p2_avg``（1-5 原始质量分）。安静降级 + 异常吞掉，与本模块其余部分同一口径。
    """
    client = _get_client()
    if client is None or not trace_id:
        return
    try:
        client.create_score(
            name="rubric_total",
            value=result.total / 100.0,  # 归一到 0-1：Langfuse 的数值筛选按此刻度更直观
            trace_id=trace_id,
            data_type="NUMERIC",
            comment=_rubric_comment(result),
        )
        client.create_score(
            name="rubric_pass",
            value=1.0 if result.overall_pass else 0.0,  # BOOLEAN 的 value 走 0/1
            trace_id=trace_id,
            data_type="BOOLEAN",
        )
        client.create_score(
            name="rubric_p2_avg", value=result.p2_avg, trace_id=trace_id, data_type="NUMERIC"
        )
    except Exception:
        logger.warning("Langfuse 记录 rubric score 失败，跳过（不影响评测结论）", exc_info=True)


def flush_traces() -> None:
    """阻塞式 flush 待发送的 trace / score。**短命进程退出前必须调**。

    Langfuse SDK 靠后台线程批量上报。评测脚本这类「跑完就 ``SystemExit``」的进程，队列里没发出去的
    score 会随进程一起消失——现象是 trace 有、score 没有，且全程零报错。长驻的 API 进程不受影响
    （后台线程有的是时间 flush），所以这个坑只在 ``scripts/`` 里踩得到。
    """
    client = _get_client()
    if client is None:
        return
    try:
        client.flush()
    except Exception:
        logger.warning("Langfuse flush 失败，可能有 score 未上报", exc_info=True)


def tracing_middlewares() -> list[Any]:
    """Agent 装配时要挂的观测中间件（未启用 / 未装包 → 空表，装配处无需判断）。"""
    if _get_client() is None:
        return []
    try:
        from agentscope.middleware import TracingMiddleware

        return [TracingMiddleware()]
    except Exception:
        logger.warning("TracingMiddleware 构造失败，本次降级无观测", exc_info=True)
        return []


@contextmanager
def enqueue_span(
    *,
    task_id: str,
    session_id: str | None = None,
    user_id: str | None = None,
    kind: str = "normal",
) -> Iterator[TraceParent]:
    """API 进程入队那一段的 span，交出要随消息带走的 :class:`TraceParent`。

    **这是整条 trace 的根**：队列模式下 ``run_agent`` 跑在另一个进程，OTel 上下文靠 ContextVar
    过不去，只能把 traceparent 写进消息、由 worker 用 :func:`turn_span` 的 ``parent`` 接上。
    没这一段时一次请求在 trace 里断成两截，排队等了多久在任何一截里都看不见。

    无 client 时交出 :func:`new_traceparent` 造的那个——观测关了，日志关联照样要有。
    异常处理口径与 :func:`turn_span` 相同（业务异常原样穿透、观测自身故障降级）。
    """
    client = _get_client()
    if client is None:
        yield new_traceparent()
        return
    yielded = False
    body_exc: BaseException | None = None
    try:
        from langfuse import propagate_attributes

        with client.start_as_current_observation(
            name="shoppingx.enqueue",
            as_type="span",
            metadata={"task_id": task_id, "kind": kind},
        ):
            with propagate_attributes(session_id=session_id or None, user_id=user_id or None):
                parent = _current_otel_traceparent() or new_traceparent()
                yielded = True
                try:
                    yield parent
                except BaseException as exc:  # noqa: BLE001 —— 只做标记，紧接着原样抛回
                    body_exc = exc
                    raise
    except Exception as exc:
        if exc is body_exc:
            raise
        logger.warning("Langfuse 入队 span 创建失败，本次降级无观测", exc_info=True)
        if yielded:
            return
        yield new_traceparent()


@contextmanager
def turn_span(
    session_id: str | None = None,
    user_id: str | None = None,
    prompt_version: str | None = None,
    ab_bucket: int | None = None,
    parent: str = "",
) -> Iterator[Any]:
    """把一轮 ``run_agent`` 包成一条 trace 的根 span（无 client 时是个空壳，不改变行为）。

    根 span 必须由 Langfuse 自己的 tracer 起：它没有 ``gen_ai.*`` 属性，若用 AgentScope 的
    tracer 起会被 Langfuse 的 span 过滤器丢掉——子 span 照样上报，但 trace 少了根，UI 里
    看到的是一堆没有归属的观测。``propagate_attributes`` 负责把 session / user 顺着上下文
    抹到本轮所有子 span 上（Langfuse 的聚合查询按这两个维度做，只设在根上是不够的）。

    ``prompt_version`` / ``ab_bucket`` 走同一条 propagate 通道：前者用 Langfuse
    原生的 ``version`` 维度（UI 里能直接按版本切分对比），后者进 metadata。**两个都要**——只有
    版本时看不出「这个人是被分进来的还是手工钉的」，桶号是把线上 trace 与离线 A/B 报告对上账的
    唯一钥匙。

    ``parent``：队列消息带来的 traceparent（见 :func:`enqueue_span`）。有它时本轮挂在 API 那段
    span 下面、与它同一个 trace_id；空串 / 解析不了就照旧自成一条 trace（直连模式、离线脚本）。
    上游没采样（flags=00，API 侧观测关着）时只沿用 trace_id、不挂父 span——那个父 span 并不存在。
    """
    client = _get_client()
    if client is None:
        yield None
        return
    remote = parse_traceparent(parent)
    trace_context: dict[str, str] | None = None
    if remote is not None:
        trace_context = {"trace_id": remote.trace_id}
        if remote.sampled:
            trace_context["parent_span_id"] = remote.span_id
    yielded = False
    body_exc: BaseException | None = None
    try:
        from langfuse import propagate_attributes

        with client.start_as_current_observation(
            name="shoppingx.turn",
            as_type="agent",
            trace_context=trace_context,
        ) as span:
            metadata = {"ab_bucket": ab_bucket} if ab_bucket is not None else None
            with propagate_attributes(
                session_id=session_id or None,
                user_id=user_id or None,
                version=prompt_version or None,
                metadata=metadata,
            ):
                _current_trace_id.set(client.get_current_trace_id())
                yielded = True
                try:
                    yield span
                except BaseException as exc:  # noqa: BLE001 —— 只做标记，紧接着原样抛回
                    body_exc = exc
                    raise
    except Exception as exc:
        # 业务异常（with 体里抛的，被 contextlib throw 回这个 yield 点）必须**原样穿透**：
        # 早先这里连它一起吞掉又 yield 了第二次，Python 只好报 "generator didn't stop after
        # throw()" 的 RuntimeError，把真实报错盖死——批 3 验收时 q03 撞上游内容审核，终端只见
        # RuntimeError，真凶 ``openai.APIError: ...inappropriate content`` 要往上翻 50 行栈。
        if exc is body_exc:
            raise
        # 剩下的才是观测自身的毛病（起 span 失败 / span 收尾报错），绝不反噬主链路。
        logger.warning("Langfuse 根 span 创建失败，本轮降级无观测", exc_info=True)
        if yielded:
            return  # span 早交出去了，主链路已跑完；再 yield 一次就是上面那个 RuntimeError
        yield None
