"""Langfuse 在线观测接入（v4 / OpenTelemetry）——主链路调试用。

**范围**：只 trace 主对话链路（``run_agent`` 主 loop + 派出去的 worker），不碰 ``eval/`` 与
judge LLM——评测链路进了 trace 只会把线上数据搅浑。

**零胶水的由来**：Langfuse v4 本身就是 OTel SDK 的包装，client 初始化时会把自己的
TracerProvider 设成全局；而 AgentScope 原生的 ``TracingMiddleware`` 打的是标准 ``gen_ai.*``
语义属性，正好落进 Langfuse 的 span 过滤器（``is_genai_span``）放行的那一类。所以主链路的观测
= 装配时挂一个框架自带的中间件，**不需要自建 exporter，也不需要手工传 trace_id**：OTel 上下文
本身就是 ContextVar，worker 的 span 天然挂在父 span 下。

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
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from app.utils.env import env_bool

if TYPE_CHECKING:  # 只为类型标注；运行时不 import，避免 agent → eval 的反向依赖
    from app.eval.rubric import RubricResult

logger = logging.getLogger("shoppingx.tracing")


# 本轮（一次 run_agent）的 trace_id：主 loop 在 root 处生成并写入，fork 子 loop 读出来复用，
# 从而把主 + 所有子的多次独立 ainvoke 归并到同一条 trace。ContextVar 天然按 async 上下文隔离
# （多用户并发各有各的），且新建的 asyncio.Task（如 parallel_dispatch 的 gather）建时即拷贝当前
# 上下文 → 子 task 能读到父设的值。每轮 run_agent 都重新生成，不跨轮复用、无需手动 reset。
_current_trace_id: ContextVar[str | None] = ContextVar("langfuse_trace_id", default=None)

# score comment 的截断：单条 rationale 与整段 comment 各设上限，防 judge 长篇大论灌爆 UI 那一栏。
_RATIONALE_CLIP = 200
_COMMENT_CLIP = 1500


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
    主 loop 与所有 fork 共用。轻量的 ``CallbackHandler`` 则**每次 invoke 现建**（见
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

    这是 refdocs 16-3 §5「5 分钟定位 badcase」的 Step 2 所依赖的那一栏——只写
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
    """把 Rubric 评测结论作为 score 挂回**被评的那条 trace**（refdocs 16-3 §2.4）。

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


def record_trace_scores(scores: dict[str, float]) -> None:
    """把数值指标作为 score 挂到**本轮 trace**（携带量 / 缓存命中率等，见 usage.py）。

    安静降级：无 client（未启用 / 缺 key）或本轮没有 trace_id（root 未挂 tracing）一律跳过；
    任何异常吞掉，绝不反噬主链路。score 用 ``create_score``（langfuse v4）按 ``trace_id`` 关联。
    """
    client = _get_client()
    if client is None:
        return
    trace_id = _current_trace_id.get()
    if not trace_id:
        return
    try:
        for name, value in scores.items():
            client.create_score(
                name=name, value=float(value), trace_id=trace_id, data_type="NUMERIC"
            )
    except Exception:
        logger.warning("Langfuse 记录 score 失败，跳过（不影响主链路）", exc_info=True)


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
def turn_span(
    session_id: str | None = None,
    user_id: str | None = None,
    prompt_version: str | None = None,
    ab_bucket: int | None = None,
) -> Iterator[Any]:
    """把一轮 ``run_agent`` 包成一条 trace 的根 span（无 client 时是个空壳，不改变行为）。

    根 span 必须由 Langfuse 自己的 tracer 起：它没有 ``gen_ai.*`` 属性，若用 AgentScope 的
    tracer 起会被 Langfuse 的 span 过滤器丢掉——子 span 照样上报，但 trace 少了根，UI 里
    看到的是一堆没有归属的观测。``propagate_attributes`` 负责把 session / user 顺着上下文
    抹到本轮所有子 span 上（Langfuse 的聚合查询按这两个维度做，只设在根上是不够的）。

    ``prompt_version`` / ``ab_bucket`` 走同一条 propagate 通道（批 4 / 18-3）：前者用 Langfuse
    原生的 ``version`` 维度（UI 里能直接按版本切分对比），后者进 metadata。**两个都要**——只有
    版本时看不出「这个人是被分进来的还是手工钉的」，桶号是把线上 trace 与离线 A/B 报告对上账的
    唯一钥匙。
    """
    client = _get_client()
    if client is None:
        yield None
        return
    yielded = False
    body_exc: BaseException | None = None
    try:
        from langfuse import propagate_attributes

        with client.start_as_current_observation(name="shoppingx.turn", as_type="agent") as span:
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
