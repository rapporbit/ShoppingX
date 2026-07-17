"""Langfuse 在线观测接入（v4 / OpenTelemetry）——主链路调试用。

**范围**：只 trace 主对话链路（``run_agent`` 主 loop + fork 出去的子 Agent），不碰
``eval/`` 与 judge LLM。落地手段就是「挂在 **invoke 层** 的 ``config["callbacks"]``，
**不挂在 ``get_llm()`` 工厂**」——工厂被 judge / 评测共享，挂那儿会把离线评测链路一并卷进 trace。

**一轮 = 一条 trace（关键设计）**：langfuse v4 的 CallbackHandler 对**每一次 ``ainvoke``** 都
按「顶层 run（``parent_run_id is None``）即新 root」起一条**独立 trace**；它建 span 用的是
``start_observation`` 而非 ``start_as_current_observation``，**不会把 span 设成 OTEL 活跃 span**，
所以子 Agent 的 ``ainvoke`` 跨边界拿不到父的活跃上下文，会各自另起一条顶层 trace（实测：一轮带
3 个平台 fork 的对话 → 4 条 root trace，散落难看）。

修法：**主 loop 生成一个 ``trace_id`` 放进 ContextVar，fork 子 loop 读出来复用**——给各自的
CallbackHandler 传 ``trace_context={"trace_id": <同一个>}``（v4 显式 trace 关联口径），主 + 所有
子归并到**同一条 trace**。多轮对话再靠 ``session_id``（=thread_id）聚成一个 **Session**（UI 的
Sessions 视图按会话看；Tracing 的 Observations 表是把所有 span 摊平，不是「一对话一行」的地方）。

**fork 双记坑（务必留意）**：子 loop **不能**在父 handler 已在场时再挂一个自己的 handler。
LangGraph 的 ``ensure_config`` 对 ``callbacks`` 走的是**合并**（``_merge_callbacks``），而非
LangChain 原生 ``ensure_config`` 的**覆盖**——工具体内 ``var_child_runnable_config`` 里躺着父
loop 的 handler（LangChain 执行工具时写入的「本次 tool run 的子 callback manager」），子 Agent 的
``ainvoke`` 于是同时带着父 + 子两个 handler，**同一次 LLM 调用被记两遍**（实测：5 平台 fork 的
generation 与 item_search span 全部成对出现，trace 里 token 虚高 65%，成本分析全歪）。故
:func:`apply_tracing` 在 ``root=False`` 且上下文里已有 Langfuse handler 时**直接返回**、不再追加：
父 handler 本就绑着同一个 ``trace_id``，且子 run 挂在 ``dispatch_tool`` 的 span 下 → 继承它不但不丢
观测，层级还更准（子 Agent 嵌在工具里，而非平铺在 trace 根）。

**安静降级铁律**：观测是调试附属品，绝不能反噬主链路。没装 langfuse 包 / ``LANGFUSE_ENABLED``
非真 / 缺 PUBLIC|SECRET key / client 初始化失败，一律让 :func:`apply_tracing` 原样返回 config 且
**绝不抛**，主链路照跑、只是没 trace。

**host 坑**：SDK 不设 host 会**静默发到 EU 默认区**。这里显式把 ``LANGFUSE_BASE_URL``
（缺省退 ``LANGFUSE_HOST``）喂给 client，``.env.example`` 用的是 ``https://us.cloud.langfuse.com``。

**v4 API 形状**：session/user 不传给 ``CallbackHandler()``，经 config 的 metadata 特殊键
``langfuse_session_id`` / ``langfuse_user_id`` 关联；trace 归并经 ``trace_context=`` 传同一 id。
"""

from __future__ import annotations

import logging
import os
from contextvars import ContextVar
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import var_child_runnable_config

from app.utils.env import env_bool

if TYPE_CHECKING:  # 只为类型标注；运行时不 import，避免 agent → eval 的反向依赖
    from app.eval.rubric import RubricResult

logger = logging.getLogger("shoppingx.tracing")

# v4 经 runnable config 的 metadata 这两个特殊键关联 session / user（CallbackHandler 内部识别）。
_SESSION_META_KEY = "langfuse_session_id"
_USER_META_KEY = "langfuse_user_id"

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


def _make_handler(client: Any, trace_id: str) -> Any | None:
    """为本次 invoke 现建一个绑定 ``trace_id`` 的 CallbackHandler；失败则 None（安静降级）。"""
    try:
        from langfuse.langchain import CallbackHandler

        return CallbackHandler(trace_context={"trace_id": trace_id})
    except Exception:
        logger.warning("构造 Langfuse CallbackHandler 失败，本次降级无观测", exc_info=True)
        return None


def _inherited_handlers() -> list[Any]:
    """当前 runnable 上下文里**会被子 run 继承**的那批 callback handler。

    LangChain 执行工具时会把「本次 tool run 的子 callback manager」写进
    ``var_child_runnable_config``，父 loop 的 Langfuse handler 就在里面（作为 inheritable
    handler）。``dispatch_tool`` 的函数体正跑在这个上下文里，故这里读得到。
    """
    ambient = var_child_runnable_config.get()
    if not ambient:
        return []
    callbacks = ambient.get("callbacks")
    if callbacks is None:
        return []
    if isinstance(callbacks, list):
        return list(callbacks)
    # BaseCallbackManager：只有 inheritable_handlers 会往子 run 传（handlers 可能含本层专属的）。
    return list(getattr(callbacks, "inheritable_handlers", []))


def _langfuse_handler_inherited() -> bool:
    """父 run 的 Langfuse handler 是否已在上下文里 —— 在，则子 loop 会自动继承它，别再挂第二个。"""
    try:
        from langfuse.langchain import CallbackHandler
    except Exception:  # 没装包 → 谈不上继承；与本模块「安静降级」同一口径
        return False
    return any(isinstance(h, CallbackHandler) for h in _inherited_handlers())


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


def trace_config_extra(session_id: str | None = None, user_id: str | None = None) -> dict[str, Any]:
    """构造挂进 invoke config 的 langfuse 元数据（session / user 关联）。

    空值不写——避免把空串当成一个 session/user 关联键污染 trace。
    """
    meta: dict[str, Any] = {}
    if session_id:
        meta[_SESSION_META_KEY] = session_id
    if user_id:
        meta[_USER_META_KEY] = user_id
    return {"metadata": meta} if meta else {}


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


def apply_tracing(
    config: RunnableConfig,
    *,
    session_id: str | None = None,
    user_id: str | None = None,
    root: bool = False,
) -> RunnableConfig:
    """把 callbacks + session/user 元数据 + 本轮 trace_id 就地合并进 config；无 client 原样返回。

    - ``root=True``（主 loop）：生成本轮 trace_id 写入 ContextVar，并设 ``session_id`` /
      ``user_id``，trace 落在对应 session / user 下。
    - ``root=False``（fork 子 loop）：父 handler 已在上下文里（LangGraph 会合并继承）时**直接返
      回**，不追加第二个 handler——否则同一次 LLM 调用被父 + 子各记一遍（见模块 docstring
      「fork 双记坑」）。父 handler 不在场时（examples / 测试直调子 Agent）才自己挂一个，并**复用**
      ContextVar 里父设的 trace_id，使子 Agent 的独立 ``ainvoke`` 归并进同一条 trace。

    合并而非覆盖：``callbacks`` 追加（不踩调用方已挂的别的回调），``metadata`` 增量更新。
    """
    client = _get_client()
    if client is None:
        return config

    if not root and _langfuse_handler_inherited():
        return config

    if root:
        # 每轮一个新 trace_id。
        trace_id = client.create_trace_id()
    else:
        # 复用父设的 trace_id；理论上 fork 必在 root 之后，缺失则保守另起一个（退化为独立 trace）。
        trace_id = _current_trace_id.get() or client.create_trace_id()

    handler = _make_handler(client, trace_id)
    if handler is None:
        return config

    if root:
        # **建 handler 成功之后**才写进 ContextVar，供本轮内所有 fork 复用。顺序要紧：先 set 再建，
        # 一旦 handler 构造失败就会留下一个「有 id、没 trace」的 ContextVar，随后 record_*_scores
        # 会把分数挂到一条根本不存在的 trace 上（孤儿 score，UI 里查无此人）。
        _current_trace_id.set(trace_id)

    callbacks = config.setdefault("callbacks", [])
    if isinstance(callbacks, list):
        callbacks.append(handler)
    extra = trace_config_extra(session_id=session_id, user_id=user_id)
    if extra.get("metadata"):
        config.setdefault("metadata", {}).update(extra["metadata"])
    return config
