"""一轮任务的**运行时无关**周边：当轮上下文拼装、产物落盘、配额记账。

这三件事都不碰 Agent 框架——它们吃的是 query / P_t / 会话目录 / 用量快照，产出的是给模型看的
一段文本、磁盘上的两个文件、账本里的一条记录。批 0 迁移期它们住在 ``main_agent.py``（LangChain 版
run_agent 的私有函数），``orchestrator.py`` 反向 import 过去用，理由是「两份实现漂移是迁移期最难
查的 bug」。L8 摘掉 LangChain 后 main_agent 整体删除，这些函数搬到这里成为独立模块——它们本来
就与「跑在哪个运行时」无关，塞在某个运行时的实现文件里只是历史包袱。

搬迁时顺手改掉一处**与工具表脱节的提示词**：``_render_platform_block`` 曾指挥模型
``parallel_dispatch_tool``，而那个工具在 L3 已被 ``task_dispatch``（同轮多派 + 框架并发）取代。
给模型的动机文本必须跟着工具表走，否则它照着念一个不存在的名字，只能撞 tool-not-found 白烧一轮。
"""

import asyncio
import contextlib
import json
import logging
from collections.abc import Sequence
from pathlib import Path

from app.db.quota import add_usage
from app.memory.injector import HISTORY_EMPTY
from app.memory.session_state import SessionPrefState
from app.tools.shopping_summary import ShoppingSummaryOutput
from app.utils.env import env_int

logger = logging.getLogger("shoppingx.session_io")

# 一轮任务的总时限（防失控之③，报错收场）。看门狗在远早于它的位置先给用户提示，见
# harness/hooks/watchdog.py。可经 env 覆盖以适配不同模型时延。
MAIN_AGENT_TIMEOUT_SEC = env_int("MAIN_AGENT_TIMEOUT_SEC", 300)


def render_platform_block(enabled: tuple[str, ...]) -> str:
    """渲染 ``<enabled_platforms>``——本轮允许检索的平台（用户在前端设置里勾的，默认只 amazon）。

    单平台时告诉主 Agent **跨平台泛搜那条路别派 worker**：一个平台派不出并行，只剩开销
    （一个子 Agent 的完整上下文 + 一轮往返）——这正是「单干优先」要挡住的场景。

    **但禁令只针对「按平台切分」这一种派发**。原先这里写的是「不要派 task_dispatch 检索」，
    一句话把并行来源钉死在平台维度上：线上默认单平台（语料 99.75% 在 amazon），于是这条注入
    每轮都在关掉 worker 的总闸——多品类并列（跑鞋 + 耳机分头查，与平台数无关）也被它拦下，
    实测模型老老实实串行搜了三次（评测 pl02）。这是「零派发」的直接成因之一。

    这只是给模型的**动机**；真正的硬保证在机制层（``task_dispatch`` 丢弃未启用平台的 demand、
    item_search 的 Qdrant filter 收口到启用集合）——prompt 打动机、机制打保证。
    """
    names = " / ".join(enabled)
    if len(enabled) == 1:
        return (
            f"<enabled_platforms>\n本次只启用 **{names}** 一个平台（用户未开启多平台比价）。\n"
            f"- 跨平台泛搜**不要**派 task_dispatch：只有一个平台，按平台切分没有并行收益，"
            f'直接在主流程 item_search(platform="{names}") 检索、精挑、收尾。\n'
            f"- **但多类并列照常并行派**（plan 的 slot_mode=parallel）：那是按**品类**切分，"
            f"与平台数无关——一类一条 task_dispatch、同一轮里一起发。\n"
            f"- 比价 / 到手价照常算，但只在该平台内部的候选之间比。\n"
            f"- 收尾时如实说明「本次只搜了 {names}」，不要暗示比过其它平台。\n"
            "</enabled_platforms>"
        )
    return (
        f"<enabled_platforms>\n本次启用 {len(enabled)} 个平台：{names}。\n"
        f"- 跨平台泛搜按 <tool_policy>：**同一轮发出 {len(enabled)} 个 task_dispatch**"
        f"（subagent_type=\"search\"、一平台一条、只列这些平台），框架会并发执行；"
        f"不要派未启用的平台，也不要一条条串行发。\n"
        "</enabled_platforms>"
    )


def inject_runtime_context(
    query: str,
    history_block: str,
    pt: SessionPrefState,
    enabled_platforms: tuple[str, ...] = (),
    prior_candidates: str = "",
    image_paths: Sequence[str] = (),
) -> str:
    """把运行时用户上下文（启用平台 + 近期行为历史 + 会话级 P_t）拼进本轮 query 前，组成当轮
    用户消息——而不是塞进 system prompt。

    它们都**每轮必变**：历史每轮收尾覆盖、P_t 每轮更新。system prompt 在请求里排在 messages 之前，
    把任何每轮变的东西混进去，都会连累它自己 + 它后面「本该跨轮稳定」的全部历史一起打断 prompt
    cache 前缀。这条用户消息排在**干净的** ``prior_turns``(q,a) 之后、是缓存断点之后永不缓存的
    部分（对齐 refdocs/05 §4.4「按易变性分层，越易变越靠后」）。空的块跳过（不塞「暂无」占位，
    省 token 也不给模型噪声）；全空则原样返回 query。

    **长期偏好不在这里注入**（这是 P_t 重构改掉的）。它曾经拼在这条消息的最前面，而那时 planner
    还没跑、``session_domains`` 还是空的——``injector._in_scope`` 对空域一律放行，于是模型看到的
    偏好块**必然是跨域全量**的：「买跑鞋时不要皮革」会出现在买旅行包的这一轮，模型很自觉地把
    leather 转述进 ``item_picker(exclude_keywords=...)``，硬淘汰就这么绕过域闸生效了。
    改由 ``harness.hooks.preference_inject`` 在 planner **之后**注入域内偏好——那时域才存在。
    """
    parts: list[str] = []
    # 启用平台随用户设置而变（默认单平台 amazon），同属「每轮可变」——与历史/P_t 一样走
    # 用户消息，不进 system prompt（否则打断跨轮稳定的 cache 前缀）。
    if enabled_platforms:
        parts.append(render_platform_block(enabled_platforms))
    if history_block and history_block != HISTORY_EMPTY:
        parts.append(f"<user_recent_history>\n{history_block}\n</user_recent_history>")
    if not pt.is_empty():
        parts.append(f"<session_constraints>\n{pt.render()}\n</session_constraints>")
    # 上一轮已检索、已登记的候选：让「只要防水的」这类追问能直接在既有候选上过滤（item_picker），
    # 而不是把 planner → item_search → price_compare 整条链重跑一遍。候选体本身仍在工具内 hydrate，
    # 这里只给模型看 item_id + 决策字段（compact 投影）。
    if prior_candidates:
        parts.append(
            "<prior_candidates>\n"
            "上一轮已检索并登记的候选（本会话内可直接按 item_id 复用，无需重新检索）：\n"
            f"{prior_candidates}\n"
            "</prior_candidates>"
        )
    # 参考图（M20）：只报**文件名**，图本身不进 messages——主模型是纯文本的，多模态消息塞进来只会
    # 报错或被静默忽略。图关在 image_understand 工具里，它的识别结果已由 Harness 在开局预跑写进上文
    # （先于 planner）。这条块只交代「用户是拿图来买东西的」这个意图，免得模型把上文那条
    # image_understand 结果当成无主的噪声。
    if image_paths:
        names = "\n".join(f"- {name}" for name in image_paths)
        parts.append(
            "<reference_images>\n"
            "用户本轮上传了参考图，想买「和图里这个类似的商品」。图已由系统识别，结论见上文的 "
            "image_understand 工具结果——按它的 search_query / keywords 检索即可，不必重复调用。\n"
            "若识别结果为降级（degraded=true），说明图没看成：如实告诉用户并请他用文字描述，别硬编。\n"
            "**多主体消歧**：识别结果的 multi_subject=true 时，图里有多件可买的商品（见 objects），"
            "而 subject / search_query 只描述了其中一件——直接拿去搜就是在替用户瞎猜。此时：\n"
            "- 用户消息里已指明要哪件（如「找图里那个包」「这双鞋多少钱」，或「黑色那个」而 "
            "objects 里只有一件是黑的）→ 按用户指的那件重写检索词直接搜，**不要多问**。\n"
            "- 用户消息没提是哪件（只说了预算、平台等与选件无关的信息，或什么都没说）→ 先调 "
            "ask_user 列出 objects 问清楚要找哪件，拿到回复再搜。别默认挑最大的那件。\n"
            f"{names}\n"
            "</reference_images>"
        )
    if not parts:
        return query
    return "\n\n".join(parts) + f"\n\n用户本轮消息：\n{query}"


def write_session_artifacts(
    session_dir: Path, final_text: str, summary: ShoppingSummaryOutput | None
) -> None:
    """把本次任务产物落到会话目录，供 ``GET /api/files/<thread_id>/<name>`` 下载（M10）。

    - ``summary.md``：购物清单文案。**优先用 shopping_summary 的结构化 ``summary`` 字段**
      （那才是精挑清单本体），只有没终结产物时（闲聊兜底）才退回 ``final_text``——否则模型
      收尾若多说一句「希望对你有帮助」当作 final_text，下载到的 md 就只剩那句废话。
    - ``result.json``：完整结构化结果（机器读，前端商品卡 / 二次处理用），仅在有终结结果时写。

    产物落 ``output/<thread_id>/``（已 gitignore）。写文件失败不该拖垮主任务——产物是附带
    交付物，记日志降级即可，主链路的偏好写回与 task_result 上报照常进行。故捕获面放宽到
    ``Exception``：除磁盘 OSError，``model_dump``/``json.dumps`` 万一抛也得照样降级，不反噬主链路。
    """
    md = summary.summary if summary is not None else final_text
    try:
        (session_dir / "summary.md").write_text(md or "", encoding="utf-8")
        if summary is not None:
            (session_dir / "result.json").write_text(
                json.dumps(summary.model_dump(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    except Exception:
        logger.warning("写会话产物失败（session_dir=%s），降级跳过", session_dir, exc_info=True)


# 在途的配额记账 task：只为保一个强引用——detached task 若无人引用会被 GC 掉，账就丢了。
_pending_charges: set[asyncio.Task[None]] = set()


async def charge_quota(
    user_id: str | None, snap: dict[str, float | int], prompt_version: str = ""
) -> None:
    """把本轮全树成本记进用户配额账本，**取消路径下也要记完**。

    为什么绕这一圈而不是直接 ``await add_usage(...)``：本函数跑在 run_agent 的 finally 里，而这条
    路径最常见的触发者恰恰是「用户点了取消」——此时本 task 已被 cancel，直接 await 会在第一个挂起
    点就抛 CancelledError，记账协程连库都摸不到，用户就凭「烧完 token 再取消」白嫖了一整轮。
    ``shield`` 让记账在独立 task 里跑到完，外层的取消信号照常传播（suppress 只吞掉 shield 这个
    await 点二次抛出的 CancelledError，不影响 run_agent 里原本那条 raise）。
    """
    task = asyncio.create_task(
        add_usage(
            user_id,
            float(snap["cost_usd"]),
            int(snap["input_tokens"]),
            int(snap["output_tokens"]),
            prompt_version=prompt_version,
        )
    )
    _pending_charges.add(task)
    task.add_done_callback(_pending_charges.discard)
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.shield(task)
