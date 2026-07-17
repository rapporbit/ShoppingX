"""断点之后的压缩动作 + 断点之前的 cache_control 标记（L2 Cache-Aware 微压缩）。

两件事，对应 ROADMAP M6：
  - :func:`compress_after_breakpoint`：压缩**较旧区** ``[:breakpoint]`` 里超量的工具结果，
    保留**最近 K 轮** ``[breakpoint:]`` 全文。这样累计 token 不随轮数线性爆炸（每条旧的大
    工具结果都被截到上限），而模型当前正在用的近几轮仍是全文。
  - :func:`apply_cache_control`：给较旧区末尾消息打 ephemeral 缓存标记，落实
    「单请求 ≤4 个标记 / 最小写入阈值（1024 token）」两条硬约束。

命名说明（承接 breakpoint.py 对 refdocs 的更正）：函数名 ``compress_after_breakpoint`` 沿用
ROADMAP，但**实际压缩的是断点之前的较旧区**——因为缓存前缀必须逐字稳定，能动的只有不再
被频繁复用的旧历史；最近 K 轮要留全文给模型当工作集。详见 breakpoint.py 顶部说明。

cache_control 落地口径（对 refdocs/05 的实现更正）：标记写进消息的 **content-block**
（``{"type":"text","text":...,"cache_control":{"type":"ephemeral"}}``），不挂 ``additional_kwargs``
——后者只有 langchain-anthropic 会读，``langchain_openai`` 不转发它，在 OpenAI 兼容端点发不到线上。
本项目 LLM 走 DashScope（OpenAI 兼容），其 Qwen / DeepSeek 消费 content-block 上的 ``cache_control``
（显式缓存，命中计费 10%、5min 内保证命中；实测 ``cached_tokens`` 命中）。注意打标记会把该条
消息 content 从字符串改写成 block 列表（DashScope 显式缓存要求的格式），转换确定性、幂等。
是否启用由上层开关控制（``COMPRESS_CACHE_CONTROL``）。
"""

import json
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage

from app.utils.tokens import count_tokens, truncate_to_token_budget

# 每条工具结果在「较旧区」的 token 上限。比 fork 边界的硬截断
# （middleware.MAX_TOOL_RESULT_TOKENS=4000）更紧——旧历史细节已不重要，压更狠以省 token。
# token 数走 app.utils.tokens（Qwen 本地分词器为主、CJK 启发式兜底），不再用 char/4 粗估
# （后者对中文低估约 3 倍，见 tokens.py）。
DEFAULT_MAX_TOOL_TOKENS = 1500
_TRUNCATE_HINT = "\n\n[…较旧工具结果已精简；如需细节可用更窄查询重取]"

# ---- JSON 字段抽取（较旧区工具结果智能压缩）----
# 工具返回的候选列表字段名（item_search/price_compare/shipping_calc/item_picker）。
_CANDIDATE_LIST_KEYS = frozenset({"candidates", "ranked", "items", "picks"})
# 较旧区只保留决策相关字段：身份+标题+价格+评分+到手价+入选理由。
_KEEP_FIELDS = frozenset(
    {
        "item_id",
        "platform",
        "title",
        "price_usd",
        "rating",
        "landed_usd",
        "pick_reason",
    }
)


def _extract_candidate_fields(candidate: dict) -> dict:
    """从单个候选 dict 抽取决策字段，丢弃冗余（brand/score/weight_kg/…）。"""
    return {k: v for k, v in candidate.items() if k in _KEEP_FIELDS and v}


def _smart_compress_json(text: str, max_tokens: int) -> str | None:
    """尝试 JSON 字段抽取压缩：解析工具返回的 JSON，对候选列表做字段精简。

    成功返回压缩后的 JSON 字符串（保证合法 JSON + 在 token 预算内）；
    解析失败 / 无候选列表 / 压缩后仍超限则返回 None，调用方回退到尾部截断。
    """
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(data, dict):
        return None

    compressed_any = False
    for key in _CANDIDATE_LIST_KEYS:
        if key in data and isinstance(data[key], list) and data[key]:
            data[key] = [_extract_candidate_fields(c) for c in data[key] if isinstance(c, dict)]
            compressed_any = True

    if not compressed_any:
        return None

    result = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    if count_tokens(result) <= max_tokens:
        return result
    return None


# Anthropic cache_control 硬约束（refdocs/05 §4.4）。
MAX_CACHE_MARKERS = 4
# Sonnet 写入缓存的最小前缀 token 阈值；不足则写了也不会被缓存，白占一个标记额度。
MIN_CACHE_PREFIX_TOKENS = 1024


def _content_text(content: object) -> str:
    """把消息 content 归一成可估长度的字符串（content 可能是 str 或 block 列表）。"""
    return content if isinstance(content, str) else str(content)


def compress_after_breakpoint(
    messages: Sequence[AnyMessage],
    breakpoint_idx: int,
    *,
    max_tool_tokens: int = DEFAULT_MAX_TOOL_TOKENS,
) -> list[AnyMessage]:
    """压缩较旧区 ``[:breakpoint_idx]`` 中超量的工具结果；最近 K 轮 ``[breakpoint_idx:]`` 全文保留。

    只截断 ToolMessage 的**字符串载荷**（尾部截断 + 留提示），不删条目、不改顺序、不碰其它消息
    类型。这样消息结构稳定、不重排。

    缓存语义（诚实说明，别夸大）：真正逐字稳定、能持续命中缓存的是**较旧区 ``[:bp]``**——它里面的
    工具结果一旦被截断就不再变，且这个区只增不减，所以缓存前缀逐轮单调变长。代价是**边界处会有
    一次性变化**：一条工具结果在「最近 K 轮」时是全文，等它被新轮次挤出窗口、跨入较旧区那一轮会被
    截断一次（此后冻结）。这是滚动式增量缓存的正常现象（跨界消息只重算一次），不是「整段已发前缀
    永不变」。保留最近 K 轮全文是**刻意取舍**：让模型看到刚拿到的工具结果全貌（精度），用边界处的
    一次性重算换取它——而不是为严格字节稳定把最新结果也压了。

    非字符串 content（如多模态 block 列表）一律跳过，避免把结构化内容 ``str()`` 化后截断造成损坏。
    ``breakpoint_idx`` 会被夹紧到 ``[0, len(messages)]``，越界也不崩。
    """
    bp = max(0, min(breakpoint_idx, len(messages)))

    compressed: list[AnyMessage] = []
    for i, msg in enumerate(messages):
        # content 先取出：下面会重绑定 msg，绑定后 isinstance 收窄的类型就丢了。
        content = msg.content
        # 只压较旧区、且只压「字符串载荷」的工具结果（block 列表不碰，防结构损坏）。
        # 已带截断提示的不再压（幂等护栏：避免 token 估算微漂移导致二次截断、破坏前缀稳定）。
        if (
            i < bp
            and isinstance(msg, ToolMessage)
            and isinstance(content, str)
            and not content.endswith(_TRUNCATE_HINT)
            and count_tokens(content) > max_tool_tokens
        ):
            extracted = _smart_compress_json(content, max_tool_tokens)
            if extracted is not None:
                msg = msg.model_copy(update={"content": extracted})
            else:
                truncated = truncate_to_token_budget(content, max_tool_tokens, _TRUNCATE_HINT)
                msg = msg.model_copy(update={"content": truncated})
        compressed.append(msg)
    return compressed


_EPHEMERAL: dict[str, str] = {"type": "ephemeral"}


def _block_has_marker(content: object) -> bool:
    """content 是否已是带 cache_control 的 block 列表。"""
    return isinstance(content, list) and any(
        isinstance(b, dict) and "cache_control" in b for b in content
    )


def _count_existing_markers(messages: Sequence[AnyMessage]) -> int:
    return sum(1 for m in messages if _block_has_marker(m.content))


def _to_marked_blocks(content: object) -> list[Any] | None:
    """把消息 content 转成带 ephemeral cache_control 的 content-block 列表。

    - 非空字符串 → 单个 text block 带标记。
    - 已是 block 列表 → 给末尾的 dict block 补标记（已带则原样返回）。
    - 空串 / 空列表 / 其它标量 → 返回 None（无法安全标记，调用方应跳过该条）。

    DashScope / OpenAI 兼容端点的显式缓存就认 content-block 上的 cache_control（见模块 docstring）。
    """
    if isinstance(content, str):
        if not content:
            return None
        return [{"type": "text", "text": content, "cache_control": dict(_EPHEMERAL)}]
    if isinstance(content, list) and content:
        blocks: list[Any] = list(content)
        for j in range(len(blocks) - 1, -1, -1):
            if isinstance(blocks[j], dict):
                if "cache_control" in blocks[j]:
                    return blocks  # 已标记，幂等返回
                blocks[j] = {**blocks[j], "cache_control": dict(_EPHEMERAL)}
                return blocks
        return None
    return None


def apply_cache_control(
    messages: Sequence[AnyMessage],
    breakpoint_idx: int,
    *,
    max_markers: int = MAX_CACHE_MARKERS,
    min_prefix_tokens: int = MIN_CACHE_PREFIX_TOKENS,
) -> list[AnyMessage]:
    """给缓存前缀 ``[:breakpoint_idx]`` 的末尾消息打 content-block ephemeral cache_control。

    落实两条硬约束：
      - **≤max_markers**：含已有标记在内超额则不再打（第 5 个会被忽略、白写）。
      - **最小写入阈值**：前缀 token 不足 ``min_prefix_tokens``（DashScope 显式缓存为 1024）不打。

    从断点前最后一条往前找第一条能安全标记的消息（跳过空 content 的 AIMessage 等），把它的
    content 改写成带 cache_control 的 block 列表——这是 DashScope/OpenAI 兼容端点真正消费的格式
    （``additional_kwargs`` 上的标记不会被 ``langchain_openai`` 转发）。返回新列表（浅拷贝 +
    被标记那条做 model_copy），不原地改入参；转换确定性、幂等。
    """
    out = list(messages)
    if breakpoint_idx <= 0 or breakpoint_idx > len(out):
        return out
    if _count_existing_markers(out) >= max_markers:
        return out

    prefix = out[:breakpoint_idx]
    prefix_tokens = sum(count_tokens(_content_text(m.content)) for m in prefix)
    if prefix_tokens < min_prefix_tokens:
        return out

    for i in range(breakpoint_idx - 1, -1, -1):
        marked = _to_marked_blocks(out[i].content)
        if marked is not None:
            out[i] = out[i].model_copy(update={"content": marked})
            return out
    return out


def mark_system_cache(
    system_message: SystemMessage | None,
    *,
    min_prefix_tokens: int = MIN_CACHE_PREFIX_TOKENS,
) -> SystemMessage | None:
    """给 **system message** 打 content-block ephemeral cache_control——它是全天不变、跨轮 / 跨
    会话都字节稳定的**最长缓存层**（对齐 refdocs/05 §4.4「system + tools 独立缓存层」）。

    与 :func:`apply_cache_control` 各管一段、互补：后者标记 ``messages`` 里的早期历史前缀（每轮
    往后挪），本函数专标记独立的 ``request.system_message`` 字段（LangChain 把 system prompt
    放这里、**不在** ``messages`` 内，故 apply_cache_control 够不到它）。两者各打 1 个标记，
    合计 2 个、远在 :data:`MAX_CACHE_MARKERS`（4）之内。

    落实「最小写入阈值」：system prompt 恒远超 1024 token，但仍判一道（few-shot 被关等极端情况下
    可能偏小），不足则不打（写了也不会被缓存、白占标记额度）。已带标记的原样返回（幂等）；无法安全
    标记（空 content）或入参为 None 返回 None，调用方跳过 system_message 的 override。
    """
    if system_message is None:
        return None
    if _block_has_marker(system_message.content):
        return system_message
    if count_tokens(_content_text(system_message.content)) < min_prefix_tokens:
        return None
    marked = _to_marked_blocks(system_message.content)
    if marked is None:
        return None
    return system_message.model_copy(update={"content": marked})
