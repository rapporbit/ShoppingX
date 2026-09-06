"""AgentScope 侧的 Cache Breakpoint：断点与压缩都下沉到 **block 粒度**。

**为什么不能复用 :mod:`app.compress.breakpoint` 的消息级断点。** 两个运行时的历史形态根本不同：

- LangChain：一次 Observe = 一条独立的 ``ToolMessage``，整条 loop 是几十条消息，断点是消息下标。
- AgentScope：**一次 reply（一整轮用户交互）= 一条 assistant ``Msg``**，轮内所有
  thinking / tool_call / tool_result 全部 ``content.extend`` 进这同一条消息
  （见 ``AgentState.append_context``）。一条 15 步的购物链就是一条 30+ block 的消息。

所以在 AgentScope 下，消息级断点只有两种结果：要么整轮都算「最近区」（一个字都压不掉、token 照
样线性爆炸），要么整轮都算「较旧区」（把模型刚拿到的工具结果也截了）。断点必须落进消息内部。

断点语义与 LangChain 版逐字对齐（见 ``breakpoint.py`` 顶部对 refdocs/05 的更正）：
``[:bp]`` 是较旧区（压缩、可当缓存前缀），``[bp:]`` 是最近 ``keep_recent`` 个工具结果构成的
工作集（留全文）。只是坐标从 ``int`` 变成了 ``(msg_idx, block_idx)`` 二元组。

cache_control 标记**不在这一层打**：AgentScope 的 content block 是 pydantic 强类型
（``TextBlock`` 不收未知字段），塞不进 ``cache_control``。标记落在
:mod:`app.harness.formatter`——那一层的 dict 序列才是真正发出去的 payload。
"""

import json
from collections.abc import Sequence

from agentscope.message import Msg, TextBlock, ToolResultBlock

from app.utils.tokens import count_tokens, truncate_to_token_budget

# 最近多少次工具结果留全文（= 工作集大小）。3 是实测口径：模型真正会回头引用的就是最后几次
# Observe，再往前的细节它已经转写进自己的推理里了。
DEFAULT_KEEP_RECENT = 3

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


# 断点坐标：(消息下标, 消息内 block 下标)。字典序比较即「谁在前」。
BlockPos = tuple[int, int]


def _tool_result_positions(messages: Sequence[Msg]) -> list[BlockPos]:
    """按出现顺序列出所有 tool_result block 的坐标（= 每一次 Observe）。"""
    positions: list[BlockPos] = []
    for i, msg in enumerate(messages):
        content = msg.content
        if not isinstance(content, list):
            continue
        for j, block in enumerate(content):
            if isinstance(block, ToolResultBlock):
                positions.append((i, j))
    return positions


def compute_block_breakpoint(
    messages: Sequence[Msg], keep_recent: int = DEFAULT_KEEP_RECENT
) -> BlockPos:
    """定位断点：``[:bp]`` 较旧可压缩区，``[bp:]`` 最近 ``keep_recent`` 次工具结果。

    与 :func:`app.compress.breakpoint.compute_breakpoint` 同规则：
      - ``keep_recent <= 0`` → 整段可压缩，返回末尾坐标。
      - 工具结果数 ≤ ``keep_recent`` → 还没有「较旧」历史，返回 ``(0, 0)``（啥也不压）。
      - 否则返回倒数第 ``keep_recent`` 个 tool_result block 的坐标。
    """
    if keep_recent <= 0:
        return (len(messages), 0)
    positions = _tool_result_positions(messages)
    if len(positions) <= keep_recent:
        return (0, 0)
    return positions[-keep_recent]


def _block_text(block: ToolResultBlock) -> str | None:
    """取 tool_result 的**纯文本载荷**；多模态或非纯文本一律返回 None（不碰）。

    ``output`` 有两种形态：裸字符串，或 ``[TextBlock | DataBlock]``。本仓工具走
    ``ToolChunk(content=[TextBlock(...)])``，命中后者且恒为单个 TextBlock。混进 DataBlock
    （图片等）的一律跳过——压缩是给文字瘦身，不是把结构化内容 ``str()`` 化后截断。
    """
    output = block.output
    if isinstance(output, str):
        return output
    if isinstance(output, list) and len(output) == 1 and isinstance(output[0], TextBlock):
        return output[0].text
    return None


def _rewrite_output(block: ToolResultBlock, text: str) -> ToolResultBlock:
    """把压缩后的文本写回 block（保持原 output 形态：字符串进字符串出）。"""
    output: str | list = text
    if not isinstance(block.output, str):
        output = [TextBlock(type="text", text=text)]
    return block.model_copy(update={"output": output})


def compress_blocks_before(
    messages: Sequence[Msg],
    breakpoint_pos: BlockPos,
    *,
    max_tool_tokens: int = DEFAULT_MAX_TOOL_TOKENS,
) -> list[Msg]:
    """压缩断点前超量的 tool_result；断点后原样保留。

    **绝不原地改**：``messages`` 里的 Msg 与 block 就是 ``agent.state.context`` 里的那批对象，
    原地改会把持久化历史也一起截了——压缩只改「这一次请求送给模型的视图」。所以命中压缩的
    block 走 ``model_copy``，其宿主 Msg 也 ``model_copy`` 换一份新的 content 列表；没命中的
    消息**原对象返回**（省拷贝，也让「视图未变」在 ``is`` 层面就看得出来）。

    压缩策略与 LangChain 版共用一套：先试 JSON 字段抽取（``_smart_compress_json``），失败再
    尾部截断；已带截断提示的不再压（幂等护栏，防 token 估算微漂移造成二次截断、破坏前缀稳定）。
    """
    out: list[Msg] = []
    for i, msg in enumerate(messages):
        content = msg.content
        if not isinstance(content, list) or (i, 0) > breakpoint_pos:
            out.append(msg)
            continue

        new_content = list(content)
        changed = False
        for j, block in enumerate(content):
            if (i, j) >= breakpoint_pos or not isinstance(block, ToolResultBlock):
                continue
            text = _block_text(block)
            if (
                text is None
                or text.endswith(_TRUNCATE_HINT)
                or count_tokens(text) <= max_tool_tokens
            ):
                continue
            extracted = _smart_compress_json(text, max_tool_tokens)
            if extracted is None:
                extracted = truncate_to_token_budget(text, max_tool_tokens, _TRUNCATE_HINT)
            new_content[j] = _rewrite_output(block, extracted)
            changed = True
        out.append(msg.model_copy(update={"content": new_content}) if changed else msg)
    return out


def post_step_compress(
    messages: list[Msg],
    *,
    keep_recent: int = DEFAULT_KEEP_RECENT,
    max_tool_tokens: int = DEFAULT_MAX_TOOL_TOKENS,
) -> list[Msg]:
    """AgentScope 侧的 ``post_step_compress``：定位断点 → 压缩较旧区。纯函数、幂等。

    比 LangChain 版少一步「打 cache_control」——那件事挪到了 formatter（见模块 docstring）。
    """
    return compress_blocks_before(
        messages,
        compute_block_breakpoint(messages, keep_recent),
        max_tool_tokens=max_tool_tokens,
    )
