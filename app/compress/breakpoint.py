"""Cache Breakpoint 定位：把对话切成「可压缩的较旧区」和「保留全文的近 K 轮」。

一句话：断点是一条分界线，线**之前**（较旧）的历史可以压缩 + 打缓存标记，线**之后**
（最近 keep_recent 轮工具调用）保留全文，因为它们是模型当前正在用的「工作集」。

对 refdocs/05 的更正（它正文里有一句自相矛盾）：refdocs 正文有句「保留最近 K 个工具调用在断点
之前」，但它的图和代码都是「缓存较旧区、压缩后挪进缓存」。本实现按**缓存第一性原理**统一：
  - 缓存要求一个**逐字稳定的前缀**。较旧区 ``[:breakpoint]`` 一旦压缩就冻结、且只增不减，
    最适合当缓存前缀（打 cache_control）。
  - 最近 ``keep_recent`` 轮工具调用 ``[breakpoint:]`` 是易变的工作集，保留全文给模型看全貌、不缓存。
因此本函数返回的 ``breakpoint`` = **最近 keep_recent 个工具消息里第一个的下标**：
``[:breakpoint]`` 较旧（压），``[breakpoint:]`` 最近 K 轮（留）。

**别夸大稳定性**：逐字稳定、能持续命中缓存的是较旧区 ``[:bp]``（它逐轮单调变长）。断点会随对话推进
**前移**，所以一条工具结果跨过断点（从最近区进较旧区）那一轮会被截断一次、此后冻结——这是滚动增量
缓存的正常代价（跨界消息只重算一次），不是「整段已发前缀永不变」。详见
:func:`app.compress.compressor.compress_after_breakpoint`。
"""

from collections.abc import Sequence

from langchain_core.messages import AnyMessage, ToolMessage

DEFAULT_KEEP_RECENT = 3


def compute_breakpoint(
    messages: Sequence[AnyMessage], keep_recent: int = DEFAULT_KEEP_RECENT
) -> int:
    """计算 Cache Breakpoint 下标：``[:bp]`` 较旧可压缩区，``[bp:]`` 最近 keep_recent 轮全文区。

    以「工具消息」为锚定轮次的标志（一次 Observe = 一条 ToolMessage）：
      - 攒够的工具调用 ≤ keep_recent：还没有「较旧」历史，全部保留全文 → 返回 ``0``
        （可压缩区为空，啥也不压；缓存交给 system prompt 那层兜，见 M9）。
      - 否则：断点设在「倒数第 keep_recent 个工具消息」的下标，使最近 K 轮工具调用落在
        ``[bp:]`` 全文区，更早的历史落在 ``[:bp]`` 压缩区。

    返回值是 messages 的下标，恒在 ``[0, len(messages)]`` 内，便于直接切片。
    """
    if keep_recent <= 0:
        # 不保留任何近轮 → 整段都可压缩。
        return len(messages)

    tool_indices = [i for i, msg in enumerate(messages) if isinstance(msg, ToolMessage)]
    if len(tool_indices) <= keep_recent:
        return 0
    return tool_indices[-keep_recent]
