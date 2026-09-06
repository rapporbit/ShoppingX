"""``CacheAwareOpenAIFormatter``：在**真正发出去的 payload** 上打 cache_control。

为什么标记非落在这一层不可：AgentScope 的 content block 是 pydantic 强类型（``TextBlock``
不收未知字段），``cache_control`` 塞不进 ``Msg``；``Msg.metadata`` 又是**消息级**的，而
AgentScope 一整轮就是一条 assistant 消息（见 :mod:`app.compress.blocks`），标记落在它上面
等于标了一整轮。formatter 输出的 dict 序列则**就是**线上 payload 本身，标在这里所见即所得。

**别把命中率归到这个标记头上**（L0 的 S1 实测）：本仓网关（DashScope OpenAI 兼容）是**隐式**
前缀缓存——带标记组与不带标记的对照组第二次都命中 1024 token。标记继续打，因为成本为零、且
换 Anthropic 直连时它是必需品；但缓存收益的真正来源是「断点前的字节逐轮不变」，那是压缩
（``as_blocks``）与注入纪律（注入随 state 长驻、不每轮重发）挣来的，不是这一行 JSON。

两条硬约束照 refdocs/05 §4.4 落实：单请求 ≤4 个标记、前缀不足最小写入阈值不打。
"""

from typing import Any

from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import Msg

from app.compress.blocks import DEFAULT_KEEP_RECENT, MIN_CACHE_PREFIX_TOKENS
from app.utils.tokens import count_tokens

_EPHEMERAL: dict[str, str] = {"type": "ephemeral"}


def _payload_text(entry: dict[str, Any]) -> str:
    """一条 OpenAI 消息 dict 的可计长文本（够用即可，不追求与网关分词逐字一致）。"""
    content = entry.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return ""


def _mark(entry: dict[str, Any]) -> bool:
    """给这条 dict 打 ephemeral 标记；打不了返回 False。

    ``role="tool"`` 一律跳过：OpenAI 兼容端点要求 tool 消息的 content 是**字符串**，改成
    block 列表会被网关判 400。跳过它只是让缓存前缀短几条，不影响正确性。
    """
    if entry.get("role") == "tool":
        return False
    content = entry.get("content")
    if isinstance(content, str) and content:
        entry["content"] = [{"type": "text", "text": content, "cache_control": dict(_EPHEMERAL)}]
        return True
    if isinstance(content, list) and content:
        for j in range(len(content) - 1, -1, -1):
            if isinstance(content[j], dict):
                if "cache_control" in content[j]:
                    return True  # 已标记，幂等
                content[j] = {**content[j], "cache_control": dict(_EPHEMERAL)}
                return True
    return False


class CacheAwareOpenAIFormatter(OpenAIChatFormatter):
    """``OpenAIChatFormatter`` + **system 那一条**上的一个 ephemeral 标记。

    **为什么标记不跟着压缩断点走**（第一版这么写，被实测打回）：断点是「倒数第 keep_recent 个
    工具结果」，每轮都往后挪。标记跟着挪，就意味着上一轮被标记的那条这一轮不再带标记——而打标记
    会把该条的 ``content`` 从字符串改写成 block 列表（``"abc"`` → ``[{"type":"text",...}]``）。
    于是**前缀的字节逐轮都在变**，缓存从第二轮起就接不上。为了一个在隐式缓存下收益为零的标记，
    赔掉整段前缀的稳定性，是笔烂账。

    system 段是全天不变、跨轮跨会话都字节稳定的最长缓存层（本仓 system prompt 纯静态、无运行时
    注入，见 ``agents._assemble``），标记钉死在它上面：位置不动、形态不动，前缀就还是那份前缀。
    与 LangChain 侧的 ``mark_system_cache`` 同一口径（那边 system 是独立字段，不在 messages 里）。

    ``keep_recent`` 仍然收下，但只用来解释「压缩断点在哪」——formatter 不再依赖它做落点决策。
    """

    def __init__(self, *args: Any, keep_recent: int = DEFAULT_KEEP_RECENT, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._keep_recent = keep_recent

    async def format(self, msgs: list[Msg]) -> list[dict[str, Any]]:
        formatted = await super().format(msgs)
        if not formatted or formatted[0].get("role") != "system":
            return formatted
        if count_tokens(_payload_text(formatted[0])) < MIN_CACHE_PREFIX_TOKENS:
            # 不足最小写入阈值：写了也不会被缓存，白占一个标记额度。
            return formatted
        _mark(formatted[0])  # 单个标记，恒在 MAX_CACHE_MARKERS（4）之内
        return formatted
