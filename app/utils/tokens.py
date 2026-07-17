"""Token 计数与按 token 预算截断：Qwen 本地分词器为主，CJK 感知启发式兜底。

**为什么不沿用 char/4**：旧实现统一按「1 token≈4 字符」估算，对中文严重失真。M6 实测
（DashScope ``get_tokenizer``）：中文 ≈ **1.33 字/token**、英文 ≈ **4.2 字/token**——``/4`` 把中文
当 4 字/token，对中文低估约 3 倍。对全球电商（Lazada/Shopee/Shein 大量中文/东南亚语料）会让
「token 上限」形同虚设：旧工具结果该压的没压、缓存前缀阈值误判。

**为什么不直接用响应里的 usage**：``usage.prompt_tokens`` 是真值，但它是**发送后**才有的**聚合**
数字；而压缩要在**发送前**逐条判断（哪条工具结果超 cap、前缀够不够阈值），那时响应还不存在、
也没有逐条粒度。故预检必须本地估算；usage 仅适合事后校准/观测缓存命中（另行接入）。

**分层策略**：
  1. 主：``dashscope[tokenizer]`` 的 ``get_tokenizer``——Qwen 系列共用词表，本地 byte-level BPE，
     无网络、无 API key。实测单次 encode 0.1ms 级（见 M6 性能基准）。
  2. 兜底：CJK 感知启发式（按字符类别分桶估）——当未装 tokenizer extra、或走非 Qwen 模型
     （如 DeepSeek 子链路，``get_tokenizer`` 不支持）时自动降级。

**性能**：首次 ``get_tokenizer`` 约 3.3s（import dashscope + 载词表），**一次性**——故分词器实例用
``lru_cache(maxsize=1)`` 只建一次，并提供 :func:`warm_tokenizer` 供服务启动时预热，别让这 3.3s
落在某个用户请求里。逐条计数再 ``lru_cache`` 一层（消息冻结后内容不变，命中率高）。
"""

import logging
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

# Qwen 系列共用同一词表，固定用 qwen-turbo 取分词器即可（与实际部署的 qwen-max/plus 同分词）。
_TOKENIZER_MODEL = "qwen-turbo"

# CJK 感知启发式的 token/字符 系数（M6 实测：中文≈1.33 字/token、英文≈4.2 字/token）。
_CJK_TOKENS_PER_CHAR = 0.75  # 1 / 1.33
_OTHER_TOKENS_PER_CHAR = 0.25  # 1 / 4.0


@lru_cache(maxsize=1)
def _qwen_tokenizer() -> Any | None:
    """惰性取 Qwen 本地分词器，失败（未装 extra / 非 Qwen 模型）则返回 None 并只告警一次。"""
    try:
        from dashscope import get_tokenizer

        return get_tokenizer(_TOKENIZER_MODEL)
    except Exception as exc:  # ImportError / 不支持的模型 / 词表缺失
        logger.warning("Qwen 分词器不可用，token 计数降级为 CJK 启发式：%s", exc)
        return None


def warm_tokenizer() -> bool:
    """服务启动时预热分词器（吃掉首次约 3.3s 的初始化），避免落在用户请求里。

    返回是否成功加载到真实分词器（False 表示将走启发式兜底）。
    """
    return _qwen_tokenizer() is not None


def _is_cjk(ch: str) -> bool:
    """是否为 CJK / 假名 / 谚文 / 全角等「高 token 密度」字符（约 0.75 token/字）。"""
    code = ord(ch)
    return (
        0x3000 <= code <= 0x303F  # CJK 符号与标点
        or 0x3040 <= code <= 0x30FF  # 平假名 / 片假名
        or 0x3400 <= code <= 0x4DBF  # CJK 扩展 A
        or 0x4E00 <= code <= 0x9FFF  # CJK 统一表意
        or 0xAC00 <= code <= 0xD7AF  # 谚文音节
        or 0xF900 <= code <= 0xFAFF  # CJK 兼容表意
        or 0xFF00 <= code <= 0xFFEF  # 全角 / 半角
    )


def _estimate_tokens(text: str) -> int:
    """CJK 感知启发式：按字符类别分桶估 token 数（无分词器时的兜底）。"""
    cjk = sum(1 for c in text if _is_cjk(c))
    other = len(text) - cjk
    return round(cjk * _CJK_TOKENS_PER_CHAR + other * _OTHER_TOKENS_PER_CHAR)


@lru_cache(maxsize=4096)
def count_tokens(text: str) -> int:
    """文本的 token 数：有 Qwen 分词器走精确 encode，否则走 CJK 启发式。

    结果缓存（消息冻结后内容不变，重复 step 直接命中）。空串返回 0。
    """
    if not text:
        return 0
    tok = _qwen_tokenizer()
    if tok is not None:
        return len(tok.encode(text))
    return _estimate_tokens(text)


def truncate_to_token_budget(text: str, max_tokens: int, suffix: str = "") -> str:
    """把 ``text`` 截到约 ``max_tokens`` 个 token（含 ``suffix``），未超则原样返回。

    实现按「本段实际 字符/token 比例」换算字符预算后用 str 切片——切点落在码点边界，**不会产生
    半个 UTF-8 字 / 切坏多字节字符**；且对「真实分词器」和「启发式」两条路径统一（都靠
    :func:`count_tokens`），调用方无需关心后端。这是估算式截断（目的是把体积压到上限附近，不要求
    逐 token 精确），对「控住体积」足够。
    """
    if max_tokens <= 0:
        return suffix
    total = count_tokens(text)
    if total <= max_tokens:
        return text
    # 极小 cap：预算比提示语还短时退化为「硬切、不留提示」，保证输出永远短于原文。
    budget = max_tokens - count_tokens(suffix)
    if budget <= 0:
        return text[: max(0, int(len(text) * max_tokens / total))]
    return text[: max(0, int(len(text) * budget / total))] + suffix
