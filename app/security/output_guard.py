"""L4 输出审核：最终回答推给用户之前，脱掉内部信息。

**刻意窄。** refdocs 16-6 §2.4 的示例把 ``item_id`` / ``thread_id`` / 内部工具名一起脱敏——本项目
**主动更正**：脱敏范围收窄到「泄露了对用户无用、对攻击者有用」的东西。理由是这层的失败代价不
对称：

- 漏放一个 ``item_id``：用户看到一串商品编号，最多是噪声——而它往往正是用户想要的（拿去平台
  搜同款）。
- 误杀一个 ``item_id``：推荐清单里的商品标识变成 ``[已脱敏]``，功能直接坏掉。
- 漏放一个 API Key：真出事。

所以只脱三类**确定无害可删**的东西：密钥格式的字符串、内部服务地址、服务器绝对路径。工具名不脱
（``dispatch_tool`` 这个名字对攻击者毫无价值，模型偶尔提到它也不构成泄露）；``item_id`` / 商品编号
不脱（同 ``session_hooks`` 里既有的取向：宁可漏放也不误杀）。

与 :mod:`app.harness.hooks.session_hooks` 的 ``output_guard`` Hook 的分工：那个洗的是 **Harness
自己的控制文案**被模型鹦鹉学舌抄进回答（``[强制收尾] …``），是「内部噪声」问题；这里洗的是
**敏感信息泄露**，是安全问题。两者都挂在 ``on_session_end``，前者先跑（priority 10）后者后跑
（priority 20）——先去噪再脱敏，顺序无关紧要但这样日志更好读。
"""

from __future__ import annotations

import logging
import re

from app.security.learned_rules import learned_output_patterns

logger = logging.getLogger("shoppingx.security.output")

REDACTED = "[已脱敏]"

# 只脱「确定无害可删」的三类。每条都配一个「为什么它对用户无用」的理由——加新模式前先问这个问题。
_SENSITIVE_PATTERNS: tuple[tuple[str, str], ...] = (
    # 密钥：任何形如 sk-xxx / Bearer xxx 的长串。用户永远不需要看到它。
    ("api_key", r"\b(?:sk|pk|api|key)[-_][A-Za-z0-9]{16,}\b"),
    ("bearer_token", r"(?i)\bbearer\s+[A-Za-z0-9._\-]{20,}"),
    # 内部服务地址：向用户暴露内网拓扑，纯风险无收益。
    (
        "internal_endpoint",
        r"(?i)\bhttps?://(?:localhost|127\.0\.0\.1|0\.0\.0\.0|"
        r"opensearch|qdrant|redis|vllm|reranker)(?::\d+)?\S*",
    ),
    # 服务器绝对路径：会话目录 / 项目路径泄露磁盘结构。产物下载走 /api/files/，用不着绝对路径。
    # 前置 lookbehind 挡住 URL 里的同名路径段（``https://x.com/home/foo`` 不该被当成本机路径）。
    ("filesystem_path", r"(?<![\w:/])(?:/Users/|/home/|/var/|/opt/)[\w./-]{4,}"),
    ("windows_path", r"[A-Za-z]:\\(?:Users|Windows|Program)[\w.\\-]{2,}"),
)

_COMPILED = tuple((name, re.compile(pattern)) for name, pattern in _SENSITIVE_PATTERNS)


def audit_output(text: str) -> tuple[bool, str, list[str]]:
    """审核最终回答，返回 ``(是否干净, 脱敏后文本, 命中的模式名)``。

    ``是否干净`` 为 ``False`` 时**不拦截回答**，只是用脱敏后的文本替代原文——用户拿不到答案比
    看到一行 ``[已脱敏]`` 糟糕得多。命中即打 warning，供事后追查「模型为什么会说出这个」。

    异常一律原样返回（安全层不反噬主链路）。
    """
    if not text:
        return True, text, []
    try:
        hits: list[str] = []
        cleaned = text
        # 策划基线（in-code，事实来源）+ 飞轮学习到的字面规则（learned 叠加层）。后者从 bad case
        # 沉淀、落盘持久化、可人工审核，见 :mod:`app.security.learned_rules`。
        for name, pattern in (*_COMPILED, *learned_output_patterns()):
            cleaned, n = pattern.subn(REDACTED, cleaned)
            if n:
                hits.append(name)
        if hits:
            logger.warning("输出审核命中敏感模式并已脱敏：%s", ", ".join(hits))
        return not hits, cleaned, hits
    except Exception:
        logger.warning("输出审核异常，原样放行", exc_info=True)
        return True, text, []
