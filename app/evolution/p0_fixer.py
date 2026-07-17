"""对 LEAK 类 bad case 抽字面串、产候选脱敏规则（refdocs 18-2 §4）。

**两道闸，都过才提议规则：**

1. **可脱类别闸**：只提议 ``output_guard`` 自己认可「确定无害可删」的类别——内网 endpoint。
   ``item_id`` / 商品编号 / 工具名**明令不脱**（output_guard docstring：宁可漏放不误杀、工具名对
   攻击者无价值），所以 fixer 也绝不为它们产规则。哪怕 judge 判词点名了它们。

2. **真泄露闸**：候选串必须真的出现在**用户可见文本**（``BadCase.final_text``）里。所以提取直接
   在 final_text 上做，而不是 judge 判词——从源头堵掉「judge 把评估轨迹当成 Agent 回答」的假阳性
   （q08 就是这么被拦下的：judge 说泄露 landed_usd/item_id，但 summary.md 里一个都没有）。

**为什么不让 LLM 抽/生成正则：** 贪婪正则会把整段回复脱成 ``[已脱敏]``。这里只用确定性正则从
用户文本里捞出 endpoint 形态的串，取 ``scheme://host[:port]``（真正敏感的拓扑部分），存字面。

**为什么只提议 curated 没覆盖的 endpoint：** curated 的内网 host 列表是写死的（qdrant/redis/…），
新上的内部服务（``pricing-svc:8080``）它抓不到——那正是飞轮要补的增量。已被 curated 覆盖的就不
重复提议。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlsplit

from app.evolution.collector import BadCase
from app.evolution.router import Route, classify
from app.security.output_guard import audit_output

# 用户文本里的 URL（到第一个空白 / 中英文标点 / 右括号止）。
_URL = re.compile(r"https?://[^\s)）,，。；;]+")
# 私网 IP 段（RFC1918 + 环回 + 通配）。
_PRIVATE_IP = re.compile(r"^(?:127\.|0\.0\.0\.0|10\.|192\.168\.|172\.(?:1[6-9]|2\d|3[01])\.)")
# 公网顶级域（有这些的是给用户看的正经 URL，如 amazon.com——绝不脱）。
_PUBLIC_TLD = re.compile(r"\.(?:com|net|org|io|cn|co|shop|store|app)(?::\d+)?$", re.IGNORECASE)


@dataclass
class ProposedRule:
    """一条待落盘的候选脱敏规则。``literal`` 是要脱掉的字面串。"""

    name: str
    literal: str
    source_case: str
    #: 产出这条规则的那次执行的 Langfuse trace id。人工确认规则时点开即可看到泄露是怎么发生的
    #: ——判词只说「暴露了内部端点」，trace 里能看到是哪个工具的返回把它带出来的。
    evidence_trace: str = ""


def _host_port(url: str) -> str | None:
    """取 ``scheme://host[:port]``（丢 path/query）——内网拓扑的敏感部分。取不到返回 None。"""
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    if not parts.scheme or not parts.hostname:
        return None
    netloc = parts.hostname + (f":{parts.port}" if parts.port else "")
    return f"{parts.scheme}://{netloc}"


def _looks_internal(url: str) -> bool:
    """这个 URL 像内网地址（该脱）还是给用户的正经外链（不该脱）？

    保守偏向「不脱」：只有明确像内网的才判 True——私网 IP、裸主机名（无点，如 ``qdrant``）、
    或带端口且非公网 TLD 的主机。``https://amazon.com/dp/xxx`` 命中公网 TLD → False，绝不误脱。
    """
    parts = urlsplit(url)
    host = parts.hostname or ""
    if not host:
        return False
    if _PRIVATE_IP.match(host):
        return True
    if _PUBLIC_TLD.search(f"{host}:{parts.port}" if parts.port else host):
        return False
    if "." not in host:  # 裸主机名：qdrant / pricing-svc
        return True
    return parts.port is not None  # 有点但带端口、非公网 TLD → 多半是内部服务


def _already_covered(literal: str) -> bool:
    """curated 规则已经能脱掉这个串吗？能的话不重复提议（audit_output 命中即为已覆盖）。

    注意：此时 learned 叠加层可能已有这条候选——所以「已覆盖」也涵盖「本轮早先已提议过」，
    天然去重。"""
    clean, _, _ = audit_output(literal)
    return not clean


def propose_rules(case: BadCase) -> tuple[list[ProposedRule], dict[str, Route]]:
    """对一条 bad case 产候选规则 + 返回每条 P0 fail 的分流结果（供上报）。

    只有被判 LEAK 的 P0，才在 ``final_text`` 上扫未覆盖的内网 endpoint。BANNED / JUDGMENT 只体现在
    返回的分流字典里，不产任何规则。
    """
    routes = {dim: classify(dim, rat) for dim, rat in case.p0_failures.items()}
    if Route.LEAK not in routes.values():
        return [], routes
    if not case.final_text:
        return [], routes  # 真泄露闸：拿不到用户可见文本，一律不提议

    proposals: list[ProposedRule] = []
    seen: set[str] = set()
    for m in _URL.finditer(case.final_text):
        hostport = _host_port(m.group(0))
        if hostport is None or hostport in seen:
            continue
        if not _looks_internal(m.group(0)):
            continue  # 公网外链，给用户的，不脱
        if _already_covered(hostport):
            continue  # curated / 已有候选能脱 → 不重复
        seen.add(hostport)
        host = urlsplit(hostport).hostname or "endpoint"
        safe = re.sub(r"[^a-z0-9]+", "_", host.lower()).strip("_")
        proposals.append(
            ProposedRule(
                name=f"learned_endpoint_{safe}",
                literal=hostport,
                source_case=case.id,
                evidence_trace=case.trace_id,
            )
        )
    return proposals, routes
