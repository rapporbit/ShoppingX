"""脱敏规则的「学习叠加层」：飞轮自动沉淀的规则住这里，与 in-code 策划规则分层。

**为什么要分层，而不是往 ``output_guard._SENSITIVE_PATTERNS`` 里 append（18-2 §4 的天真做法）：**

1. ``_SENSITIVE_PATTERNS`` 是模块级元组、import 时编译一次——运行时 append 一个 module global，
   进程重启就没了、多进程也不共享。规则必须**落盘持久化**。
2. in-code 那份是**人工策划的基线**，是事实来源，飞轮**永不写它**。飞轮只往本叠加层加东西，
   两者在 ``audit_output`` 里合并生效。这样「代码是基线的事实来源」这条不破。

**为什么只存字面串（``kind="literal"``），不存正则：** 让 LLM 从坏轨迹里自由生成正则是脚枪——
一个贪婪的 ``.*`` 能把整段回复脱成 ``[已脱敏]``。飞轮抽到的是**真实泄露的那个具体串**
（某个 endpoint / 内部字段名），本层负责 ``re.escape`` 成字面匹配。正则规则只允许人工在
in-code 那份里写。

**候选态（``state="candidate"``）立即生效**：P0 是红线，宁可误杀不可漏放（18-2 §4.3）。但候选规则
进 ``list_candidates()`` 待人工确认转 ``active`` 或删除——因为触发它的「泄露」判定往往来自 judge，
judge 会系统性假阳性（见 memory rubric-judge-calibration-pitfalls）。``disabled`` 态不生效。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

logger = logging.getLogger("shoppingx.security.learned")

# 落盘位置：/data/* 已 gitignore，learned 规则不入库——靠飞轮从 bad case 复现（同 data/eval/）。
RULES_PATH = Path("data/security/learned_rules.json")

RuleState = Literal["candidate", "active", "disabled"]
# 生效的态：candidate 立即生效（P0 宁可误杀），disabled 是人工否掉的、不生效。
_LIVE_STATES: frozenset[str] = frozenset({"candidate", "active"})


@dataclass
class LearnedRule:
    """一条学习到的脱敏规则。``pattern`` 存**原始字面串**，编译时才 ``re.escape``。"""

    name: str
    pattern: str  # 原始字面串（真实泄露的那个具体串），不是正则
    source_case: str  # 来自哪条 bad case（可回溯「这规则为什么加」）
    state: RuleState = "candidate"
    created: str = ""  # ISO 时间戳，由调用方传入（本模块不碰时钟——对齐工作区禁用 Date.now 的约定）
    kind: Literal["literal"] = "literal"
    # 那次执行的 Langfuse trace id。source_case 只告诉你「哪条 query」，trace 告诉你「怎么泄露的」
    # ——人工确认规则时能点开看是哪个工具的返回把内网串带进了用户可见文本。
    # 有默认值：旧 learned_rules.json 缺这个键也能照常 LearnedRule(**r) 解析出来。
    evidence_trace: str = ""


# ── 落盘读写（mtime 缓存，guard 每次调用不重复 parse + compile）──

_cache: tuple[float, list[LearnedRule]] | None = None
_compiled_cache: tuple[float, tuple[tuple[str, re.Pattern[str]], ...]] | None = None


def _mtime() -> float:
    try:
        return RULES_PATH.stat().st_mtime
    except OSError:
        return -1.0


def load_rules() -> list[LearnedRule]:
    """读全部规则（含 disabled）。文件不存在返回空。mtime 未变则走缓存。"""
    global _cache
    mt = _mtime()
    if _cache is not None and _cache[0] == mt:
        return _cache[1]
    rules: list[LearnedRule] = []
    if mt >= 0:
        try:
            raw = json.loads(RULES_PATH.read_text(encoding="utf-8"))
            rules = [LearnedRule(**r) for r in raw.get("rules", [])]
        except Exception:
            logger.warning("读取 learned 规则失败，当作空表", exc_info=True)
            rules = []
    _cache = (mt, rules)
    return rules


def save_rules(rules: list[LearnedRule]) -> None:
    """全量写盘。写完清缓存，令下次 load / 编译看到最新。"""
    RULES_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_comment": "飞轮自动沉淀的脱敏规则（scripts/eval/evolve_p0.py 生成）。勿手改 pattern；"
        "人工只改 state（candidate→active 确认 / →disabled 否掉）。基线在 output_guard.py。",
        "rules": [asdict(r) for r in rules],
    }
    RULES_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    global _cache, _compiled_cache
    _cache = None
    _compiled_cache = None


def add_candidate(
    name: str,
    literal: str,
    source_case: str,
    *,
    created: str = "",
    evidence_trace: str = "",
) -> bool:
    """加一条候选规则。按**字面串**去重（同一个泄露串已在表里就不重复加）。返回是否真的加了。"""
    literal = literal.strip()
    if not literal:
        return False
    rules = list(load_rules())
    if any(r.pattern == literal for r in rules):
        return False
    rules.append(
        LearnedRule(
            name=name,
            pattern=literal,
            source_case=source_case,
            created=created,
            evidence_trace=evidence_trace,
        )
    )
    save_rules(rules)
    logger.info("新增候选脱敏规则 %s（来自 %s）：%r", name, source_case, literal)
    return True


def list_candidates() -> list[LearnedRule]:
    """当前待人工确认的候选规则。"""
    return [r for r in load_rules() if r.state == "candidate"]


def learned_output_patterns() -> tuple[tuple[str, re.Pattern[str]], ...]:
    """生效（candidate + active）的规则，编译成 ``(name, Pattern)``——供 ``audit_output`` 合并。

    字面串走 ``re.escape``，永不把飞轮抽到的串当正则解释。mtime 未变走编译缓存。
    """
    global _compiled_cache
    mt = _mtime()
    if _compiled_cache is not None and _compiled_cache[0] == mt:
        return _compiled_cache[1]
    compiled = tuple(
        (r.name, re.compile(re.escape(r.pattern))) for r in load_rules() if r.state in _LIVE_STATES
    )
    _compiled_cache = (mt, compiled)
    return compiled


def reset_for_test() -> None:
    """测试用：清缓存（配合 monkeypatch RULES_PATH 到 tmp）。"""
    global _cache, _compiled_cache
    _cache = None
    _compiled_cache = None
