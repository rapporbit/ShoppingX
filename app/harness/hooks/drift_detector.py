"""Silent Drift 漂移检测：每 N 轮识别四类偏离信号，分级纠正。

LoopDetector 检测"重复做同一件事"；Silent Drift 检测"做不同的事但在偏离目标"——
每步看着合理，累积 5 步后已彻底跑偏。二者互补，不替代。

四类信号（全部由 ``_computational_precheck`` 确定性预检，命中即出判定）：
1. 目标遗忘 —— 最近 N 轮行为里一个 query 关键词都没提到
2. 探索发散 —— 连续多次检索返回空结果（方向错了，找不到东西）
3. 偏好丢失 —— 推荐结果命中用户黑名单属性（会话级 P_t 的硬 dislike）
4. 成本失控 —— 最近几轮 token 消耗远超均值（浪费在无效工具调用）

纠正策略：轻微 → 注入提醒；严重 → 注入强纠正；连续严重 → 强制收尾（并把阶段机推到
CONCLUDING 授权收尾，否则 phase gate 会把被强制的 shopping_summary 拦下来）。
预算面不在这里管——budget_router 每轮 pre_think 都会按全树成本定档，无需旁路信号。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.harness.middleware import harness_hook
from app.harness.signals import blacklist_hits
from app.utils.env import env_bool, env_int
from app.utils.terms import normalize_terms, term_hits

logger = logging.getLogger("shoppingx.harness.drift")

CHECK_INTERVAL = env_int("HARNESS_DRIFT_INTERVAL", 3)
DRIFT_ENABLED = env_bool("HARNESS_DRIFT_ENABLED", True)

# 目标遗忘：最近 N 轮行为里**一个** query 关键词都没提到，才算遗忘。
#
# 这里刻意不用「命中率 < 20%」这类比率判据：关键词集含 CJK bigram，分母被撑到几十，而单轮行为
# 摘要只可能覆盖其中少数几个词——比率判据恒真，会把「目标遗忘」变成每轮必报的假阳性；又因为预检
# 一命中就 return，还会连带遮蔽掉后三类信号和 LLM 判定。refdocs 的原话是「连续 3 轮都没提到原始
# 需求」，`hits == 0` 才是它的忠实表达。
_GOAL_MIN_HITS = 1
# 探索发散：连续空结果次数
_EMPTY_RESULT_THRESHOLD = 3
# 偏好丢失：命中黑名单属性次数（一次即违规——硬 dislike 本该被 item_picker 机械淘汰在前）
_BLACKLIST_THRESHOLD = 1
# 成本失控：最近 N 轮均值 > 历史均值 × 倍数
_COST_SPIKE_FACTOR = 2.0

# 严重度序：预检可能同时命中多个信号，取**最严重**的那个，而不是第一个命中的。
_SEVERITY: dict[str, int] = {"正常": 0, "轻微偏离": 1, "严重偏离": 2}

_DRIFT_CHECK_PROMPT = """你是购物 Agent 的漂移检测器。
用户原始需求：{original_query}
Agent 最近 {n} 轮行为摘要：{recent_actions}
请判断 Agent 是否仍朝用户需求方向前进。
只回答以下之一：
- "正常"
- "轻微偏离"
- "严重偏离"
不要解释。"""


class DriftState:
    """单会话的漂移检测状态（与 HarnessAgentMiddleware 同生命周期）。

    ``token_history`` 由 :class:`HarnessAgentMiddleware` 在每次模型调用后按 tree_snapshot 的
    增量追加；``blacklist_violations`` 由 ``track_result_signals`` 在工具返回后累加。
    """

    def __init__(self) -> None:
        self.round_counter: int = 0
        self.consecutive_severe: int = 0
        self.consecutive_empty_results: int = 0
        self.blacklist_violations: int = 0
        self.violated_terms: list[str] = []
        self.recent_tool_names: list[str] = []
        self.token_history: list[int] = []
        # 目标词桥：planner 每轮写完 P_t 后从中刷新（约束词双语 + 品类）。「目标遗忘」信号拿
        # 用户 query 关键词去匹行为摘要，但摘要里可匹的是英文检索词——中文 bigram 裸匹恒 0，
        # planner(原话) 一滑出窗口就每 3 轮必报假阳性。P_t 是现成的确定性跨语言词面。
        self.goal_terms: set[str] = set()

    def reset(self) -> None:
        self.round_counter = 0
        self.consecutive_severe = 0
        self.consecutive_empty_results = 0
        self.blacklist_violations = 0
        self.violated_terms.clear()
        self.recent_tool_names.clear()
        self.token_history.clear()
        self.goal_terms.clear()


def _extract_keywords(query: str) -> set[str]:
    """从 query 中提取关键词（简单分词，中英文混合）。

    中文按单字 / 双字滑窗（无分词器依赖），英文按空白 + 非字母分割。
    """
    text = query.lower()
    # 英文 token
    en_tokens = re.findall(r"[a-z]{2,}", text)
    # 中文：提取所有连续汉字段，每段做 bigram 滑窗
    cjk_spans = re.findall(r"[一-鿿]{2,}", text)
    cjk_tokens: list[str] = []
    for span in cjk_spans:
        for i in range(len(span) - 1):
            cjk_tokens.append(span[i : i + 2])
        if len(span) >= 2:
            cjk_tokens.append(span)  # 整段也留
    # 数字 token（预算金额等）
    num_tokens = re.findall(r"\d+", text)

    stop = {
        "的",
        "了",
        "在",
        "是",
        "和",
        "我",
        "要",
        "买",
        "想",
        "个",
        "不",
        "有",
        "一",
        "找",
        "帮",
        "用",
        "给",
        "能",
        "可以",
        "什么",
        "怎么",
        "吗",
        "the",
        "an",
        "is",
        "are",
        "me",
        "my",
        "want",
        "to",
        "for",
        "and",
        "or",
        "with",
        "not",
        "no",
        "some",
        "any",
    }
    all_tokens = en_tokens + cjk_tokens + num_tokens
    return {t for t in all_tokens if t not in stop}


def _computational_precheck(
    context: dict[str, Any], state: DriftState
) -> tuple[str | None, list[str]]:
    """确定性预检：不调 LLM。返回 ``(漂移级别 | None, 命中信号名列表)``。

    ``None`` 表示四类信号都没命中 → 交给 LLM 做二段推断（预检只能抓「看得见的」偏离，抓不到
    「每步都合理但方向在滑」的语义漂移）。四类信号全部评估完再取最严重级别——早 return 会让
    先检查的信号永久遮蔽后面的。
    """
    original_query = context.get("original_query", "")
    recent_actions = context.get("recent_actions_summary", "")

    verdicts: list[tuple[str, str]] = []  # (信号名, 级别)

    # 信号 1: 目标遗忘——最近 N 轮行为里一个目标词都没提到。目标词面 = query 关键词过
    # normalize_terms（中文材质/功能词补出英文变体，能匹上英文检索参数）∪ P_t 词桥
    # （state.goal_terms）。命中判定走 term_hits 而非裸 in：英文词要词边界（"in" 不该撞进
    # "insulated"），中文词照旧子串。
    keywords = set(normalize_terms(sorted(_extract_keywords(original_query)))) | state.goal_terms
    if keywords and recent_actions:
        actions_lower = recent_actions.lower()
        hits = sum(1 for kw in keywords if term_hits(kw, actions_lower))
        if hits < _GOAL_MIN_HITS:
            verdicts.append(("目标遗忘", "轻微偏离"))

    # 信号 2: 探索发散——连续空结果
    if state.consecutive_empty_results >= _EMPTY_RESULT_THRESHOLD:
        verdicts.append(("探索发散", "严重偏离"))

    # 信号 3: 偏好丢失——推荐结果里出现了黑名单属性
    if state.blacklist_violations >= _BLACKLIST_THRESHOLD:
        verdicts.append(("偏好丢失", "严重偏离"))

    # 信号 4: 成本失控——最近 3 轮 token 均值远超历史均值
    if len(state.token_history) >= 6:
        recent_avg = sum(state.token_history[-3:]) / 3
        total_avg = sum(state.token_history) / len(state.token_history)
        if total_avg > 0 and recent_avg > total_avg * _COST_SPIKE_FACTOR:
            verdicts.append(("成本失控", "轻微偏离"))

    if not verdicts:
        return None, []

    worst = max(verdicts, key=lambda v: _SEVERITY[v[1]])[1]
    signals = [name for name, _ in verdicts]
    logger.info("Drift 预检命中 %s → %s", signals, worst)
    return worst, signals


def _force_conclude_phase() -> None:
    """把主 loop 的阶段机推到 CONCLUDING，为强制收尾**授权**。

    阶段白名单禁令已撤，但 ``phase_check`` 仍留一道底线：阶段还在 PLANNING 时拒绝
    shopping_summary（本轮未规划/精挑不许交卷）。强制收尾是漂移恶化下的兜底通路，必须
    压过这道底线——否则注入「立即调 shopping_summary」的同时又拦下它，模型被一边逼着
    收尾、一边不许收尾。推进 CONCLUDING 同时也让遥测如实反映「已进入收尾」。
    """
    try:
        from app.agent.fork_guard import current_fork_depth
        from app.harness.phase_machine import Phase, get_phase_machine

        if current_fork_depth() >= 1:
            return
        machine = get_phase_machine()
        if machine is not None and machine.phase != Phase.CONCLUDING:
            machine.set_phase(Phase.CONCLUDING)
    except Exception:
        logger.debug("强制收尾时推进阶段机失败", exc_info=True)


def _apply_correction(
    context: dict[str, Any],
    verdict: str,
    state: DriftState,
    signals: list[str] | None = None,
) -> dict[str, Any]:
    """根据漂移判定级别注入分级纠正。``signals`` 为预检命中的信号名（LLM 判定时为空）。"""
    original_query = context.get("original_query", "")
    signals = signals or []

    if "严重偏离" in verdict:
        state.consecutive_severe += 1

        if state.consecutive_severe >= 2:
            _force_conclude_phase()
            context.setdefault("inject_messages", []).append(
                {
                    "role": "system",
                    "content": (
                        "[强制收尾] 连续检测到严重漂移。"
                        "请立即基于已有结果调用 shopping_summary 给出回答，不要再发起新的检索。"
                    ),
                }
            )
            logger.warning("Drift: 连续严重 %d 次，强制收尾", state.consecutive_severe)
        elif "偏好丢失" in signals:
            terms = "、".join(state.violated_terms[:3])
            context.setdefault("inject_messages", []).append(
                {
                    "role": "system",
                    "content": (
                        f"[偏好丢失] 当前结果里出现了用户明确排除的属性「{terms}」。"
                        "请剔除这些商品后再继续，不要把它们放进最终清单。"
                    ),
                }
            )
            logger.warning("Drift: 偏好丢失，命中黑名单 %s", state.violated_terms[:3])
        else:
            context.setdefault("inject_messages", []).append(
                {
                    "role": "system",
                    "content": (
                        f"[漂移纠正] 你已明显偏离用户原始需求「{original_query[:60]}」。"
                        "请立即回到该需求，停止无关的检索方向。"
                    ),
                }
            )
            logger.info("Drift: 严重偏离 (consecutive=%d)", state.consecutive_severe)

    elif "轻微偏离" in verdict:
        state.consecutive_severe = 0
        context.setdefault("inject_messages", []).append(
            {
                "role": "system",
                "content": f"[漂移提醒] 注意保持在用户原始需求「{original_query[:60]}」的方向上。",
            }
        )
        logger.info("Drift: 轻微偏离 %s", signals)

    else:
        state.consecutive_severe = 0

    return context


@harness_hook("post_reflect", name="drift_detector", priority=20)
async def detect_drift(context: dict[str, Any]) -> dict[str, Any] | None:
    """每 N 轮检测一次 Agent 是否偏离目标。

    先做 Computational 预检（确定性、零成本），大部分情况预检即出结论；
    预检不确定时才走轻量 LLM（三选一判定）。
    """
    if not DRIFT_ENABLED:
        return None

    state: DriftState | None = context.get("_drift_state")
    if state is None:
        return None

    state.round_counter += 1
    if state.round_counter % CHECK_INTERVAL != 0:
        return None

    original_query = context.get("original_query", "")
    if not original_query:
        return None

    # 第一关：Computational 预检
    comp_verdict, signals = _computational_precheck(context, state)
    if comp_verdict is not None:
        ctx = _apply_correction(context, comp_verdict, state, signals)
        # 违规计数清零：已经就本次违规发过纠正了。留着会让后续每次检测都重复判「偏好丢失」，
        # consecutive_severe 一路涨到强制收尾——模型明明已经改正了也会被硬收。
        state.blacklist_violations = 0
        return ctx

    # 第二关：LLM 推断（仅在预检未决时）
    recent_actions = context.get("recent_actions_summary", "")
    if not recent_actions:
        return None

    try:
        # 用 fast 档（默认关推理），不是 judge 强模型：refdocs 17-3 §3.4 要求漂移判定
        # 「用 lite 模型、单次 < $0.0001」。实测 judge 档单次 3.8-5.4s——每 3 轮一次就给主链路白加
        # 近 10s，而这里只要一个三选一的标签。judge 档留给 Rubric 离线评测（要的是评分稳定性，
        # 不在线上关键路径）。
        from app.agent.llm import get_fast_llm

        llm = get_fast_llm()
        resp = await llm.ainvoke(
            [
                (
                    "user",
                    _DRIFT_CHECK_PROMPT.format(
                        original_query=original_query,
                        recent_actions=recent_actions,
                        n=CHECK_INTERVAL,
                    ),
                ),
            ]
        )
        raw = resp.content
        verdict = raw.strip() if isinstance(raw, str) else str(raw)
    except Exception:
        logger.debug("Drift LLM 判定失败，跳过", exc_info=True)
        return None

    return _apply_correction(context, verdict, state)


# 产出「要推给用户的结果」的工具——只在这些工具的返回上查黑名单。
# 不查 item_search：召回阶段捞到黑名单商品是正常的（后面 item_picker 会淘汰），
# 在那里报违规是假阳性。
_RECOMMEND_TOOLS = frozenset({"item_picker", "shopping_summary"})
_SEARCH_TOOLS = frozenset({"item_search", "web_search"})


def _pt_goal_terms() -> set[str]:
    """从 P_t 取目标词面（约束词 + 品类，经 normalize_terms）——「目标遗忘」的跨语言词桥。

    P_t 由 planner 单写者维护，keywords 按 prompt 契约中英双给，是现成的确定性词面；planner
    返回即刷新（本 hook 在它 post_tool_call 时调用）。读失败返回空集：信号退回 query 词面，
    不中断 Agent。"""
    try:
        from app.api.context import get_session_pt

        pt = get_session_pt()
        if pt is None:
            return set()
        terms = pt.like_terms() + pt.dislike_terms() + pt.soft_dislike_terms()
        if pt.category:
            terms.append(pt.category)
        return set(normalize_terms(terms))
    except Exception:
        logger.debug("读取 P_t 目标词失败", exc_info=True)
        return set()


def _is_empty_result(result_text: str) -> bool:
    """检索是否**真的**空手而归——优先解析 JSON 字段，解析不动才退回文本特征。

    初版是 ``'"candidates": []' in text`` 的格式耦合匹配，两个坑：① item_search 会把已入池
    候选折叠成 ``already_in_pool``（追问轮复用候选时 fresh 为空 ≠ 空结果），字符串匹配把复用
    轮数成「连续空」，三轮就够触发「严重偏离→强制收尾」；② 序列化格式一改，信号静默死。
    判空以 ``total_recall`` 为准：candidates 为空但总召回 > 0 是「被约束筛光」，方向没错，
    不算探索发散。"""
    try:
        data = json.loads(result_text)
    except (ValueError, TypeError):
        data = None
    if isinstance(data, dict) and "candidates" in data:
        if data.get("candidates") or data.get("already_in_pool"):
            return False
        return not data.get("total_recall")
    return "未找到" in result_text or "0 条" in result_text


@harness_hook("post_tool_call", name="drift_result_tracker", priority=50)
async def track_result_signals(context: dict[str, Any]) -> dict[str, Any] | None:
    """从工具返回里累加两类漂移信号：连续空结果（信号 2）、黑名单命中（信号 3）。"""
    state: DriftState | None = context.get("_drift_state")
    if state is None:
        return None

    tool_name = context.get("tool_name", "")
    tool_result = context.get("tool_result", "")
    if not isinstance(tool_result, str):
        tool_result = str(tool_result)

    # 目标词桥刷新：planner 刚写完 P_t（单写者，返回即最新）
    if tool_name == "planner":
        state.goal_terms = _pt_goal_terms()

    # 信号 2: 探索发散——检索类工具连续返回空
    if tool_name in _SEARCH_TOOLS:
        state.recent_tool_names.append(tool_name)
        if _is_empty_result(tool_result):
            state.consecutive_empty_results += 1
        else:
            state.consecutive_empty_results = 0

    # 信号 3: 偏好丢失——最终推荐面上出现用户硬排除的属性
    if tool_name in _RECOMMEND_TOOLS:
        hits = blacklist_hits(tool_result)
        if hits:
            state.blacklist_violations += len(hits)
            for term in hits:
                if term not in state.violated_terms:
                    state.violated_terms.append(term)
            logger.info("Drift: %s 结果命中黑名单 %s", tool_name, hits)

    return None
