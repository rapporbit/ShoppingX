"""Rubrics as Rewards（RaR）：按 query 动态生成评分细则，judge 模型逐项打分。refdocs 08。

**与 ``recall_metrics`` 的分工：** 召回评测是确定性指标（recall@k / nDCG，无需 LLM）；本模块评的是
**Agent 端到端回答的质量**——「针对这条特定需求，回答满不满足关键维度」，必须靠强 judge 模型动态生成
细则再逐项打分。三档结构（refdocs 08 §2-3）：

- **P0 业务红线**（一票否决）：超预算、性别/品类冲突、推违禁品、泄露内部 id/工具名。
  任一 fail 即整体 fail。
- **P1 执行规范**（每违反扣 ``P1_PENALTY`` 分）：该调的工具没调、无收尾、死循环、商品卡缺要素。
- **P2 质量打分**（1-5）：需求覆盖度、场景洞察力、决策建议价值。

**为什么 Rubric 要动态生成：** 换一条 query（「工业螺丝批量采购」vs「送闺蜜伴手礼」）红线与维度
完全不同，通用模板打分会失真。所以每条 query 先让 judge 生成专属细则，再据此打分。

judge 调用全走 :func:`app.agent.llm.get_judge_llm`（更强、temperature=0，评分稳定可复现）。
生成与打分各一次结构化输出（``with_structured_output``），与 ``planner`` 等工具同款。
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import AIMessage, AnyMessage
from pydantic import BaseModel, Field

# 顶层 import 安全：tracing 只在 TYPE_CHECKING 下反向引用本模块，运行时无循环依赖。
from app.agent.tracing import record_rubric_scores

logger = logging.getLogger(__name__)

# Rubric 缓存目录：同一 query 的细则**只生成一次**并复用，让回归对照的「尺子刻度」固定——
# 否则 judge 每次重新生成细则（方差），改 Agent 前后的分数变化分不清是 Agent 变了还是尺子抖了。
# 缓存 key 掺入生成 prompt 的 hash：prompt 一改 key 即变、缓存自动失效重建（尺子定义变了就该重建）。
RUBRIC_CACHE_DIR = Path("data/eval/rubric_cache")

# ── 聚合权重（可调）。P0 是闸不是分：任一 fail 直接 total=0、overall_pass=False。
# 质量分以 P2 均值归一到 0-100 为底，再按 P1 违规数扣分。阈值仅作「高分轨迹」筛选锚点（飞轮用）。
P1_PENALTY = 10.0  # 每条 P1 违规扣的分（0-100 制）
HIGH_SCORE_THRESHOLD = 70.0  # ≥ 此分且 P0 全过，算「高分轨迹」（可沉淀 few-shot）
_ARG_CLIP = 120  # 轨迹里单个工具入参的渲染截断


# ────────────────────────────── 数据模型 ──────────────────────────────
class RubricCriterion(BaseModel):
    """一条评分细则。P0/P1 是判定（pass/fail），P2 是 1-5 打分。"""

    tier: Literal["P0", "P1", "P2"]
    dimension: str = Field(description="维度名，如『负向约束与人群匹配』")
    criterion: str = Field(
        description="判定描述：什么算 fail（P0/P1）或 1 分到 5 分各代表什么（P2）"
    )


class Rubric(BaseModel):
    """针对单条 query 动态生成的整套评分细则。"""

    criteria: list[RubricCriterion] = Field(default_factory=list)


class CriterionScore(BaseModel):
    """judge 对单条细则的打分结果。"""

    id: str = Field(description="细则编号，回填用，如 P0-1 / P2-2")
    tier: Literal["P0", "P1", "P2"]
    dimension: str = ""
    passed: bool | None = Field(default=None, description="P0/P1 的判定；P2 留空")
    score: int | None = Field(default=None, description="P2 的 1-5 分；P0/P1 留空")
    rationale: str = Field(default="", description="判定/打分依据，定位 bad case 用")


class _ScoreSheet(BaseModel):
    """judge 一次性返回的全部细则打分（结构化输出载体）。"""

    scores: list[CriterionScore] = Field(default_factory=list)


class RubricResult(BaseModel):
    """单条 query 的完整评测结论。"""

    query: str
    overall_pass: bool  # P0 全过即 True（业务红线没破）
    total: float  # 0-100 质量分；P0 破则为 0
    is_high_score: bool  # 飞轮筛选锚点：P0 全过且 total ≥ 阈值
    p0_failures: list[str] = Field(default_factory=list)  # fail 的 P0 维度名
    p1_violations: list[str] = Field(default_factory=list)  # 违规的 P1 维度名
    p2_avg: float = 0.0  # P2 维度均分（1-5）
    scores: list[CriterionScore] = Field(default_factory=list)
    rubric: Rubric | None = None


# ────────────────────────────── 轨迹抽取 ──────────────────────────────
def extract_tool_calls(messages: list[AnyMessage]) -> list[dict[str, Any]]:
    """从 run_agent 返回的 messages 里抽出（父 loop 的）工具调用序列。

    自洽地从 messages 取，不依赖 monitor/WS（离线评测脚本没有连接）。fork 子 Agent 的内部调用在
    各自 thread 的 messages 里、不进父序列，这里看到的是主 loop 的编排轨迹——正是 P1 要评的对象。
    """
    calls: list[dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            for tc in msg.tool_calls or []:
                calls.append({"name": tc.get("name", "?"), "args": tc.get("args", {})})
    return calls


def render_trajectory(calls: list[dict[str, Any]]) -> str:
    """把工具调用序列渲染成给 judge 看的紧凑文本（带序号，入参截断防灌爆）。

    **入参脱敏**：丢掉键名含 ``id`` 的参数（如 ``item_ids``）。否则 judge 会把轨迹里的内部
    item_id 误当成「Agent 向用户泄露内部 id」判 P0 信息安全 fail——轨迹是评估用的内部日志，
    不是 Agent 给用户的输出（基线首轮的系统性假阳性根因）。
    """
    if not calls:
        return "（本轮未调用任何工具）"
    lines = []
    for i, c in enumerate(calls, 1):
        arg_str = ", ".join(
            f"{k}={str(v)[:_ARG_CLIP]}"
            for k, v in (c["args"] or {}).items()
            if "id" not in k.lower()
        )
        lines.append(f"{i}. {c['name']}({arg_str})")
    return "\n".join(lines)


def render_agent_output(final_text: str, items: list[dict[str, Any]], trajectory: str) -> str:
    """把 Agent 这一轮的可评材料拼成单段文本：最终回复 + 商品卡 + 工具轨迹。

    轨迹段显式标注为「内部执行日志」并声明不得据此判信息泄露——judge 评信息安全红线时只该看
    **最终回复正文**有没有对用户暴露内部 id/工具名，不该把评估流程自带的内部材料算作泄露。
    """
    head = "【最终回复（这是 Agent 给用户看的内容，信息安全红线只评这一段）】\n"
    parts = [head + (final_text or "（空）")]
    if items:
        # landed_usd=到手价，是给用户的正常商品卡字段，不是「内部字段泄露」。
        card_lines = [
            f"- {it.get('title', '?')} | {it.get('platform', '?')} | "
            f"到手价${it.get('landed_usd')} | 理由={it.get('reason', '')}"
            for it in items
        ]
        parts.append("【商品卡（给用户展示的卡片）】\n" + "\n".join(card_lines))
    parts.append(
        "【内部执行日志·非用户可见·仅供核对工具调用是否合规，禁止据此判定信息泄露】\n" + trajectory
    )
    return "\n\n".join(parts)


# ────────────────────────────── judge prompt ──────────────────────────────
_GEN_PROMPT = """你是电商购物 Agent 的评测细则设计专家。针对下面这条**具体的购物意图**，设计一套\
专属评分细则（Rubrics as Rewards），分三档：

- P0 业务红线（一票否决）：违反即整体判 fail。如主体商品超预算、把女士用品推给男性礼物场景、\
**最终回复正文**对用户暴露内部 item_id 或工具名。请针对本条 query 的硬约束具体化。
- P1 执行规范（每违反扣分）：如**用户要求的**该调工具没调（要比价才需 price_compare、要到手价\
才需 shipping_calc）、没有收尾清单、同一工具反复调用形成死循环、商品卡缺标题/价格/理由。
- P2 质量打分（1-5）：需求覆盖度、场景洞察力、决策建议价值等，写清 1 分与 5 分各代表什么。

**红线设置纪律（避免误伤，务必遵守）：**
1. 价格类一票否决红线**只在已知约束里有明确 budget 时才设**，且单位一律按 **USD（美元）**，\
不要自行折算人民币、不要凭空假设预算上限。
2. 软偏好（如「便宜」「小众」「抗造」）**不得**升级成 P0 一票否决，至多作 P2 质量维度。
3. 信息安全红线**只针对最终回复正文**是否对用户暴露内部 id/工具名；评测材料里的「内部执行日志」\
「商品卡的到手价字段」是正常内容，**不算泄露**，不要据此设红线。
4. **用户明确提出的检索意图**（如大牌平替、仿大牌风、低价替代款）属正常购物需求，本系统按用户\
明确意图检索，**不得**因「仿/平替/像大牌」设合规或安全红线判 fail。
5. **执行规范只按用户本轮明确要求的任务来设，不要强加用户没要的步骤**：用户只说「推荐/看看有\
什么」时，**不得**因未调 price_compare / shipping_calc（没比价、没算到手价）判 P1 违规——比价、\
到手价只在用户明确要「比价 / 性价比 / 到手价 / 含税含运」时才算「该调的工具」。评价类（「这款\
好不好 / 值不值」）重点评是否给出有依据的好坏判断（相对品类基准 + 是否命中用户偏好）+ 是否附\
推荐卡，而非是否比价。
{intent_rule}
要求：细则要**具体到这条 query**（能引用其预算数字、人群、材质排除等），不要写放之四海皆准的空话。\
每档 2-4 条，总数控制在 6-10 条。

购物意图：{query}
已知约束（客观事实，供你具体化 P0 红线；若为空忽略）：{constraints}

只输出一个 json 对象，形如：
{{"criteria": [{{"tier": "P0", "dimension": "价格红线", "criterion": "主体>80美元即fail"}}]}}
tier 取 P0/P1/P2，dimension 是维度名，criterion 是判定描述。不要输出 json 以外的任何文字。"""

# 非购物意图（闲聊/拒绝）的细则导向：评收尾是否得体，而非强加检索/比价/商品卡规范。
_INTENT_RULE_NON_SHOPPING = """6. **本条不是商品检索意图**（闲聊或应拒绝的请求）：不要设置\
「该调 item_search/price_compare 等检索工具」「该输出商品卡」类规范。重点评估：是否用 \
chat_fallback 恰当收尾、是否礼貌诚实地说明能力边界或拒绝、有没有强行带货/答非所问。
"""

_SCORE_PROMPT = """你是严格、公正的电商购物 Agent 评测官。下面给出一条购物意图、一套评分细则、以及 \
Agent 的实际回答（含最终回复、商品卡、工具调用轨迹）。请逐条细则打分。

打分规则：
- P0/P1 细则：判定 passed=true（满足）或 passed=false（违反），score 留空。
- P2 细则：给 1-5 的整数 score，passed 留空。
- 每条都写一句 rationale 说明依据，引用回答里的具体内容。证据不足时从严（宁可判不满足）。
- id 用细则前的编号（如 P0-1）。务必每条细则都返回一条打分，不要遗漏、不要新增。
- **「内部执行日志」段是评测内部材料、非 Agent 给用户的输出**：评信息安全/隐私类细则时只看\
「最终回复」正文，日志里出现 item_id、工具名、到手价字段都不算泄露，不得据此判 fail。

购物意图：{query}

评分细则：
{rubric_text}

Agent 的实际回答：
{agent_output}

只输出一个 json 对象，形如：
{{"scores": [{{"id": "P0-1", "tier": "P0", "passed": true, "score": null, "rationale": "依据…"}}, \
{{"id": "P2-1", "tier": "P2", "passed": null, "score": 4, "rationale": "依据…"}}]}}
P0/P1 填 passed（布尔）、score 置 null；P2 填 score（1-5 整数）、passed 置 null。\
每条细则对应一条打分，id 用细则前的编号。不要输出 json 以外的任何文字。"""


def _enumerate_rubric(rubric: Rubric) -> tuple[str, dict[str, RubricCriterion]]:
    """给细则编号（P0-1/P1-1/...），返回渲染文本 + id→细则映射（聚合时回填 dimension 用）。"""
    lines: list[str] = []
    by_id: dict[str, RubricCriterion] = {}
    counters: dict[str, int] = {}
    for c in rubric.criteria:
        counters[c.tier] = counters.get(c.tier, 0) + 1
        cid = f"{c.tier}-{counters[c.tier]}"
        by_id[cid] = c
        lines.append(f"[{cid}]（{c.tier}｜{c.dimension}）{c.criterion}")
    return "\n".join(lines), by_id


# ────────────────────────────── 主流程 ──────────────────────────────
def _rubric_cache_key(query: str, constraints: dict[str, Any] | None, intent: str) -> str:
    """缓存 key：query + 约束 + intent + 生成 prompt 文本，全量 hash。

    掺入 prompt 文本，使「改了细则生成 prompt」自动让旧缓存失效——回归对照里，尺子定义没变就复用、
    变了就重建，二者都正确。
    """
    payload = {
        "query": query,
        "constraints": constraints or {},
        "intent": intent,
        "prompt": _GEN_PROMPT + _INTENT_RULE_NON_SHOPPING,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _read_cached_rubric(cache_file: Path) -> Rubric | None:
    """命中则读回缓存细则，否则 None。同步小文件 IO（离线评测，阻塞可忽略）。"""
    if not cache_file.exists():
        return None
    return Rubric.model_validate_json(cache_file.read_text(encoding="utf-8"))


def _write_cached_rubric(cache_file: Path, rubric: Rubric) -> None:
    """把生成的细则写入缓存（同步小文件 IO）。"""
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(rubric.model_dump_json(indent=2), encoding="utf-8")


async def generate_rubric(
    query: str,
    constraints: dict[str, Any] | None = None,
    intent: str = "shopping",
    use_cache: bool = True,
    cache_dir: Path = RUBRIC_CACHE_DIR,
) -> Rubric:
    """调 judge 模型为这条 query 动态生成 P0/P1/P2 细则（命中缓存则直接复用，不调 judge）。

    ``intent`` 非 ``shopping``（闲聊/拒绝）时切换细则导向：评收尾是否得体，不强加检索/比价规范。
    ``use_cache=True``：命中缓存直接返回；未命中生成后写缓存。``use_cache=False``：跳过读、强制
    重新生成并覆盖写（``--refresh-rubric`` 走这条，用于刷新尺子）。
    """
    from app.agent.llm import get_judge_llm

    cache_file = cache_dir / f"{_rubric_cache_key(query, constraints, intent)}.json"
    if use_cache:
        cached = _read_cached_rubric(cache_file)
        if cached is not None:
            return cached

    intent_rule = "" if intent == "shopping" else _INTENT_RULE_NON_SHOPPING
    # json_mode（非 function_calling）：本仓库 judge 走 DashScope/Qwen 兼容端点，不支持
    # tool_choice=required，故用 json_object 模式——prompt 里已自带 json 字样与结构说明。
    structured = get_judge_llm().with_structured_output(Rubric, method="json_mode")
    result = await structured.ainvoke(
        _GEN_PROMPT.format(
            query=query, constraints=constraints or "（无）", intent_rule=intent_rule
        )
    )
    rubric = result if isinstance(result, Rubric) else Rubric.model_validate(result)
    if not rubric.criteria:
        raise ValueError(f"judge 未生成任何细则（query={query!r}）")

    _write_cached_rubric(cache_file, rubric)
    return rubric


async def score_against_rubric(
    query: str, rubric: Rubric, agent_output: str
) -> list[CriterionScore]:
    """调 judge 模型，按细则给 Agent 回答逐条打分，并把 dimension 回填到每条打分上。"""
    from app.agent.llm import get_judge_llm

    rubric_text, by_id = _enumerate_rubric(rubric)
    structured = get_judge_llm().with_structured_output(_ScoreSheet, method="json_mode")
    result = await structured.ainvoke(
        _SCORE_PROMPT.format(query=query, rubric_text=rubric_text, agent_output=agent_output)
    )
    sheet = result if isinstance(result, _ScoreSheet) else _ScoreSheet.model_validate(result)
    for s in sheet.scores:
        if s.id in by_id and not s.dimension:
            s.dimension = by_id[s.id].dimension
    return sheet.scores


def aggregate(query: str, rubric: Rubric, scores: list[CriterionScore]) -> RubricResult:
    """把逐条打分聚合成结论：P0 闸 + P1 扣分 + P2 均分归一。纯函数，可单测。"""
    p0_failures = [s.dimension or s.id for s in scores if s.tier == "P0" and s.passed is False]
    p1_violations = [s.dimension or s.id for s in scores if s.tier == "P1" and s.passed is False]
    p2_scores = [s.score for s in scores if s.tier == "P2" and s.score is not None]
    p2_avg = sum(p2_scores) / len(p2_scores) if p2_scores else 0.0

    overall_pass = not p0_failures
    if not overall_pass:
        total = 0.0
    else:
        p2_pct = (p2_avg / 5.0) * 100.0 if p2_scores else 100.0
        total = max(0.0, p2_pct - P1_PENALTY * len(p1_violations))

    return RubricResult(
        query=query,
        overall_pass=overall_pass,
        total=round(total, 1),
        is_high_score=overall_pass and total >= HIGH_SCORE_THRESHOLD,
        p0_failures=p0_failures,
        p1_violations=p1_violations,
        p2_avg=round(p2_avg, 2),
        scores=scores,
        rubric=rubric,
    )


async def evaluate(
    query: str,
    run_result: dict[str, Any],
    constraints: dict[str, Any] | None = None,
    intent: str = "shopping",
    use_cache: bool = True,
) -> RubricResult:
    """端到端评测一条 query：生成细则 → 渲染回答 → 打分 → 聚合。

    ``run_result`` 为 :func:`app.agent.main_agent.run_agent` 的返回
    （含 final_text / messages / items / trace_id）。``intent`` 透传给细则生成（闲聊类换导向）。
    ``use_cache`` 控制细则是否复用缓存（回归对照固定尺子用 True，刷新尺子用 False）。
    生成与打分分两次 judge 调用，各自结构化；任一异常向上抛，由跑批脚本捕获并记为该条评测失败。

    收尾把结论作为 score 挂回 ``run_result["trace_id"]`` 那条 Langfuse trace（refdocs 16-3 §2.4），
    于是「低分」与「当时怎么跑的」落在同一处，可在 UI 里按 ``rubric_total`` 筛 bad case trace。
    未启用观测（trace_id 为 None）时该调用零副作用；:func:`aggregate` 仍是纯函数，单测不受影响。
    """
    final_text = run_result.get("final_text", "")
    items = run_result.get("items", []) or []
    messages = run_result.get("messages", []) or []

    trajectory = render_trajectory(extract_tool_calls(messages))
    agent_output = render_agent_output(final_text, items, trajectory)

    rubric = await generate_rubric(query, constraints, intent, use_cache=use_cache)
    scores = await score_against_rubric(query, rubric, agent_output)
    result = aggregate(query, rubric, scores)
    record_rubric_scores(run_result.get("trace_id"), result)
    return result
