"""M23 planner RL 的 reward 函数（S1 环境的核心件）。

```
R = 0.45 × R_retrieval   # keywords 真打进 Qdrant：must_have 命中率 + 品类纯度
  + 0.30 × R_field       # category / domains / budget 与 golden 的字段级一致
  + 0.15 × R_format      # 枚举合法 + exclude_terms 带 evidence
  + 0.10 × R_econ        # 检索词过长 / 过短 / 同义堆砌的惩罚
schema parse 失败 → R = -1.0（一票否决）
```

**为什么 R_retrieval 占 45%**：它是这件事的 agentic 内核。refdocs 08-2 用 BERTScore 比字面
语义，而本项目有它没有的条件——138 万点的 Qdrant 就在本地、item_search 只要 0.3s、完全
确定性。能直接问「这组词到底搜到东西没有」，就没理由退回去比字面。没有这一维，planner RL
就退化成普通 RLHF。

**三条硬纪律**：

1. **弃权维度不计分，权重重分配**（不是当 0 分罚）。golden 里 `category=None`（无上文碎片）、
   `budget_uncertain=True`（中文数词口语区间）都是「这题无解」，拿 0 分去罚等于教模型
   在无解的题上瞎猜。见 `scripts/train/build_planner_golden.py` 的弃权语义。
2. **命中口径必须复用 `app.utils.terms.term_hits`**——全链路「命中怎么算」一个口径。裸 `in`
   会让 "watch" 命中 "watching"，reward 里错一次，模型会顺着这个错误梯度一路跑偏。
3. **反 hacking 门禁前置**（对齐 08-2 §2.4）：keywords 照抄原 query 是最容易被 RL 发现的
   捷径——中文 query 打英文库召回本来就差，但「照抄」在字面指标上不吃亏。必须在 reward
   层面掐死，否则训出来的 planner 就是个复读机。

reward 是**纯函数**：输入 PlanOutput 的 dict 与 golden 的 dict，输出 (总分, 分项)。不依赖任何
RL 框架、不碰会话副作用——S3 换 ms-swift / verl / 自研 loop 都只要包一层。检索那一维需要外部
注入 `retrieve` 回调，本地测试可传桩。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from app.memory.domains import ALL_DOMAINS
from app.utils.terms import term_hits

WEIGHTS = {"retrieval": 0.45, "field": 0.30, "format": 0.15, "econ": 0.10}
PARSE_FAIL_REWARD = -1.0

# top20 里有一半命中 must_have 锚就算满分。要求 100% 是不现实的——库里同品类商品本来就
# 掺着配件与周边（accessory_flood 那一族的根因是数据不是算法），把标准定在够不着的地方，
# 梯度就永远指向「再堆几个同义词」这种没用的方向。
HIT_RATE_FULL = 0.5
TOP_K = 20

# 检索词条数的合理区间。少于 2 个通常是只给了个品类词（召回太宽），多于 6 个是同义词堆砌
# （Qdrant dense 检索里堆同义词并不会更准，只会把 query 向量拖向词表中心）。
KW_MIN, KW_MAX = 2, 6
KW_TOKEN_MAX = 4  # 单个检索词超过 4 个 token，基本是把整句塞进来了

_WORD = re.compile(r"[a-z0-9]+")
_CJK = re.compile(r"[一-鿿]")


@dataclass
class RewardBreakdown:
    """分项明细。训练时只用 `total`，调参与 debug 全靠这些分项——只看总分是查不出
    「涨分到底涨在哪一维」的，M22 端到端 A/B 无结论就吃过这个亏。"""

    total: float = 0.0
    retrieval: float | None = None
    # 不叫 field：类体里一旦出现 `field: ... = None`，就把 dataclasses.field 遮蔽成 None，
    # 下面两行的 field(default_factory=...) 直接 TypeError。踩过，别改回去。
    field_score: float | None = None
    fmt: float = 0.0
    econ: float = 0.0
    penalties: list[str] = field(default_factory=list)
    detail: dict = field(default_factory=dict)


def _norm(s: str) -> str:
    return re.sub(r"[\s\-_/、,，.。]+", "", (s or "").lower())


def _tokens(s: str) -> set[str]:
    return {w for w in _WORD.findall((s or "").lower()) if len(w) > 2}


# ── R_format：schema 之外的「合法性」，parse 本身由调用方兜（失败直接 -1）────────────
def score_format(plan: dict) -> tuple[float, list[str]]:
    """枚举合法 + exclude_terms 带 evidence + 系统回填字段没被模型抢着填。

    第三项容易被忽略但很重要：`currency` / `budget_usd` / `exclude_keywords` / `dest_country`
    是**系统确定性回填**的（prompt 里写明「模型不要填」）。模型填了不会报错，只会被覆盖——
    但那说明它没读懂分工，且白烧 token。这里给一点扣分，让它学会闭嘴。
    """
    issues: list[str] = []
    score = 1.0

    doms = plan.get("domains") or []
    if bad := [d for d in doms if d not in ALL_DOMAINS]:
        issues.append(f"域不在枚举内：{bad}")
        score -= 0.5
    if "global" in doms:  # 偏好侧「跨品类底线」专用标记，planner 填它等于污染全局
        issues.append("填了 global")
        score -= 0.5

    for t in plan.get("exclude_terms") or []:
        if not (t.get("evidence") or "").strip():
            issues.append(f"exclude_terms 缺 evidence：{t.get('word')}")
            score -= 0.3
            break

    system_owned = ("currency", "budget_usd", "exclude_keywords", "dest_country")
    if filled := [k for k in system_owned if plan.get(k)]:
        issues.append(f"抢填系统回填字段：{filled}")
        score -= 0.1 * len(filled)

    return max(0.0, score), issues


# ── R_econ：检索词形态。它治的是「话说得对但搜不动」──────────────────────────────
def score_econ(plan: dict) -> tuple[float, dict]:
    """条数落在 [2,6]、单词别超 4 token、别堆同义词。

    同义堆砌的判法是**词面重叠**（"laptop bag" 与 "laptop backpack" 共享 laptop）：dense 检索里
    堆同义词不会更准，只会把 query 向量拖向词表中心。这不是理论——M21 的「正例全展开」有效、
    「同义扩写」无效，是同一枚硬币的两面。
    """
    kws = [k for k in (plan.get("keywords") or []) if str(k).strip()]
    n = len(kws)
    if n == 0:
        return 0.0, {"reason": "无检索词"}

    count_score = 1.0 if KW_MIN <= n <= KW_MAX else max(0.0, 1.0 - 0.25 * min(
        abs(n - KW_MIN), abs(n - KW_MAX)
    ))
    long_ratio = sum(1 for k in kws if len(_WORD.findall(str(k).lower())) > KW_TOKEN_MAX) / n
    # 重叠率：所有词的 token 总数 vs 去重后的 token 数，越接近 1 说明各词越独立
    all_toks = [t for k in kws for t in _tokens(str(k))]
    overlap = 1.0 - (len(set(all_toks)) / len(all_toks)) if all_toks else 0.0

    score = count_score * (1 - 0.5 * long_ratio) * (1 - 0.6 * min(1.0, overlap * 2))
    return max(0.0, min(1.0, score)), {
        "n": n, "长词占比": round(long_ratio, 2), "词面重叠": round(overlap, 2)
    }


# ── R_field：与 golden 的字段级一致。**弃权维度不计分**，不是当 0 分罚 ──────────────
def _cat_match(pred: str, gold: str) -> float:
    """品类宽松匹配。三票投票里「剪刀 / 工具」「VR 眼镜 / 虚拟现实眼镜」都算对，
    比字面就是在罚合理答案（这是标注阶段就看清楚的事，reward 侧必须跟上同一口径）。"""
    p, g = _norm(pred), _norm(gold)
    if not p or not g:
        return 1.0 if p == g else 0.0
    if p == g or p in g or g in p:
        return 1.0
    # 中文按字集合、英文按词集合算 Jaccard，给部分分而不是一刀切 0
    pc, gc = (set(p), set(g)) if _CJK.search(g) else (_tokens(pred), _tokens(gold))
    inter = len(pc & gc)
    return round(inter / max(1, len(pc | gc)), 3) if inter else 0.0


def _budget_match(plan: dict, gold: dict) -> float | None:
    """预算：金额 + clear_budget。`budget_uncertain` 的样本返回 None（跳过计分）。"""
    if gold.get("budget_uncertain"):
        return None
    gb, pb = gold.get("budget_amount"), plan.get("budget_amount")
    amount_ok = (
        1.0 if gb is None and pb is None
        else 0.0 if gb is None or pb is None
        else 1.0 if abs(float(pb) - float(gb)) < 1e-6 else 0.0
    )
    clear_ok = 1.0 if bool(plan.get("clear_budget")) == bool(gold.get("clear_budget")) else 0.0
    return round(0.7 * amount_ok + 0.3 * clear_ok, 3)


def score_field(plan: dict, gold: dict) -> tuple[float | None, dict]:
    """category / domains / budget 三维加权。三维全弃权则整维返回 None。"""
    parts: dict[str, float] = {}

    if (gc := gold.get("category")) is not None:
        parts["category"] = _cat_match(str(plan.get("category") or ""), str(gc))
    if (gd := gold.get("domains")) is not None:
        pred, goldset = set(plan.get("domains") or []), set(gd)
        if not pred and not goldset:
            parts["domains"] = 1.0
        elif not pred or not goldset:
            parts["domains"] = 0.0
        else:  # 集合 F1：漏判与多判都要罚，只算交集会奖励「把所有域都填上」
            inter = len(pred & goldset)
            p, r = inter / len(pred), inter / len(goldset)
            parts["domains"] = round(2 * p * r / (p + r), 3) if inter else 0.0
    if (b := _budget_match(plan, gold)) is not None:
        parts["budget"] = b

    if not parts:
        return None, {"reason": "三维全弃权"}
    # 等权平均：三维都是 planner 的核心判定，没有理由厚此薄彼
    return round(sum(parts.values()) / len(parts), 3), parts


# ── R_retrieval：45% 的大头。拿模型产的 keywords 真打一次 Qdrant ────────────────────
def score_retrieval(titles: Sequence[str], gold: dict) -> tuple[float | None, dict]:
    """给定「用 policy 的 keywords 搜回来的 top20 标题」，算命中率与品类纯度。

    刻意**只吃标题列表**，不吃 Candidate 对象、更不自己发起检索：这样它在单测里能用桩数据
    跑完（不必起 Qdrant），在 rollout 里由环境侧注入真实检索结果。reward 与 I/O 分家。

    弱锚样本（golden 的 must_have 无 2 票交集）不额外降权——降权逻辑放调用方，这里只负责
    如实算分并把 `weak` 标出来。
    """
    must = [m for m in (gold.get("must_have") or []) if str(m).strip()]
    anchor = str(gold.get("category_anchor") or "")
    if not must and not anchor:
        return None, {"reason": "无锚可判"}
    if not titles:
        return 0.0, {"reason": "召回为空"}

    lowered = [t.lower() for t in titles[:TOP_K]]
    n = len(lowered)
    hit = sum(1 for t in lowered if any(term_hits(m.lower(), t) for m in must)) if must else 0
    atoks = _tokens(anchor)
    pure = sum(1 for t in lowered if atoks & _tokens(t)) / n if atoks else 0.0
    hit_rate = hit / n

    if must and atoks:
        score = 0.6 * min(1.0, hit_rate / HIT_RATE_FULL) + 0.4 * pure
    elif must:
        score = min(1.0, hit_rate / HIT_RATE_FULL)
    else:
        score = pure
    return round(min(1.0, score), 3), {
        "top_k": n, "命中数": hit, "命中率": round(hit_rate, 3), "品类纯度": round(pure, 3)
    }


# ── 反 hacking 门禁（对齐 refdocs 08-2 §2.4）───────────────────────────────────────
def _is_copycat(plan: dict, user_text: str) -> bool:
    """keywords 是不是在照抄原 query。

    这是 RL 最容易发现的捷径：把用户原话整句塞进 keywords，字面上「什么都没漏」，
    可中文口语打英文商品库召回本来就差——它在 R_field / R_format 上一分不丢，全靠
    R_retrieval 那一维扛。掐死它，否则训出来的 planner 就是个复读机。
    """
    kws = [str(k) for k in (plan.get("keywords") or []) if str(k).strip()]
    if not kws or not user_text:
        return False
    joined, src = _norm(" ".join(kws)), _norm(user_text)
    if not src:
        return False
    # 单个检索词就是整句、或拼起来与原句高度重合，都算照抄
    if any(_norm(k) == src or (len(_norm(k)) > 8 and _norm(k) in src) for k in kws):
        return True
    return joined in src or src in joined


def _evidence_faked(plan: dict, user_text: str) -> bool:
    """exclude_terms 的 evidence 必须是**本轮原话的片段**。编一句话出来当证据，
    等于凭空获得硬淘汰权——golden 那边只标了「本轮该不该有排除」，成不成立在这里查。"""
    src = _norm(user_text)
    for t in plan.get("exclude_terms") or []:
        ev = _norm(t.get("evidence") or "")
        if ev and ev not in src:
            return True
    return False


def compute_reward(
    plan: dict | None,
    gold: dict,
    user_text: str,
    titles: Sequence[str] | None = None,
) -> RewardBreakdown:
    """总入口。``plan=None`` 表示 schema parse 失败 → 一票否决 -1.0。

    ``titles`` 是用 plan 的 keywords 检索回来的 top20 标题；传 None 表示这次 rollout 没跑检索
    （调试 / 消融），R_retrieval 弃权、权重顺延给其余三维。

    **权重重分配**是这里唯一容易写错的地方：某一维弃权时，不能把它当 0 分算进分母，否则
    「无解的题」会系统性拉低所有样本的分，GRPO 的组内优势就被这个常数偏置污染了。正确做法
    是从分母里把它拿掉——只在**实际参与的维度**上归一。
    """
    if plan is None:
        return RewardBreakdown(total=PARSE_FAIL_REWARD, penalties=["schema parse 失败"])

    br = RewardBreakdown()
    br.fmt, fmt_issues = score_format(plan)
    br.econ, econ_detail = score_econ(plan)
    br.field_score, field_detail = score_field(plan, gold)
    br.retrieval, retr_detail = (
        score_retrieval(titles, gold) if titles is not None else (None, {"reason": "未跑检索"})
    )

    # 门禁：先算分再打折，顺序不能反——打折的是「这一维的得分」，不是它的权重。
    if _is_copycat(plan, user_text):
        br.penalties.append("keywords 照抄原 query → R_retrieval 折半")
        if br.retrieval is not None:
            br.retrieval = round(br.retrieval * 0.5, 3)
    if _evidence_faked(plan, user_text):
        br.penalties.append("exclude evidence 不在原话里 → R_field 置 0")
        if br.field_score is not None:
            br.field_score = 0.0

    scored = {k: v for k, v in
              (("retrieval", br.retrieval), ("field", br.field_score),
               ("format", br.fmt), ("econ", br.econ)) if v is not None}
    wsum = sum(WEIGHTS[k] for k in scored) or 1.0
    br.total = round(sum(WEIGHTS[k] * v for k, v in scored.items()) / wsum, 4)
    br.detail = {
        "format_issues": fmt_issues, "econ": econ_detail,
        "field": field_detail, "retrieval": retr_detail,
        "参与计分的维度": sorted(scored),
    }
    return br
