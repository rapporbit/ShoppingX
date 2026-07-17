"""item_picker —— 在合流候选集里按用户偏好二次精挑。

召回 + 比价 + 到手价之后，候选还是一堆；这步按用户的硬约束（必须满足，违反就淘汰）和
软偏好（尽量满足，命中就加分）挑出最终的几件，并给每件写「为什么选它」的理由——直接喂给
``shopping_summary`` 收尾。

**规则精挑，不调 LLM**：精挑是「过滤 + 打分排序」（对齐论文 Planner 的 Filter / Matcher /
Attenuator / Aggregator 四步），规则化既确定可测、又零额外 token / 延迟。模型在调用前
（通常已跑过 planner）把自然语言偏好解析成结构化入参传进来：
  - ``budget_usd`` + ``exclude_keywords``：**负向硬** + 结构化硬约束（**Filter**），超预算 / 命中
    排除词即淘汰。材质/颜色等可枚举属性已在入库时抽成结构化字段、检索环节先行过滤，这里只填自由文本。
  - ``must_have``：**正向硬**约束（**Matcher 高权重**），如「必须金属」——比软偏好重、强力上浮，但
    **不淘汰不匹配的**（自由文本正向 keep-only 会误杀→空结果；用权重强度而非二值 drop 表达硬软）。
  - ``prefer_keywords``：**正向软**偏好，命中加分（**Matcher**），只装结构化覆盖不到的自由文本。
  - ``deprioritize_keywords``：**负向软**避讳词，命中**减分不淘汰**（**Attenuator**，负向软档）。

**四格子 = 论文四工具**（对齐 arXiv:2509.21317 §3.3）：负硬→Filter 淘汰、正硬→Matcher 高权重、
正软→Matcher 加分、负软→Attenuator 减分。正软/正硬/负软的打分**各两路**：① 关键词命中（字面）+
② **embedding 语义**（对意图算候选 cosine：正向 ``+W·sim``、负向 ``−W·sim``），抓「精致高级感」
「塑料感」这类字面漏网的近邻。负向语义作独立减分项、不进正向向量，躲开否定语义弱的反向召回。

打分 = 正硬命中×W_MATCH_HARD + 正硬语义×W_MATCH_HARD_SEM + 正软命中×W_PREF + 正软语义×W_MATCH_SEM
       + **行为亲和命中×W_AFFINITY** + 评分×W_RATING + 便宜度×W_CHEAP
       − 负软命中×W_ATTEN − 负软语义×W_ATTEN_SEM
       （**Aggregator** 加权求和），排序取前 ``top_k``。

**行为亲和**（``W_AFFINITY``，见 :mod:`app.memory.affinity`）是唯一一路不来自「模型转述用户的话」的
证据：它从用户**收藏过什么**里数出来（零 LLM），补的是上面四格子共同的盲区——那四格全都建立在「用户
说得出、且模型转述得对」之上，而人对自己口味的自陈往往既不全也不准。权重刻意压在软偏好之下：行为是
弱证据，只配在同等条件下让对味的那件上浮，不该盖过用户当下明说的诉求。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Annotated, NamedTuple

from langchain_core.tools import InjectedToolArg, tool
from pydantic import BaseModel

from app.api import monitor
from app.api.context import (
    get_original_query,
    get_session_dir,
    get_session_domains,
    get_user_id,
)
from app.memory.assemble import assemble
from app.memory.domains import infer_domains_from_text
from app.memory.session_state import load_pt
from app.recall.reranker import get_reranker
from app.recall.towers import get_tower_client
from app.tools._args import StrListArg
from app.tools._bundle import (
    combine_bundle,
    get_session_bundle,
    prospective_slot,
    render_allocation,
    slot_display,
    slot_query,
)
from app.tools._candidates import (
    compact_candidates,
    register_updates,
    registry_snapshot,
    set_last_picks,
    update_fields,
)
from app.tools._diagnostics import report_diagnostics
from app.tools.schemas import ItemCandidate
from app.utils.env import env_float, env_int
from app.utils.terms import (
    classify_constraint,
    display_term,
    extract_attrs,
    normalize_terms,
    parse_spec_term,
    spec_verdict,
)
from app.utils.terms import term_hits as _hits  # 别名保住既有调用点与测试的引用名

logger = logging.getLogger("shoppingx.item_picker")

# 以下打分权重与展示上限由后台管理页面热更新（见 app/config/registry.py）：值住在模块全局、由
# _load_params() 从 env 求值，改完 env 回调它即重新生效。**故意保持模块级常量形态**而非改成
# param() 式函数调用——既有测试大量 monkeypatch.setattr(mod, "_W_MATCH_HARD_SEM", 0.0) 构造场景，
# 改成函数会让那些 patch 设到没人读的属性上、测试静默失去约束力。下方类型声明让静态检查知道它们
# 存在（赋值在函数里）。不走 env 的三项（_W_PREF/_W_RATING/_W_CHEAP）一并留在函数内，保住整段
# 权重注释的连贯性——它们的相对量级才是这段代码要讲的事。
_W_PREF: float
_W_RATING: float
_W_CHEAP: float
_W_MATCH_HARD: float
_W_MATCH_HARD_SEM: float
_W_ATTEN: float
_W_MATCH_SEM: float
_W_ATTEN_SEM: float
_W_AFFINITY: float
_W_SPEC_CONFLICT: float
_RERANK_FLOOR: float
_W_RERANK_MISS: float
_SLOT_RERANK_FLOOR: float
_W_SLOT_RERANK: float
PICK_DISPLAY_CAP: int
PICK_REL_SHOW_RATIO: float


def _load_params() -> None:
    """从 env 求值本模块的可调参数（导入时跑一次；后台改参数后由覆盖层回调）。"""
    global _W_PREF, _W_RATING, _W_CHEAP, _W_MATCH_HARD, _W_MATCH_HARD_SEM, _W_ATTEN
    global _W_MATCH_SEM, _W_ATTEN_SEM, _W_AFFINITY, _W_SPEC_CONFLICT
    global _RERANK_FLOOR, _W_RERANK_MISS, _SLOT_RERANK_FLOOR, _W_SLOT_RERANK
    global PICK_DISPLAY_CAP, PICK_REL_SHOW_RATIO

    # 打分权重：软偏好命中最重（这是「按偏好精挑」的本职），评分次之，价格便宜度再次。
    _W_PREF = 1.0
    _W_RATING = 0.6
    _W_CHEAP = 0.4
    # 正向**硬**约束（must_have，如「必须金属」）的加分权重——比软偏好重，强力上浮匹配项。
    # **不做二值淘汰**：库无可靠材质/颜色字段，keep-only 会误杀「是金属但标题没写金属」的候选、
    # 导致空结果；改用「更高权重 Matcher」表达硬软之别——硬拉得更狠、但浮不上来也不淘汰
    # （无空结果风险）。关键词路（字面命中）与语义路（embedding sim）各一份权重，均比对应
    # 软档（_W_PREF/_W_MATCH_SEM）重。
    _W_MATCH_HARD = env_float("PICK_W_MATCH_HARD", 2.0)
    _W_MATCH_HARD_SEM = env_float("PICK_W_MATCH_HARD_SEM", 1.0)
    # 软性避讳（Attenuator）**关键词**命中的减分权重：与 _W_PREF 对称——命中一个软偏好 +1、命中一个
    # 软避讳 -1，方向相反、量级相当。软避讳只压低排序、不淘汰（淘汰是 exclude 硬约束的职责）。
    _W_ATTEN = 1.0
    # 语义 Matcher / Attenuator（对齐论文 arXiv:2509.21317 §3.3 式(6)/(8)）的加/减分权重。
    # 关键词只抓字面；这两路对正/负软意图算候选**语义相似度**：正向 ``+W·sim`` 加分（式6 Matcher）、
    # 负向 ``−W·sim`` 减分（式8 Attenuator），补「塑料感」「精致高级感」这类没写字面词但语义
    # 近的近邻。负向作独立减分项、不进正向 query 向量，躲开 embedding 否定语义弱导致的反向召回。
    # **0.7 由消融标定**（scripts/ablation_semantic_pick.py：跨语言中文偏好词对英文标题，
    # 5 场景确定性 gt，NDCG@10 在 0.5→0.8 单调升、峰约 0.8；取 0.7 为「捕获多数增益又不过拟合
    # n=5 噪声」的保守值）。
    # 设 0 即关闭该路（省编码、零额外延迟）。可经 env 覆盖。
    _W_MATCH_SEM = env_float("PICK_W_MATCH_SEM", 0.7)
    _W_ATTEN_SEM = env_float("PICK_W_ATTEN_SEM", 0.7)
    # 行为亲和（收藏聚合出的属性词，见 app.memory.affinity）命中的加分权重。
    # **刻意低于 _W_PREF（1.0）**：这是从行为里**推断**的取向，比用户本轮亲口说的软偏好弱一档
    # ——它该做的是「同等条件下让对味的那件上浮」，而不是盖过用户当下的诉求。设 0 即关闭这一路
    # （与 AFFINITY_MAX_TERMS=0 等效，留两个开关是因为一个关消费、一个关聚合）。语义路**不给**：
    # 亲和词本就是从标题词表里数出来的字面 token（canvas / nylon），再对它算 embedding 相似度
    # 是拿弱证据放大弱证据。
    #
    # **0.2 的来历（诚实标注：调出来的，不是消融标定的）**：初版取 0.5，真实链路 A/B 打脸——25 件短袖
    # 真候选、B 组收藏 2 件亚麻，结果 7 件亚麻款霸占前 7，一件 3.8 分的压过了 4.6 分的非亚麻款。因为
    # 一个亲和命中(+0.5)几乎等于评分项拉满(_W_RATING×1.0=0.6)：弱证据事实上成了主导排序因子，越过了
    # 「同等条件下才上浮」的设计意图。0.2 的量级 ≈ 评分差 0.33 分（0.2÷0.6×1.0×5）——够在其它条件接近
    # 时决出胜负，不够盖过一个明显更好的评分。要定死该像 _W_MATCH_SEM 那样跑一次消融
    # （scripts/ablation_semantic_pick.py 的做法），本次没做。
    _W_AFFINITY = env_float("PICK_W_AFFINITY", 0.2)
    # 数值规格（尺寸/容量，如「16寸」「16 inch」）冲突的减分权重。规格是**互斥枚举**：标题明确标了
    # 另一个更小的档位（14 inch 对 16 寸要求）就是摆在明面上的不合格证据，与「标题没写」完全两回事
    # ——后者不奖不罚（unknown），前者按此权重沉底（量级对齐品类门 miss 的 _W_RERANK_MISS：都是
    # 「明确证据表明不对路」级别的降权，不硬淘汰，容解析噪声）。匹配/冲突判定在 utils.terms 的
    # spec 专道（确定性数值比对，含 up-to 上界语义与 ±5% 容差），字面路和语义路对数字都失明，
    # 见 terms.py 的「数值规格专道」段。
    _W_SPEC_CONFLICT = env_float("PICK_W_SPEC_CONFLICT", 2.0)

    # 品类一致性相关性分（cross-encoder，见记忆定稿「picker 合流后一次批量 rerank」）的两种用法：
    # ① 普通轮：低于 FLOOR 判跨品类混入，降权沉底不剔除（定稿场景：炊具池混进背包，实测分离
    #    0.97 vs 0.006，0.2 留足余量）。
    # ② 套装轮：作**槽内排序信号**（W_SLOT_RERANK 加分）——真品在场时把蹭词垃圾压下去。
    # 槽位**逐出门**（SLOT_FLOOR）默认关闭：badcase 4c0ac682 真实数据标定证伪了绝对阈值——
    # 「品类内配件蹭词」场景分数完全交叠（贴纸 own 0.30~0.60 vs 真笔袋 0.055），任何阈值要么
    # 放垃圾要么杀真品。垃圾占槽的最终兜底在 shopping_summary 的 slot off-intent（LLM 带槽位
    # 语境判，摘除后按缺货报）。若未来语料/模型换代想再试门，开这个 env 前先重跑标定。
    _RERANK_FLOOR = env_float("PICK_RERANK_FLOOR", 0.2)
    _W_RERANK_MISS = env_float("PICK_W_RERANK_MISS", 2.0)
    _SLOT_RERANK_FLOOR = env_float("PICK_SLOT_RERANK_FLOOR", 0.0)  # 0=逐出门关闭（标定结论）
    _W_SLOT_RERANK = env_float("PICK_W_SLOT_RERANK", 1.0)

    # 最终清单展示硬上限：合适的候选（过预算 + 排除词后的 survivors）全部展示，但封顶此值件——
    # 避免一次甩出几十张商品卡灌爆前端 / 上下文。语义是「合适的都给，但有上限」，不是「固定取
    # N 件」。
    # 与 item_search 的 MAX_TOP_K 同思路：模型传的 top_k 只当上界，机制再封一道顶（可经 env 调）。
    # 20→8（2026-07 相机 bad case 的 token 审计）：一次吐 20 件 ×（长标题 + 理由句）≈2,700 tokens，
    # 是整条链最大的单条工具结果、且此后每步解码都要重读；而池子通常 10~30 件、最终清单只要 5~6 件。
    # 只收紧**渲染给模型的件数**——登记表仍是全量，收尾按 id hydrate 不受影响。
    PICK_DISPLAY_CAP = env_int("PICK_DISPLAY_CAP", 8)

    # 展示相对门：cross-encoder 品类相关分显著低于池内头部的候选，判为「品类不够相符」，宁缺毋滥
    # **不凑数展示**（哪怕没填满 PICK_DISPLAY_CAP）——单平台池小时尤要紧：老的「按分填满」策略会把
    # 跨品类蹭词货塞进末位卡（10 挑 8 只淘汰 2，跑题货几乎必然露脸）。相对而非绝对：绝对阈值被
    # badcase 4c0ac682 标定证伪（分数场景间不可比、交叠）；这里用「池内最高相关分 × 本比例」当门，
    # peak 低（整池都一般）时门也低、几乎不砍——不硬造空。仅普通轮 + 真跑了相关性门（rerank_on）时
    # 生效；缺信号退化回「按分填满」，不因缺信号误杀。**保底至少留最高分 1 件**（空清单是空召回诚实
    # 路径的事，不该由展示门主动制造）。0.35 是保守初值（badcase 普通轮真品/垃圾分离 0.97 vs 0.006，
    # 0.35×peak 足够挡垃圾又不误杀边缘真品）——未消融标定，误杀代价高于放垃圾故取低值；设 0 关闭。
    PICK_REL_SHOW_RATIO = env_float("PICK_REL_SHOW_RATIO", 0.35)


_load_params()


class ItemPickerOutput(BaseModel):
    """item_picker 的结构化返回。"""

    picks: list[ItemCandidate]  # 已填 pick_reason，按综合分降序
    excluded: list[str]  # 命中排除词被淘汰的 item_id
    over_budget: list[str]  # 超预算被淘汰的 item_id
    # 本轮模型传的 must_have 在幸存池内命中的**件数**；None = 本轮没传 must_have（不适用）。
    # 供 refine_backfill 判「复用的旧池对本轮新硬条件有没有货」——picks 件数看不出这个
    # （must_have 是加分不淘汰，全不匹配时照样原样返回一整批）。
    must_have_hits: int | None = None
    # 品类一致性门（cross-encoder 相关性分）的池内计数：offcat = 判为跨品类混入、降权沉底的
    # 件数；oncat = 其余（品类相符）。None = 本轮没跑相关性门（无干净品类 query / 服务失败）。
    # 供补搜闸（refine_backfill 的污染分支）判「池子是不是被检索词里的场景词稀释吃空了」——
    # 手表 badcase：formal dress watch 召回 10 条里 8 条西装皮鞋，门正确沉底后只剩 2 件真手表，
    # 但没有任何机制补货，径直收尾出了 2 件的清单。
    oncat_count: int | None = None
    offcat_count: int | None = None
    # 套装组合报告（「一套齐」轮才有，见 app.tools._bundle）：预算分配 / 放弃的可选槽 /
    # 缺货的必备槽 / 每槽升降级备选。None = 本轮不是套装，普通精挑。
    bundle: dict | None = None

    def __str__(self) -> str:
        """喂给模型的紧凑形态：JSON 且丢 url/image_url（见 _candidates.compact_candidates）。

        这里的每个字段都只是**给模型看的**：harness 要的诊断（picks 件数 / must_have_hits /
        oncat / offcat）已在工具返回前走结构化侧信道登记（见 app/tools/_diagnostics.py），
        文本被截断 Hook 砍尾不影响信号。小字段仍放头部：截断后模型至少还看得到计数。
        """
        payload: dict[str, object] = {}
        if self.must_have_hits is not None:
            payload["must_have_hits"] = self.must_have_hits
        # 只在真有跨品类混入时带（常态 offcat=0，写进去纯烧 token）。
        if self.offcat_count:
            payload["oncat_count"] = self.oncat_count
            payload["offcat_count"] = self.offcat_count
        payload.update(
            {
                # 标题截短回显：picks 的完整标题在上下文里的检索结果中已出现过，这里只要
                # 短 handle + item_id 够模型对上号（见 compact_candidates 的 title_chars 说明）。
                "picks": compact_candidates(self.picks, title_chars=60),
                "excluded": self.excluded,
                "over_budget": self.over_budget,
            }
        )
        if self.bundle is not None:
            b = self.bundle
            # rows 不回显（与 picks 同一批货，重复烧 token）；空集合字段全丢；feasible/total
            # 必带（模型要据此讲「超没超预算、砍了谁」）。
            payload["bundle"] = {
                "total_usd": b.get("total_usd"),
                "feasible": b.get("feasible"),
                **({"budget_usd": b["budget_usd"]} if b.get("budget_usd") is not None else {}),
                **({"over_usd": b["over_usd"]} if b.get("over_usd") else {}),
                **(
                    {"skipped_optional": b["skipped_optional"]} if b.get("skipped_optional") else {}
                ),
                **(
                    {"missing_essential": b["missing_essential"]}
                    if b.get("missing_essential")
                    else {}
                ),
                **({"not_included": b["not_included"]} if b.get("not_included") else {}),
                **({"alternatives": b["alternatives"]} if b.get("alternatives") else {}),
            }
        return json.dumps(payload, ensure_ascii=False, default=str)


def _searchable(c: ItemCandidate) -> str:
    """候选的可匹配文本（标题 + 品牌 + 品类，小写）。"""
    return f"{c.title} {c.brand} {c.category}".lower()


# 词命中判定（词边界 + 否定修饰）已上收到 :func:`app.utils.terms.term_hits`——item_search
# （召回阶段的记忆硬排除）与 /api/similar（搜同款的 blocking 过滤）用的是同一套判定：
# 「命中怎么算」全链路必须一个口径，否则同一条黑名单在 picker 挡得住、在别的通路挡不住。


def _split_specs(terms: list[str]) -> tuple[list[str], list[tuple[float, str, str]]]:
    """把归一后的词表切成（普通词, 数值规格），规格附原词做展示。

    数值规格（「16寸」「16 inch」）走 utils.terms 的 spec 专道（确定性数值比对），不进字面
    term_hits（匹不中 "16-inch" 变体）也不进语义 intent（embedding 对数字失明，会给 14 寸
    发同样的分）。同一 ``(值, 单位)`` 去重：planner 常常中英双给（「16寸」+「16 inch」），
    归一后是同一条规格，重复计分等于同一个理由加两次。
    """
    plain: list[str] = []
    specs: list[tuple[float, str, str]] = []  # (值, 归一单位, 展示用原词)
    seen: set[tuple[float, str]] = set()
    for t in terms:
        s = parse_spec_term(t)
        if s is None:
            # 类型学留痕：范围/尺码/装量/代际这些 gap 类约束（见 terms.CONSTRAINT_LANES）
            # 现状回退 topic 通路 = 对该算子双盲。不改行为，只记日志——「哪类裸奔约束
            # 实际出现了多少」是巡检的现实证据，补专道的优先级按它排。
            kind = classify_constraint(t)
            if kind != "topic":
                logger.info("约束类型学：%r 判为 %s（无专道，回退 topic 通路）", t, kind)
            plain.append(t)
        elif s not in seen:
            seen.add(s)
            specs.append((s[0], s[1], t))
    return plain, specs


def _effective_price(c: ItemCandidate) -> float | None:
    """精挑用的有效价：优先到手价，退而求其次用归一货价。"""
    if c.landed_usd is not None:
        return c.landed_usd
    return c.price_usd


# 近重复判定的标题 token Jaccard 阈值。刻意保守：只杀「同一商品的变体」（颜色 / 翻新 / 包装，
# 标题几乎全同），不碰「同规格不同品」——相机 bad case 里两支 Meike 35mm f1.7（Fuji X 口 vs
# Sony E 口）各自的兼容机型列表完全不同，Jaccard≈0.24，是**不同商品**，不该在确定性层被合并；
# 这类「品类对但与主推不搭」的上下文冗余归 shopping_summary 的 off_intent 判断（LLM 带上下文判）。
_NEAR_DUP_JACCARD = 0.6


def _near_duplicate(a: ItemCandidate, b: ItemCandidate) -> bool:
    """同价 + 标题 token 高重合 → 同一商品的变体，最终清单只留一件（留综合分高的）。"""
    if a.price_usd is None or a.price_usd != b.price_usd:
        return False
    ta = set(re.findall(r"[a-z0-9]+", a.title.lower()))
    tb = set(re.findall(r"[a-z0-9]+", b.title.lower()))
    if not ta or not tb:
        return False
    return len(ta & tb) / len(ta | tb) >= _NEAR_DUP_JACCARD


async def _category_relevance(
    survivors: list[ItemCandidate],
) -> tuple[dict[str, float], bool, bool]:
    """候选 × **干净品类 query** 的 cross-encoder 相关性分（相关性门的信号源）。

    返回 ``(item_id → 分数, 门是否可执法, 锚是否分歧)``。锚分歧 = 普通轮的 category 锚
    （planner 的 LLM 输出）与用户原文的词面域判定（独立信号）对不上——「合法但错」的锚
    会让门**反着杀**（把真命中的沉底），此时 fail-open 不执法，比带错锚执法安全。
    三条硬约束（背包 badcase 实测踩出，勿破）：
    ① query 只用干净品类词——套装轮 = 槽 keywords（:func:`slot_query`），普通轮 = P_t 的英文
       品类（planner 判的）；**绝不拼 prefer 偏好词**，拼了实测排序反转。
    ② 任一批打分降级到本地 token 重叠（used_remote=False）→ 整门本轮停用：token 重叠会给
       蹭词垃圾打高分（标题真含 "water bottle"），反向执法比不执法更糟。
    ③ 分数随 query 回写登记表做增量缓存：补搜轮 picker 重跑只给新候选打分。
    没有干净 query（槽 keywords 空 / P_t 无品类）→ 该批不执法，失效方向 = 维持现状。
    """
    if not survivors:
        return {}, False, False
    slots = get_session_bundle()
    jobs: list[tuple[str, list[ItemCandidate]]] = []
    if len(slots) >= 2:  # 套装轮：按「将归入的槽」分批，各槽用各自的干净 query
        by_slot: dict[str, list[ItemCandidate]] = {}
        for c in survivors:
            sid = prospective_slot(c, slots)  # 归槽判定统一返回槽 id
            if sid:
                by_slot.setdefault(sid, []).append(c)
        jobs = [
            (slot_query(s), by_slot[s.id]) for s in slots if slot_query(s) and by_slot.get(s.id)
        ]
    else:  # 普通轮：全池一批，query = planner 判的英文主品类
        sd = get_session_dir()
        category = load_pt(sd).category.strip() if sd is not None else ""
        if category:
            # 锚核验（解锚）：category 与 domains 同出 planner 一张嘴，互证无意义；能反证的
            # 只有用户原文词面。两侧词表域都判得出且交集为空 → 锚不可信 → 本轮不执法。
            # 任一侧空集 = 词表覆盖不到 = 无反证证据，照旧执法（宁漏勿错，见 DOMAIN_TERMS）。
            # 套装轮不做：槽 keywords 经 ask_user 组成确认，已有人工核验通路。
            anchor_d = infer_domains_from_text(category)
            query_d = infer_domains_from_text(get_original_query())
            if anchor_d and query_d and not (anchor_d & query_d):
                logger.warning(
                    "品类门锚分歧：category=%r 判 %s、用户原文判 %s——锚不可信，本轮不执法",
                    category,
                    sorted(anchor_d),
                    sorted(query_d),
                )
                return {}, False, True
            jobs = [(category, list(survivors))]
    if not jobs:
        return {}, False, False

    scores: dict[str, float] = {}

    async def _score(query: str, cands: list[ItemCandidate]) -> bool:
        for c in cands:  # 增量缓存命中：query 没变的候选不重复打分
            if c.rerank_score is not None and c.rerank_query == query:
                scores[c.item_id] = float(c.rerank_score)
        fresh = [c for c in cands if c.item_id not in scores]
        if not fresh:
            return True
        batch, used_remote = await get_reranker().score_detailed(
            query, [_searchable(c) for c in fresh]
        )
        if not used_remote:
            return False
        for c, s in zip(fresh, batch, strict=True):
            scores[c.item_id] = float(s)
            update_fields(c.item_id, rerank_score=float(s), rerank_query=query)
        return True

    try:
        oks = await asyncio.gather(*(_score(q, cands) for q, cands in jobs))
    except Exception:
        logger.warning("相关性门打分失败，本轮停用（失效方向=维持现状）", exc_info=True)
        return {}, False, False
    if not all(oks):
        logger.warning("相关性门：远程精排降级为本地回退，本轮停用（宁不执法勿反向执法）")
        return {}, False, False
    return scores, True, False


@tool
async def item_picker(
    budget_usd: float | None = None,
    exclude_keywords: StrListArg | None = None,
    prefer_keywords: StrListArg | None = None,
    must_have: StrListArg | None = None,
    deprioritize_keywords: StrListArg | None = None,
    top_k: int = PICK_DISPLAY_CAP,
    candidates: Annotated[list[ItemCandidate] | None, InjectedToolArg] = None,
) -> ItemPickerOutput:
    """在候选集里按用户偏好二次精挑出最终清单，并给每件写入选理由。

    何时调用：召回 / 比价 / 到手价之后，要按用户硬约束 + 软偏好选出最终几件时。

    **候选不用你传**：本轮该精挑的商品（新搜到的那批 / 追问轮上一轮那批）系统自己拿，你只负责把
    用户的偏好翻译成下面这些结构化条件。

    参数：
      - budget_usd：预算上限（按到手价硬筛），不传则不限价。
      - exclude_keywords：硬约束排除词，命中即淘汰。可枚举结构化属性（材质/颜色等）已由检索前
        的结构化过滤处理，这里传的是枚举不了的自由文本排除点（如 ["做工廉价感"]）。
      - prefer_keywords：软偏好词，命中加分，同理只传结构化覆盖不到的自由文本（如 ["小众设计"]）。
      - must_have：**正向硬约束**（自由文本，如 ["金属","真皮"]）——用户「必须 / 一定要 X」的 X。
        比 prefer 权重更高、强力上浮匹配项，但**不淘汰不匹配的**（库无可靠材质/颜色字段，二值
        keep-only 会误杀「是金属却没在标题写金属」的候选、造成空结果）。可枚举的结构化硬约束
        （预算/评分/品类）走各自参数与检索期 filter；这里只放枚举不了的自由文本正向硬约束。
      - deprioritize_keywords：**软性避讳词**，命中减分但**不淘汰**（Attenuator）。用于用户
        「不太喜欢 / 尽量避免」这类非绝对排斥（如 ["塑料感","太花哨"]），通常来自 planner 的
        soft_dislikes。绝对不要的走 exclude_keywords（硬淘汰），别混。
      - top_k：最终保留件数上界，默认 = 展示上限 20。语义是「把所有合适候选都留下，
        但最多 20 件」——不要传一个很小的数把合适的候选砍掉；机制会再封一道 20 的顶。

    注：用户的长期偏好与本轮会话约束（P_t）都会**确定性并入**，不依赖模型每轮自觉转述——装配与
    授权规则见 :func:`app.memory.assemble.assemble`（硬淘汰只给用户亲手勾的黑名单 + 本轮亲口说的
    「不要 X」；模型推断的一贯取向、以及「尽量别」这类弱表达一律只减分）。
    """
    # 候选来源：会话登记表的**全部**候选——它自身就是「本轮该精挑的全集」（换品类那轮，planner
    # 判 search 时已把上一轮读回的旧候选清掉；追问轮不清，全集即上一轮那批）。
    #
    # **模型不再传 item_ids**（这个参数已删）。它以前的活是把 10 个 id 逐个抄进入参——纯搬运，不含
    # 任何决策（该精挑哪批是 planner 的 retrieval 早已定死的事），却要模型实打实解码一长串 id（实测
    # 追问轮首次思考因此花掉 ~20s）。更糟的是：一个可选的 id 列表，就是给模型开了「我再自己筛一遍」
    # 的口子，而它筛的时候既没有分数也没有权重——筛选是本工具的职责（同一课在 shopping_summary 上
    # 已经学过一遍）。
    #
    # candidates 是 InjectedToolArg（模型侧不可见）：仅供直接调用 / 单测注入现成候选，绕开登记表。
    if candidates is None:
        candidates = registry_snapshot()

    # 记忆的**唯一**入口：长期库（域内）+ 会话级 P_t，硬 / 软的授权规则全在 assemble 里判一次
    # （见 app.memory.assemble 的模块 docstring）。这里只负责把它和模型本轮传的词并起来。
    mem = await assemble(get_user_id() or "")
    # **匹配词一律归一成英文**（normalize_terms）：商品库是纯英文的（实测 amazon 样本 300 条标题
    # 0 条含中文），而下面的 _hits 是字符串命中——一条中文原子词（「塑料」）去匹 "Plastic Packing
    # Cubes" 永远不命中，用户的「不要塑料」在机制层就是空转。此前它看着能用，全靠 planner 每轮
    # **自觉**多产一个英文变体（exclude_keywords=['塑料','plastic']）；而记忆那条路（curator 从中文
    # 对话抽的原子词原样存库）连这个自觉都没有，一直在空转。跨语言不能指望模型每轮记得——归一放在
    # 匹配前、确定性地做（见 app.utils.terms）。
    # 归一后再拆一刀：数值规格（「16寸」「16 inch」）从每个桶里拆出来走确定性 spec 专道——
    # 字面路匹不中 "16-inch" 变体、语义路对数字失明（会把 14 寸当 16 寸加分），见 _split_specs。
    exclude, exclude_specs = _split_specs(
        normalize_terms(_merge_terms(exclude_keywords, mem.exclude))
    )
    attenuate, attenuate_specs = _split_specs(
        normalize_terms(_merge_terms(deprioritize_keywords, mem.penalty))
    )
    must, must_specs = _split_specs(normalize_terms(_merge_terms(must_have, mem.must)))
    prefer, prefer_specs = _split_specs(normalize_terms(_merge_terms(prefer_keywords)))
    # 行为亲和：单独一路弱加分，**与本轮显式词去重**——一个词既是用户本轮说的软偏好、又出现在他的
    # 收藏里时，只按显式的那档算分。不去重就成了「同一个理由加两次分」（1.0 + 0.5），命中它的商品会
    # 无理由地压过命中另一个同级偏好的商品，而这个差额纯粹是记账重复造成的。
    affinity = [
        t for t in normalize_terms(_merge_terms(mem.affinity)) if t not in prefer and t not in must
    ]
    # 把「长期记忆本轮如何影响了结果」摆到台面上——记忆最危险的失败是**静默**的：一条偏好误杀
    # 了一批商品，用户看不到任何提示，只会觉得「怎么搜不出东西」，且归因不到记忆头上。
    #
    # **上报的是长期库那部分的全量**（mem.memory_*），不是「去重后新增的那部分」，也不含 P_t
    # （那是用户本轮刚说的，不算「记忆生效」）。去重口径一旦兼职当上报口径，事件就恰好在最常见
    # 的路径上静默：偏好本来就注入了 prompt，模型多半会照着转述一遍，于是「新增」为空 → 不发事件
    # → 商品被记忆杀了、用户却什么都看不到。
    await monitor.report_memory_applied(
        get_session_domains(), mem.memory_exclude, mem.memory_penalty
    )
    # 本轮没显式传预算、但会话累积过预算 → 用 P_t 兜底（续聊改颜色不该丢掉上一轮的预算上限）。
    if budget_usd is None:
        budget_usd = mem.budget_usd
    await monitor.report_tool_start("item_picker", count=len(candidates), budget=budget_usd)

    excluded: list[str] = []
    over_budget: list[str] = []
    survivors: list[ItemCandidate] = []
    for c in candidates:
        text = _searchable(c)
        hit = next((kw for kw in exclude if _hits(kw, text)), None)
        if hit is None:
            # 排除桶里的数值规格（「不要14寸」）：标题标称尺寸容差内命中即淘汰——「14 inch」
            # 写法千变（14-inch/14in/14"），字面路挡不全，spec 专道按数值判。
            hit = next(
                (d for v, u, d in exclude_specs if spec_verdict(v, u, text) == "match"), None
            )
        if hit is not None:
            excluded.append(c.item_id)
            # 淘汰事实只**记日志**，不再进 P_t（Mmem 删掉了 rejected_options）：它对模型没用
            # （淘汰每轮按同一套约束机制性重跑，不靠模型记）且有害（模型会把商品标题当排除词
            # 传回来，电商标题高度雷同，一个标题连坐淘汰一大片——真实 bug）。既然不喂模型，
            # 它就不是记忆、是排查日志——那就该待在日志里。
            logger.debug("item_picker 淘汰 %s：命中排除词 %s", c.title[:40], hit)
            continue
        price = _effective_price(c)
        if budget_usd is not None and price is not None and price > budget_usd:
            over_budget.append(c.item_id)
            logger.debug(
                "item_picker 淘汰 %s：超预算（%.2f > %.2f）", c.title[:40], price, budget_usd
            )
            continue
        survivors.append(c)

    # 硬杀词执行力标注（诚实性）：整池 0 命中的硬排除词 = 「没产生任何过滤效果」。不标出来，
    # 用户会以为「花哨」被处理了——实际它匹不中任何英文标题，一件商品都没挡。只做标注不做
    # 拦截：0 命中常是正常的（池子里本来就没有塑料款），异常与否用户自己看得懂；命中过高的
    # 反方向（杀空池子）已有补搜闸兜（f057659）。
    no_effect_excludes = [
        kw for kw in exclude if not any(_hits(kw, _searchable(c)) for c in candidates)
    ]
    if no_effect_excludes:
        logger.info("硬排除词整池 0 命中（未产生过滤效果）：%s", no_effect_excludes)

    # 便宜度归一：在幸存集合里，最便宜得 1、最贵得 0（无价信息的记 0.5 中性）。
    priced = [p for p in (_effective_price(c) for c in survivors) if p is not None]
    lo, hi = (min(priced), max(priced)) if priced else (0.0, 0.0)
    span = hi - lo

    def _cheapness(c: ItemCandidate) -> float:
        price = _effective_price(c)
        if price is None or span == 0:
            return 0.5
        return (hi - price) / span

    # 品类一致性相关性门：cross-encoder 对「干净品类 query」打分。召回是向量近邻，标题蹭词的
    # 跨品类垃圾（water bottle **stickers**）向量分和真品拉不开（实测 0.60 vs 0.67），字面匹配
    # 更拦不住（标题真含关键词）——只有 cross-encoder 拉得开（实测 0.97 vs 0.006）。
    rerank_scores, rerank_on, anchor_conflict = await _category_relevance(survivors)
    # 池内品类计数（补搜闸的污染信号，见 ItemPickerOutput 字段说明）。没拿到分的候选按品类
    # 相符算——判不了不定罪，与「rr < FLOOR 才沉底」的失效方向一致。
    oncat_count: int | None = None
    offcat_count: int | None = None
    if rerank_on:
        offcat_count = sum(
            1
            for c in survivors
            if (rr := rerank_scores.get(c.item_id)) is not None and rr < _RERANK_FLOOR
        )
        oncat_count = len(survivors) - offcat_count

    # 语义 Matcher + Attenuator（对齐论文式6/8）：对正/负软意图算候选的 embedding 相似度——正向 sim
    # 加分（Matcher，抓「精致高级感」这类字面漏网的正向近邻）、负向 sim 取负减分（Attenuator，抓「塑
    # 料感」这类负向近邻）。负向作独立减分项、不进正向 query 向量，躲开否定语义反向召回。标题**只
    # 编码一次**、正负意图各编一次。仅当对应软意图非空 + 权重>0 + 有幸存候选时才编码；编码任一环失败
    # 降级为纯关键词（不反噬主链路）。cosine clamp 到 ≥0（只「像→加/减」，不因不像反向奖惩）。
    sem_match: dict[str, float] = {}  # 正软语义加分
    sem_hard: dict[str, float] = {}  # 正硬语义强加分（must_have）
    sem_penalty: dict[str, float] = {}  # 负软语义减分
    # intent 只拼**普通词**（prefer/must/attenuate 已被 _split_specs 剥掉数值规格）：embedding
    # 对数字失明——"16 inch" 进了 hard_intent，14 寸候选照样拿接近满分的语义硬加分（badcase）。
    pos_intent = " ".join(prefer) if _W_MATCH_SEM > 0 else ""
    hard_intent = " ".join(must) if _W_MATCH_HARD_SEM > 0 else ""
    neg_intent = " ".join(attenuate) if _W_ATTEN_SEM > 0 else ""

    def _sims(item_mat: object, intent_vec: object) -> dict[str, float]:
        return {
            c.item_id: max(0.0, float(s))
            for c, s in zip(survivors, item_mat @ intent_vec, strict=True)  # type: ignore[operator]
        }

    if survivors and (pos_intent or hard_intent or neg_intent):
        try:
            tower = get_tower_client()
            # 候选标题编码（向量已 L2 归一，点积即 cosine）；正软/正硬/负软意图复用这份 item_mat。
            item_mat = await tower.encode_texts([_searchable(c) for c in survivors])
            if pos_intent:
                sem_match = _sims(item_mat, (await tower.encode_texts([pos_intent]))[0])
            if hard_intent:
                sem_hard = _sims(item_mat, (await tower.encode_texts([hard_intent]))[0])
            if neg_intent:
                sem_penalty = _sims(item_mat, (await tower.encode_texts([neg_intent]))[0])
        except Exception:
            logger.warning("item_picker 语义打分编码失败，降级纯关键词", exc_info=True)
            sem_match, sem_hard, sem_penalty = {}, {}, {}

    # scored 用 4 元组：亲和命中（matched_aff）单独一路带出去，理由措辞与「你要的 X」区分开。
    scored: list[tuple[float, list[str], list[str], ItemCandidate]] = []
    # 槽无关的 base 分（**不含便宜度**）：套装组合优选要按「槽内」重新归一便宜度——床垫
    # （$200 档）在全局归一里永远垫底、台灯（$20 档）永远满分，跨槽求和会被价格档位而非
    # 商品优劣主导（见 _bundle.combine_bundle）。
    base_scores: dict[str, float] = {}
    matched_map: dict[str, list[str]] = {}
    for c in survivors:
        text = _searchable(c)
        matched = [kw for kw in prefer if _hits(kw, text)]
        matched_must = [kw for kw in must if _hits(kw, text)]  # 正向硬命中（高权重）
        penalized = [kw for kw in attenuate if _hits(kw, text)]
        matched_aff = [kw for kw in affinity if _hits(kw, text)]  # 行为亲和命中（弱加分）
        # 数值规格三态计分：match 并入对应档的命中（照常加分、进理由），conflict 减分沉底
        # （must 档按 _W_SPEC_CONFLICT，prefer 档与软避讳同权重），unknown 不奖不罚。
        spec_conflicts = 0
        for v, u, d in must_specs:
            verdict = spec_verdict(v, u, text)
            if verdict == "match":
                matched_must.append(d)
            elif verdict == "conflict":
                spec_conflicts += 1
        for v, u, d in prefer_specs:
            verdict = spec_verdict(v, u, text)
            if verdict == "match":
                matched.append(d)
            elif verdict == "conflict":
                penalized.append(d)
        penalized += [d for v, u, d in attenuate_specs if spec_verdict(v, u, text) == "match"]
        # 评分缺失记中性 0.5（同 _cheapness 的处理），避免把「没人评过」误判成「评分极低」。
        rating_norm = (c.rating / 5.0) if c.rating is not None else 0.5
        base = (
            _W_PREF * len(matched)
            + _W_MATCH_HARD * len(matched_must)  # 正硬：关键词命中强加分（不淘汰不匹配的）
            + _W_AFFINITY * len(matched_aff)  # 行为亲和：从收藏推断的弱正向
            + _W_RATING * rating_norm
            + _W_MATCH_SEM * sem_match.get(c.item_id, 0.0)
            + _W_MATCH_HARD_SEM * sem_hard.get(c.item_id, 0.0)  # 正硬：语义强加分
            - _W_ATTEN * len(penalized)
            - _W_ATTEN_SEM * sem_penalty.get(c.item_id, 0.0)
            - _W_SPEC_CONFLICT * spec_conflicts  # 硬规格冲突：标题标称尺寸明确不合要求
        )
        # 相关性门（普通轮口径）：低于阈值判蹭词垃圾，**降权沉底不剔除**（容打分噪声——误杀
        # 一件真品的代价高于让垃圾多沉几名）。套装轮的「逐出槽位」在 combine_bundle 里另判。
        rr = rerank_scores.get(c.item_id) if rerank_on else None
        if rr is not None and rr < _RERANK_FLOOR:
            base -= _W_RERANK_MISS
        base_scores[c.item_id] = base
        # 入选理由用「软偏好 + 命中的正硬」一起展示（正硬更该出现在理由里）。
        matched_map[c.item_id] = matched_must + matched
        # 亲和命中单独一路带出去（第 3 位）：它在理由里的措辞必须和「你要的 X」区分开——用户没
        # 说过这个词，冒充成他说的就是编造事实。
        scored.append((base + _W_CHEAP * _cheapness(c), matched_must + matched, matched_aff, c))

    scored.sort(key=lambda t: t[0], reverse=True)

    # 套装轮（会话里登记了 ≥2 槽，见 app.tools._bundle）优先走跨槽组合优选：总预算内每槽
    # 选一件（essential 必选、optional 可砍），代替「全池排序取前 N」。不构成套装（槽 <2 /
    # 打标全失败只剩一组有货）返回 None，照常走普通精挑——失效方向安全：最差退化成现状行为。
    outcome = combine_bundle(
        survivors,
        base_scores,
        matched_map,
        budget_usd,
        w_cheap=_W_CHEAP,
        w_slot_pref=_W_PREF,
        slot_relevance=rerank_scores if rerank_on else None,
        relevance_floor=_SLOT_RERANK_FLOOR,
        w_relevance=_W_SLOT_RERANK,
    )
    picks: list[ItemCandidate] = []
    if outcome is not None:
        # 跨槽没有「本批最低价」这类可比统计（床垫和台灯比价没有意义），理由只写属性 /
        # 命中偏好 / 评分价格；「哪槽花钱哪槽省」的相对叙事由组合报告承担（bundle + summary 注入）。
        empty_stats = _BatchStats(None, None, None)
        for p in outcome.chosen:
            item = p.cand.model_copy()
            # 归槽结果回写到 slot 字段（**槽 id**）——盖章缺失、靠 keywords 兜底归槽的候选
            # （主循环补搜没传 slot 的那批）全靠这行把槽位带到收尾卡片，否则前端落「其他」组
            # （badcase 75aa84）。
            item.slot = p.slot.id
            # 套装理由不单列行为亲和（弱信号、组合叙事已够满，且避免把推断词冒充成用户明说的
            # 偏好）——传空 affinity 列表；亲和仍通过 base_scores 影响了组合选择。
            reason = _build_reason(item, p.matched, [], empty_stats)
            item.pick_reason = f"【{p.slot.name}】{reason}"
            item.pref_matched = bool(p.matched)
            picks.append(item)
    else:
        # 合适的（survivors）全部展示，但封顶 PICK_DISPLAY_CAP——模型传的 top_k 只当上界，再硬封
        # 一道。近重复合并：同价 + 标题几乎全同的变体（颜色/翻新/包装）只留分最高的一件，名额顺延
        # 给下一名——两件一模一样的商品各占一张卡，对用户零信息增量，对上下文是双倍 token。
        limit = min(max(1, top_k), PICK_DISPLAY_CAP)
        # 展示相对门（见 PICK_REL_SHOW_RATIO）：基准 = 池内最高 cross-encoder 相关分。仅普通轮 +
        # 真跑了相关性门时算；否则 None（退化回按分填满，不因缺信号误杀）。
        rel_floor: float | None = None
        if rerank_on and rerank_scores and PICK_REL_SHOW_RATIO > 0:
            rel_floor = max(rerank_scores.values()) * PICK_REL_SHOW_RATIO
        final: list[tuple[float, list[str], list[str], ItemCandidate]] = []
        dup_dropped: list[str] = []
        gate_dropped: list[str] = []
        for row in scored:
            if len(final) >= limit:
                break
            cand = row[3]
            if any(_near_duplicate(cand, kept[3]) for kept in final):
                dup_dropped.append(cand.item_id)
                continue
            # 相对门：品类相关分显著低于头部 → 宁缺毋滥不凑数。**final 为空时放行**——保底留最高
            # 综合分 1 件，不让展示门主动产空清单（空是空召回诚实路径的职责，不是这里）。
            if rel_floor is not None and final:
                rr = rerank_scores.get(cand.item_id)
                if rr is not None and rr < rel_floor:
                    gate_dropped.append(cand.item_id)
                    continue
            final.append(row)
        if dup_dropped:
            logger.info("item_picker 近重复合并：丢弃 %d 件（%s）", len(dup_dropped), dup_dropped)
        if gate_dropped:
            logger.info(
                "item_picker 展示相对门：%d 件品类相关分低于头部 %.0f%%，不凑数展示（%s）",
                len(gate_dropped),
                PICK_REL_SHOW_RATIO * 100,
                gate_dropped,
            )
        chosen = [c for _score, _matched, _aff, c in final]
        # 批内统计要在**定稿的这一批**上算（不是全部 survivors）：用户看到的是这几张卡，「本批
        # 最低价」说的就该是这几张里的最低——拿一个他看不见的更大集合算，标签会和眼前价格对不上。
        stats = _batch_stats(chosen)
        for (_score, matched, matched_aff, _c), src in zip(final, chosen, strict=True):
            item = src.model_copy()
            item.pick_reason = _build_reason(item, matched, matched_aff, stats)
            # pref_matched **只认显式偏好**，不含行为亲和：它喂给 shopping_summary 的 prompt 当
            # 「命中了用户偏好」的判据，而亲和是我们从收藏里推断的、用户从没说过。让推断冒充明说，
            # 收尾文案就会写出「按你的要求选了帆布款」——用户根本没提过帆布。
            item.pref_matched = bool(matched)  # 结构化判据，供 summary 的 prompt 用（见 schemas）
            picks.append(item)

    # 把入选理由回写登记表：shopping_summary 收尾改按 id hydrate 时，能直接取到 pick_reason
    # （卡片理由与 LLM 兜底都复用它），不必让模型把理由再重吐一遍。
    register_updates(picks)
    # 记下「本轮精选是哪几件」：收尾的 shopping_summary 缺省就用这批，模型不必把 id 逐个抄一遍
    # （抄的过程中它还会顺手再砍掉几件——筛选是这里的职责，不是收尾时凭印象再来一遍）。
    set_last_picks(picks)

    # 本轮 must_have 的池内命中件数——退回补搜闸（refine_backfill）的质量信号。两条口径都是刻意的：
    # ① 只统计**模型本轮传的** must_have、不并记忆的 must：记忆里的旧硬条件（如「纯棉」）往往整池
    #    全命中，一并计数会把「本轮新条件（刺绣）0 命中」的信号整个稀释掉。
    # ② 只走关键词路、不用语义分（sem_hard）豁免：同品类候选对任何购物意图的 cosine 地板都不低，
    #    绝对阈值难标定；而两个方向的代价不对称——误报 0（有货但标题没写词）只多付一次补搜合流，
    #    结果不会变差；漏报（池里真没货却照常推进收尾）是「挑不出→不许再搜→承认失败」的死路。
    model_must, model_must_specs = _split_specs(normalize_terms(_merge_terms(must_have)))
    must_have_hits: int | None = None
    if model_must or model_must_specs:
        must_have_hits = sum(
            1
            for c in survivors
            if any(_hits(kw, _searchable(c)) for kw in model_must)
            or any(spec_verdict(v, u, _searchable(c)) == "match" for v, u, _d in model_must_specs)
        )

    out = ItemPickerOutput(
        picks=picks,
        excluded=excluded,
        over_budget=over_budget,
        must_have_hits=must_have_hits,
        oncat_count=oncat_count,
        offcat_count=offcat_count,
        bundle=outcome.report if outcome is not None else None,
    )
    # 思考结果摘要：精选了哪几件、各自入选理由（供前端展开看这一步「挑出了什么、为什么」）。
    if outcome is not None:
        # 套装轮直接给分配表：哪槽花了多少、砍了谁、缺了谁——比逐件理由更是用户要的答案。
        picker_result = render_allocation(outcome.report)
    elif picks:
        pick_lines = [
            f"· {c.title}" + (f" —— {c.pick_reason}" if c.pick_reason else "") for c in picks[:5]
        ]
        picker_result = f"精选 {len(picks)} 件：\n" + "\n".join(pick_lines)
    else:
        picker_result = "无符合条件的候选（可能被排除词 / 预算筛掉）"
    # 先出货、后出文案：清单是**哪几件**在这一刻就已经定了，但用户还得等主 loop 收尾那轮解码 +
    # shopping_summary 内部生成才看得见（好几秒）。把卡片现在就推出去，文案随后由 summary_delta
    # 逐字补上。收尾的 task_result 会用定稿那批原样覆盖，两者同构、顺序一致，不会跳动。
    # 商品卡的字段全是**确定性的**（价格/图/链接/理由都在 picks 里），提前推不存在「先给一版、
    # 收尾又换一版」的风险——真正还没定的只有那段文案。
    if picks:
        await monitor.report_items_preview([_preview_item(c) for c in picks])
    await monitor.report_tool_end(
        "item_picker",
        picked=len(picks),
        excluded=len(excluded),
        auto_excluded=len(mem.memory_exclude),  # 本轮域内、来自长期硬黑名单的排除词数（可观测）
        auto_attenuated=len(mem.memory_penalty),  # 本轮域内、来自长期软避讳的减分词数
        semantic=bool(sem_match or sem_hard or sem_penalty),  # 走了语义打分（论文式6/8）
        # 整池 0 命中的硬排除词（诚实性标注：这些词本轮没挡下任何商品，别让用户以为生效了）
        **({"no_effect_excludes": no_effect_excludes} if no_effect_excludes else {}),
        result=picker_result,
    )
    # 给 harness 的诊断走结构化侧信道（middleware 消费）；模型可见文本里带不带、怎么截断
    # 都不再影响这些信号——见 app/tools/_diagnostics.py 的受众分离说明。
    report_diagnostics(
        "item_picker",
        {
            "picks": len(picks),
            "must_have_hits": must_have_hits,
            "oncat_count": oncat_count,
            "offcat_count": offcat_count,
            # 硬淘汰归因计数（补搜闸的杀池信号）：「池子为什么空」决定补搜指路指哪条——
            # 超预算杀的 → 带 price_usd_max 重搜；排除词杀的 → 换开检索词。
            "excluded_count": len(excluded),
            "over_budget_count": len(over_budget),
            # 品类门锚分歧（planner category 与用户原文词面对不上 → 门本轮 fail-open）。
            # 暂无 harness 消费方，先进侧信道供评测/日志盯复发率——加字段=dict 加一个 key。
            "anchor_conflict": anchor_conflict,
        },
    )
    return out


def _preview_item(c: ItemCandidate) -> dict[str, object]:
    """一件候选压成商品卡预览（字段与收尾 ``SummaryItem`` 同构，前端复用同一个 ProductCards）。

    ``reason`` 用 picker 算好的确定性理由：收尾时前几件会被 LLM 重写成叙事句，但那是**文风**的
    差别，不是内容的差别——预览这一刻不该为了等一句更好听的话把整批卡片压着不发。
    价格两个字段照实透传、互不兜底（同 SummaryItem：没跑 shipping_calc 就是没有到手价）。
    """
    return {
        "item_id": c.item_id,
        "platform": c.platform,
        "title": c.title,
        "price_usd": c.price_usd,
        "landed_usd": c.landed_usd,
        "reason": c.pick_reason,
        "image_url": c.image_url,
        "url": c.url,
        # 套装轮非空：前端预览阶段就能按槽分组（与收尾 SummaryItem 同构）。内部盖章是槽 id，
        # 出前端这一步映射回展示名——前端与旧会话数据全程只见名字。
        "slot": slot_display(c.slot),
    }


def _merge_terms(*groups: list[str] | None) -> list[str]:
    """把模型本轮传的词与装配出的记忆词并起来：小写、去空、保序去重（词是拿去匹标题的）。"""
    out: list[str] = []
    seen: set[str] = set()
    for g in groups:
        for t in g or []:
            low = t.strip().lower()
            if low and low not in seen:
                seen.add(low)
                out.append(low)
    return out


class _BatchStats(NamedTuple):
    """一批 picks 的横向统计——「本批最低价 / 评分最高」这类相对信息的来源。

    **相对信息是白捡的、且比任何形容词都硬**：picker 手里本来就攥着全批候选，算一次最小值 / 最大值
    /均价就能告诉用户「这件是这批里最便宜的」——这正是他要的判断依据，而且不需要模型写一个字。
    """

    min_price: float | None
    avg_price: float | None
    max_rating: float | None


def _batch_stats(cands: list[ItemCandidate]) -> _BatchStats:
    prices = [p for p in (_effective_price(c) for c in cands) if p is not None]
    ratings = [c.rating for c in cands if c.rating is not None]
    return _BatchStats(
        min_price=min(prices) if prices else None,
        avg_price=(sum(prices) / len(prices)) if prices else None,
        max_rating=max(ratings) if ratings else None,
    )


def _build_reason(
    c: ItemCandidate, matched: list[str], affinity: list[str], stats: _BatchStats
) -> str:
    """拼一句「为什么选它」——全部来自真实字段，零 LLM。

    素材五类：① 这批里的相对位置（最便宜 / 评分最高 / 价格偏低）；② 标题属性（7 件套、尼龙、
    防水）；③ 命中的偏好（中文展示：命中的往往是英文词 canvas / durable，别让用户自己翻译）；
    ④ 命中的行为亲和（措辞与 ③ **必须分开**，见下）；⑤ 评分 + 价格。

    **写成人话，不是把打分特征倒给用户**：这句话直接印在商品卡上，也喂给 shopping_summary 的模型。
    早先它是分号串起来的字段罗列（「低于本批均价；6 件套；评分 4.7；货价 $19.99」），一眼就是日志——
    而且模型会把「本批」「货价」这些内部口径词原样抄进面向用户的文案里。所以：前三类拼成一个主句，
    评分 / 价格降级成句尾的补充；「批」「货价」这类只有工程师懂的词一律不出现。
    """
    head: list[str] = []

    price = _effective_price(c)
    if price is not None and stats.min_price is not None and price <= stats.min_price:
        head.append("这几件里最便宜的")
    elif price is not None and stats.avg_price is not None and price < stats.avg_price:
        head.append("这几件里价格偏低的一款")
    if c.rating is not None and stats.max_rating is not None and c.rating >= stats.max_rating:
        head.append("评分也是最高的" if head else "这几件里评分最高的")

    attrs = extract_attrs(c.title)
    if attrs:
        head.append("、".join(attrs))

    # 命中的偏好：① 中英命中同一条 → 只显示一次；② 已经作为标题属性说过的词不再重复一遍
    # （「帆布，正好是你要的帆布」）；③ 用引号括住——命中的常是形容词（耐用 / durable），
    # 裸接在「你要的」后面读不通。
    shown = [t for t in dict.fromkeys(display_term(m) for m in matched) if t not in attrs]
    if shown:
        head.append(f"正是你要的「{'、'.join(shown)}」")

    # 行为亲和命中：**归因必须说清是从收藏推断的**，不能混进上面那句「正是你要的」——用户压根没说过
    # 这个词，把推断说成他的要求就是编造事实（P0 诚实红线）。写明出处还有第二个作用：记忆最危险的
    # 失败是静默生效，用户看到排序变了却不知道为什么、也就无从纠正。这句话本身就是纠错入口——他一看
    # 「你收藏过的都是帆布」，就知道该去收藏夹里删掉那几件不作数的。
    # **不像 shown 那样剔除已在 attrs 里的词**：亲和命中的前提就是标题含这个词，attrs 几乎必然已经
    # 说过它一遍（「帆布」），照搬那条去重规则会把归因整个过滤成空——留下的只有一个被悄悄改过的排序。
    # 措辞上重一次「帆布」是小代价，说不清「凭什么给它加分」才是大问题。
    aff_shown = [t for t in dict.fromkeys(display_term(a) for a in affinity) if t not in shown]
    if aff_shown:
        head.append(f"和你此前收藏的是同一路（{'、'.join(aff_shown)}）")

    tail: list[str] = []
    if c.rating is not None:
        # 只报评分，**不报评论数**：本仓库数据集没有 reviews_count 这一列，恒为 0（见
        # _candidates._MODEL_NOISE_FIELDS）。拼一句「4.9（0 评）」等于两头误导：模型会当成零评价的
        # 高分货去编「口碑极佳」，用户看到的也是自相矛盾的一行。数据里真有评论数的那天再放回来。
        tail.append(f"评分 {c.rating}")
    if price is not None:
        label = "到手价" if c.landed_usd is not None else "售价"
        tail.append(f"{label} ${price:.2f}")

    if head:
        return "，".join(head) + "。" + ("，".join(tail) + "。" if tail else "")
    if tail:
        return "，".join(tail) + "，可以作为备选。"
    return "各方面表现都算均衡，可以作为备选。"
