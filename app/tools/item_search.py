"""item_search —— 单平台商品检索（Qdrant dense 召回 + filter，语义+个性化双通道融合）。

主链路的「检索」一环。封装 :mod:`app.recall` 的编码 + Qdrant dense 召回：把用户这次搜索意图
（query，已并入本轮域内的 like 偏好词）编码成 dense「请求向量」做召回，返回 top_k 归一候选。
精确命中/硬约束走 Qdrant payload filter（非 sparse 打分，见 `docs/plans/召回引擎选型思路.md` §4）；
filter 维度：platform + price_usd_max + min_rating（Qdrant Range）+ brand_exclude（后置过滤）。

精排取舍（对齐 refdoc）：refdoc 的 item_search 只做 dense 召回 + 双通道本地融合，cross-encoder
精排是 CategoryInsight/RAG 链路的事（refdocs/11 vs 13-1）。本工具据此**不再做 cross-encoder
精排**——候选的二次质量把关交给下游 item_picker（按用户偏好精挑）。这样每次检索少一次 rerank
网络往返，跨平台 fork 放大时收益明显。

**单平台**：一次只搜一个平台。跨平台 / 多槽位并行检索由主 loop 同轮多发本工具（一平台或一槽
一条、由框架批并发）来完成，本工具不自己循环多平台——把「要不要并行」的决策权留给主 loop。

**个性化改走「拼进检索词」，不再走 user 塔向量画像**（Mmem）：本工具把用户本轮域内的 like
偏好原子词并进 query 文本再编码。原来那条路（把所有 like 加权平均成一个 user 向量、按 β 融进
请求向量）的问题是**不可观测**——召回结果怪了，你既无法归因、无法调试，也没法向用户解释；而且
不裁剪时，几十条偏好平均出的「口味质心」基本就是噪声。拼进检索词后，个性化出现在工具上报、
前端思考过程和日志里，看得见、调得动。

dislike 偏好**绝不进**这条通路：embedding 对否定算子编码极弱、对主题词极强，「不要皮革」拼进
query 等于把请求向量往「皮革」那片推，召回**更多**皮革。它另走 item_picker 的确定性词匹配。
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Literal

from pydantic import BaseModel

from app.agent.platform_scope import resolve_search_platforms
from app.api import monitor
from app.api.context import get_user_id
from app.harness.retrieval_budget import note_filtered_probe, note_item_search
from app.memory.assemble import assemble
from app.recall import get_recall_client, get_tower_client
from app.recall.schemas import RecallCandidate
from app.recall.search_cache import cached_recall
from app.tools._args import StrListArg
from app.tools._bundle import note_slot_searched, register_slot
from app.tools._candidates import compact_candidates, enrich, register
from app.tools._diagnostics import report_diagnostics
from app.tools._shell import tool
from app.tools.schemas import FilteredOutItem, ItemCandidate
from app.utils.env import env_float, env_int
from app.utils.terms import normalize_terms, term_hits

# 以下 6 个参数由后台管理页面热更新（见 app/config/registry.py）：值住在模块全局、由 _load_params()
# 从 env 求值，改完 env 回调它即重新生效。**故意保持模块级常量形态**而非改成 param() 式函数调用——
# 既有测试大量 monkeypatch.setattr(mod, "RELEVANCE_FLOOR", ...) 构造场景，改成函数会让那些 patch
# 设到没人读的属性上、测试静默失去约束力。下方类型声明让静态检查知道它们存在（赋值在函数里）。
DEFAULT_TOP_K: int
MAX_TOP_K: int
SINGLE_PLATFORM_POOL_K: int
RENDER_CAP: int
RELEVANCE_FLOOR: float
RETRY_MIN_HITS: int
EXCLUDE_FETCH_BUFFER: int
PROBE_LIMIT: int

# 回给模型的 filtered_out 条数上限。它是**证据**不是候选池：3 条不足以让模型判断「差得多还是差
# 一点」（价格分布看不出来），10 条纯烧 token 且这批货本就不该被推荐。5 条够说明问题。
FILTERED_OUT_CAP = 5


def _load_params() -> None:
    """从 env 求值本模块的可调参数（导入时跑一次；后台改参数后由覆盖层回调）。

    赋值顺序即源码顺序，故 MAX_TOP_K 能安全地拿 DEFAULT_TOP_K 当默认值。
    """
    global DEFAULT_TOP_K, MAX_TOP_K, SINGLE_PLATFORM_POOL_K, RENDER_CAP
    global RELEVANCE_FLOOR, RETRY_MIN_HITS, EXCLUDE_FETCH_BUFFER
    global PROBE_LIMIT

    # 召回条数默认值：跨平台 fork 时**每个**子 Agent 都要吃一份这么大的候选 JSON——20 条 ≈ 3.7K token
    # fresh（缓存必 miss，因为它是新内容），5 个平台就是 ~18K。实测一条 query 最终只出 4~5 件
    # 商品，20 条里过半是子 Agent 自己都会剔掉的跑题货（「餐具/背包/桌子」），故收到 10：候选池
    # 仍是 5 平台 × 10 = 50 条进 item_picker，够精挑。
    DEFAULT_TOP_K = env_int("ITEM_SEARCH_TOP_K", 10)

    # 召回 top_k 上界（**硬封顶，不是建议**）：默认与 DEFAULT_TOP_K 同值，即模型传多少都按 10 收。
    #
    # 为什么是硬的：实测模型会**无视默认值自己传 top_k=20**（prompt 里那句给 item_picker 写的
    # 「top_k 缺省即 20」被它套到了本工具头上；且换成没有这句话的 ultra 变体后它照样传 20）。默认值
    # 是「建议」，建议拦不住模型——照本仓库一贯口径（边界用机制兜、不靠 prompt，见 fork 深度闸 /
    # item_picker 的 PICK_DISPLAY_CAP），这里改成机制封顶：模型传的 top_k 只当**上界的候选**，
    # 真正生效的是 min(top_k, MAX_TOP_K)。
    #
    # 真需要更大候选池时调 env（``ITEM_SEARCH_MAX_TOP_K``），别指望改 prompt 说服模型。
    MAX_TOP_K = env_int("ITEM_SEARCH_MAX_TOP_K", DEFAULT_TOP_K)

    # 单平台召回池：跨平台靠「平台数 × top_k」堆出大候选池（5×10=50）供 item_picker 精排；单平台没有
    # 这个乘数——若同样只召 10 条，池子≈展示上限（彼时 CAP=8），精排几乎无筛除空间（10 挑 8
    # 只淘汰 2 件，跨品类蹭词货照样露脸）。故单平台把召回池单独放大，让 cross-encoder 精排有料可挑。
    # **只放大「进登记表供 picker 精排」的池子，不放大「进模型上下文」的渲染量**（见 RENDER_CAP）。
    SINGLE_PLATFORM_POOL_K = env_int("ITEM_SEARCH_SINGLE_POOL_K", 30)

    # 渲染给模型上下文的候选条数上限（与「进登记表的召回池」解耦）：召回池可以大（30 供 picker
    # 精排），但模型自己不精挑——精挑是 item_picker 的职责。让模型上下文吃满整池候选 JSON 是纯烧
    # token（30 条 ≈5.5K，且这条 ToolMessage 后续每轮都要重读）。故 register 全池、只渲染头部 +
    # 一句「其余已入池待精挑」。同 item_picker PICK_DISPLAY_CAP 的「登记全量、渲染收敛」思路。
    # **默认 5 而非 10**：这几条的作用不是给模型精挑（那是 item_picker），是给主 loop Observe「本轮
    # 检索质量」——判断要不要换词 / 补搜、有没有品类漂移。头部 5 条足够形成这个判断：1 条方差大
    # （召回是向量近邻序、非质量序，头部恰好对后面全歪会误导）、0 条会在计数信号撒谎时失明
    # （category 字段脏，见 phone→配件 badcase，只有看到标题模型才可能察觉）。再多是保守浪费。
    # 未标定初值，可经 env 调。
    RENDER_CAP = env_int("ITEM_SEARCH_RENDER_CAP", 5)

    # 相关性下限（余弦相似度）：dense 召回永远返回 top-k 最近邻，**不管多远**——库里没货时也吐一堆
    # 「最近的垃圾」，total_recall=20 假装召回满满。floor 滤掉低于阈值的召回，让 total_recall 反映
    # **相关**召回数（纯垃圾如实报 0 → 触发 web_search 兜底 / 空召回硬路径）。
    # ⚠️ **诚实标注实测局限**：BGE-M3 在本千级杂货库里给**任何真实英文 query 都打 ≥0.48**
    # （连「挖掘机/处方药/活体金鱼」这类库里根本没有的也 0.48-0.53），只有纯乱码 ≤0.40。而「库稀缺但
    # 沾边」0.49-0.60、「命中良好」0.56-0.70 —— 三段严重重叠，**单一绝对阈值无法把「库里没货」和
    # 「有但一般」分开**。故 floor=0.45 实际只挡**乱码级**无关，**挡不住「品类缺货」**（数据稀疏
    # 问题，靠 prompt 的「不编造 / 没货就如实说」诚实兜，不是阈值能治）。可经 env 调。
    RELEVANCE_FLOOR = env_float("RELEVANCE_FLOOR", 0.45)

    # 「召回够不够用」的条数判据：少于这么多条就自动摘掉评分门槛重搜一次（见下面的放宽段）。
    # 取 3 而非 0：只召回一两条时模型照样会自己发起一轮重搜，那一轮 Think 的解码开销正是要省掉的。
    RETRY_MIN_HITS = env_int("ITEM_SEARCH_RETRY_MIN_HITS", 3)

    # 记忆硬排除生效时，多召这么多条补偿被杀的名额。硬排除（blocking 黑名单 + 本轮硬 dislike）在
    # **召回阶段**就过滤（不是等 item_picker 事后杀），top_k 硬封 10 的候选池才不会对黑名单用户
    # 永远残缺——「不要皮革」的用户搜公文包，10 条里 8 条皮的，杀完只剩 2 条还无处补货。
    # Qdrant 多取 10 条近邻的成本可忽略，而少取的代价是残缺候选池。
    EXCLUDE_FETCH_BUFFER = env_int("ITEM_SEARCH_EXCLUDE_BUFFER", 10)

    # 探测召回条数（filtered_out 用）：命中不足且**确有硬过滤条件**时，再打一次不带 price /
    # rating / 记忆排除的召回，与正式结果做差，回答「库里到底是没货，还是有货但被挡了」。
    # 只多一次 Qdrant 近邻查询（请求向量复用，不重编码，实测 ~10ms 级），不额外调模型。
    # 设 0 关闭探测（返回体里就不再有 filtered_out）。
    PROBE_LIMIT = env_int("ITEM_SEARCH_PROBE_LIMIT", 8)


_load_params()

# 平台枚举：用 Literal 而非 str，从 schema 层挡住模型生成 Amazon/AMAZON/amzn 等等价但不规范的
# 串（refdocs/11 §2.2）。全集对齐 utils.clean.PLATFORMS —— **eBay 不在其中**：它的 CSV 卖家名近全
# 掩码、描述列全空，清洗阶段就整体剔除了，召回库里一条 eBay 商品都没有，列进来只会诱导模型去搜一个
# 必然空召回的平台。"all" 跨**本次启用的**平台合流（见 agent.platform_scope）。
# 深层防御另有 qdrant_store.search 的 strip().lower() 归一兜底（测试/直连调用方）。
Platform = Literal["all", "amazon", "walmart", "shein", "lazada", "shopee"]


def _searchable(rc: RecallCandidate) -> str:
    """召回候选的可匹配文本（与 item_picker 的 ``_searchable`` 同口径：标题+品牌+品类，小写）。"""
    return f"{rc.title} {rc.brand} {rc.category}".lower()


def _apply_filters(
    recalled: list[RecallCandidate],
    *,
    floor: float,
    brand_exclude: list[str] | None,
    exclude_terms: list[str] | None = None,
) -> tuple[list[RecallCandidate], int]:
    """按语义门槛 + 用户硬约束过滤召回结果，返回 (候选, 记忆排除掉的条数)。

    抽成函数是为了能用**不同的 floor 复跑**：召回不足时降档重过滤，不必重新打一次检索。

    ``exclude_terms`` 是记忆装配出的硬排除词（blocking 黑名单 + 本轮硬 dislike，已归一），在
    **这里**就过滤而不是等 item_picker：total_recall 必须反映「用户真能要的召回数」——此前它记
    的是排除前的条数，web_search 兜底闸（仅在召回全空时放行）被这个假数字拦死，候选全被黑名单
    杀光时用户拿到空清单还没有任何兜底。命中判定与 picker 同一套 ``term_hits``（词边界 + 否定
    修饰），"vegan leather" 不会被「不要皮革」误杀。
    """
    relevant = [rc for rc in recalled if rc.score >= floor]
    if brand_exclude:
        excl = {b.lower() for b in brand_exclude}
        relevant = [rc for rc in relevant if rc.brand.lower() not in excl]
    memory_dropped = 0
    if exclude_terms:
        before = len(relevant)
        relevant = [
            rc for rc in relevant if not any(term_hits(kw, _searchable(rc)) for kw in exclude_terms)
        ]
        memory_dropped = before - len(relevant)
    return relevant, memory_dropped


def _blocked_reason(
    rc: RecallCandidate,
    *,
    price_usd_max: float | None,
    min_rating: float | None,
    brand_exclude: list[str] | None,
    exclude_terms: list[str] | None,
) -> tuple[str, bool] | None:
    """这条探测候选是被哪个硬条件挡下的？返回 (人话原因, 是否「只差预算」)；不是被挡的返回 None。

    **判定顺序是刻意的**：先结构性排除（品牌黑名单 / 排除词 / 评分），最后才判价格。一条既踩
    排除词又超预算的货，若报成「超预算」，会让上游得出「放宽预算就能买到」的错结论——放宽了它
    照样被排除词杀。所以第二个返回值（``price_only``）只在**其余条件都不沾**时才为真，
    :func:`app.harness.retrieval_budget.budget_relax_due` 的「只差预算」判据全靠它干净。

    ``price_usd`` 缺失（历史索引里出现过，见 price-usd-filter-empty-bug）不判超预算：拿不到价格
    就不敢说它超没超，宁可不报——报错原因比不报更坏。
    """
    if brand_exclude and rc.brand and rc.brand.lower() in {b.lower() for b in brand_exclude}:
        return f"品牌 {rc.brand} 在你的排除清单里", False
    if exclude_terms:
        hit = next((kw for kw in exclude_terms if term_hits(kw, _searchable(rc))), "")
        if hit:
            return f"命中你的排除偏好「{hit}」", False
    if min_rating is not None and rc.rating is not None and rc.rating < min_rating:
        return f"评分 {rc.rating} 低于门槛 {min_rating}", False
    if price_usd_max is not None and rc.price_usd is not None and rc.price_usd > price_usd_max:
        # 预算按原样显示（整数不拖小数尾，小数不四舍五入）：这句话模型会照抄给用户，把
        # $1.5 的预算写成 $2 就是当着用户的面改他的硬约束。
        budget = (
            f"{price_usd_max:.0f}" if float(price_usd_max).is_integer() else f"{price_usd_max:.2f}"
        )
        return f"超预算（${rc.price_usd:.2f} > ${budget}）", True
    return None


def _probe_filtered_out(
    probed: list[RecallCandidate],
    kept_ids: set[str],
    *,
    price_usd_max: float | None,
    min_rating: float | None,
    brand_exclude: list[str] | None,
    exclude_terms: list[str] | None,
) -> tuple[list[FilteredOutItem], int, int]:
    """探测召回 → 差集 → 被挡样本，返回 (样本 ≤CAP, 只差预算的条数, 其余原因的条数)。

    相关度红线（``RELEVANCE_FLOOR``）在探测里照旧生效——挡不住的乱码级无关货本来就不该被当成
    「库里有」的证据。已进候选池的、以及说不清为什么没进池的（名次外等），都不算被挡。
    """
    items: list[FilteredOutItem] = []
    price_blocked = 0
    other_blocked = 0
    for rc in probed:
        if rc.item_id in kept_ids or rc.score < RELEVANCE_FLOOR:
            continue
        verdict = _blocked_reason(
            rc,
            price_usd_max=price_usd_max,
            min_rating=min_rating,
            brand_exclude=brand_exclude,
            exclude_terms=exclude_terms,
        )
        if verdict is None:
            continue
        reason, price_only = verdict
        if price_only:
            price_blocked += 1
        else:
            other_blocked += 1
        if len(items) < FILTERED_OUT_CAP:
            items.append(
                FilteredOutItem(
                    item_id=rc.item_id, title=rc.title, price_usd=rc.price_usd, reason=reason
                )
            )
    return items, price_blocked, other_blocked


class ItemSearchOutput(BaseModel):
    """item_search 的结构化返回。"""

    platform: str
    candidates: list[ItemCandidate]
    total_recall: int  # 本次实际召回条数
    truncated: bool  # 是否因 top_k 上限截断（下游可据此决定要不要换更窄的 query）
    # 本次是否触发了「召回不足→自动放宽门槛重试」。为 True 时这批候选可能不满足 min_rating
    # （**只有评分门槛被放宽**；语义相关度 RELEVANCE_FLOOR 是诚实红线，任何情况都不降档），
    # 模型据此知道「已经放宽过了，别再自己重搜一轮」。预算 / 品牌黑名单等用户硬约束同样没放宽。
    relaxed: bool = False
    # 命中用户硬排除偏好（blocking 黑名单 + 本轮硬 dislike）被过滤掉的召回条数。>0 时模型据此
    # 知道「召回少不是库里没货，是用户自己的偏好筛的」——零候选时该如实说「符合的都被你的排除
    # 偏好筛掉了」，而不是「库里没有」。
    memory_excluded: int = 0
    # 本次召回中**登记表里已有**的 item_id（重试检索 / 跨轮读回时的重复召回）。渲染时这部分
    # 折叠成 id 列表、不再全文回显——完整字段模型早看过一遍，重复回显纯烧 token（相机 bad case
    # 实测第 4 次检索 10 条里 5 条是重复全文）。
    known_ids: list[str] = []
    # 库里有、但被本次硬条件挡在池外的样本（≤5 条，见 FilteredOutItem）。空列表＝没探测或
    # 确实没被挡的货。**这些不是候选**，不能进清单/商品卡，只作「不是没货，是被 X 挡了」的证据。
    filtered_out: list[FilteredOutItem] = []
    # 本次召回实际走了哪条路（可观测）：基线 "dense"，按实际生效的过滤 / 放宽 / 探测追加后缀。
    # 只在偏离基线时回给模型（见 __str__），常态不占 token。
    recall_strategy: str = "dense"

    def __str__(self) -> str:
        """喂给模型的紧凑投影（见 :func:`compact_candidates`：只留决策要用的字段）。

        单平台检索时每条候选里的 ``platform`` 是纯冗余——顶层已经写了一次平台名，20 条再各重复
        一遍只是烧 token。``platform="all"`` 合流时必须留：那时模型得知道每件货来自哪个平台。
        """
        single_platform = self.platform != "all"
        known = set(self.known_ids)
        fresh = [c for c in self.candidates if c.item_id not in known]
        # 渲染收敛（与「进登记表的召回池」解耦）：召回池全量已进登记表供 item_picker 精排，
        # 模型上下文只看头部 RENDER_CAP 条——模型不精挑，灌满整池纯烧 token。其余以
        # pooled_for_pick 计数告知，让模型知道「池里还有货、已交给精挑」，不必自己在这里筛。
        shown = fresh[:RENDER_CAP]
        pooled = len(fresh) - len(shown)
        return json.dumps(
            {
                "platform": self.platform,
                "total_recall": self.total_recall,
                "truncated": self.truncated,
                # 只在真放宽过时才带这个键：没放宽时是默认状态，写进去纯属每轮多烧 token。
                **({"relaxed": True} if self.relaxed else {}),
                # 同理：与基线一致时不写。
                **(
                    {"recall_strategy": self.recall_strategy}
                    if self.recall_strategy != "dense"
                    else {}
                ),
                # 「库里有但被挡了」的证据：模型据此说清是没货还是超预算，别把后者说成前者。
                **(
                    {"filtered_out": [f.model_dump(exclude_none=True) for f in self.filtered_out]}
                    if self.filtered_out
                    else {}
                ),
                # 同理：只在真有排除时才带（常态是 0，写进去纯烧 token）。
                **({"memory_excluded": self.memory_excluded} if self.memory_excluded else {}),
                "candidates": compact_candidates(
                    shown, drop={"platform"} if single_platform else ()
                ),
                # 召回池比展示多出来的那部分：已入登记表待 item_picker 精排，此处只报数不展开。
                **({"pooled_for_pick": pooled} if pooled else {}),
                # 与已入池候选重复的部分只报 id：全量字段模型已看过，可直接按 id 复用。
                **(
                    {"already_in_pool": [c.item_id for c in self.candidates if c.item_id in known]}
                    if known
                    else {}
                ),
            },
            ensure_ascii=False,
            default=str,
        )


@tool
async def item_search(
    query: str,
    platform: Platform = "all",
    top_k: int = DEFAULT_TOP_K,
    price_usd_max: float | None = None,
    min_rating: float | None = None,
    brand_exclude: StrListArg | None = None,
    slot: str = "",
) -> ItemSearchOutput:
    """在单个平台检索商品（dense 召回，长期偏好与硬排除已由系统并入）；跨平台同轮多发、一平台一条。
    参数：query 用品类核心词（场景/人群词交给 item_picker 的 prefer）；platform 见
    <enabled_platforms>；price_usd_max / min_rating / brand_exclude 召回期过滤；top_k 不用传；
    slot 只在多槽位轮传槽名（一槽一条、同轮发）。
    返回 filtered_out = 库里有但被条件挡住（不是候选，如实说被哪个条件挡的）。
    """
    # 个性化**不再由系统悄悄拼词**（M4）：长期偏好每轮注入给模型看，由模型自己决定要不要写进
    # `query` / `brand_exclude` / `price_usd_max`。个性化因此出现在**工具入参**里——上报、前端
    # 思考过程、日志三处都看得见，且能归因到是模型哪一步加的。原来那条「系统把 like 词拼进
    # query」的腿，和注入给模型的文本是两份来源，模型转述一遍就会重复拼，谁也说不清最终检索词
    # 是怎么来的。
    effective_query = query
    mem = await assemble(get_user_id() or "")
    # 会话级 P_t 的「不要 X」在**召回阶段**就生效（不等 item_picker 事后杀）：见 _apply_filters。
    # 归一与 picker 同口径（中文词补英文变体），命中判定同一套 term_hits。
    mem_exclude = normalize_terms(mem.exclude)
    # 平台收口（机制层，不靠模型自觉）：把入参落到本轮「启用平台」集合内——"all" 只等于全部启用平台，
    # 模型点名一个未启用的平台则回落到启用集合。用户没勾的平台不该被搜（见 agent.platform_scope）。
    search_platforms = resolve_search_platforms(platform)
    platform = search_platforms[0] if len(search_platforms) == 1 else "all"  # type: ignore[assignment]
    await monitor.report_tool_start(
        "item_search",
        query=effective_query,  # 上报**实际**用于检索的词——检索词必须看得见
        platform=platform,
        top_k=top_k,
        price_usd_max=price_usd_max,
        min_rating=min_rating,
        brand_exclude=brand_exclude,
    )

    # 单平台放大召回池（供 picker 精排，不进上下文——渲染仍由 __str__ 的 RENDER_CAP 收敛）；跨平台
    # 每平台仍按上限收，避免 5×30=150 灌爆登记表。判据是「实际要搜的平台数」——不管模型传的是某个
    # 具体平台，还是 "all" 但本轮只启用了一个平台，只要落地只搜一个，就放大它的池子。
    if len(search_platforms) == 1:
        capped_k = SINGLE_PLATFORM_POOL_K
    else:
        capped_k = max(1, min(top_k, MAX_TOP_K))
    # 有硬排除词时多召一个 buffer 补偿被杀的名额，别让黑名单用户永远拿到残缺候选池。
    fetch_k = capped_k + 1 + (EXCLUDE_FETCH_BUFFER if mem_exclude else 0)
    tower = get_tower_client()
    recall = get_recall_client()

    # 编码做成惰性（阶段 3）：下面三次召回（主 / 放宽重试 / 探测）都经两级缓存，全命中时
    # 一次 embedding 往返都不该发——而它是这条链路上更贵的那一跳。真回源时只编码一次。
    vec_box: list[Any] = []

    async def _vec() -> Any:
        if not vec_box:
            vec_box.append(await tower.encode_query(effective_query))
        return vec_box[0]

    async def _recall(
        top_k: int, *, price_max: float | None, rating_min: float | None
    ) -> list[RecallCandidate]:
        async def _fetch() -> list[RecallCandidate]:
            return await asyncio.to_thread(
                recall.search,
                await _vec(),
                top_k,
                search_platforms,
                price_usd_max=price_max,
                min_rating=rating_min,
            )

        return await cached_recall(
            effective_query,
            top_k,
            search_platforms,
            price_usd_max=price_max,
            min_rating=rating_min,
            fetch=_fetch,
        )

    recalled = await _recall(fetch_k, price_max=price_usd_max, rating_min=min_rating)
    relevant, memory_dropped = _apply_filters(
        recalled,
        floor=RELEVANCE_FLOOR,
        brand_exclude=brand_exclude,
        exclude_terms=mem_exclude,
    )

    # 召回太贫瘠时**工具自己放宽重试**，不把这个判断甩给模型——让模型「发现召回不足→决定松一档→
    # 再调一次 item_search」要多烧一整轮 Think（十几秒解码），而重搜只要零点几秒。判据是条数规则，
    # 不需要语义理解，没道理花一次 LLM 决策去做。
    #
    # **能放宽的只有 min_rating 这一个软偏好**，其余一概不动，因为它们各自守着一条线：
    #   · price_usd_max（预算）/ brand_exclude（「不要 X 牌」）—— 用户说死的硬约束，松了就是拿人家
    #     明确不要的东西充数；
    #   · RELEVANCE_FLOOR —— 「宁可如实说没找到，也不给跑题货」的 P0 诚实红线（见 prompt 的空召回
    #     硬路径）。它卡光了恰恰说明库里真没有相关商品，此时降档只会把跑题商品捞回来凑数，是幻觉的
    #     温床——空手而归远好过给一堆不沾边的货。
    relaxed = False
    if len(relevant) < RETRY_MIN_HITS and min_rating is not None:
        # 评分门槛是 Qdrant 召回阶段的 filter，只能重搜一次才能摘掉（预算照旧硬卡）。
        widened = await _recall(fetch_k, price_max=price_usd_max, rating_min=None)
        candidates_wo_rating, mem_dropped_wo = _apply_filters(
            widened,
            floor=RELEVANCE_FLOOR,  # 相关度红线照旧
            brand_exclude=brand_exclude,
            exclude_terms=mem_exclude,  # 记忆硬排除是用户授权的硬约束，放宽时同样不动
        )
        # 真捞到更多才算「放宽过」：评分不是瓶颈时结果不会变，那没必要向模型多报一个 relaxed 标记。
        if len(candidates_wo_rating) > len(relevant):
            relevant, memory_dropped = candidates_wo_rating, mem_dropped_wo
            relaxed = True
    truncated = len(relevant) > capped_k

    # ── 探测召回：命中不足时问一句「库里到底是没这类货，还是有货但被硬条件挡了」 ──
    # 两者在返回体里长得一模一样（total_recall 都很小），模型只能猜，实测常把「都超预算」说成
    # 「没找到」。再打一次**不带 price / rating / 记忆排除**的召回做差集，把被挡的样本如实回给
    # 模型。只在确有硬过滤条件时跑——没有过滤，差集必然为空，那一次查询纯属白花。
    strategy = ["dense"]
    if price_usd_max is not None:
        strategy.append("price_filter")
    if min_rating is not None:
        strategy.append("rating_relaxed" if relaxed else "rating_filter")
    if mem_exclude:
        strategy.append("memory_exclude")
    if brand_exclude:
        strategy.append("brand_exclude")
    filtered_out: list[FilteredOutItem] = []
    has_hard_filter = (
        price_usd_max is not None
        or min_rating is not None
        or bool(mem_exclude)
        or bool(brand_exclude)
    )
    if PROBE_LIMIT > 0 and has_hard_filter and len(relevant) < capped_k:
        probed = await _recall(PROBE_LIMIT, price_max=None, rating_min=None)
        filtered_out, price_blocked, other_blocked = _probe_filtered_out(
            probed,
            {rc.item_id for rc in relevant},
            price_usd_max=price_usd_max,
            min_rating=min_rating,
            brand_exclude=brand_exclude,
            exclude_terms=mem_exclude,
        )
        strategy.append("probe")
        # 全树登记：「预算内到底有没有货」是跨平台合流后的结论，供补搜闸判「该补搜还是该
        # 建议放宽预算」（见 retrieval_budget.budget_relax_due）。
        note_filtered_probe(
            hits=len(relevant[:capped_k]),
            price_blocked=price_blocked,
            other_blocked=other_blocked,
        )

    candidates = [ItemCandidate.from_recall(rc) for rc in relevant[:capped_k]]
    out = ItemSearchOutput(
        platform=platform,
        candidates=candidates,
        total_recall=len(candidates),
        truncated=truncated,
        relaxed=relaxed,
        memory_excluded=memory_dropped,
        # 必须在 register() 之前判：登记完这批自己就全「已在池内」了。
        known_ids=[c.item_id for c in candidates if enrich(c.item_id) is not None],
        filtered_out=filtered_out,
        recall_strategy="+".join(strategy),
    )
    # 套装槽位盖章：slot 入参经 register_slot 解析成本轮槽表里的规范槽名（没登记的按规则补登，
    # 见其 docstring）。盖在 register 之前——登记表存的就是带槽标的全量候选。解析不出 → 不盖章，
    # 候选落 picker 的 keywords 兜底归槽。
    slot_name = register_slot(slot)
    if slot_name:
        note_slot_searched(slot_name)  # 「搜了但没货」与「压根没搜」要分得开，组合报告如实说
        for c in candidates:
            c.slot = slot_name
    # 登记召回信号到全树检索状态：供 web_search 兜底门判定（仅在召回全空时才放行 web_search）。
    note_item_search(out.total_recall)
    # 把全量候选（含真实 url/image_url）按 item_id 登记到会话：url/image 不再随候选喂模型，
    # 收尾由 shopping_summary 从这里按 item_id 回填卡片（见 _candidates.py）。
    register(candidates)
    # 思考结果摘要：召回了多少、头部几条长啥样（供前端展开看这一步「找到了什么」）。
    top_titles = "、".join(c.title for c in out.candidates[:3] if c.title)
    search_result = f"召回 {out.total_recall} 条" + (f"，Top：{top_titles}" if top_titles else "")
    if relaxed:
        search_result += "（首轮召回不足，已自动放宽评分门槛重试；相关度/预算/品牌黑名单不放宽）"
    if memory_dropped:
        search_result += f"（另有 {memory_dropped} 条命中用户硬排除偏好，已在召回阶段过滤）"
    if filtered_out:
        search_result += f"（探测到库内另有 {len(filtered_out)} 条相关商品被硬条件挡下）"
        # 结构化诊断走侧信道给 harness（result_nudges 据此提示模型别把「被挡」说成「没货」）：
        # 与模型可见文本解耦，截断 / 措辞改动都伤不到信号。
        report_diagnostics(
            "item_search",
            {
                "filtered_out": [f.model_dump() for f in filtered_out],
                "filtered_price_only": all(f.reason.startswith("超预算") for f in filtered_out),
            },
        )
    await monitor.report_tool_end(
        "item_search",
        platform=platform,
        total_recall=out.total_recall,
        memory_excluded=memory_dropped,  # 记忆在召回阶段杀了几条——静默失效是记忆最危险的失败
        result=search_result,
    )
    return out
