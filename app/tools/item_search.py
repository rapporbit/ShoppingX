"""item_search —— 单平台商品检索（Qdrant dense 召回 + filter，语义+个性化双通道融合）。

主链路的「检索」一环。封装 :mod:`app.recall` 的编码 + Qdrant dense 召回：把用户这次搜索意图
（query，已并入本轮域内的 like 偏好词）编码成 dense「请求向量」做召回，返回 top_k 归一候选。
精确命中/硬约束走 Qdrant payload filter（非 sparse 打分，见 `docs/plans/召回引擎选型思路.md` §4）；
filter 维度：platform + price_usd_max + min_rating（Qdrant Range）+ brand_exclude（后置过滤）。

精排取舍（对齐 refdoc）：refdoc 的 item_search 只做 dense 召回 + 双通道本地融合，cross-encoder
精排是 CategoryInsight/RAG 链路的事（refdocs/11 vs 13-1）。本工具据此**不再做 cross-encoder
精排**——候选的二次质量把关交给下游 item_picker（按用户偏好精挑）。这样每次检索少一次 rerank
网络往返，跨平台 fork 放大时收益明显。

**单平台**：一次只搜一个平台。跨平台并行检索由主 loop 通过 ``parallel_dispatch_tool`` fork 多个
同质子 Agent、每个子 Agent 搜一个平台来完成（fork 三件事之「能并行」），本工具不自己
循环多平台——把「要不要并行」的决策权留给主 loop 的 fork 判断。

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
import re
from typing import Literal

import numpy as np
from langchain_core.tools import tool
from pydantic import BaseModel

from app.agent.platform_scope import resolve_search_platforms
from app.agent.retrieval_budget import note_item_search
from app.api import monitor
from app.api.context import get_user_id
from app.memory.assemble import assemble
from app.recall import get_recall_client, get_tower_client
from app.recall.schemas import RecallCandidate
from app.recall.text import tail_category
from app.recall.towers import TowerClient
from app.tools._args import StrListArg
from app.tools._bundle import current_slot, note_slot_searched, register_slot
from app.tools._candidates import compact_candidates, enrich, register
from app.tools.schemas import ItemCandidate
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
CATEGORY_MATCH_FLOOR: float
RETRY_MIN_HITS: int
EXCLUDE_FETCH_BUFFER: int


def _load_params() -> None:
    """从 env 求值本模块的可调参数（导入时跑一次；后台改参数后由覆盖层回调）。

    赋值顺序即源码顺序，故 MAX_TOP_K 能安全地拿 DEFAULT_TOP_K 当默认值。
    """
    global DEFAULT_TOP_K, MAX_TOP_K, SINGLE_PLATFORM_POOL_K, RENDER_CAP
    global RELEVANCE_FLOOR, CATEGORY_MATCH_FLOOR, RETRY_MIN_HITS, EXCLUDE_FETCH_BUFFER

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
    # 这个乘数——若同样只召 10 条，池子≈展示上限（PICK_DISPLAY_CAP=8），精排几乎无筛除空间（10 挑 8
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

    # 品类一致性过滤（target_name + expected_category 搭配用）：型号过滤挡不住"配件类商品标题里
    # 带宿主型号"这种假阳性（如"给 XM5 用的耳机壳""QC45 充电线"——型号 token 对得上，但压根不是
    # 耳机本体）。这类配件在数据源里的 category 字段本就不属于目标品类（如充电线归在
    # "Televisions & Video Products"，不在"Headphones & Earbuds"下）——真实复现过：定点调查 Sony
    # WH-1000XM5 / Bose QC45 时各自召回到一件同型号配件，型号过滤没拦住，误判"库内已找到候选"，
    # 连带把本该放行的 web_search 兜底也拦死了（见 retrieval_budget.web_search_allowed）。
    # 用已有的 TowerClient 编码 expected_category 与候选自身 category（tail_category 去掉顶层
    # 泛词噪声）比余弦相似度，而非再拉一份"配件关键词黑名单"——黑名单枚举不完，品类字段是数据
    # 自带的干净信号。**诚实标注**：阈值未跑线上真实 embedding 校准（远程模型才有语义度量意义，
    # 本地 n-gram 回退只在同语言/有公共子串时近似有效），先给个保守初值，可经 env 调。
    CATEGORY_MATCH_FLOOR = env_float("CATEGORY_MATCH_FLOOR", 0.5)

    # 「召回够不够用」的条数判据：少于这么多条就自动摘掉评分门槛重搜一次（见下面的放宽段）。
    # 取 3 而非 0：只召回一两条时模型照样会自己发起一轮重搜，那一轮 Think 的解码开销正是要省掉的。
    RETRY_MIN_HITS = env_int("ITEM_SEARCH_RETRY_MIN_HITS", 3)

    # 记忆硬排除生效时，多召这么多条补偿被杀的名额。硬排除（blocking 黑名单 + 本轮硬 dislike）在
    # **召回阶段**就过滤（不是等 item_picker 事后杀），top_k 硬封 10 的候选池才不会对黑名单用户
    # 永远残缺——「不要皮革」的用户搜公文包，10 条里 8 条皮的，杀完只剩 2 条还无处补货。
    # Qdrant 多取 10 条近邻的成本可忽略，而少取的代价是残缺候选池。
    EXCLUDE_FETCH_BUFFER = env_int("ITEM_SEARCH_EXCLUDE_BUFFER", 10)


_load_params()

# 平台枚举：用 Literal 而非 str，从 schema 层挡住模型生成 Amazon/AMAZON/amzn 等等价但不规范的
# 串（refdocs/11 §2.2）。全集对齐 utils.clean.PLATFORMS —— **eBay 不在其中**：它的 CSV 卖家名近全
# 掩码、描述列全空，清洗阶段就整体剔除了，召回库里一条 eBay 商品都没有，列进来只会诱导模型去搜一个
# 必然空召回的平台。"all" 跨**本次启用的**平台合流（见 agent.platform_scope）。
# 深层防御另有 qdrant_store.search 的 strip().lower() 归一兜底（测试/直连调用方）。
Platform = Literal["all", "amazon", "walmart", "shein", "lazada", "shopee"]

# 定点型号过滤（target_name 用）：语义召回分不清"库里没货"和"有但不是这个型号"（见 _load_params
# 里 RELEVANCE_FLOOR 的诚实标注），对定点商品调查这种"要精确型号"的场景，用字符串型号匹配
# 兜一道——型号号段对不上的直接不算候选，别让 total_recall 假装找到了。
_TARGET_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STRIP_RE = re.compile(r"[\s\-]+")


def _matches_target_name(title: str, target_name: str) -> bool:
    """粗粒度型号过滤（不追求完美，只挡最明确的"型号对不上"）：
    - 目标名解析出「字母+数字混合」token（如 "1000xm5"/"qc45"，区分度最高、单独即可判定）：
      命中一个就算匹配。
    - 没有这类 token（型号是纯数字，如 "651"）：退化成"数字 token（长度≥2）+ 品牌/系列词
      （长度≥3 的纯字母 token）同时出现在标题里"才算匹配——单独数字太容易撞价格/尺寸等噪声。
    - 两类 token 都解析不出（商品名没有具体型号，纯描述性文本）：不过滤，避免误伤。
    标题与目标名都先去空格/连字符再比对，兼容 "WH-1000XM5"/"WH1000XM5" 等写法差异。

    **诚实标注局限**：纯字符串匹配，"给 XX 型号用的配件"这类标题会因为同样包含型号词被
    误判为匹配——比语义召回的「完全不沾边」好得多，但不是精确匹配，别指望它完美。这一类
    "型号对得上但是配件"的假阳性由 :func:`_category_relevant_mask` 另外兜（见其说明）。
    """
    tokens = _TARGET_TOKEN_RE.findall(target_name.lower())
    norm_title = _STRIP_RE.sub("", title.lower())

    mixed = [
        t
        for t in tokens
        if len(t) >= 3 and any(c.isdigit() for c in t) and any(c.isalpha() for c in t)
    ]
    if mixed:
        return any(t in norm_title for t in mixed)

    digits = [t for t in tokens if t.isdigit() and len(t) >= 2]
    words = [t for t in tokens if t.isalpha() and len(t) >= 3]
    if digits and words:
        return any(d in norm_title for d in digits) and any(w in norm_title for w in words)

    return True


async def _category_relevant_mask(
    tower: TowerClient, expected_category: str, categories: list[str]
) -> list[bool]:
    """批量判断候选自身的 category 是否与 ``expected_category`` 语义相符。

    对齐用户点子：与其枚举"配件关键词黑名单"（case/cover/cable/charger/...，枚举不完、
    换个品类又要重列一份），不如直接比对候选自带的 category 字段——数据里配件类商品本就
    不会被分进目标品类（如耳机充电线在源数据里是 "Televisions & Video Products"，压根不在
    "Headphones & Earbuds" 下），这是比标题关键词干净得多的信号。

    没有 category 数据的候选不参与判定（返回 True，不误伤——跟 :func:`_matches_target_name`
    "解析不出型号就不过滤"同一口径）。一次批量编码（expected_category + 各候选的
    ``tail_category``），省去逐条网络往返。
    """
    idx = [i for i, c in enumerate(categories) if c.strip()]
    mask = [True] * len(categories)
    if not idx:
        return mask
    texts = [expected_category] + [tail_category(categories[i]) for i in idx]
    vecs = await tower.encode_texts(texts)
    expected_vec = vecs[0]
    for pos, i in enumerate(idx, start=1):
        # encode_texts 已 L2 归一化，内积即余弦相似度。
        sim = float(np.dot(expected_vec, vecs[pos]))
        mask[i] = sim >= CATEGORY_MATCH_FLOOR
    return mask


def _searchable(rc: RecallCandidate) -> str:
    """召回候选的可匹配文本（与 item_picker 的 ``_searchable`` 同口径：标题+品牌+品类，小写）。"""
    return f"{rc.title} {rc.brand} {rc.category}".lower()


def _apply_filters(
    recalled: list[RecallCandidate],
    *,
    floor: float,
    brand_exclude: list[str] | None,
    target_name: str | None,
    exclude_terms: list[str] | None = None,
) -> tuple[list[RecallCandidate], int, int]:
    """按语义门槛 + 用户硬约束过滤召回结果，返回 (候选, 型号过滤掉的条数, 记忆排除掉的条数)。

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
    target_dropped = 0
    if target_name:
        before = len(relevant)
        relevant = [rc for rc in relevant if _matches_target_name(rc.title, target_name)]
        target_dropped = before - len(relevant)
    return relevant, target_dropped, memory_dropped


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

    def __str__(self) -> str:
        """喂给模型的紧凑投影（见 :func:`compact_candidates`：只留决策要用的字段）。

        单平台检索时每条候选里的 ``platform`` 是纯冗余——顶层已经写了一次平台名，20 条再各重复
        一遍只是烧 token。``platform="all"`` 合流时必须留：那时模型得知道每件货来自哪个平台。
        """
        single_platform = self.platform != "all"
        known = set(self.known_ids)
        fresh = [c for c in self.candidates if c.item_id not in known]
        # 渲染收敛（与「进登记表的召回池」解耦）：召回池全量已进登记表供 item_picker 精排，模型上下文
        # 只看头部 RENDER_CAP 条——模型不精挑，灌满整池纯烧 token。其余以 pooled_for_pick 计数告知，
        # 让模型知道「池里还有货、已交给精挑」，不必自己在这里筛。
        shown = fresh[:RENDER_CAP]
        pooled = len(fresh) - len(shown)
        return json.dumps(
            {
                "platform": self.platform,
                "total_recall": self.total_recall,
                "truncated": self.truncated,
                # 只在真放宽过时才带这个键：没放宽时是默认状态，写进去纯属每轮多烧 token。
                **({"relaxed": True} if self.relaxed else {}),
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
    target_name: str | None = None,
    expected_category: str | None = None,
    slot: str = "",
) -> ItemSearchOutput:
    """在【单个】平台检索商品（dense 召回；用户长期偏好词已由系统自动并入检索词）。

    何时调用：需要在某平台搜商品时。跨多个平台请用 parallel_dispatch_tool 并行 fork、每个
    子任务搜一个平台，不要自己串行多次调本工具。
    用户的硬排除偏好（「绝不推荐」黑名单 + 本轮明说的「不要 X」）已由系统在召回阶段自动过滤
    （返回的 memory_excluded 计数），不需要你转述进参数；total_recall 为 0 且 memory_excluded>0
    时，如实告诉用户「符合的商品都被你的排除偏好筛掉了」，而不是「库里没有」。
    参数：
      - query：这次要搜什么（自然语言意图，走 dense 语义检索）。**用品类核心词**（如
        men's wristwatch / laptop backpack），场景 / 人群 / 正式度词（formal、business、
        men 这类）**不要堆进来**——它们会把同场景的其他品类一并召回、稀释目标品类（实测
        formal dress watch men business 召回的大半是西装皮鞋）；这类词交给 item_picker
        的 prefer_keywords 加分。
      - platform：平台名（amazon/walmart/shein/lazada/shopee）；"all" 跨**本次启用的**平台
        合流（启用集合见当轮 <enabled_platforms>，用户未勾选的平台一律不搜——传了也会被收口）。
      - top_k：召回条数。**不用传**——服务端按固定上限收口，传更大的数不会拿到更多候选。
      - price_usd_max：预算上限（USD），在 Qdrant 召回阶段直接过滤超预算商品。
        planner 拆出 budget_usd 后传入，省得召回一堆超预算的浪费配额。
      - min_rating：最低评分（如 4.0），在召回阶段过滤低分商品。
      - brand_exclude：要排除的品牌列表（如 ["Nike", "Adidas"]），大小写不敏感。
      - target_name：**定点商品调查时传**被点名的商品名/型号原文（如 "Sony WH-1000XM5"）。
        传入后额外按型号过滤召回，语义相关但型号不符的候选不算命中（如搜索该型号时召回到
        同品牌其它便宜型号），避免"库里没这型号却拿相似品硬凑"。跨平台泛搜
        （parallel_dispatch_tool 场景）不传。
      - expected_category：**定点商品调查时配合 target_name 一起传**目标商品所属品类
        （如 planner 拆出的 "降噪耳机"）。用于过滤"型号 token 对得上但其实是配件/耗材"的
        假阳性（如该型号的充电线、保护壳——标题必然带宿主型号，型号过滤拦不住，但配件在
        数据里的品类跟本体不是一类）：候选自身 category 与此语义不符时不算命中。不传则
        跳过这层过滤（沿用旧行为）。
      - slot：**套装（一套齐）检索时传** demands 里「套装槽位：X」标注的槽名 X（照抄）。
        系统按它给候选打标，供跨槽组合优选分组；非套装检索不传。
    """
    # 个性化：把用户**本轮域内**的 like 偏好原子词拼进检索词（见 memory.assemble.search_terms）。
    # 这条通路取代了原来的 user 塔向量画像——那条路把所有 like 加权平均成一个向量塞进召回，结果
    # 无法归因、无法调试、也无法向用户解释。拼进 query 文本后，个性化**看得见**：它出现在下面
    # 的 report_tool_start 上报里、前端的思考过程里、日志里。可观测的弱个性化，胜过不可观测的
    # 强个性化——后者出问题时你连从哪儿查都不知道。
    mem = await assemble(get_user_id() or "")
    pref_terms = mem.search_terms
    effective_query = " ".join([query, *pref_terms]) if pref_terms else query
    # 记忆硬排除在**召回阶段**生效（不等 item_picker 事后杀）：见 _apply_filters 的说明。
    # 归一与 picker 同口径（中文词补英文变体），命中判定同一套 term_hits。
    mem_exclude = normalize_terms(mem.exclude)
    # 平台收口（机制层，不靠模型自觉）：把入参落到本轮「启用平台」集合内——"all" 只等于全部启用平台，
    # 模型点名一个未启用的平台则回落到启用集合。用户没勾的平台不该被搜（见 agent.platform_scope）。
    search_platforms = resolve_search_platforms(platform)
    platform = search_platforms[0] if len(search_platforms) == 1 else "all"  # type: ignore[assignment]
    await monitor.report_tool_start(
        "item_search",
        query=effective_query,  # 上报**实际**用于检索的词（含偏好词）——个性化必须看得见
        platform=platform,
        top_k=top_k,
        personalization=("+".join(pref_terms) if pref_terms else "inactive"),
        price_usd_max=price_usd_max,
        min_rating=min_rating,
        brand_exclude=brand_exclude,
        target_name=target_name,
        expected_category=expected_category,
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

    request_vec = await tower.encode_query(effective_query)  # dense 通路（偏好词已并入检索词）

    recalled = await asyncio.to_thread(
        recall.search,
        request_vec,
        fetch_k,
        search_platforms,
        price_usd_max=price_usd_max,
        min_rating=min_rating,
    )
    relevant, target_dropped, memory_dropped = _apply_filters(
        recalled,
        floor=RELEVANCE_FLOOR,
        brand_exclude=brand_exclude,
        target_name=target_name,
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
    #     温床——空手而归远好过给一堆不沾边的货；
    #   · target_name（定点调查）—— 返回错型号比「没找到」更糟。
    relaxed = False
    if len(relevant) < RETRY_MIN_HITS and not target_name and min_rating is not None:
        # 评分门槛是 Qdrant 召回阶段的 filter，只能重搜一次才能摘掉（预算照旧硬卡）。
        widened = await asyncio.to_thread(
            recall.search,
            request_vec,
            fetch_k,
            search_platforms,
            price_usd_max=price_usd_max,
            min_rating=None,
        )
        candidates_wo_rating, dropped_wo_rating, mem_dropped_wo = _apply_filters(
            widened,
            floor=RELEVANCE_FLOOR,  # 相关度红线照旧
            brand_exclude=brand_exclude,
            target_name=target_name,
            exclude_terms=mem_exclude,  # 记忆硬排除是用户授权的硬约束，放宽时同样不动
        )
        # 真捞到更多才算「放宽过」：评分不是瓶颈时结果不会变，那没必要向模型多报一个 relaxed 标记。
        if len(candidates_wo_rating) > len(relevant):
            relevant, target_dropped, memory_dropped = (
                candidates_wo_rating,
                dropped_wo_rating,
                mem_dropped_wo,
            )
            relaxed = True
    category_dropped = 0
    if target_name and expected_category and relevant:
        before = len(relevant)
        mask = await _category_relevant_mask(
            tower, expected_category, [rc.category for rc in relevant]
        )
        relevant = [rc for rc, keep in zip(relevant, mask, strict=True) if keep]
        category_dropped = before - len(relevant)
    truncated = len(relevant) > capped_k

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
    )
    # 套装槽位盖章：显式入参优先（用户确认后新加的槽走这条），退回 dispatch 派发时从 demand
    # 确定性解析并经 ContextVar 传下来的槽名（机制主通路，不依赖子 Agent 转述）。盖在 register
    # 之前——登记表存的就是带槽标的全量候选，跨轮落盘 / 读回都带着。
    slot_ref = slot.strip() or current_slot()
    if slot_ref:
        # 槽引用（id / 精确名 / 漂移名）→ 稳定槽 id：全链路唯一解析点（_bundle.register_slot，
        # 内置懒读回；确属新品类时补登发号）。解析不出（非套装轮 / 野 id / 用户删过的槽）
        # → 空串，不盖章——候选落 keywords 兜底，绝不让模型的字符串自成一档身份。
        slot_id = register_slot(slot_ref)
        if slot_id:
            note_slot_searched(slot_id)  # 「搜了但没货」与「压根没派」要分得开，组合报告如实说
            for c in candidates:
                c.slot = slot_id
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
    if target_dropped:
        search_result += f"（另有 {target_dropped} 条语义相关但型号不符，已排除）"
    if category_dropped:
        search_result += f"（另有 {category_dropped} 条型号相符但品类不符，疑似配件，已排除）"
    await monitor.report_tool_end(
        "item_search",
        platform=platform,
        total_recall=out.total_recall,
        memory_excluded=memory_dropped,  # 记忆在召回阶段杀了几条——静默失效是记忆最危险的失败
        result=search_result,
    )
    return out
