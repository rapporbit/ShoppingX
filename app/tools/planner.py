"""planner —— 把购物意图拆成结构化字段。

复杂多约束的需求（「便宜又抗造的旅行三件套，预算 300，不要塑料，喜欢小众」）先过这一步，
拆成预算 / 品类 / 偏好等结构化字段，给后续检索、精挑当锚点。尤其是它顺手把自然语言偏好拆成
**三个原子词桶**——``exclude_terms``（硬淘汰）/ ``soft_dislikes``（减分）/ ``prefer_keywords``
（加分）——让下游的 ``item_search`` 和 ``item_picker`` 直接吃结构化入参，不必各自再解析一遍
自然语言。

**planner 是会话态 P_t 的唯一写者（产生与撤销都归它）。** 它每轮都跑、看的是用户原话，产出得早
（当轮就落进 P_t、当轮被 item_picker 执行）且带 blocking 档位；撤销走「模型照抄约束 id 进
``retract_ids`` + ``merge_pt`` 存在性/词面双闸核验」。curator 只判长期库，不碰 P_t——双写者
时代它曾用更差的输出覆盖 planner（无档位 draft、脑内换汇），教训见 ``app.memory.curator``。

**档位（blocking）由机制判，不由模型判**：模型判「尽量别太花哨」算硬排除还是软避讳，实测约 1/4
的概率判错，但它**转述用户原话**是稳的。所以 ``exclude_terms`` 的每个词都必须附 evidence（原话
片段），档位交给 :func:`_is_weak` 扫原话里的弱表达标记确定性地判。见 :class:`ExcludeTerm`。

用 LLM 做意图理解（``with_structured_output`` 强约束成 Pydantic）。走 ``get_fast_llm``
（同模型关 reasoning）：结构化抽取不吃思维链——实测主档 12~17s（reasoning 占 70%）vs 快档
~3s，tasks / 排除词 / 预算解析产出一致（对照见 latency-audit round2），与 summary / curator /
parser 同档。模块级引用便于测试 monkeypatch 成假模型，从而离线可测。
"""

from __future__ import annotations

import logging
import os
import re
from typing import Literal

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.tools import tool
from pydantic import BaseModel, Field, model_validator

from app.agent.fork_guard import current_fork_depth
from app.agent.llm import get_fast_llm
from app.agent.prompts import get_planner_prompt
from app.agent.token_budget import charge_tool_llm_usage
from app.api import monitor
from app.api.context import (
    get_original_query,
    get_session_dir,
    get_session_pt,
    get_user_id,
    set_dest_country,
    set_retrieval_mode,
    set_session_domains,
    set_session_pt,
    set_session_tasks,
)
from app.memory.domains import (
    DOMAIN_GLOBAL,
    PrefDomain,
    domain_menu,
    reconcile_domains,
)
from app.memory.session_state import (
    SessionConstraint,
    SessionPrefState,
    constraint_verb,
    merge_pt,
    save_pt,
)
from app.recall.fx import to_base_or_none
from app.recall.geo import (
    DEFAULT_DEST_COUNTRY,
    DEST_COUNTRY_SLOT,
    match_country_name,
    resolve_dest_country,
)
from app.tools._args import drop_none_values
from app.tools._bundle import (
    MAX_SLOTS,
    BundleSlot,
    reset_session_bundle,
    set_session_bundle,
)
from app.tools._candidates import registry_snapshot, reset_candidates

logger = logging.getLogger("shoppingx.planner")

# 用户本轮想让 Agent 做的事（意图信号，驱动 <workflow> 按需组合能力，而非走死一条链）：
#   recommend      —— 挑 / 推荐商品（要检索 + 精挑）
#   evaluate       —— 评某商品好不好（品类基准 + 评分 + 口碑）
#   price_compare  —— 跨平台比价
#   landed_cost    —— 算关税 + 运费（到手价）
#   category_intel —— 只问品类行情（热卖 / 价位 / 该看哪些维度），不一定要具体商品
ShoppingTask = Literal["recommend", "evaluate", "price_compare", "landed_cost", "category_intel"]

# 本轮该怎么拿候选——**「要不要重新检索」由 planner 判，不由「候选池有没有货」猜**：
#   reuse   —— 用户只是在上一轮结果上**收紧**条件（「只要防水的」「把塑料的去掉」）。新结果是旧
#              候选的子集，重搜一遍只会拿回同一批商品还多花近百秒 → 直接 item_picker 过滤。
#   augment —— 用户**放宽**条件或要更多（「预算提到 500」「再多给几个」）。旧池子是在旧约束下召回
#              的，压根不含新放开的那部分 → 必须重新检索，新老候选合流后再精挑。
#   search  —— 换品类 / 换意图 / 首轮。旧候选与本轮无关，正常检索——且**当场把旧候选清掉**（见
#              planner 工具体）：下游 item_picker 吃的是登记表全集，留着上一轮的鞋就会混进这一轮
#              的沙发。清完，登记表自身即「本轮该精挑的全集」，模型不必也无法再指定是哪几件。
RetrievalMode = Literal["reuse", "search", "augment"]

# 意图接地——「这个购物意图能不能可靠翻译成品类 / 检索词」由 planner 显式判，给主 loop 一个
# 机制可见的信号（动机层提示，不是硬闸）：
#   internal —— 经典 / 常识意图（送父母礼物、买跑鞋），模型参数内知识足以列品类假设。模糊但
#               经典（「送女朋友礼物」）也算 internal——缺的是澄清维度，不是世界知识。
#   web      —— 含模型**没把握的新说法 / 潮流词 / 时效性诉求**（「谷子」「痛包」「今年流行的
#               那种」）→ 主 loop 检索前先 web_search 一次，把它翻译成品类词 / 检索词再搜。
#               判据是「有没有把握理解这个说法」，不是「意图模不模糊」。
IntentGrounding = Literal["internal", "web"]

# 用户没写币种时钉死的默认预算币种（确定性，禁止模型每轮自由猜）。默认人民币，可经 env 调。
DEFAULT_BUDGET_CURRENCY = (os.getenv("DEFAULT_BUDGET_CURRENCY", "CNY") or "CNY").strip().upper()

# 预算币种确定性解析表：(ISO 码, 正则)，**按特异性排序**，第一个命中即返回。
# 关键消歧靠顺序：带 $ 的 S$/HK$ 必须先于裸 $→USD；含「元」的 美元/欧元/日元 必须先于「元」→CNY，
# 否则「美元」会被尾部 CNY 规则（含「元」）抢先错判。
# ¥ 在中文市场默认按 CNY（真要日元写「日元/円/JPY」）。
_CURRENCY_PATTERNS: list[tuple[str, str]] = [
    ("SGD", r"S\$|新加坡元|新元|SGD"),
    ("HKD", r"HK\$|港币|港元|HKD"),
    ("JPY", r"日元|日币|円|JPY"),
    ("USD", r"美元|美刀|美金|US\$|USD|\$"),
    ("EUR", r"欧元|EUR|€"),
    ("GBP", r"英镑|GBP|£"),
    ("INR", r"印度卢比|卢比|INR|₹"),
    ("CNY", r"人民币|RMB|CNY|￥|¥|块钱|块|元"),
]


def resolve_budget_currency(text: str) -> tuple[str, bool]:
    """从用户原始意图里**确定性**解析预算币种，返回 ``(ISO 码, 是否明示)``。

    命中任一符号/词即「明示」（``True``）；都没命中则落 :data:`DEFAULT_BUDGET_CURRENCY`
    （默认 CNY）且「非明示」（``False``）——供答案标注「已按 ¥ 理解」或触发一次澄清。
    纯规则、不调模型：同一句话永远解析出同一币种（修掉「预算 500 每轮被猜成 ₹/¥/$」的抖动）。
    """
    for code, pattern in _CURRENCY_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return code, True
    return DEFAULT_BUDGET_CURRENCY, False


_NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")
_CJK_NUM_RE = re.compile(r"[零一二两三四五六七八九十百千万亿]")
# 数字后紧跟的量级缩写（"1万"/"2k"）：数字 token 本身对不上时，接受乘上量级后的值。
_SCALE_SUFFIXES = {
    "k": 1_000.0,
    "K": 1_000.0,
    "千": 1_000.0,
    "w": 10_000.0,
    "W": 10_000.0,
    "万": 10_000.0,
}


def budget_amount_grounded(intent: str, amount: float) -> bool:
    """模型填的 ``budget_amount`` 是否真在**本轮原话**里出现过（机制闸，防抄上文重折）。

    追问轮（「不要皮革的」）里模型会把 P_t 渲染出的「≤ $80」抄进 budget_amount，而币种解析
    只看本轮原话——没提币种就落默认 CNY，80 美元被当 80 人民币折成 $11.2，P_t 预算悄悄缩水
    7 倍（真实 e2e 复现）。规则核对：本轮原话里找得到这个数（含 "1,000" 逗号形式与 k/千/万
    量级缩写）才算「本轮提过」；原话没有任何数字也没有中文数词 → 判「本轮没提预算」（budget
    置 None，merge_pt 语义即「保持上一轮不变」）；含中文数词（「预算三百」）无法确定性核对，
    放行模型的值——宁可放过，不误删真预算。
    """
    values: set[float] = set()
    for m in _NUM_RE.finditer(intent):
        v = float(m.group().replace(",", ""))
        values.add(v)
        nxt = intent[m.end() : m.end() + 1]
        if nxt in _SCALE_SUFFIXES:
            values.add(v * _SCALE_SUFFIXES[nxt])
    if not values:
        return bool(_CJK_NUM_RE.search(intent))
    return any(abs(amount - v) < 1e-6 for v in values)


async def resolve_dest_country_layered(text: str) -> tuple[str, bool, bool]:
    """四层确定性决定本轮收货国，返回 ``(ISO 码, 是否为「假设值」, 是否本轮明示)``。

    优先级（高 → 低），**任一层命中即停**：
    1. 本轮用户明说（「寄到日本」）—— 纯规则解析，见 :func:`app.recall.geo.resolve_dest_country`。
    2. 会话级 P_t 的 ``slots["dest_country"]`` —— 本会话前几轮说过一次，后面一直生效。
    3. 长期记忆里 ``category="location"`` 的偏好 —— 跨会话记住常用收货地。
    4. env ``DEFAULT_DEST_COUNTRY`` 默认值。

    只有走到第 4 层才算「假设」（返回 ``assumed=True``）——前三层都有用户依据。1~3 层里
    **只有第 1 层是「本轮明示」**（``stated_now=True``），但 2/3 层同样源于用户自己说过的话，
    不该每轮都去骚扰他确认，故不标 assumed；标 assumed 的目的只是提醒模型「这是系统替你猜的，
    得在回复里讲明」。``stated_now`` 单独返回是给 slots 写入门槛用的：**只有本轮带收货语境的
    明示才值得固化成会话事实**——第 2 层写回是自我循环，第 3 层写入则会把长期记忆「快照」进
    会话，用户中途改了记忆本会话也不跟着变。
    """
    country, explicit = resolve_dest_country(text)
    if explicit:  # 第 1 层：本轮原话（门控明示）
        return country, False, True

    pt = get_session_pt()  # 第 2 层：会话级 slots
    if pt is not None:
        slot = (pt.slots or {}).get(DEST_COUNTRY_SLOT, "").strip().upper()
        if slot:
            return slot, False, False

    user_id = get_user_id()  # 第 3 层：长期记忆（跨会话常用收货地）
    if user_id:
        try:
            from app.memory.store import get_store

            entries = await get_store().read(user_id)
            for e in entries:
                if e.category == "location" and e.polarity == "like":
                    # 记忆条目语义已确定是收货地（category 即门控），用无门控匹配——
                    # 「常用收货地：中国」若再要求语境词反而可能漏掉。
                    code = match_country_name(e.content)
                    if code:
                        return code, False, False
        except Exception:  # noqa: BLE001 —— 记忆后端挂了不该崩掉 planner，降级到默认国即可
            pass

    return DEFAULT_DEST_COUNTRY, True, False  # 第 4 层：系统默认 → 必须在回复里标注假设


# 弱表达标记：命中即说明用户那句话是「避讳」而非「排除」，对应的词必须降级到 soft_dislikes。
#
# 只收**修饰否定强度**的词，不收否定词本身（「不」「别」既能组成「不要」也能组成「不太喜欢」，
# 单看它区分不出档位）。英文那几个是给英文 query 兜的，同理只收修饰语。
_WEAK_MARKERS = (
    "尽量",
    "尽可能",
    "最好别",
    "最好不",
    "不太",
    "不是很",
    "不怎么",
    "别太",
    "太过",
    "有点",
    "稍微",
    "能不要",
    "可以的话",
    "如果可以",
    "倾向",
    "prefer not",
    "rather not",
    "ideally",
    "if possible",
    "not too",
    "a bit",
)


def _is_weak(evidence: str) -> bool:
    """用户原话里带弱表达修饰 → 这不是硬排除。"""
    low = evidence.lower()
    return any(m in low for m in _WEAK_MARKERS)


class ExcludeTerm(BaseModel):
    """一个硬排除词 + 它在用户原话里的依据。

    **evidence 不是留给人看的注释，是机制的输入**：模型对「尽量别太花哨」该算硬排除还是软避讳的
    判断本身就不稳（实测约 1/4 的概率归错档），但它**转述用户原话**是稳的。于是不再问模型「这算
    哪一档」，只问它「用户在哪儿说的」——档位由 :func:`_is_weak` 扫原话里的弱表达标记确定性地判。
    把不稳的判断换成稳的转述，是这个字段存在的全部理由。

    附带好处：中英对照自动兜住。「flashy」这个英文词自己看不出强弱，但它的 evidence 同样是
    「尽量别太花哨」，于是跟着中文词一起被降级——纯字面去重的老办法在这里是失效的。
    """

    word: str = Field(description="排除词本身（原子词，如「塑料」/「plastic」）")
    evidence: str = Field(
        default="",
        description="用户原话里说这句话的**片段**（如「不要塑料的」）。照抄，不要改写、不要引申。",
    )


class PlanOutput(BaseModel):
    """购物意图拆解结果（字段缺失留空 / None，不臆造）。"""

    # 模型把「没有这项」写成显式 null（badcase cdee1d6d：5 个 list 字段全 null 打挂 planner）
    # → 丢键让 default_factory 接管，见 drop_none_values。
    _null_is_absent = model_validator(mode="before")(staticmethod(drop_none_values))

    tasks: list[ShoppingTask] = Field(
        default_factory=list,
        description=(
            "用户本轮想让你做的事（可多选）：recommend=挑/推荐、evaluate=评好坏、"
            "price_compare=比价、landed_cost=算关税运费、category_intel=只问品类行情。"
            "**按用户明确表达来判**：只说「推荐/看看有啥」→ [recommend]；说「哪个便宜/多少钱」"
            "→ 加 price_compare；说「到手/含税含运多少」→ 加 landed_cost；说「这款值不值/好不好」"
            "→ evaluate；只问「这类东西行情/该看哪些维度」→ [category_intel]。别塞用户没要的。"
        ),
    )
    retrieval: RetrievalMode = Field(
        default="search",
        description=(
            "本轮怎么拿候选（见 RetrievalMode）：reuse=用户只是在上一轮结果上**收紧**条件"
            "（「只要防水的」「去掉塑料的」「只留 4 分以上」）→ 不重新检索，直接在既有候选上过滤；"
            "augment=**放宽**条件或要更多（「预算提到 500」「再多给几个」「还有别的吗」）→ 旧池子"
            "不含新放开的那部分，必须重新检索、新老候选合流；search=换品类 / 换意图 / 本轮"
            "没有既有候选 → 正常检索。**没有既有候选时一律 search**（系统会强制回填，别填 reuse）。"
        ),
    )
    target_refs: list[str] = Field(
        default_factory=list,
        description=(
            "用户**点名的具体商品**（要评价/比较的对象），如型号名、商品标题、贴的链接原文。"
            "「评价这款 XX」「比较这两个：A 和 B」时填；泛泛「推荐点耳机」这种没有具体对象则留空。"
        ),
    )
    category: str = Field(default="", description="主品类，如「旅行收纳」「跑鞋」")
    intent_grounding: IntentGrounding = Field(
        default="internal",
        description=(
            "这个购物意图你能不能可靠翻译成品类和检索词：internal=经典/常识意图（送礼、买跑鞋），"
            "你自己理解得了——**模糊但经典也是 internal**（「送女朋友礼物」缺的是澄清，不是世界"
            "知识）；web=含你**没把握的新说法/潮流词/时效性诉求**（「谷子」「痛包」「今年最流行的"
            "那种」「最近很火的」）→ 主流程会先 web_search 把它翻译成品类词再检索。判据是「你有没有"
            "把握理解这个说法」，拿不准且带时效词才填 web，别把普通模糊需求都推给搜索。"
        ),
    )
    domains: list[PrefDomain] = Field(
        default_factory=list,
        description=(
            "本轮在买哪些**品类域**（可多选，如「旅行三件套」= bags + apparel）。它决定用户的长期"
            "偏好哪些在本轮生效——「买鞋时不喜欢皮革」不该在买沙发时也把皮沙发全排掉。\n"
            "**只要本轮涉及具体商品，就必须至少填一个**：买跑鞋 → [footwear]；买降噪耳机 → "
            "[electronics]；买手表/腕表 → [jewelry_watches]（**不是** furniture——线上真实误判过，"
            "按商品本体归域，别被使用场景带偏）；归不进任何具体域（如「送人的小礼物」）→ [other]。"
            "只有纯闲聊、完全不涉及商品时才留空。\n"
            "**绝不要填 global**——那是偏好侧「跨品类底线」专用的标记，不是「本轮什么都买」的意思。"
            "可选值：\n" + domain_menu()
        ),
    )
    budget_amount: float | None = Field(
        default=None, description="用户原话给的预算金额（**不要换算**，照原数填），无则 None"
    )
    currency: str = Field(
        default="", description="预算币种 ISO 码——由系统规则确定性回填，**模型不要填**"
    )
    currency_assumed: bool = Field(
        default=False, description="True=用户未明示币种、已用默认币种——由系统回填，模型不要填"
    )
    budget_usd: float | None = Field(
        default=None,
        description="预算折算成 USD——由 budget_amount+currency 确定性折算回填，**模型不要填**",
    )
    clear_budget: bool = Field(
        default=False,
        description=(
            "用户本轮**明确取消 / 放开**了预算（「算了不限预算」「贵点也行，不设上限」「直接上"
            "最好的别管价格」）→ true。只是本轮没提预算 → false（那是「保持不变」，不是「取消」）。"
        ),
    )
    dest_country: str = Field(
        default="", description="收货国 ISO 码——由系统规则确定性回填，**模型不要填**"
    )
    dest_country_assumed: bool = Field(
        default=False,
        description="True=用户未明示收货国、已按默认国估算——由系统回填，**模型不要填**",
    )
    # 偏好只有三个桶，**全是原子词**（拿去和商品标题做匹配的），按「方向 × 力度」正交切分：
    # 负硬 → exclude_keywords（淘汰）、负软 → soft_dislikes（减分）、正向 → prefer_keywords
    # （加分）。
    # 正向没有「硬」档：数据没有可靠的材质 / 风格字段，正向二值淘汰（keep-only）会误杀一大片。
    #
    # **刻意不给「整句」桶**（原先的 hard_constraints / soft_preferences）。整句拿去匹商品标题永远
    # 匹不上，机制根本不消费它——可它一旦存在，模型就会把「不要塑料」老实地放进去（实测），于是这
    # 条硬约束静默失效。删掉这个桶，模型没地方丢，只能填原子词。**别给它错的选项，胜过教它别选错。**
    # material_pref / style_pref 同样删掉：它们和 prefer_keywords 是同一件事（正向原子词），
    # 三个桶只会让模型每轮纠结往哪个填。
    bundle_slots: list[BundleSlot] = Field(
        default_factory=list,
        description=(
            "「一套齐」跨品类套装需求专用（「新生入学一套」「露营装备一套」「旅行三件套」）：把这"
            "一套拆成 2~6 个子品类槽位（name 中文槽名 / keywords 英文检索词 / prefer 槽级偏好词 / "
            "essential 少了它这套是否就不成立）。**单品类需求一律留空**（哪怕买多件同类）。"
            "**每槽附 evidence**：用户原话点名了这件就照抄那个片段；是你按常识推断补的就留空——"
            "系统据此判断「套装组成要不要先跟用户确认」。"
        ),
    )
    keywords: list[str] = Field(default_factory=list, description="检索关键词，供 item_search")
    exclude_terms: list[ExcludeTerm] = Field(
        default_factory=list,
        description=(
            "**绝对排除**的原子词（命中即淘汰）：用户**本轮新说**的「不要 X / 不能 X」里的那个 X"
            "（已累积约束系统自动带轮，不要重放）。\n"
            "**材质、颜色也填这里**——商品数据里没有材质字段，拿关键词匹标题是这条约束唯一的"
            "执行通路。**英文必给**（「不要塑料」→ plastic，可另附中文原词）：**商品标题基本都是"
            "英文**，只给中文这条硬约束几乎等于没写（系统另有词表兜底，但覆盖不到的只能靠你给）。\n"
            "**每个词必须附上 evidence（用户原话片段）**：说不出用户在哪儿说过，就不该硬淘汰他的"
            "商品。弱表达（「尽量别」「不太喜欢」）放 soft_dislikes，别放这里。"
        ),
    )
    exclude_keywords: list[str] = Field(
        default_factory=list,
        description="硬排除词的扁平列表——**由系统从 exclude_terms 派生回填，模型不要填**",
    )
    soft_dislikes: list[str] = Field(
        default_factory=list,
        description=(
            "**软性避讳**的原子词（命中减分、**不淘汰**）：用户**本轮新说**的「不太喜欢 / "
            "尽量避免 / 能不要就不要」"
            "这类非绝对排斥（如「太花哨」「塑料感」）。绝对不要的放 exclude_keywords，别混。"
            "**同样优先给英文**（商品标题是英文）。"
        ),
    )
    prefer_keywords: list[str] = Field(
        default_factory=list,
        description=(
            "正向偏好的原子词（命中加分，**只填本轮新说的**）：材质 / 功能 / 做工填这里。"
            "**优先给会出现在英文商品标题里**"
            "的具象词**（「抗造」→ durable、「帆布」→ canvas、「防水」→ waterproof），"
            "而不是 niche / unique 这类抽象词——电商标题不会写 niche，给了也命中不了。"
            "抽象风格取向照样可以给（另有语义打分通道消费它），但别只给抽象词。"
        ),
    )
    retract_ids: list[str] = Field(
        default_factory=list,
        description=(
            "用户本轮**明确撤回 / 改口**的既有会话约束（「算了塑料也行」「不用非得蓝色」）：把"
            "【上一轮的会话状态】里该约束方括号中的 id（如 c2）**照抄**进来，不要自己编。"
            "只是补充新条件、没推翻旧的 → 留空。撤回的同时改成别的（「不要蓝色了改黑色」）→ "
            "旧 id 进这里，新的照常进三个词桶。"
        ),
    )

    @model_validator(mode="after")
    def _resolve_exclude_strength(self) -> PlanOutput:
        """**档位由机制判，不由模型判**：扫 evidence 里的弱表达标记，把该软的踢出硬排除。

        模型判「尽量别太花哨」算哪一档，实测约 1/4 的概率判错（把它硬归进排除），且错法有两种：
        要么两个桶都填，要么坚定填进 exclude、soft 里另填近义词——后者纯字面比对根本抓不住。
        所以不问模型「这算哪档」（它不稳），只问「用户在哪儿说的」（它稳），档位交给 _is_weak
        扫原话确定性地判。提示词里已经写过「弱表达放 soft_dislikes」，它照样混——**护栏要用机制
        兜，不能靠 prompt 求。**

        两道闸，按可靠性排序：
        1. evidence 带弱表达标记 → 降级（主闸，中英一起兜住：flashy 的 evidence 同样是那句中文）；
        2. 词同时出现在 soft_dislikes 里 → 降级（兜底闸，接住 evidence 缺失/照抄失败的漏网）。

        降级不是丢弃：词落进 soft_dislikes，仍然减分，只是不再淘汰。方向是安全的——硬淘汰误杀的
        代价，远大于软避讳漏放。
        """
        soft_lower = {w.strip().lower() for w in self.soft_dislikes if w.strip()}
        hard: list[str] = []
        for t in self.exclude_terms:
            word = t.word.strip()
            if not word:
                continue
            if _is_weak(t.evidence) or word.lower() in soft_lower:
                if word.lower() not in soft_lower:  # 降级进软桶，保序去重
                    self.soft_dislikes.append(word)
                    soft_lower.add(word.lower())
                continue
            hard.append(word)
        # exclude_keywords 是派生字段，这里**整个覆盖**。模型偶尔会无视「不要填」的说明直接填它
        # （老 schema 的惯性），那些词没有 evidence、逃得过上面的判断——但也不能直接扔掉（「不要
        # 塑料」整条硬约束凭空消失，用户明说的东西不该被静默吞掉）。降级进软桶：**不误杀，也不
        # 丢信息**，代价只是这一个词从淘汰变成减分。
        for w in self.exclude_keywords:
            word = w.strip()
            if word and word.lower() not in soft_lower and word not in hard:
                self.soft_dislikes.append(word)
                soft_lower.add(word.lower())
        self.exclude_keywords = hard
        return self

    @model_validator(mode="after")
    def _clean_bundle_slots(self) -> PlanOutput:
        """套装槽位的机制收口：去空名、按名去重、封顶 MAX_SLOTS；**不足 2 槽直接清空**。

        「是不是套装」由 ``len(bundle_slots) >= 2`` 这一个机制判据决定（下游 item_picker 据
        会话里有没有 ≥2 槽切组合模式），不设 is_bundle 布尔让模型另判一遍——单槽的「套装」
        就是普通单品类需求，留着只会让下游多一个半激活的歧义态。
        """
        seen: set[str] = set()
        cleaned: list[BundleSlot] = []
        for s in self.bundle_slots:
            name = s.name.strip()
            if not name or name in seen:
                continue
            seen.add(name)
            s.name = name
            # 槽 id 是机制发的身份（set_session_bundle 发号），模型无权自造——schema 里带
            # 这个字段只是复用模型（BundleSlot），这里一律清空，防幻觉 id 撞号/冒充既有槽。
            s.id = ""
            cleaned.append(s)
        self.bundle_slots = cleaned[:MAX_SLOTS] if len(cleaned) >= 2 else []
        return self


# 至少含一个「词字符」（字母 / 数字 / CJK）才算原子词。LLM 偶尔往桶里吐 "." "-" 这类纯标点
# token（真实评测抓到），而 term_hits 对非 ASCII-word 走子串路径——"." 对任何标题**永远命中**：
# 落 like 桶是全池均匀加分（无害但脏），落 exclude 桶就是整池屠杀。在入口挡掉。
_WORD_CHAR_RE = re.compile(r"[0-9a-zA-Z一-鿿぀-ヿ가-힯]")


def _atoms(words: list[str]) -> list[str]:
    """原子词清洗：小写、去空、剔纯标点、保序去重。P_t 的 terms 是拿去和商品标题做字符串匹配的。"""
    out: list[str] = []
    seen: set[str] = set()
    for w in words:
        low = w.strip().lower()
        if low and low not in seen and _WORD_CHAR_RE.search(low):
            seen.add(low)
            out.append(low)
    return out


def _plan_constraints(plan: PlanOutput, intent: str, turn: int) -> list[SessionConstraint]:
    """planner 的三个原子词桶 → P_t 的三条会话约束（**一桶一条**，不是一词一条）。

    档位记的是**用户自己的语气**（planner 做识别，不替用户授权）：exclude_keywords（「不要 X」）
    → ``blocking=True`` 命中即淘汰；soft_dislikes（「尽量别」）→ ``blocking=False`` 只减分——
    硬淘汰匹不准的词就是拿误杀去赌。

    **一桶一条**：一条约束的中英同义词（塑料 / plastic）属于同一件事，拆成两条 constraint 只会
    让 P_t 渲染给模型时出现两行几乎一样的噪声（「不要塑料」「不要plastic」）。keywords 里放全部
    原子词即可——下游 ``_terms()`` 本来就是摊平去重后拿去匹标题的。

    **不发 id**（跨轮身份由 ``merge_pt`` 统一发号，LLM/本函数都无权造）；``source_quote`` 取本轮
    原话前 60 字当锚——比逐词回溯 evidence 粗，但用途（撤回核验参照 / 面板溯源）只要「指得回
    那句话」即可。
    """
    specs = (
        ("dislike", True, plan.exclude_keywords),
        ("dislike", False, plan.soft_dislikes),
        # 正向永远 blocking=False：数据里没有可靠的材质 / 风格字段，正向做二值淘汰（keep-only）
        # 会误杀一大片。prefer_terms() 眼下不看 blocking，但别在这里留一颗等人踩的雷。
        ("like", False, plan.prefer_keywords),
    )
    out: list[SessionConstraint] = []
    for polarity, blocking, words in specs:
        atoms = _atoms(words)
        if not atoms:
            continue
        out.append(
            SessionConstraint(
                content=f"{constraint_verb(polarity, blocking)}{'、'.join(atoms)}",
                polarity=polarity,  # type: ignore[arg-type]
                keywords=atoms,
                category="other",
                source_quote=intent[:60],
                turn_added=turn,
                blocking=blocking,
            )
        )
    return out


def _sync_session_pt(plan: PlanOutput, intent: str, *, dest_stated_now: bool = False) -> None:
    """把 planner 刚识别出的本轮约束**当轮**写进 P_t —— 短期记忆的机制执行通路。

    **这是在补一条断掉的链。** 改造前 P_t 只由 curator 在**会话收尾之后**写，于是本轮 item_picker
    读到的 P_t 永远是上一轮的：用户这轮亲口说的「不要塑料」，机制侧一条都没执行，全靠主 loop 的
    模型自觉把它转述进 ``item_picker(exclude_keywords=...)``。也就是说，长期记忆有机制路（确定性
    并入 + 域闸），**短期记忆反倒没有**——而短期约束的明确性远高于长期推断。

    planner 本来就产出了这些结构化字段、也本来就在跑 LLM，这里零额外调用：识别完立刻落 P_t，
    item_picker 当轮就能硬执行（``pt.dislike_terms()`` → 淘汰，``pt.like_terms()`` → 强加分）。

    **``turn`` 在这里递增**（``prev.turn + 1`` 写回）：planner 现在是 P_t 的唯一写者，递增点
    自然随写权一起挪过来（曾归 curator——它退出 P_t 时这个缝差点漏掉）。``turn_added`` 语义
    不变：约束记的就是它落进 P_t 的那一轮号。
    """
    session_dir = get_session_dir()
    if session_dir is None:
        return  # 没会话（单测 / examples 直调工具）→ 无 P_t 可言，退化成纯拆解
    # 「还没有 P_t」的正确语义是**空 P_t**，不是「不写 P_t」——首轮本来就没有，正是要在这里开第一份。
    prev = get_session_pt() or SessionPrefState()
    turn = prev.turn + 1  # 本轮号：constraint.turn_added 与写回的 pt.turn 用同一个值
    merged = merge_pt(
        prev,
        _plan_constraints(plan, intent, turn),
        # LLM 只提议增量：撤回引用渲染文本里的 id（merge_pt 内有存在性 + 词面呼应双闸核验），
        # 换代与否由 retrieval 判据机制决定——四条不变量全在 merge_pt 的确定性代码里兑现。
        retract_ids=plan.retract_ids,
        user_utterance=intent,
        retrieval=plan.retrieval,
        budget_usd=plan.budget_usd,
        # 撤销权与产生权同归 planner（单写者）。机制闸：本轮原话里有落地的新预算数字时无视
        # clear_budget——「预算改成 500」被模型顺手多勾一个 clear 不该把新预算清掉；两者同真时
        # 新值为准，clear 只在「放开且没给新数」时生效（失效方向=宁松，预算误清可见可纠）。
        clear_budget=plan.clear_budget and plan.budget_usd is None,
        category=plan.category,
        current_intent=intent[:80],
        turn=turn,
        # slots 只固化「本轮带收货语境的明示」——线上事故教训：一次误判写进 slots 就毒化整个
        # 会话（第 2 层从此短路长期记忆）。第 2/3 层的值本就有各自的持久层，不需要再进 slots。
        slots_patch={DEST_COUNTRY_SLOT: plan.dest_country}
        if plan.dest_country and dest_stated_now
        else None,
    )
    # 双写：ContextVar 供本轮 item_picker 即时消费；pt.json 供续聊轮 load_pt 读回。planner 是
    # P_t 的唯一写者——curator 只读不写，没有第二个写者需要协调时序。
    set_session_pt(merged)
    save_pt(session_dir, merged)


def _render_pt_constraints(pt: SessionPrefState) -> list[str]:
    """把既有约束渲染成**带 id** 的紧凑列表——planner 撤回通道（retract_ids）的引用基准。

    形如 ``[c1] 硬排除：塑料、plastic（原话：「不要塑料的」）``。id 必须展示：LLM 撤回时只被
    允许**照抄**这里的 id（merge_pt 再核验），不展示它就只能凭空编。原话锚帮模型把「算了塑料
    也行」对到 c1 上，不必靠词形猜。
    """
    tag = {
        (True, True): "硬排除",
        (True, False): "软避讳",
        (False, True): "偏好",
        (False, False): "偏好",
    }
    out = []
    for c in pt.constraints:
        kind = tag[(c.polarity == "dislike", c.blocking)]
        quote = f"（原话：「{c.source_quote}」）" if c.source_quote else ""
        out.append(f"[{c.id}] {kind}：{'、'.join(c.keywords) or c.content}{quote}")
    return out


def _render_prior_context(*, sample: int = 5) -> str:
    """拼出 planner 判 ``retrieval``/``retract_ids`` 所需的上一轮上下文：旧意图 / 带 id 的旧约束
    + 手上候选的概貌。

    **从 ContextVar 与登记表自己读，不让主 loop 的模型转述**——转述既费 token 又会失真，而这些
    状态本来就在进程里躺着。没有会话上下文（单测 / examples）时返回空串，planner 退化成纯拆解。

    **不要求模型重放约束全集**（曾经的「快照替换」方案，已废弃）：一次漏吐 = 约束永久静默消失，
    存续必须不过 LLM 的手。这里只让它做两件小事：新说的进桶、撤回的抄 id。
    """
    pt = get_session_pt()
    cands = registry_snapshot()
    if not cands and (pt is None or pt.is_empty()):
        return ""
    lines: list[str] = ["【上一轮的会话状态（判 retrieval 用，不是本轮用户的话）】"]
    if pt is not None and not pt.is_empty():
        if pt.current_intent:
            lines.append(f"- 当前意图：{pt.current_intent}")
        if pt.budget_usd is not None:
            lines.append(f"- 预算：≤ ${pt.budget_usd:.0f}")
        if pt.category:
            lines.append(f"- 品类：{pt.category}")
        for k, v in pt.slots.items():
            lines.append(f"- {k}：{v}")
        if pt.constraints:
            lines.append("已累积约束（**不必重放**，系统自动带到下一轮）：")
            lines.extend(_render_pt_constraints(pt))
            lines.append(
                "→ 三个词桶只填用户**本轮新说**的；用户明确撤回 / 改口上面某条时，"
                "把它方括号里的 id 照抄进 retract_ids。"
            )
    if cands:
        titles = "；".join(c.title[:40] for c in cands[:sample])
        lines.append(f"手上已有候选 {len(cands)} 件（示例：{titles}）")
    else:
        lines.append("手上没有既有候选")
    lines.append("\n【本轮用户原话】")
    return "\n".join(lines) + "\n"


@tool
async def planner(intent: str) -> PlanOutput:
    """把用户的购物意图拆解成结构化字段（预算/品类/材质/风格/硬约束/软偏好 + 检索关键词）。

    何时调用：面对复杂多约束的购物需求时，先调本工具拆解，再据此检索与精挑。
    参数：
      - intent：用户的原始购物意图（自然语言）。
    """
    await monitor.report_tool_start("planner", intent=intent)
    prior = _render_prior_context()
    # 结构化输出把 AIMessage 吞成 Pydantic 对象，usage 只能经 callback 拿到；不挂就是漏账
    # （成本与预算闸都少算这一笔，见 token_budget.charge_tool_llm_usage）。
    usage_cb = UsageMetadataCallbackHandler()
    try:
        # method 必须显式钉死：默认值由 LangChain 的模型能力画像推断，qwen 系会被判成不支持
        # tools → 回退 response_format=json_object，而 DashScope 要求该模式下 messages 里必须
        # 出现 "json" 字样（本 prompt 没有）→ 400 直接打挂拆解。后台管理页能热更新 LLM_MAIN，
        # 不钉死就等于「随手换个模型就炸」（实测 qwen3.5-flash 触发，deepseek-v4-flash 不触发）。
        structured = get_fast_llm().with_structured_output(PlanOutput, method="function_calling")
        result = await structured.ainvoke(
            [("system", get_planner_prompt()), ("user", prior + intent if prior else intent)],
            config={"callbacks": [usage_cb]},
        )
        plan = result if isinstance(result, PlanOutput) else PlanOutput.model_validate(result)
    except Exception:
        # 模型调用失败也要补一条 end 事件，否则前端（M8）会看到工具「永远在跑」。
        await monitor.report_tool_end("planner", error=True)
        raise
    finally:
        # 放 finally：模型调用成功但下游解析抛错时，token 也已真实花掉，照样入账。
        charge_tool_llm_usage(usage_cb.usage_metadata)
    # 没有既有候选就没什么可复用/可合流的——无视模型的自由发挥，钉死 search。首轮判成 reuse 会让
    # 主 loop 直奔 item_picker 精挑一个空池子，产出空清单（和币种回填同一套「确定性兜底」思路）。
    if not registry_snapshot():
        plan.retrieval = "search"
    set_retrieval_mode(plan.retrieval)
    # 判了 search（换品类 / 新需求）→ **把上一轮读回的旧候选清掉**。planner 跑在本轮 item_search
    # 之前，此刻登记表里躺着的全是 load_candidates 从盘上读回的旧货——它们属于上一个品类，留着
    # 就是污染：下游按 id 捞候选时会把「上一轮的鞋」混进「这一轮的沙发」。
    #
    # 清在这里，是因为「这批旧候选算不算数」本来就只有 planner 判得了（它是 retrieval 判定的同一
    # 件事）。清干净之后，登记表**自身**就等于「本轮该考虑的全部候选」——下游因此不需要任何
    # 「哪些是本轮新召回的」这类额外账本，item_picker 直接吃登记表全集即可。
    #
    # 只主 loop 清（depth 0）：子 Agent 与主共享同一份会话登记表，若它也走到这儿，清掉的是**主**
    # 检索到一半的候选。prompt 本就规定子不调 planner，这里再上一道机制闸。
    if plan.retrieval == "search" and current_fork_depth() == 0:
        reset_candidates()
        # 套装槽位的生命周期 = 候选池生命周期：换品类 / 新需求时旧套装一起清（含盘上那份）。
        reset_session_bundle(clear_file=True)
    # 套装登记：validator 已收口成「≥2 槽或空」。只主 loop 写（同上 reset 的深度闸——子 Agent
    # 本就不该调 planner，真调了也不能让它覆盖主 loop 的套装定义）。
    if plan.bundle_slots and current_fork_depth() == 0:
        set_session_bundle(plan.bundle_slots)
    # 货币确定性：无视模型对 currency / budget_usd 的自由猜测，用规则解析币种 + fx 静态表折算回填。
    # 这是修「预算 500 每轮被猜成不同币种 → 预算内空召回退化」的关键一步（确定性，可复现）。
    code, explicit = resolve_budget_currency(intent)
    plan.currency = code
    plan.currency_assumed = not explicit
    # 预算落地闸：本轮原话里不存在这个数 → 模型是从上文（P_t 渲染的「≤ $80」）抄来的，置 None
    # 让 merge_pt 按「本轮未提及」沿用上一轮已折算好的 USD 预算——否则抄来的 80 会按本轮解析出的
    # 币种（默认 CNY）重折成 $11.2，预算悄悄缩水（见 budget_amount_grounded）。
    if plan.budget_amount is not None and not budget_amount_grounded(intent, plan.budget_amount):
        plan.budget_amount = None
    plan.budget_usd = to_base_or_none(plan.budget_amount, code, "USD")
    # 收货国确定性：同一套范式（规则解析 > 会话 slots > 长期记忆 > 默认国），模型同样无权自由填。
    # 收货国决定关税免征额（US $0 / CN $7 / AU $660，差两个数量级），判错整条到手价就废了。
    # 写进 ContextVar 供 shipping_calc 机制兜底——不指望模型每次都记得把参数传对。
    dest, assumed, dest_stated_now = await resolve_dest_country_layered(intent)
    plan.dest_country = dest
    plan.dest_country_assumed = assumed
    set_dest_country(dest, assumed)
    # 要推荐 → **一律补 landed_cost**（用户没开口也算到手价）。
    #
    # 跨境购物里用户真正想知道的数字是「寄到我这儿一共多少钱」，可他往往不会主动问——因为他不
    # 知道我们会算。于是最有价值的能力被藏在了「用户得先说出『到手价』三个字」后面，默认给出的
    # 是一个他还得自己心算运费关税的平台标价。
    #
    # 收货国缺失不是不算的理由：上面的四层解析保证它**永远有值**（兜底 DEFAULT_DEST_COUNTRY=CN），
    # 而 assumed 标记会让收尾文案讲明「按寄往中国估算，实际收货地不同请告诉我」——按默认国算完
    # 再告诉他口径，比让他先回答一句「你寄哪」多等一轮往返要好。
    #
    # 放在代码里而不是 prompt 里：planner 的模型侧纪律仍是「只填用户明确表达的 tasks」（否则它
    # 会顺手把 price_compare 也加上，把轻推荐拖成全流程）。「该不该替他算到手价」是产品决策，
    # 不该每轮重新指望模型判对——确定性回填，与币种 / 收货国同一套路子。
    if "recommend" in plan.tasks and "landed_cost" not in plan.tasks:
        plan.tasks.append("landed_cost")
    # 品类域：与币种 / 收货国不同，品类**无法纯规则解析**（「旅行三件套」映射到哪几个域是语义
    # 判断），故由 LLM 填、代码只做一道净化：剔掉 global——它是偏好侧「跨品类底线」的标记，不是
    # 「本轮什么都买」的意思，模型填了也不认（否则域集合里混个 global，等于把域隔离整个短路掉）。
    plan.domains = [d for d in plan.domains if d != DOMAIN_GLOBAL]
    # 域反证（badcase：手表 query 判成 apparel，prompt 反例已被证伪治不住）：用「用户原文 +
    # 主品类」的词面命中（DOMAIN_TERMS 高精度词表，独立于 LLM 的信号）核验，漏判的域**并入**。
    # 并入不替换——词表也可能误判：并入的失效方向是「多注入一个域的偏好」（软性减分/加分，
    # 可见可纠），替换的失效方向是「把 planner 判对的域丢了」（域隔离静默失效）。
    reconciled = reconcile_domains(plan.domains, f"{get_original_query()} {plan.category}".strip())
    if reconciled != plan.domains:
        logger.info("域反证：词面证据补入 %s（planner 判 %s）", reconciled, plan.domains)
        plan.domains = reconciled
    set_session_domains(plan.domains)
    # 任务清单同样落 session 级（同 retrieval / domains 的聚合方式）：阶段机的转移通告读它，
    # 在「无比价 / 到手价诉求」的轮次提示模型跳过 price_compare / shipping_calc——动机层提示，
    # 不是硬闸（COMPARING 阶段这两个工具仍然可用，用户中途改口还能调）。
    set_session_tasks(plan.tasks)
    # 本轮约束当轮落 P_t —— 短期记忆的机制执行通路（见 _sync_session_pt）。放在币种 / 收货国 /
    # 品类域全部确定性回填**之后**：P_t 要存的是这些回填后的最终值，不是模型的原始猜测。
    _sync_session_pt(plan, intent, dest_stated_now=dest_stated_now)
    # 约束集变化推给前端偏好面板（可见可纠）。只主 loop 推：子 Agent 的 planner 调用（prompt
    # 禁、机制上也不该）不该刷新用户面板。无会话（单测直调）时 _sync 没写 P_t，也就不推。
    if current_fork_depth() == 0 and (pt_now := get_session_pt()) is not None:
        await monitor.report_session_constraints(pt_now)
    # 给前端「思考过程」展开看的人读摘要：这一步把自然语言意图拆成了哪些结构化字段。
    plan_lines: list[str] = []
    if plan.retrieval != "search":
        plan_lines.append(
            {
                "reuse": "取候选：复用上一轮候选，不重新检索",
                "augment": "取候选：重新检索并与旧候选合流",
            }[plan.retrieval]
        )
    if plan.tasks:
        plan_lines.append("任务：" + "、".join(plan.tasks))
    if plan.target_refs:
        plan_lines.append("指定商品：" + "、".join(plan.target_refs))
    if plan.category:
        plan_lines.append(f"品类：{plan.category}")
    if plan.intent_grounding == "web":
        # 动机层提示（同 tasks 的路子，不设硬闸）：意图里有模型没把握的新说法 / 时效词，
        # 检索前先 web_search 翻译成品类词——此阶段 item_search 未跑，web_search 门控本就放行。
        plan_lines.append("意图接地：含新说法/时效词，建议检索前先 web_search 翻译成品类词")
    if plan.bundle_slots:
        slot_bits = [f"{s.name}({'必备' if s.essential else '可选'})" for s in plan.bundle_slots]
        bundle_line = "套装槽位：" + "、".join(slot_bits)
        # 有槽位是推断补的（用户没逐一点名）→ 明示出来：主 loop 据此决定要不要先 ask_user
        # 让用户对组成增删确认（「一套」的说法本就不唯一，别替用户拍板）。
        if any(not s.evidence.strip() for s in plan.bundle_slots):
            bundle_line += "（组成含推断项，用户未逐一点名——建议先与用户确认增删）"
        plan_lines.append(bundle_line)
    # 品类域摆进思考过程：它决定「哪些长期偏好本轮生效」，判错了用户得看得见——记忆最怕的就是
    # 静默失效（域判错 → 偏好没生效 → 用户只觉得「搜出来的东西不对」，却归因不到记忆头上）。
    plan_lines.append(
        "品类域：" + ("、".join(plan.domains) if plan.domains else "判不出（本轮全部偏好生效）")
    )
    if plan.budget_usd is not None:
        plan_lines.append(f"预算：≤ ${plan.budget_usd:.0f}")
    if "landed_cost" in plan.tasks:  # 只在要算到手价时显示，否则是噪音
        suffix = "（默认，用户未指定）" if plan.dest_country_assumed else ""
        plan_lines.append(f"收货国：{plan.dest_country}{suffix}")
    # 三个偏好桶按「会怎么影响结果」展示，而不是按「是什么维度」——前端「思考过程」里的用户
    # 关心的是「这条约束会淘汰商品还是只压排序」，材质 / 风格的分类对他没有意义。
    if plan.exclude_keywords:
        plan_lines.append("硬排除（命中即淘汰）：" + "、".join(plan.exclude_keywords))
    if plan.soft_dislikes:
        plan_lines.append("软避讳（命中减分）：" + "、".join(plan.soft_dislikes))
    if plan.prefer_keywords:
        plan_lines.append("偏好（命中加分）：" + "、".join(plan.prefer_keywords))
    if plan.keywords:
        plan_lines.append("检索词：" + "、".join(plan.keywords))
    await monitor.report_tool_end(
        "planner",
        category=plan.category,
        budget_usd=plan.budget_usd,
        currency=code,
        currency_assumed=plan.currency_assumed,
        result="\n".join(plan_lines),
    )
    return plan
