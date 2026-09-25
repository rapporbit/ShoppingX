"""planner —— 把购物意图拆成结构化字段。

复杂多约束的需求（「便宜又抗造的旅行三件套，预算 300，不要塑料，喜欢小众」）先过这一步，
拆成预算 / 品类 / 偏好等结构化字段，给后续检索、精挑当锚点。尤其是它顺手把自然语言偏好拆成
**三个原子词桶**——``exclude_terms``（硬淘汰）/ ``soft_dislikes``（减分）/ ``prefer_keywords``
（加分）——让下游的 ``item_search`` 和 ``item_picker`` 直接吃结构化入参，不必各自再解析一遍
自然语言。

**planner 是本轮约束 P_t 的唯一写者，且无状态。** 它每轮都跑，输入是「前几轮用户原话 + 本轮
原话」，**整体重算**本轮仍生效的全部约束（当轮落进 P_t、当轮被 item_picker 执行）。撤回、换品类
都随重算自然生效，没有跨轮合并逻辑（2026-09-25 删，见 :mod:`app.memory.turn_constraints`）。
curator 只判长期库，不碰 P_t——双写者时代它曾用更差的输出覆盖 planner（无档位 draft、脑内换汇），
教训见 ``app.memory.curator``。

**档位（blocking）由机制判，不由模型判**：模型判「尽量别太花哨」算硬排除还是软避讳，实测约 1/4
的概率判错，但它**转述用户原话**是稳的。所以 ``exclude_terms`` 的每个词都必须附 evidence（原话
片段），档位交给 :func:`_is_weak` 扫原话里的弱表达标记确定性地判。见 :class:`ExcludeTerm`。

用 LLM 做意图理解（结构化输出强约束成 Pydantic）。走 ``get_planner_llm``：结构化抽取不吃思维
链——实测主档 12~17s（reasoning 占 70%）vs 快档 ~3s，tasks / 排除词 / 预算解析产出一致
（对照见 latency-audit round2）。

**这一档可以单独换模型**（``LLM_PLANNER``，不配即快档）：planner 是链路外的一次性调用，有自己
的 prompt 前缀，换模型名不打断主 loop 的前缀缓存——主 loop 内换名会，所以那里只能切 thinking。
选型实测见 `docs/plans/baseline-artifacts/planner_model_eval.json`（dev92 + planner_reward）。
模块级引用便于测试 monkeypatch 成假模型，从而离线可测。
"""

from __future__ import annotations

import logging
import os
import re
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.agent.invoke import call_structured
from app.agent.llm import get_planner_llm
from app.agent.prompts import get_planner_prompt
from app.api import monitor
from app.api.context import (
    get_prior_queries,
    get_session_dir,
    get_user_id,
    set_dest_country,
    set_session_tasks,
    set_turn_constraints,
)
from app.memory.turn_constraints import TurnConstraints
from app.recall.fx import to_base_or_none
from app.recall.geo import (
    DEFAULT_DEST_COUNTRY,
    match_country_name,
    resolve_dest_country,
)
from app.tools._args import drop_none_values
from app.tools._bundle import (
    MAX_SLOTS,
    SLOT_MODE_BUNDLE,
    SLOT_MODE_PARALLEL,
    BundleSlot,
    reset_session_bundle,
    set_session_bundle,
)
from app.tools._shell import tool

logger = logging.getLogger("shoppingx.planner")

# 用户本轮想让 Agent 做的事（意图信号，驱动 <workflow> 按需组合能力，而非走死一条链）：
#   recommend      —— 挑 / 推荐商品（要检索 + 精挑）
#   evaluate       —— 评某商品好不好（品类基准 + 评分 + 口碑）
#   price_compare  —— 跨平台比价
#   landed_cost    —— 算关税 + 运费（到手价）
#   category_intel —— 只问品类行情（热卖 / 价位 / 该看哪些维度），不一定要具体商品
#   place_order / query_order / cancel_order —— 交易意图（批 1 的交易域）。它们与前五个正交：
#     检索类任务判的是「要给什么」，交易类判的是「要动哪张单」，一轮里可以只有后者（「我的订单
#     呢」不需要任何检索）。
ShoppingTask = Literal[
    "recommend",
    "evaluate",
    "price_compare",
    "landed_cost",
    "category_intel",
    "place_order",
    "query_order",
    "cancel_order",
]

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
    """模型填的 ``budget_amount`` 是否真在这句原话里出现过（机制闸，防编造 / 防错配币种）。

    规则核对：原话里找得到这个数（含 "1,000" 逗号形式与 k/千/万量级缩写）才算「这句提过」；
    原话没有任何数字也没有中文数词 → 判「没提」；含中文数词（「预算三百」）无法确定性核对，
    放行模型的值——宁可放过，不误删真预算。调用方见 :func:`budget_source`。
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


def budget_source(amount: float, utterances: list[str]) -> str | None:
    """找出 ``amount`` 出自哪句原话（新 → 旧），币种按**那句**解析；都找不到返回 None。

    预算可能是前几轮说的（「预算 80 美元」→ 本轮「不要皮革的」）。币种若按本轮原话解析，没提
    币种就落默认 CNY，80 美元被当 80 人民币折成 $11.2，预算悄悄缩水 7 倍（旧版真实 e2e 复现）。
    所以先认数字确实对得上的那句；中文数词那条放行路径只作后备，否则「换一个」里的「一」会抢先。
    """
    for u in reversed(utterances):
        if _NUM_RE.search(u) and budget_amount_grounded(u, amount):
            return u
    for u in reversed(utterances):
        if not _NUM_RE.search(u) and budget_amount_grounded(u, amount):
            return u
    return None


async def resolve_dest_country_layered(text: str) -> tuple[str, bool]:
    """四层确定性决定本轮收货国，返回 ``(ISO 码, 是否为「假设值」)``。

    优先级（高 → 低），**任一层命中即停**：
    1. 本轮用户明说（「寄到日本」）—— 纯规则解析，见 :func:`app.recall.geo.resolve_dest_country`。
    2. 前几轮用户原话（新 → 旧）里的明示 —— 本会话说过一次，后面一直生效。
    3. 长期记忆里 key 为 ``default_ship_to`` 的事实 —— 跨会话记住常用收货地。
    4. env ``DEFAULT_DEST_COUNTRY`` 默认值。

    只有走到第 4 层才算「假设」（返回 ``assumed=True``）——前三层都有用户依据，不该每轮都去
    骚扰他确认；标 assumed 的目的只是提醒模型「这是系统替你猜的，得在回复里讲明」。
    """
    country, explicit = resolve_dest_country(text)
    if explicit:  # 第 1 层：本轮原话（门控明示）
        return country, False

    for prior in reversed(get_prior_queries()):  # 第 2 层：前几轮原话
        country, explicit = resolve_dest_country(prior)
        if explicit:
            return country, False

    user_id = get_user_id()  # 第 3 层：长期记忆（跨会话常用收货地）
    if user_id:
        try:
            from app.memory.fact_store import get_fact_store
            from app.memory.facts import SHIP_TO_KEY, memory_enabled

            for fact in (await get_fact_store().get_facts(user_id)) if memory_enabled() else []:
                if fact.key == SHIP_TO_KEY:
                    # key 本身已确定这条讲的是收货地，用无门控匹配——「常用收货地：中国」
                    # 若再要求语境词反而可能漏掉。M2 起按 key 取，不再按已废的 category/polarity。
                    code = match_country_name(fact.value)
                    if code:
                        return code, False
                    break  # 有这条但解析不出国家（写成「欧洲」之类）→ 不再找别条，退默认
        except Exception:  # noqa: BLE001 —— 记忆后端挂了不该崩掉 planner，降级到默认国即可
            pass

    return DEFAULT_DEST_COUNTRY, True  # 第 4 层：系统默认 → 必须在回复里标注假设


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
            "交易类（与上面正交，可单独出现）：说「买/下单/就要这个」→ place_order；"
            "说「我的订单/那单怎么样」→ query_order；说「取消/不要了」→ cancel_order。"
        ),
    )
    category: str = Field(default="", description="主品类：用户本轮要买的那类东西（中文品类名）")
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
    topic_switch: bool = Field(
        default=False,
        description=(
            "本轮要买的东西是否换成了**另一类商品**（对照【前几轮用户原话】）："
            "双肩包 → 颈枕、耳机 → 沙发 = true；"
            "双肩包 → 更轻的双肩包、加预算、换平台、追问比较 = false。"
            "没有前几轮原话时填 false。true 时上一轮的套装槽位会被清掉，别轻易填。"
        ),
    )

    budget_amount: float | None = Field(
        default=None,
        description=(
            "当前仍生效的预算金额（本轮或前几轮原话里给的，以最新一次为准；**不要换算**，"
            "照原数填），无则 None"
        ),
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
            "用户**明确取消 / 放开**了预算（「算了不限预算」「贵点也行，不设上限」「直接上"
            "最好的别管价格」），且之后没再给新预算 → true。只是没提预算 → false。"
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
            "**本轮要买的东西跨了 2~6 个不同品类**时把它拆成槽位（name 中文槽名 / keywords "
            "英文检索词 / prefer 槽级偏好词 / essential 少了它这套是否就不成立）。两种情形都要拆"
            "（哪一种由 slot_mode 说明）：①「一套齐」配套需求（「新生入学一套」「旅行三件套」）；"
            "② 一次点名几类**互不相干**的东西（「想买双跑鞋，再配个降噪耳机」）。"
            "**单品类需求一律留空**（哪怕买多件同类）。"
            "**每槽附 evidence**：用户原话点名了这件就照抄那个片段；是你按常识推断补的就留空——"
            "系统据此判断「组成要不要先跟用户确认」。"
        ),
    )
    slot_mode: str = Field(
        default="bundle",
        description=(
            "bundle_slots 非空时才有意义，二选一：\n"
            "- bundle：用户要的是**配套的一整套**，各件互相搭配、共享一个总预算、锦上添花的那件"
            "预算紧时可以砍（「新生入学一套 1500」「露营装备一套」）。\n"
            "- parallel：用户一次要看**几类互不相干**的东西，各类各自推荐、不配套、也不许砍掉"
            "任何一类（「想买双跑鞋，再配个降噪耳机」「帮我看看猫粮和一个加湿器」）。\n"
            "判据是「少了其中一件，剩下的还成立吗」：一套床品缺了被子就不成套 → bundle；"
            "跑鞋和耳机各买各的、缺一个另一个照样有用 → parallel。拿不准填 parallel"
            "（并列形态不会替用户砍掉任何一类，判错的代价更小）。"
        ),
    )
    keywords: list[str] = Field(default_factory=list, description="检索关键词，供 item_search")
    exclude_terms: list[ExcludeTerm] = Field(
        default_factory=list,
        description=(
            "**绝对排除**的原子词（命中即淘汰）：用户说过的「不要 X / 不能 X」里的那个 X"
            "（本轮和前几轮说过、且没被撤回的都要给）。\n"
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
            "**软性避讳**的原子词（命中减分、**不淘汰**）：用户说过且仍生效的「不太喜欢 / "
            "尽量避免 / 能不要就不要」"
            "这类非绝对排斥（如「太花哨」「塑料感」）。绝对不要的放 exclude_keywords，别混。"
            "**同样优先给英文**（商品标题是英文）。"
        ),
    )
    prefer_keywords: list[str] = Field(
        default_factory=list,
        description=(
            "正向偏好的原子词（命中加分，本轮和前几轮说过且仍生效的都要给）："
            "材质 / 功能 / 做工填这里。"
            "**优先给会出现在英文商品标题里**"
            "的具象词**（「抗造」→ durable、「帆布」→ canvas、「防水」→ waterproof），"
            "而不是 niche / unique 这类抽象词——电商标题不会写 niche，给了也命中不了。"
            "抽象风格取向照样可以给（另有语义打分通道消费它），但别只给抽象词。"
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
        """槽位的机制收口：去空名、按名去重、封顶 MAX_SLOTS；**不足 2 槽直接清空**。

        「本轮是不是槽位轮」由 ``len(bundle_slots) >= 2`` 这一个机制判据决定（下游 item_picker
        据会话里有没有 ≥2 槽切分组模式），不设 is_bundle 布尔让模型另判一遍——单槽的「套装」
        就是普通单品类需求，留着只会让下游多一个半激活的歧义态。

        形态（``slot_mode``）另收两道口：① 非法值一律落 bundle（保持既有行为）；② parallel 下
        ``essential`` 强制 True——并列需求里「可选」这个概念不存在，用户点名的每一类都得给交代，
        留着 False 只会让报告把某一类讲成「已放弃的可选项」。
        """
        seen: set[str] = set()
        cleaned: list[BundleSlot] = []
        for s in self.bundle_slots:
            name = s.name.strip()
            if not name or name in seen:
                continue
            seen.add(name)
            s.name = name
            cleaned.append(s)
        self.bundle_slots = cleaned[:MAX_SLOTS] if len(cleaned) >= 2 else []
        if self.slot_mode not in (SLOT_MODE_BUNDLE, SLOT_MODE_PARALLEL):
            self.slot_mode = SLOT_MODE_BUNDLE
        if self.slot_mode == SLOT_MODE_PARALLEL:
            for s in self.bundle_slots:
                s.essential = True
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


#: planner 回看的前几轮用户原话条数（不含本轮）。更早的约束靠用户重说——窗口越大，planner
#: 输入越长、「旧约束被重新抽出来」的判断也越难。
PRIOR_QUERY_WINDOW = 6


def _sync_turn_constraints(plan: PlanOutput) -> None:
    """把 planner 重算出的本轮约束**当轮**写进 P_t —— 约束的机制执行通路。

    P_t 不靠模型每轮把「不要塑料」转述进 ``item_picker(exclude_keywords=...)``：planner 识别完
    立刻落 P_t，item_picker 当轮就能硬执行（``dislike_terms()`` 淘汰、``like_terms()`` 加分）。
    整体覆盖、不与上一轮合并：planner 的输入本来就含前几轮原话，产出即全集。
    """
    if get_session_dir() is None:
        return  # 没会话（单测 / examples 直调工具）→ 无 P_t 可言，退化成纯拆解
    set_turn_constraints(
        TurnConstraints.build(
            category=plan.category,
            budget_usd=plan.budget_usd,
            exclude=_atoms(plan.exclude_keywords),
            avoid=_atoms(plan.soft_dislikes),
            prefer=_atoms(plan.prefer_keywords),
        )
    )


def _render_prior_context(prior: list[str]) -> str:
    """把前几轮用户原话拼在本轮原话前面，供 planner 重算仍生效的约束。首轮返回空串。

    只给**用户原话**，不给上一轮的结构化结果：让 planner 从原话重算，撤回 / 改口 / 换品类都在
    原话里，不需要另设撤回字段；喂结构化结果则等于把上一轮的误判原样带下去。
    """
    if not prior:
        return ""
    lines = "\n".join(f"{i}. {q}" for i, q in enumerate(prior, 1))
    return (
        "【前几轮用户原话（旧 → 新，判断依据，不是本轮的话）】\n"
        + lines
        + "\n→ 输出本轮**仍生效**的全部约束：前几轮说过、本轮没撤回的照样要给；本轮撤回 / 改口的"
        "不再给；换成另一类商品时，前一类的材质 / 风格偏好不带过来，预算与收货国照旧。\n"
        "→ **品类（category）先看本轮原话**：本轮点名了要买的东西（「推荐一个保温杯」），category"
        " 就是它，与前几轮不同则 topic_switch=true；本轮只改条件、没点名东西（「商务一点的」"
        "「便宜点」），才沿用最近一轮的品类。\n"
        "\n【本轮用户原话】\n"
    )


@tool
async def planner(intent: str) -> PlanOutput:
    """把购物意图拆成结构化字段（预算/品类/硬约束/软偏好/检索词）；系统已开局预跑，通常不必再调。
    参数 intent：用户原话。
    """
    await monitor.report_tool_start("planner", intent=intent)
    prior_queries = get_prior_queries()[-PRIOR_QUERY_WINDOW:]
    prior = _render_prior_context(prior_queries)
    try:
        # 曾经这里要显式钉 ``method="function_calling"``——旧运行时按模型能力画像推断默认
        # method，qwen 系被判成不支持 tools → 回退 response_format=json_object，而 DashScope
        # 要求该模式下 messages 里必须出现 "json" 字样（本 prompt 没有）→ 400 直接打挂拆解。
        # AgentScope 的 generate_structured_output 自带策略梯（forced→auto→no_think→none），
        # 这个坑结构上不存在，也没有 method 可钉（实测）。用量由 call_structured 入账。
        plan = await call_structured(
            get_planner_llm(),
            [("system", get_planner_prompt()), ("user", prior + intent if prior else intent)],
            PlanOutput,
            # 空表闸：这三个字段一个都没出现 = 模型只回了存根（PlanOutput 全字段带默认值，
            # model_validate 照过），拆解结果恒定为空、下游直奔空池子，且全程零报错。它们是
            # 「本轮到底要买什么」的唯一载体——单品类走 category / keywords、跨品类走
            # bundle_slots，一个都没有就没有任何可执行的意图。重采样一次仍空则抛，走下面的
            # except 补 end 事件后外抛，由工具外壳转成 [error] + ERROR 让主 loop 看得见。
            required_any=("category", "keywords", "bundle_slots"),
        )
    except Exception:
        # 模型调用失败也要补一条 end 事件，否则前端（M8）会看到工具「永远在跑」。
        await monitor.report_tool_end("planner", error=True)
        raise
    # 预算落地闸：这个数在本轮和前几轮原话里都找不到 → 模型编的，置 None。找得到就记下出处，
    # 币种按**出处那句**解析（见 budget_source）。放开预算（clear_budget）只在本轮没给新数时生效。
    source = None
    if plan.budget_amount is not None:
        source = budget_source(plan.budget_amount, [*prior_queries, intent])
        if source is None or (plan.clear_budget and source != intent):
            plan.budget_amount, source = None, None
    # 货币确定性：无视模型对 currency / budget_usd 的自由猜测，用规则解析币种 + fx 静态表折算回填。
    # 这是修「预算 500 每轮被猜成不同币种 → 预算内空召回退化」的关键一步（确定性，可复现）。
    code, explicit = resolve_budget_currency(source if source is not None else intent)
    plan.currency = code
    plan.currency_assumed = not explicit
    plan.budget_usd = to_base_or_none(plan.budget_amount, code, "USD")
    # 收货国确定性：同一套范式（规则解析 > 前几轮原话 > 长期记忆 > 默认国），模型同样无权自由填。
    # 收货国决定关税免征额（US $0 / CN $7 / AU $660，差两个数量级），判错整条到手价就废了。
    # 写进 ContextVar 供 shipping_calc 机制兜底——不指望模型每次都记得把参数传对。
    dest, assumed = await resolve_dest_country_layered(intent)
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
    # 任务清单落 session 级：收线通告读它，
    # 在「无比价 / 到手价诉求」的轮次提示模型跳过 price_compare / shipping_calc——动机层提示，
    # 不是硬闸（这两个工具始终可用，用户中途改口还能调）。
    set_session_tasks(plan.tasks)
    # 换了一类商品（旅行套装 → 沙发）时旧槽表清掉。
    if plan.topic_switch:
        reset_session_bundle()
    if plan.bundle_slots:  # validator 已收口成「≥2 槽或空」
        set_session_bundle(plan.bundle_slots, mode=plan.slot_mode)
    # 本轮约束当轮落 P_t —— 约束的机制执行通路（见 _sync_turn_constraints）。放在币种确定性回填
    # **之后**：P_t 要存的是回填后的最终值，不是模型的原始猜测。
    _sync_turn_constraints(plan)
    # 给前端「思考过程」展开看的人读摘要：这一步把自然语言意图拆成了哪些结构化字段。
    plan_lines: list[str] = []
    if plan.tasks:
        plan_lines.append("任务：" + "、".join(plan.tasks))
    if plan.category:
        plan_lines.append(f"品类：{plan.category}")
    if plan.intent_grounding == "web":
        # 动机层提示（同 tasks 的路子，不设硬闸）：意图里有模型没把握的新说法 / 时效词，
        # 检索前先 web_search 翻译成品类词——此阶段 item_search 未跑，web_search 门控本就放行。
        plan_lines.append("意图接地：含新说法/时效词，建议检索前先 web_search 翻译成品类词")
    if plan.bundle_slots:
        if plan.slot_mode == SLOT_MODE_PARALLEL:
            # 并列形态不问组成：用户已经自己点名了要看哪几类，再拿一轮 ask_user 去确认
            # 「你是不是要这几类」纯属浪费一次往返。「一套齐」才有「这套包含什么」的歧义。
            plan_lines.append(
                "并列子需求（各类独立、分头检索、不砍类）："
                + "、".join(s.name for s in plan.bundle_slots)
            )
        else:
            slot_bits = [
                f"{s.name}({'必备' if s.essential else '可选'})" for s in plan.bundle_slots
            ]
            bundle_line = "套装槽位：" + "、".join(slot_bits)
            # 有槽位是推断补的（用户没逐一点名）→ 明示出来：主 loop 据此决定要不要先 ask_user
            # 让用户对组成增删确认（「一套」的说法本就不唯一，别替用户拍板）。
            if any(not s.evidence.strip() for s in plan.bundle_slots):
                bundle_line += "（组成含推断项，用户未逐一点名——建议先与用户确认增删）"
            plan_lines.append(bundle_line)
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
