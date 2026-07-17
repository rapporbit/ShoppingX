"""匹配词归一：把任何来源的原子词收敛成**英文小写**，再送去匹商品标题。

**为什么必须有这一层。** 商品库是纯英文的（实测 amazon 样本 300 条标题 0 条含中文），而
``item_picker`` 的硬淘汰 / 加分 / 减分全靠**字符串命中**（本模块的 :func:`term_hits`：词边界 +
否定修饰过滤，但本质是子串）。于是一条中文原子词（「塑料」）去匹英文标题（"Plastic Packing
Cubes"）**永远命中不了**——用户说的「不要塑料」在机制层是空转的。

这不是理论风险，是实测出来的两处静默失效：

1. **planner 侧靠模型自觉**：它这轮恰好产出了中英双份（``exclude_keywords=['塑料','plastic']``），
   真正干活的是 ``plastic``。哪一轮它忘了给英文变体，硬约束就静默失效——没有任何地方会报错。
2. **记忆侧根本没人兜**：curator 从中文对话里抽的原子词（「塑料」「皮革」）原样存进 Store，
   ``assemble`` 原样读出来并进 ``exclude``——这条路上**从来没有**英文变体，一直在空转。

按项目一贯原则（prompt 打动机、机制打保证），跨语言不能指望模型每轮自觉。**词的归一必须发生在
匹配之前，且是确定性的。**

**为什么不给 exclude 加语义通道**（用 embedding 判「这件是不是塑料的」）：硬淘汰误杀是**看不见
的**——用户不知道有商品被删了，只觉得「怎么搜不出东西」。语义阈值做二值淘汰不可靠（见
``item_search`` 的 ``RELEVANCE_FLOOR`` 实测：BGE-M3 给任何真实英文 query 都打 ≥0.48，三段分布重叠，
单一阈值分不开）。正向偏好（prefer / must / penalty）另有语义打分通道兜着，唯独 exclude 只认关键词
——那就把关键词修对，而不是给它换一套不可靠的判据。

**归一是「扩展」，不是「替换」——中文原词一律保留。** 中文词映射出英文之后，原词仍留在表里：
库里 99% 是英文标题，但 shein / lazada 这些平台掺着中文标题，把中文词换掉等于反手在那些商品上
制造出同一个空转（实测：改成替换后，`塑料收纳盒` 就不再被「塑料」淘汰了）。命中任一即命中，多留
一个词的匹配成本可以忽略，而少留一个词是**看不见的漏挡**。

**归一策略（三级，失效方向安全）：**

1. 已是 ASCII 词 → 原样小写，**再反向补出词表内的中文变体**（≥2 字，见 :data:`EN_ZH_ALL`）——
   长期库沉淀的约束可能只有英文，掺中文标题的平台不能因此漏挡；
2. 中文词 → 保留原词，**再补上** :data:`ZH_EN` 词表里的英文（可能一对多：「防水」→ waterproof /
   water-resistant，标题写法不统一，都要能命中）；
3. 词表里查不到 → 只保留原词并记一条日志。保留而非丢弃：丢了等于静默删掉一条用户约束；留着虽然
   匹不中英文库，但语义打分那条通路仍在用它（prefer / penalty 走 ``sem_*``）。日志是给我们看的
   ——「有多少词归一不掉」是这层该被持续观察的健康指标。
"""

from __future__ import annotations

import logging
import os
import re
from typing import Literal

logger = logging.getLogger("shoppingx.terms")

# 中文 → 英文匹配词。**只收「会出现在商品标题里」的词**：材质、工艺、功能、品类属性。
# 抽象风格词（「小众」「高级感」）刻意不收——电商标题不会写 niche，硬映射只会制造假命中；
# 它们本就该走语义打分通道（sem_match），不该指望字符串命中。
ZH_EN: dict[str, tuple[str, ...]] = {
    # 材质
    "塑料": ("plastic", "plastics"),
    "皮革": ("leather",),
    "真皮": ("genuine leather", "leather"),
    "人造革": ("faux leather", "pu leather"),
    "帆布": ("canvas",),
    "尼龙": ("nylon",),
    "涤纶": ("polyester",),
    "聚酯": ("polyester",),
    "棉": ("cotton",),
    "麻": ("linen",),
    "羊毛": ("wool",),
    "丝": ("silk",),
    "金属": ("metal", "metallic"),
    "不锈钢": ("stainless steel", "stainless"),
    "铝": ("aluminum", "aluminium"),
    "钢": ("steel",),
    "木": ("wood", "wooden"),
    "竹": ("bamboo",),
    "玻璃": ("glass",),
    "陶瓷": ("ceramic",),
    "硅胶": ("silicone",),
    "橡胶": ("rubber",),
    "碳纤维": ("carbon fiber",),
    "镍": ("nickel",),
    "乳胶": ("latex",),
    "羽绒": ("down",),
    "网布": ("mesh",),
    "牛津布": ("oxford",),
    # 功能 / 属性
    "防水": ("waterproof", "water-resistant"),
    "防泼水": ("water-repellent", "water-resistant"),
    "耐用": ("durable", "heavy duty"),
    "抗造": ("durable", "heavy duty"),
    "结实": ("durable", "sturdy"),
    "轻便": ("lightweight",),
    "便携": ("portable",),
    "折叠": ("foldable", "folding"),
    "可压缩": ("compression", "compressible"),
    "大容量": ("large capacity", "spacious"),
    "透气": ("breathable",),
    "保暖": ("thermal", "insulated"),
    "防盗": ("anti-theft",),
    "无线": ("wireless",),
    "降噪": ("noise cancelling", "noise-canceling"),
    "充电": ("rechargeable",),
    "无香": ("fragrance-free", "unscented"),
    "有机": ("organic",),
    "免烫": ("wrinkle-free",),
}


# 反查表：英文匹配词 → 中文展示词。命中的往往是英文（canvas / durable），但理由是给中文用户看的，
# 直接把 "契合偏好「durable」" 摆到卡片上是让用户替我们做翻译。取词表里第一个映射到它的中文词。
EN_ZH: dict[str, str] = {}

# 反向扩词表：英文匹配词 → 映射到它的**全部**中文词（仅 ≥2 字）。normalize_terms 用它做英→中
# 扩展——ZH_EN 单向（中→英）时英文约束对掺中文标题的平台（shein / lazada）是反向空转：长期库
# 沉淀的约束可能只有英文（curator 只硬性要求英文），「no plastic」挡不住「塑料收纳盒」。
# 单字中文词（丝 / 钢 / 木…）刻意不进反查：中文命中是子串语义，单字误伤面太大
# （"silk" → 「丝」会命中「螺丝刀」）。
EN_ZH_ALL: dict[str, tuple[str, ...]] = {}
for _zh, _ens in ZH_EN.items():
    for _en in _ens:
        EN_ZH.setdefault(_en, _zh)
        if len(_zh) >= 2:
            EN_ZH_ALL[_en] = (*EN_ZH_ALL.get(_en, ()), _zh)


def display_term(term: str) -> str:
    """把一个匹配词转成给用户看的样子：英文有中文对照就换成中文，否则原样。"""
    return EN_ZH.get(term.strip().lower(), term)


# 商品标题里**真实会出现**的属性 token → 中文标签。用于从标题确定性抽出「这件到底是什么」——
# 电商标题是高度模式化的（"7-Piece Waterproof Nylon Packing Cubes"），抽出来直接就是一条硬事实，
# 不需要任何模型参与。抽不到就不写，绝不臆造（P0 诚实红线）。
TITLE_ATTRS: tuple[tuple[str, str], ...] = (
    # 材质（放前面：用户最常约束的就是材质）
    ("canvas", "帆布"),
    ("genuine leather", "真皮"),
    ("faux leather", "人造革"),
    ("pu leather", "PU 皮"),
    ("leather", "皮革"),
    ("nylon", "尼龙"),
    ("polyester", "涤纶"),
    ("cotton", "纯棉"),
    ("linen", "亚麻"),
    ("wool", "羊毛"),
    ("bamboo", "竹制"),
    ("stainless steel", "不锈钢"),
    ("aluminum", "铝制"),
    ("silicone", "硅胶"),
    ("ceramic", "陶瓷"),
    ("mesh", "网布"),
    ("oxford", "牛津布"),
    ("plastic", "塑料"),
    # 功能
    ("waterproof", "防水"),
    ("water-resistant", "防泼水"),
    ("water resistant", "防泼水"),
    ("heavy duty", "耐造"),
    ("durable", "耐用"),
    ("lightweight", "轻便"),
    ("compression", "可压缩"),
    ("foldable", "可折叠"),
    ("collapsible", "可折叠"),
    ("anti-theft", "防盗"),
    ("breathable", "透气"),
    ("insulated", "保温"),
    ("rechargeable", "可充电"),
    ("noise cancelling", "降噪"),
    ("noise-canceling", "降噪"),
    ("wireless", "无线"),
    ("organic", "有机"),
)

# 「N 件套」：7-Piece / 6 Set / 8 Pcs / 7件套 都是同一个意思，正则一网打尽。件数是套装类商品最有
# 用的一条硬信息（用户问「三件套」时尤其），而它恰恰不在任何结构化字段里，只在标题里。中文写法
# 单列一个捕获组（\b 对 CJK 无意义），「件装」也收——标题两种写法混用。
_COUNT_RE = re.compile(
    r"\b(\d{1,2})\s*(?:-|\s)?\s*(?:pcs?|pieces?|piece|sets?)\b|(\d{1,2})\s*件\s*[套装]", re.I
)

# 标题属性 token 的中文变体（≥2 字，从 ZH_EN 反查）：shein / lazada 掺中文标题，「防水尼龙包」里
# 一个英文 token 都没有——抽取侧不认中文，这些商品的属性标签与亲和证据就永远是空的（匹配侧早为
# 中文标题保留了中文原词，抽取侧此前一直英文单语）。每个中文词只归到它 ZH_EN 映射里**第一个**
# 在 TITLE_ATTRS 的英文 token 名下（「防水」→ waterproof 而不再重复记给 water-resistant），
# 否则一处中文命中被数成两个 token，亲和词频跨语言分裂又虚高。
_TITLE_ATTR_TOKENS = frozenset(t for t, _ in TITLE_ATTRS)
_TITLE_ATTR_ZH: dict[str, tuple[str, ...]] = {}
for _zh, _ens in ZH_EN.items():
    if len(_zh) < 2:
        continue
    _hit_en = next((e for e in _ens if e in _TITLE_ATTR_TOKENS), None)
    if _hit_en is not None:
        _TITLE_ATTR_ZH[_hit_en] = (*_TITLE_ATTR_ZH.get(_hit_en, ()), _zh)


def _attr_hit(token: str, text: str) -> bool:
    """标题是否含此属性：英文 token 或其中文变体，一律走 :func:`term_hits`。

    裸子串在这里踩过两次坑（"vegan leather-free" 被数成皮革证据），抽取侧与匹配侧必须同一套
    命中口径。"""
    return term_hits(token, text) or any(
        term_hits(zh, text) for zh in _TITLE_ATTR_ZH.get(token, ())
    )


def extract_attrs(title: str, limit: int = 3) -> list[str]:
    """从商品标题里确定性抽出属性标签（中文），最多 ``limit`` 个。抽不到返回空列表。

    只认词表里的词与「N 件套」这一个数量模式——宁可少抽，不可臆造：这些标签会原样出现在给用户的
    选购理由里，编一个「防水」出来比不写任何属性糟糕得多。
    """
    text = (title or "").lower()
    out: list[str] = []
    m = _COUNT_RE.search(text)
    if m:
        out.append(f"{int(m.group(1) or m.group(2))} 件套")
    for token, label in TITLE_ATTRS:
        if len(out) >= limit:
            break
        if label not in out and _attr_hit(token, text):
            out.append(label)
    return out[:limit]


def title_attr_tokens(title: str) -> list[str]:
    """从标题抽出 :data:`TITLE_ATTRS` 里的**英文 token**（不封顶、不翻译）。

    与 :func:`extract_attrs` 同源不同用途：那个抽中文标签给**人**看（进选购理由，故限量 3 个、
    含「N 件套」这类数量模式）；这个抽英文 token 给**机器**用（进 :mod:`app.memory.affinity` 的
    词频统计与后续的标题匹配），因此不限量、不掺数量模式——数量不是口味偏好，「7 件套」被收藏两次
    不代表用户喜欢七件套，只代表他在买套装。

    只认词表内的 token，抽不到就返回空——绝不臆造（同 extract_attrs 的 P0 诚实红线）。

    **命中判定必须走 :func:`term_hits`，不能用裸子串**（``token in text``）。裸子串会把
    "Vegan Leather-Free Tote" 数成一条**皮革**证据——而这件商品恰恰是「不喜欢皮革的人」才会收藏的。
    学出来的偏好于是完全反了：用户越是收藏无皮革款，系统越给真皮款加分。本模块 :data:`_NEG_BEFORE`
    正是为这个坑建的（见下方注释里记的那次真实误杀），抽取侧不该再踩一遍。
    """
    text = (title or "").lower()
    hits = [token for token, _label in TITLE_ATTRS if _attr_hit(token, text)]
    # 长词吸收短词：词表里 "genuine leather" 与 "leather" 都在，一条真皮标题会同时命中两个。放着不
    # 管，同一处文本被数成两个证据（词频虚高一倍），且后续打分对同一件商品加两次分——皮革类会系统
    # 性地压过单 token 的材质（帆布 / 尼龙）。只保留最长的那个。
    return [t for t in hits if not any(t != other and t in other for other in hits)]


# ── 词命中判定（词边界 + 否定修饰）──────────────────────────────────────────────
# 原先是 item_picker 的私有机制；item_search（召回阶段的记忆硬排除）与 /api/similar（搜同款的
# blocking 过滤）也要用同一套判定，故上收到本模块——「命中怎么算」必须全链路一个口径。
#
# 「否定修饰」词表：这些词紧挨着匹配词出现时，商品其实是**替代品**，不该算命中。
# 真实误杀：用户说「不要皮革」→ keywords=["leather","皮革"] → 裸子串匹配把 "vegan leather"
# 「人造皮革」"leather-free" 全杀了——而这些**正是**不喜欢皮革的人想要的东西。同理「不要塑料」
# 会杀掉 "plastic-free"。裸子串匹配在电商标题上就是这么脆。
_NEG_BEFORE = frozenset(
    {
        "vegan",
        "faux",
        "fake",
        "synthetic",
        "artificial",
        "imitation",
        "pu",
        "non",
        "no",
        "without",
        "anti",
        "alternative",
        "substitute",
        "人造",
        "仿",
        "假",
        "合成",
        "素",
        "无",
        "不含",
        "非",
    }
)
_NEG_AFTER = frozenset({"free", "alternative", "substitute", "替代品", "替代"})

# 英文词边界：ASCII 词只在完整单词处命中，不让 "pu" 撞进 "pouch"、"popular"。中文没有词边界
# 概念（也没有空格分词），仍走子串——但同样吃下面的否定修饰检查。
_WORD_RE_CACHE: dict[str, re.Pattern[str]] = {}

_SEP_RE = re.compile(r"[\s\-_/（()）]+")


def _is_ascii_word(kw: str) -> bool:
    return kw.isascii() and kw.replace(" ", "").isalnum()


def term_hits(kw: str, text: str) -> bool:
    """``kw`` 是否**真的**命中 ``text``——带词边界 + 否定修饰过滤，不是裸 ``in``。

    两道防线：
    1. **词边界**（仅 ASCII 词）：``pu`` 不该命中 ``pouch``、``bag`` 不该命中 ``baggage``。
    2. **否定修饰**：命中位置前后紧邻 :data:`_NEG_BEFORE` / :data:`_NEG_AFTER` 里的词时，这次
       出现**不算命中**——"vegan leather" / "plastic-free" / 「人造皮革」是替代品，不是要排除的
       东西。所有出现都被否定修饰 → 整体不命中。

    失效方向是安全的：判错顶多**漏挡**一件（用户看到一件不太想要的商品，可以忽略），
    而误杀是**看不见的**——用户根本不知道有商品被删掉了，只觉得「怎么搜不出东西」。
    """
    if kw not in text:  # 快路径：连子串都不是，直接否
        return False
    ascii_kw = _is_ascii_word(kw)
    if ascii_kw:
        pattern = _WORD_RE_CACHE.get(kw)
        if pattern is None:
            pattern = re.compile(rf"\b{re.escape(kw)}\b")
            _WORD_RE_CACHE[kw] = pattern
        spans = [m.span() for m in pattern.finditer(text)]
    else:  # 中文等无词边界：退回子串定位
        spans = []
        start = text.find(kw)
        while start != -1:
            spans.append((start, start + len(kw)))
            start = text.find(kw, start + 1)
    if not spans:
        return False
    return any(not _negated(text, lo, hi, ascii_kw=ascii_kw) for lo, hi in spans)


def _negated(text: str, lo: int, hi: int, *, ascii_kw: bool) -> bool:
    """这次出现是否被否定修饰（前面是 vegan/人造… 或后面是 free/替代…）。

    **英文与中文必须用不同的匹配方式**，这是两次踩坑换来的：

    - **英文**按分隔符切成 token，做**精确相等**。不能用后缀匹配——``piano`` 以 ``no`` 结尾，
      ``"piano leather case"`` 会被误判成「否定」而漏挡。切分后还必须滤掉空串：``"faux-leather"``
      的前段是 ``"faux-"``，切完尾部留一个空串，取 ``[-1]`` 拿到 ``""`` 而不是 ``"faux"``——
      连字符复合词在电商标题里到处都是。
    - **中文没有分词**，整段前缀就是一个 token。必须做**后缀 / 前缀包含**：``"2024新款人造皮革
      手提包"`` 的前段是 ``"2024新款人造"``，精确相等永远匹配不上 ``"人造"``，于是这件人造革包
      会被「不要皮革」误杀（真实误杀，review 抓到的）。
    """
    before_parts = [p for p in _SEP_RE.split(text[:lo]) if p]
    after_parts = [p for p in _SEP_RE.split(text[hi:]) if p]
    before = before_parts[-1] if before_parts else ""
    after = after_parts[0] if after_parts else ""
    if ascii_kw:
        return before in _NEG_BEFORE or after in _NEG_AFTER
    return any(before.endswith(neg) for neg in _NEG_BEFORE) or any(
        after.startswith(neg) for neg in _NEG_AFTER
    )


def normalize_terms(words: list[str] | tuple[str, ...] | None) -> list[str]:
    """把原子词扩成**中英标题都能匹**的小写词表（原词保留 + 双向补词，去重保序）。

    一个中文词可能补出多个英文词（「防水」→ waterproof / water-resistant），全部保留——标题写法
    不统一，命中任一即算命中。英文词反向补出词表内 ≥2 字的中文变体（plastic → 塑料），只有英文
    的约束在掺中文标题的平台上才不至于反向空转。
    """
    out: list[str] = []
    seen: set[str] = set()

    def _add(term: str) -> None:
        if term and term not in seen:
            seen.add(term)
            out.append(term)

    for raw in words or ():
        word = (raw or "").strip().lower()
        if not word:
            continue
        _add(word)  # 原词一律保留（中文标题的商品仍要能被中文词挡下）
        if word.isascii():  # 已是英文 → 反向补中文（掺中文标题的平台也要能被英文约束挡下）
            for zh in EN_ZH_ALL.get(word, ()):
                _add(zh)
            continue
        mapped = ZH_EN.get(word)
        if not mapped:
            # 补不出英文：这条约束对英文库就是空转的。不报错（语义通道仍在用它），但要留痕——
            # 「有多少词补不出英文」是这层的健康指标，长期为高说明词表该扩了。
            logger.info("匹配词无英文映射（对英文标题不会命中）：%s", word)
            continue
        for en in mapped:
            _add(en)
    return out


# ---------- 数值规格（尺寸/容量）专道 ----------
#
# 「16寸笔记本背包推荐出 14 寸」badcase（2026-07-15）：数值规格是**互斥枚举**，但它在既有两条
# 通路上全都失明——
# ① 字面路（term_hits）："16 inch" 匹不中标题的 "16-inch" / "16in" / '16"'（子串快路径里
#    空格≠连字符），真 16 寸拿不到加分；而 14 寸标题里明晃晃写着冲突证据，却只是「不加分」。
# ② 语义路（sem_hard / sem_match）：embedding 对数字几乎失明、对 "inch laptop" 主题词极强——
#    "14 inch laptop backpack" 对 "16 inch" 的 cosine 接近满分，14 寸反而领着硬约束语义加分上浮。
# 数字是本仓库第三个「embedding 处理不了的算子」（前两个：否定不进 query 向量、rerank query 剥
# prefer 词）——一律走这里的确定性比对，别喂给 embedding。

#: 单位别名 → 归一单位。只比同单位，**不做跨单位换算**：标题里的 cm 多半是包体尺寸
#: （30x45cm），拿它去换算比对「16 寸屏笔记本」是张冠李戴，宁可 unknown。
_SPEC_UNIT_ALIASES: dict[str, str] = {
    "inch": "inch",
    "inches": "inch",
    "in": "inch",
    '"': "inch",
    "英寸": "inch",
    "寸": "inch",  # 电商语境「寸」即英寸（16寸笔记本），不是市寸
    "cm": "cm",
    "厘米": "cm",
    "公分": "cm",
    "l": "l",
    "升": "l",
    "liter": "l",
    "liters": "l",
    "litre": "l",
    "litres": "l",
}

# 整条原子词就是「数值+单位」：`16 inch` / `16-inch` / `16寸` / `16"` / `40L`。
_SPEC_TERM_RE = re.compile(
    r'^(\d+(?:\.\d+)?)\s*-?\s*(inch(?:es)?|in|"|英寸|寸|cm|厘米|公分|l|升|litres?|liters?)$',
    re.IGNORECASE,
)

# 标题里的规格出现。两处防误伤：
# - `in` 后跟数字不算（"2 in 1"），后跟介词/冠词也不算（"USB 3.0 in the side pocket"）；
# - 可带 up to / fits / holds 前缀 → 上界语义（"fits up to 17 inch" 对 16 寸是兼容，不是冲突）。
_SPEC_TITLE_RE = re.compile(
    r"(?:\b(up\s+to|fits?(?:\s+up\s+to)?|holds?(?:\s+up\s+to)?)\s+)?"
    r"(\d+(?:\.\d+)?)\s*-?\s*"
    r"(inch(?:es)?\b|in\b(?!\s*-?\s*\d)(?!\s+(?:the|a|an|your|to|of|for|on|with)\b)"
    r'|"|英寸|寸|cm\b|厘米|公分|l\b|升|litres?\b|liters?\b)',
    re.IGNORECASE,
)

#: 容差（相对比例，经 env 调）：15.6 寸位的包装 16 寸机器是常态（差 2.5%，兼容），
#: 14 对 16（差 12.5%）才是真冲突。
SPEC_TOLERANCE = float(os.getenv("SPEC_MATCH_TOLERANCE", "0.05"))

# 多维尺寸链（"18 x 12 x 6 inch"）的识别：单位只挂在链尾数字上，孤立看就是一条「6 寸」假规格。
# 链里的数字是**包体三维**，不是「装几寸笔记本」的档位声明——整链忽略（不算 found，不参与裁决）。
_DIM_CHAIN_RE = re.compile(r'[\d"]\s*[x×*]\s*$')


def parse_spec_term(term: str) -> tuple[float, str] | None:
    """整条原子词是「数值+单位」时解析成 ``(值, 归一单位)``，否则 None（走普通词通路）。"""
    m = _SPEC_TERM_RE.match(term.strip())
    if not m:
        return None
    unit = _SPEC_UNIT_ALIASES.get(m.group(2).lower())
    return (float(m.group(1)), unit) if unit else None


def spec_verdict(value: float, unit: str, text: str) -> str:
    """要求的规格 ``(value, unit)`` 对一段标题文本的确定性三态裁决。

    - ``"match"``：标题标了同单位规格且容差内命中，或 up-to 上界 ≥ 要求（装得下）。
    - ``"conflict"``：标题标了同单位规格，且**全部**低于要求（容差外）——「14 inch 包」对
      16 寸要求就是标题里明晃晃的冲突证据。**只罚「全部更小」**：尺寸语义不对称，17 寸的包
      装 16 寸机器不是冲突（顶多没明说兼容，判 unknown）；「全部低于才冲突」同时挡掉多维
      尺寸的误伤——"18 x 12 inch" 对 16 寸，12 低但 18 高，整体 unknown 而非 conflict。
    - ``"unknown"``：标题没标同单位规格 → 不奖不罚。失效方向安全（漏挡优于误杀）。
    """
    tol = value * SPEC_TOLERANCE
    found = False
    all_below = True
    for m in _SPEC_TITLE_RE.finditer(text):
        if _SPEC_UNIT_ALIASES.get(m.group(3).lower()) != unit:
            continue
        if _DIM_CHAIN_RE.search(text[: m.start(2)]):  # 多维尺寸链尾的假规格，整链忽略
            continue
        found = True
        v = float(m.group(2))
        if m.group(1):  # up to / fits 前缀：上界语义，覆盖到要求值即兼容
            if v >= value - tol:
                return "match"
        elif abs(v - value) <= tol:
            return "match"
        if v >= value - tol:
            all_below = False
    if not found:
        return "unknown"
    return "conflict" if all_below else "unknown"


# ---------- 约束类型学（分诊契约表）----------
#
# embedding 盲区是**成体系的**，不是零星的坑：否定不进 query 向量、rerank query 剥 prefer 词、
# 数值规格走确定性专道——三次踩的是同一类坑（「embedding 处理不了的算子」），却是逐坑发现
# 逐坑修的。本表把「约束进来先问是什么类型、该走哪条通路、失效朝哪边倒」立成契约：
# - 新增约束通路先来这里登记一行；
# - status="gap" 的类型是**已知裸奔面**：识别得出但无专道，现状回退 topic 语义道 = 对该算子
#   双盲（字面匹不中变体、embedding 对数字/枚举失明）——补哪条、先补哪条由巡检种子
#   （scripts/eval/build_eval_queries.py 的「约束类型学巡检」桶）的评测证据决定，不等下一个
#   badcase 来定优先级。
# - 「否定」不是 classify 的输出：一条词是不是否定由**入口分桶**（exclude/deprioritize）决定，
#   词面上看不出来（"塑料" 在 exclude 里是黑名单、在 prefer 里是需求）。表里登记它只为完整。

ConstraintKind = Literal[
    "topic", "numeric_spec", "numeric_range", "enum_size", "count_pack", "generation"
]

#: kind → (执行通路, unknown/失效方向, 实现状态 live|gap)
CONSTRAINT_LANES: dict[str, tuple[str, str, str]] = {
    "topic": ("embedding 召回 + term_hits 字面加分", "不奖不罚", "live"),
    "negation": (
        "入口分桶（exclude=硬淘汰 / deprioritize=减分）+ term_hits；绝不进 query 向量",
        "漏挡优于误杀",
        "live",
    ),
    "numeric_spec": (
        "parse_spec_term → spec_verdict 三态（±容差 / up-to 上界）",
        "不奖不罚",
        "live",
    ),
    "numeric_range": (
        "待实现：spec_verdict 加比较算子（以上/以下/under/over）",
        "回退 topic=双盲",
        "gap",
    ),
    "enum_size": ("待实现：尺码互斥枚举（S/M/L）等值匹配 + 冲突判据", "回退 topic=双盲", "gap"),
    "count_pack": ("待实现：装量（2-pack/两个装）等值匹配", "回退 topic=双盲", "gap"),
    "generation": ("待实现：代际型号（第3代/gen 3）等值匹配", "回退 topic=双盲", "gap"),
}

# 范围提示词（配合词内有数字才算 range——「预算以内」无数字仍是 topic）。
_RANGE_HINT_RE = re.compile(
    r"(以上|以下|以内|之内|之间|至少|最少|不超过|不小于|不低于|大于|小于)"
    r"|\b(under|over|above|below|at\s+least|at\s+most|min|max)\b",
    re.IGNORECASE,
)
# 尺码枚举：裸词只认无歧义的（xl/xxl/xs…）；单字母 s/m/l 必须带 码/号/size 语境。
_SIZE_TOKEN = r"(?:xs|s|m|l|xl|xxl|xxxl|[23]xl)"
_ENUM_SIZE_RE = re.compile(
    rf"^(?:size\s+{_SIZE_TOKEN}|{_SIZE_TOKEN}\s*(?:码|号)|x{{1,3}}[sl]|均码)$", re.IGNORECASE
)
# 装量：2-pack / 3 pcs / pack of 2 / 两个装 / 2件装。整词锚定，「旅行三件套」这类成品概念
# 不在此列（它是 topic，不是对单品装量的硬约束）。
_COUNT_PACK_RE = re.compile(
    r"^(?:\d+\s*[- ]?(?:pack|pcs|pieces)|(?:pack|set)\s+of\s+\d+"
    r"|[两三四五六]?\d*\s*(?:个|件|支|只|片|条|双)装)$",
    re.IGNORECASE,
)
# 代际：第3代 / 3代 / gen 3 / generation 2。整词锚定。
_GENERATION_RE = re.compile(r"^(?:第?\s*\d+\s*代|gen(?:eration)?\s*\d+)$", re.IGNORECASE)


def classify_constraint(term: str) -> ConstraintKind:
    """一条原子约束词的**确定性分诊**——查 :data:`CONSTRAINT_LANES` 得它该走的通路与失效方向。

    分诊只看词面、零 IO；判不出一律落 ``topic``（现状通路，行为不变）。否定不在此判
    （由入口分桶决定，见 CONSTRAINT_LANES 注释）。
    """
    t = term.strip().lower()
    if not t:
        return "topic"
    if parse_spec_term(t) is not None:
        return "numeric_spec"
    if _RANGE_HINT_RE.search(t) and re.search(r"\d", t):
        return "numeric_range"
    if _COUNT_PACK_RE.match(t):
        return "count_pack"
    if _GENERATION_RE.match(t):
        return "generation"
    if _ENUM_SIZE_RE.match(t):
        return "enum_size"
    return "topic"
