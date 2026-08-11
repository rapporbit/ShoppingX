"""S0-2：给 `planner_dataset.jsonl` 标 golden。**三组字段三种定法，不是一刀切。**

- `budget_amount` / `currency` / `clear_budget` → **纯规则**（复用线上 `resolve_budget_currency`
  + `budget_amount_grounded`）。预算是确定性可判的，数字就在原话里；让 LLM 标只会引入抖动，
  而这一维恰恰是老 bug 源（币种每轮猜、追问轮抄上文重折）。
- `category` / `domains` → **强模型 3 票扰动投票**，不一致的进人工待审。域判定是语义活儿，
  规则词表只能反证（宁漏勿错）、不能当分类器；单票 = 把一次抽样当真理。
- `must_have` / `category_anchor` → 只标**语义锚**，不标字面 golden。检索词天然多解
  （「保温杯」= thermos / vacuum flask / insulated bottle），比字面就是在惩罚同义改写。
  锚供 `R_retrieval` 在线打 Qdrant 算命中。
- `exclude_terms` → 只标 `exclude_expected` 布尔。evidence 成不成立是 reward 侧的规则校验
  （片段是否本轮原话子串），不需要预先标。

**三票必须互相扰动，否则投票是假的**：judge 默认 temperature=0，同 prompt 三次几乎必然同答案，
「一致」只能证明模型稳定、证明不了判对。故三票各带不同温度 + 不同域清单顺序 + 要不要先写判据。

**样本单位是「一次 planner 调用」不是「一个会话」**：追问轮（占 34%）线上是带着上文单独调一次
planner 的，golden 也必须逐轮标。但**投票按会话整体跑一次**——LLM 看得到完整会话才判得准追问轮
的品类继承，而且调用数从 2.2k 降到 1.6k。

用法：``uv run python scripts/train/build_planner_golden.py --limit 30``（先 smoke 再全量）
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app.memory.domains import ALL_DOMAINS, DOMAIN_LABELS, infer_domains_from_text  # noqa: E402
from app.tools.planner import budget_amount_grounded, resolve_budget_currency  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "train"
DATASET = DATA_DIR / "planner_dataset.jsonl"
OUT = DATA_DIR / "planner_golden.jsonl"
REVIEW = DATA_DIR / "planner_golden_review.jsonl"
REPORT = DATA_DIR / "planner_golden_report.json"

CONCURRENCY = 20  # 会话级并发；每个会话内还有 3 票并发，实际在飞请求数是它的 3 倍
REQ_TIMEOUT = 150  # 秒。挂起的请求会占死并发槽让 gather 永不返回（M21 实测教训）

# ── 规则组：预算 ────────────────────────────────────────────────────────────────
# 数字 + 可选量级后缀。逗号形式（1,000）与 k/千/万 都要认，口径与 budget_amount_grounded 一致。
_NUM = re.compile(r"(\d+(?:[.,]\d+)?)\s*([kK千wW万]?)")
_SCALE = {"k": 1e3, "K": 1e3, "千": 1e3, "w": 1e4, "W": 1e4, "万": 1e4}

# 数字**后面**紧跟这些就不是钱，是规格。不设这道闸，「16寸的笔记本包」会被标成预算 16。
_SPEC_UNIT = re.compile(
    r"^\s*(?:寸|吋|英寸|inch|cm|mm|kg|g\b|ml|升|L\b|GB|TB|MB|mAh|W\b|瓦|Hz|赫兹|"
    r"小时|分钟|天|岁|年|码|号|人|件|个|只|双|张|片|档|度|%|percent|寸屏)",
    re.IGNORECASE,
)
# 数字**后面**紧跟这些 = 明确是钱。币种符号在 resolve_budget_currency 里另判，这里只判「是不是钱」。
_MONEY_AFTER = re.compile(
    r"^\s*(?:元|块|刀|美元|美金|人民币|欧元|英镑|日元|港币|新元|软妹币|rmb|usd|cny|eur|gbp|jpy)",
    re.IGNORECASE,
)
# 数字**前面**（最近 8 字内）出现这些 = 预算语境。裸数字（「预算300」是 bare 维度的主力）靠它捞。
# 刻意**不要求紧邻**：真实说法是「预算提到600吧」「价格控制在 300」，中间总隔着词。放宽的误伤
# 由 _SPEC_UNIT 那道闸先挡（「预算300元，重量不超过2kg」里的 2 先被 kg 判成规格）。
_MONEY_BEFORE = re.compile(r"预算|价格|价位|售价|控制在|花|大概|不超过|低于|上限|[$¥￥€£]")
# 数字**后面**（最近 8 字内）出现这些 = 预算语境（「300以内」「500块左右」）。
_MONEY_TAIL = re.compile(r"^\s*(?:以内|以下|之内|左右|上下|封顶|以里|块钱|出头|价位|预算)")
_RANGE_LINK = re.compile(r"[到至\-~—]\s*$")
_CJK_NUM = re.compile(r"[零一二两三四五六七八九十百千万]")
# 「这句话在谈钱吗」——用于「中文数词写的预算」这种规则抽不出确定值的弃权判定。
_MONEY_CTX = re.compile(r"预算|价格|价位|售价|控制在|[元块刀]|美元|美金|人民币|[$¥￥€£]")

# 「明确取消预算」——只认强表达。「便宜点」「不要太贵」不是取消预算，是软偏好。
_CLEAR_BUDGET = re.compile(
    r"不限预算|预算不限|不设上限|上不封顶|不限价|多少钱都(?:行|可以)|"
    r"贵(?:点|一点|些)?也(?:行|可以)|别管价格|不管多少钱|(?:价格|预算)不是问题|不考虑预算"
)

# 硬排除表达。软表达（「不太喜欢」「尽量别」）先挖掉再匹——它们该进 soft_dislikes，不是 exclude。
# 「不要太贵」「别太花哨」是程度表达（→ soft_dislikes），不是「命中即淘汰」的硬排除，先挖掉。
_SOFT_NEG = re.compile(
    r"不太[喜想爱]|尽量(?:别|不要|避免)|能不要就不要|最好(?:别|不要)|不太想|稍微|"
    r"(?:不要|别|不能)太"
)
_HARD_NEG = re.compile(
    r"不要|不想要|别要|别给|不能|不可以|拒绝|排除|去掉|剔除|避免|杜绝|无\s*\w+\s*的"
)


def _amounts(text: str) -> list[float]:
    """从本轮原话里抽出**钱**（不是规格数字）。返回按出现序的候选金额。"""
    out: list[float] = []
    for m in _NUM.finditer(text):
        raw, suffix = m.group(1), m.group(2)
        # 「2kg」的 k 不是「千」——ASCII 量级后缀后面还跟着字母就是单位的一部分。不判这一下，
        # 「重量不超过2kg」会被读成 2000 元（实测踩到）。
        if suffix in "kKwW" and re.match(r"[a-zA-Z]", text[m.end() : m.end() + 1]):
            suffix = ""
        tail = text[m.end(1) + len(suffix) :]
        head = text[max(0, m.start() - 8) : m.start()]
        if not suffix and _SPEC_UNIT.match(text[m.end(1) :]):
            continue  # 16寸 / 256GB / 12小时：规格不是钱
        is_money = bool(
            _MONEY_AFTER.match(tail) or _MONEY_TAIL.match(tail) or _MONEY_BEFORE.search(head)
            # 区间上限（「预算300到500」）：语境词只挂在区间左端，右端要靠连接符继承，
            # 否则上限被漏掉、golden 变成下限——预算标小了，reward 会去奖励更省的错答案。
            or (out and _RANGE_LINK.search(head))
        )
        if not is_money:
            continue
        val = float(raw.replace(",", ""))
        out.append(val * _SCALE.get(suffix, 1.0) if suffix else val)
    return out


_CJK_DIGIT = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CJK_UNIT = {"十": 10, "百": 100, "千": 1000, "万": 10000}
_CJK_RUN = re.compile(r"[零一二两三四五六七八九十百千万]{1,8}")


def _cjk_value(run: str) -> float | None:
    """中文数词串 → 数值。**模糊表达一律返回 None**（「十五六块」是区间口语，猜哪个都是错）。

    判模糊的信号是「两个数字词相邻且中间没有量级」（五+六）。规范写法（十二 / 三百 / 一千五）
    才解析——弃权的代价只是这条样本的预算维度不计分，猜错的代价是往 golden 里灌噪声。
    """
    total = section = 0.0
    last_unit = 0.0
    prev_digit = False
    for ch in run:
        if ch in _CJK_DIGIT:
            if prev_digit:
                return None  # 「十五六」「三四百」：口语区间，不确定
            section = _CJK_DIGIT[ch]
            prev_digit = True
            continue
        unit = _CJK_UNIT[ch]
        prev_digit = False
        if unit == 10000:  # 「万」放大的是**已累计的整体**（「一万五」= 10000 + 5×1000）
            total, section = (total + section) * 10000, 0
        else:
            total += (section or 1) * unit  # 「十二」的「十」前面没数字 → 当 1×10
            section = 0
        last_unit = unit
    # 「一千五」= 1500：结尾的裸数字继承上一级量级的十分之一
    if section and last_unit >= 10:
        total += section * last_unit / 10
    elif section:
        total += section
    return total or None


def _cjk_amounts(text: str) -> list[float]:
    """中文数词写的金额（「预算十二美元」「三百块以内」）。语境判定复用阿拉伯数字那三条规则。"""
    out = []
    for m in _CJK_RUN.finditer(text):
        tail, head = text[m.end() :], text[max(0, m.start() - 8) : m.start()]
        if not (_MONEY_AFTER.match(tail) or _MONEY_TAIL.match(tail) or _MONEY_BEFORE.search(head)):
            continue
        if (v := _cjk_value(m.group())) is not None:
            out.append(v)
    return out


def budget_golden(text: str) -> dict:
    """预算三字段的 golden，**零 LLM**。多个金额取最大值 = 区间取上限（「300到500」→ 500）。

    ``budget_uncertain=True`` 是**诚实的弃权**：原话用中文数词写预算（「三百块以内」）时规则
    抽不出确定值，此时既不能标 None（模型答 300 是对的，标 None 就是在惩罚正确答案），也不能
    瞎猜一个数。标成不可判，reward 侧对这条样本的预算维度直接跳过计分。
    """
    digits = _amounts(text)
    amounts = digits or _cjk_amounts(text)
    amount = max(amounts) if amounts else None
    # 抽出来的数就在原话里，grounded 必然成立；仍然过一遍闸，等于给规则本身上了个断言。
    # **只对阿拉伯数字过闸**：中文数词金额（「预算三百」）在句里另有阿拉伯数字（「16寸」）时，
    # 该函数按数字 token 比对必然判不成立，过闸等于把刚解析对的预算又抹掉。
    if amount is not None and digits and not budget_amount_grounded(text, amount):
        amount = None
    code, explicit = resolve_budget_currency(text)
    uncertain = amount is None and bool(_CJK_NUM.search(text) and _MONEY_CTX.search(text))
    return {
        "budget_amount": amount,
        "currency": code if amount is not None else "",
        "currency_explicit": explicit if amount is not None else False,
        "clear_budget": bool(_CLEAR_BUDGET.search(text)),
        "budget_uncertain": uncertain,
        "budget_candidates": amounts,
    }


def exclude_expected(text: str) -> bool:
    """本轮该不该产出硬排除项（软表达不算）。reward 侧据此判「漏吐」与「无中生有」。"""
    return bool(_HARD_NEG.search(_SOFT_NEG.sub("　", text)))


# ── 投票组：category / domains / must_have ──────────────────────────────────────
# 三票**必须互相扰动**，否则 temperature=0 下三次同答案，「一致」证明不了判对，只证明模型稳定。
# 扰动三个正交维度：采样温度、域清单顺序（对抗 LLM 的位置偏好）、要不要先写判据（改变推理路径）。
VOTES = [
    {"name": "v0", "temperature": 0.0, "reverse_menu": False, "reason": False},
    {"name": "v1", "temperature": 0.6, "reverse_menu": True, "reason": True},
    {"name": "v2", "temperature": 0.3, "reverse_menu": False, "reason": True},
]

PROMPT = """你在给一个跨境电商购物 Agent 的 planner 标注 golden。
下面是一段中文购物会话里**用户说的话**，请**逐轮**判断这一轮 planner 应输出的
category / domains / must_have。

会话：
{turns}

判定口径（与线上 planner 一致，照它判，别按自己的习惯）：
- category：本轮用户在买的**主品类**，中文 2~10 字（「跑鞋」「保温杯」「笔记本电脑包」）。
  追问轮没换品类 → **沿用上一轮的品类照写一遍**，不要留空；用户明确换了品类 → 写新的。
  只给场景不给品类（「出差带点啥好」「给新家配齐厨房好物」）→ **按场景挑一个最贴近的品类
  假设填上**，别留空（线上 planner 也必须给出品类假设才检索得动）。
  纯闲聊 / 完全不涉及商品 → 空串。
- category_en：上面那个品类的**英文商品类目词**（小写，1~4 词，如 "running shoes"、
  "laptop backpack"）。给机器拿去比对类目用，category 为空时也留空。
- domains：从下面这份**封闭清单**里挑，可多选（旅行三件套 = bags + apparel）。
  按**商品本体**归域，别被使用场景带偏（手表 → jewelry_watches，不是 furniture；
  露营灯 → 见清单里最贴近的一项）。确实一个都对不上才填 ["other"]。**绝不要填 global**。
{menu}
- must_have：这轮要买的东西在**英文商品标题**里几乎必然出现的 1~3 个核心词（小写名词，
  如 ["backpack"]、["laptop bag"]、["thermos","insulated bottle"]）。它是给机器判「搜回来的
  东西是不是这个类目」用的，所以：**不要品牌、不要形容词（cheap / durable）、不要场景词**
  （travel 只有在它本身就是品类名一部分时才给）。判不出品类 → 空数组。

严格输出 JSON 数组，每轮一个元素，顺序与轮次一致，不要任何解释文字：
{shape}"""


def _menu(reverse: bool) -> str:
    """域清单渲染。倒序是投票扰动的一环——LLM 对清单靠前项有位置偏好，换个序才知道判定稳不稳。"""
    items = [f"  - {d}：{DOMAIN_LABELS[d]}" for d in ALL_DOMAINS if d != "global"]
    return "\n".join(reversed(items) if reverse else items)


def _shape(with_reason: bool, n: int) -> str:
    core = '"category":"...","category_en":"...","domains":["..."],"must_have":["..."]'
    reason = '"reason":"20字内判据",' if with_reason else ""
    one = '{"turn":0,' + reason + core + "}"
    return "[" + one + (", ..." if n > 1 else "") + "]"


def _parse(text: str, n: int) -> list[dict] | None:
    m = re.search(r"\[.*]", text, re.S)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(arr, list) or len(arr) != n:
        return None
    if not all(isinstance(x, dict) and "category" in x for x in arr):
        return None
    out = []
    for x in arr:
        doms = [d for d in (x.get("domains") or []) if d in ALL_DOMAINS and d != "global"]
        must = [str(w).strip().lower() for w in (x.get("must_have") or []) if str(w).strip()]
        out.append({
            "category": str(x.get("category") or "").strip(),
            "category_en": str(x.get("category_en") or "").strip().lower(),
            "domains": sorted(set(doms)),
            "must_have": must[:3],
        })
    return out


def _llm(temperature: float):
    """judge 模型 + 指定温度。不复用 ``get_judge_llm()``：它的温度钉在 env 上（0.0），
    而这里要的恰恰是**三档不同温度**——投票的扰动源之一。其余参数与线上判官一致。"""
    from langchain.chat_models import init_chat_model

    return init_chat_model(
        os.environ.get("LLM_JUDGE") or os.environ["LLM_MAIN"],
        model_provider="openai",
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
        temperature=temperature,
        timeout=REQ_TIMEOUT,
        max_retries=2,
    )


async def _one_vote(llm, session: dict, cfg: dict) -> list[dict] | None:
    turns = session["turns"]
    prompt = PROMPT.format(
        turns="\n".join(f"第{t['turn'] + 1}轮：{t['text']}" for t in turns),
        menu=_menu(cfg["reverse_menu"]),
        shape=_shape(cfg["reason"], len(turns)),
    )
    for _ in range(2):  # 只重试一次：连着两次解析失败多半是这条会话本身怪，重试第三次也白搭
        try:
            resp = await asyncio.wait_for(llm.ainvoke(prompt), timeout=REQ_TIMEOUT)
        except Exception:
            continue
        if parsed := _parse(str(resp.content), len(turns)):
            return parsed
    return None


def _norm_cat(s: str) -> str:
    """品类归一：三票字面很难完全一致（「跑鞋」/「男士跑鞋」/「跑步鞋」），比字面就全是假不一致。"""
    return re.sub(r"[\s的儿子款个件套装用品]", "", s.lower())


def _agree_cat(cands: list[str]) -> tuple[str, bool]:
    """多数票 → 取多数；无多数但两票**互为子串**（「跑鞋」⊂「男士跑鞋」）→ 取更短的那个（更泛、
    更安全）；否则判不一致。返回 ``(定稿, 是否一致)``。"""
    normed = [_norm_cat(c) for c in cands]
    top, cnt = Counter(normed).most_common(1)[0]
    if cnt >= 2:
        return next(c for c, n in zip(cands, normed, strict=True) if n == top), True
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            a, b = normed[i], normed[j]
            if a and b and (a in b or b in a):
                return (cands[i] if len(a) <= len(b) else cands[j]), True
    return cands[0], False


def _agree_domains(cands: list[list[str]]) -> tuple[list[str], bool]:
    """域集合按**整集**取多数。刻意不取交集兜底：交集是「三票都同意的最小集」，看着安全，实则
    会把多品类需求（旅行三件套 = bags + apparel）悄悄削成单域——golden 里少一个域，训练就在
    教模型漏判。判不一致就老实进人工，别用平均值糊过去。"""
    keys = ["|".join(c) for c in cands]
    top, cnt = Counter(keys).most_common(1)[0]
    if cnt >= 2:
        return (top.split("|") if top else []), True
    return cands[0], False


def _agree_must(cands: list[list[str]]) -> tuple[list[str], bool]:
    """锚词按**词级**取 ≥2 票。全票各说各的（同义词多解，正常）→ 退回首票并标弱，reward 侧
    对弱锚样本降权，而不是当成硬 golden 去罚模型。"""
    cnt = Counter(w for c in cands for w in set(c))
    strong = [w for w, n in cnt.most_common() if n >= 2][:3]
    return (strong, True) if strong else (cands[0][:3], False)


# 去复数**只削单个 s**。写成 `(?:es|s)$` 踩过一次：suitcase → suitca、suitcases → suitcas，
# 同一个词的单复数反而对不上，凭空造出一片「锚对不上句意」的假阳性（789 条里绝大半是它）。
_PLURAL = re.compile(r"s$")


def _toks(s: str) -> set[str]:
    """英文类目词切成词干集合（去复数即可——这里只做「有没有重合」的弱校验，不是匹配器）。"""
    return {_PLURAL.sub("", w) for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 2}


def _anchor_ok(anchor: str, cat_en: str, must: list[str]) -> bool | None:
    """合成源自带的英文品类**未必对得上句子**：LLM 造句时会跑题（实测「出差旅行带点啥好呢」
    挂在 gift wrapping supplies 名下）。拿投票产的 category_en / must_have 与它对一眼，
    对不上就标出来——**不改数据，只标记**，让 reward 侧决定信不信这一维。"""
    if not anchor:
        return None
    ref = _toks(cat_en) | {t for w in must for t in _toks(w)}
    return bool(ref & _toks(anchor)) if ref else None


def _rows(session: dict, votes: list[list[dict]]) -> list[dict]:
    """会话 + 三票 → 逐轮 golden 样本（一行 = 一次 planner 调用）。"""
    out = []
    for i, turn in enumerate(session["turns"]):
        text = turn["text"]
        cats = [v[i]["category"] for v in votes]
        cats_en = [v[i]["category_en"] for v in votes]
        doms = [v[i]["domains"] for v in votes]
        musts = [v[i]["must_have"] for v in votes]
        cat, cat_ok = _agree_cat(cats)
        cat_en, _ = _agree_cat(cats_en)
        dom, dom_ok = _agree_domains(doms)
        must, must_ok = _agree_must(musts)
        anchor = session.get("category", "")
        anchor_ok = _anchor_ok(anchor, cat_en, must)
        # 真实锚里有 31 条是**没有上文的追问碎片**（线上 messages 表单条抽出来的：「不要皮革的」
        # 「最好带蓝牙」）。上文不存在，品类就是判不出来的——三票分歧不是模型不行，是题目无解。
        # 这种进人工待审纯属浪费（人也判不出），直接**弃权**：category/domains 置 None，
        # reward 跳过这两维；本轮原话里明摆着的预算 / 排除照常标，那部分信息一点没丢。
        #
        # **与三票一不一致无关**——曾经写成 `and not cat_ok`，结果碎片大多被三票「一致地猜成
        # 空品类」而躲过弃权，golden 里留下一个 category=""。线上这些轮次是带 P_t 上文调用的，
        # 模型会输出继承来的品类，拿空串去罚它纯属冤枉。一致地猜 ≠ 判对。
        no_context = bool(session.get("is_followup_fragment"))
        # 域反证并入，**与线上 reconcile_domains 同口径**：词面证据（「手表」→ jewelry_watches）
        # 是确定性的，LLM 漏了就补上。golden 与线上机制不同口径，训出来的模型上线就要被机制改写。
        evidence = sorted(infer_domains_from_text(text) - set(dom))
        if no_context:
            status = "context_missing"
        elif cat_ok and dom_ok and len(votes) == 3:
            status = "agree"
        else:
            status = "review"
        out.append({
            "id": f"{session['id']}#t{turn['turn']}",
            "session_id": session["id"],
            "turn": turn["turn"],
            "source": session["source"],
            "split": session["split"],
            "family": session.get("family", ""),
            "dims": session.get("dims", {}),
            # prior 是**用户原话序列**，不是线上 _render_prior_context() 的 P_t 渲染——那份要跑完
            # 整条会话才有。S1 rollout 时用真实渲染，这里标 golden 只需让标注者/模型看得到上文。
            "prior_turns": [t["text"] for t in session["turns"][:i]],
            "text": text,
            "is_fragment": bool(session.get("is_followup_fragment")),
            "golden": {
                **budget_golden(text),
                # None ≠ 空串：None 是「这题无解，别计分」，空串是「确实没有品类（闲聊）」。
                "category": None if no_context else cat,
                "domains": None if no_context else [*dom, *evidence],
                "must_have": must,
                # 英文品类锚：默认用合成/对抗源自带的（来自 category_cards，是库里真实存在的
                # 类目串，判品类纯度时与 payload 同词表）；**校验对不上句意就回落投票值**——
                # 「出差旅行带点啥」挂在 gift wrapping supplies 名下，照用就是拿错锚去发 reward。
                # 两个值都留着，reward 侧想换口径不用重标。
                "category_anchor": (anchor if anchor_ok is not False else cat_en) or cat_en,
                "category_anchor_raw": anchor,
                "category_en_vote": cat_en,
                "anchor_agrees": anchor_ok,
                "exclude_expected": exclude_expected(text),
            },
            "vote": {
                "n": len(votes), "category_agree": cat_ok, "domains_agree": dom_ok,
                "must_have_strong": must_ok, "domains_reconciled": evidence,
                "raw": {"category": cats, "category_en": cats_en, "domains": doms,
                        "must_have": musts},
            },
            "status": status,
        })
    return out


async def annotate(sessions: list[dict], sink) -> dict:
    """三票并发跑。**逐会话落盘**——M21 的教训：没有增量落盘时任何挂起点都让整跑归零。"""
    llms = [_llm(c["temperature"]) for c in VOTES]
    sem, stat = asyncio.Semaphore(CONCURRENCY), Counter()

    async def one(s: dict) -> None:
        async with sem:
            votes = await asyncio.gather(*(
                _one_vote(llm, s, cfg) for llm, cfg in zip(llms, VOTES, strict=True)
            ))
        ok = [v for v in votes if v]
        stat[f"votes_{len(ok)}"] += 1
        if len(ok) < 2:  # 一票不成投票，直接判失败（不硬编一个「唯一意见」当 golden）
            stat["failed"] += 1
            return
        sink(_rows(s, ok))
        stat["done"] += 1
        if stat["done"] % 100 == 0:
            print(f"  已标注 {stat['done']}/{len(sessions)} 个会话", flush=True)

    await asyncio.gather(*(one(s) for s in sessions))
    return dict(stat)


def _report(rows: list[dict], stat: dict) -> dict:
    n = len(rows) or 1
    by_src: Counter = Counter(r["source"] for r in rows)
    review = [r for r in rows if r["status"] == "review"]
    g = [r["golden"] for r in rows]
    return {
        "会话": stat,
        "轮样本": len(rows),
        "按源": dict(by_src),
        "按切分": dict(Counter(r["split"] for r in rows)),
        "投票": {
            "一致(agree)": round(1 - len(review) / n, 3),
            "待审(review)": len(review),
            "待审按切分": dict(Counter(r["split"] for r in review)),
            "category 不一致": sum(1 for r in rows if not r["vote"]["category_agree"]),
            "domains 不一致": sum(1 for r in rows if not r["vote"]["domains_agree"]),
            "锚词弱(无2票交集)": sum(1 for r in rows if not r["vote"]["must_have_strong"]),
            "域反证补入": sum(1 for r in rows if r["vote"]["domains_reconciled"]),
        },
        "规则组": {
            "带预算": round(sum(1 for x in g if x["budget_amount"] is not None) / n, 3),
            "预算不可判(中文数词)": sum(1 for x in g if x["budget_uncertain"]),
            "币种明示": round(sum(1 for x in g if x["currency_explicit"]) / n, 3),
            "clear_budget": sum(1 for x in g if x["clear_budget"]),
            "该有硬排除": round(sum(1 for x in g if x["exclude_expected"]) / n, 3),
        },
        "golden 覆盖": {
            "无上文碎片弃权": sum(1 for r in rows if r["status"] == "context_missing"),
            "category 非空": round(sum(1 for x in g if x["category"]) / n, 3),
            "domains 非空": round(sum(1 for x in g if x["domains"]) / n, 3),
            "must_have 非空": round(sum(1 for x in g if x["must_have"]) / n, 3),
            "落 other 的域": sum(1 for x in g if x["domains"] == ["other"]),
            # 自带类目串与投票英文品类**词面不重合** → 已回落成投票值。抽样看，绝大多数是
            # 「库类目更泛」（shaving & hair removal products vs shaver）而非造句跑题，
            # 回落的效果是锚更贴句意；真跑题（gift wrapping supplies 挂旅行箱包）是少数。
            # 所以这个数**不是数据缺陷计数**，是「用了投票锚而非库锚」的条数。
            "品类锚回落投票值": sum(1 for x in g if x["anchor_agrees"] is False),
        },
        "域分布": dict(Counter(d for x in g for d in (x["domains"] or ())).most_common()),
    }


def _apply_overrides(rows: list[dict]) -> int:
    """套上 dev / test 的人工裁决（``planner_golden_overrides.OVERRIDES``）。

    裁决过的行状态改 ``resolved``，不再算待审；投票原值仍在 ``vote.raw`` 里，改判可追溯。
    """
    from scripts.train.planner_golden_overrides import OVERRIDES

    hit = 0
    for r in rows:
        if fix := OVERRIDES.get(r["id"]):
            r["golden"].update(fix)
            r["status"] = "resolved"
            r["resolved_by"] = "human"
            hit += 1
    return hit


def _finish(rows: list[dict], stat: dict, out_path: Path) -> None:
    """落 review 文件 + 报告。标注跑与 ``--report-only`` 复算共用同一段收尾，避免两处走样。"""
    stat = {**stat, "人工裁决生效": _apply_overrides(rows)}
    out_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    # 待审按 dev / test / train 排序：**人工只该审前两档**（dev 92 条是量线上分布的尺子、
    # test 是验收靶，尺子不准后面全白做）。train 里的 review 量太大，人工过不完也不值得过——
    # 训练时按 status 降权 / 排除即可，那点样本换不来一小时人工。
    order = {"dev": 0, "test": 1, "train": 2}
    review = sorted(
        (r for r in rows if r["status"] == "review"), key=lambda r: (order[r["split"]], r["id"])
    )
    REVIEW.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in review), encoding="utf-8"
    )
    report = _report(rows, stat)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n→ {out_path.relative_to(PROJECT_ROOT)}（{len(rows)} 行）")
    print(f"→ {REVIEW.relative_to(PROJECT_ROOT)}（{len(review)} 条待人工裁决，dev/test 排在最前）")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只标前 N 个会话（smoke 用，0=全量）")
    ap.add_argument("--source", default="", help="只标某一源：real / synth / adversarial")
    ap.add_argument("--out", default=OUT.name)
    ap.add_argument("--resume", action="store_true", help="接着已有产物跑，跳过标完的会话")
    ap.add_argument(
        "--report-only", action="store_true",
        help="不调 LLM：拿已有产物重算规则派生字段 + 报告（改了规则/校验后复算用）",
    )
    args = ap.parse_args()

    sessions = [json.loads(x) for x in DATASET.open(encoding="utf-8") if x.strip()]
    if args.source:
        sessions = [s for s in sessions if s["source"] == args.source]
    if args.limit:
        # 取每源的前 N/3：smoke 必须三源都覆盖，顺序切片会全落在 synth 上（数据集按源排布）。
        per, picked = max(1, args.limit // 3), []
        for src in ("real", "synth", "adversarial"):
            picked += [s for s in sessions if s["source"] == src][:per]
        sessions = picked
    # resume：4.8k 次调用跑一小时，中途挂掉不该从零再来。已标完的会话原样读回，只补没标的。
    out_path, rows = DATA_DIR / args.out, []
    if args.report_only:
        rows = [json.loads(x) for x in out_path.open(encoding="utf-8") if x.strip()]
        frag = {s["id"] for s in sessions if s.get("is_followup_fragment")}
        for r in rows:  # 规则派生字段就地重算，**投票结果一个字不动**（那是 LLM 花钱买的）
            g = r["golden"]
            r["is_fragment"] = r["session_id"] in frag
            if r["is_fragment"]:
                g["category"], g["domains"], r["status"] = None, None, "context_missing"
            g.update(budget_golden(r["text"]), exclude_expected=exclude_expected(r["text"]))
            g["anchor_agrees"] = _anchor_ok(
                g["category_anchor_raw"], g["category_en_vote"], g["must_have"]
            )
            g["category_anchor"] = (
                g["category_anchor_raw"] if g["anchor_agrees"] is not False else ""
            ) or g["category_en_vote"]
        _finish(rows, {"report_only": True}, out_path)
        return
    if args.resume and out_path.exists():
        rows = [json.loads(x) for x in out_path.open(encoding="utf-8") if x.strip()]
        done_ids = {r["session_id"] for r in rows}
        sessions = [s for s in sessions if s["id"] not in done_ids]
        print(f"resume：已有 {len(rows)} 行 / {len(done_ids)} 个会话，还差 {len(sessions)} 个")

    turns = sum(len(s["turns"]) for s in sessions)
    print(f"待标注 {len(sessions)} 个会话 / {turns} 轮，三票 = {len(sessions) * 3} 次 LLM 调用")

    fh = out_path.open("a" if rows else "w", encoding="utf-8")

    def sink(batch: list[dict]) -> None:
        rows.extend(batch)
        for r in batch:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        fh.flush()

    stat = await annotate(sessions, sink)
    fh.close()
    rows.sort(key=lambda r: r["id"])
    _finish(rows, stat, out_path)


if __name__ == "__main__":
    asyncio.run(main())
