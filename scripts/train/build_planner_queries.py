"""S0-1 源②③：合成 planner 训练用的**购物意图会话**（不是搜索词）。

与 `build_synth_queries.py` 的区别，别搞混——那个是 M21 检索侧的「商品 → 检索词」反推
（L1~L4，喂 embedding 对比学习）；**这个是「品类 → 带约束的购物意图」**，喂 planner 的
SFT / GRPO。两者输入输出方向相反，产物不能互用。

**素材选 `data/rag/category_cards.jsonl` 而不是 138 万 corpus**，三个理由：
1. 247 个去重品类**全部带中文别名**——线上是中文口语，corpus 是英文标题，直接拿 corpus
   造 query 会造出一堆英文关键词，与线上分布错位；
2. 每张卡带 `raw_evidence`（真实商品 + 价格 + 评分 + 月销），能取**品类真实价位中位数**当
   预算锚。这一条直接对上 `planner.budget_amount_grounded`——「预算 300 买保温杯」和
   「预算 300 买手机」的 grounded 判定天差地别，脱离真实价位造预算，golden 就是错的；
3. `attribute` 卡带该品类的典型属性，造「16 寸 / 256GB / 12 小时」这类**数值规格**约束时
   有据可依（数值是 embedding 处理不了的算子，是已知 bad case 族）。

**维度由代码确定性抽，LLM 只负责措辞**。让 LLM 自己发挥约束组合，分布必然塌到
「预算 + 不要塑料」（真实锚里最高频的那两个），剩下的维度一条都不会出现。配额对齐
`planner_anchors.jsonl` 的实测画像：带预算 50% / 带排除 25% / 追问轮 34% / 长度中位数 17 字。

**产物是会话不是单 query**：34% 的线上请求是追问轮碎片（"不要皮革的" / "预算提到600吧"），
这类 query 单看无法标注 category——planner 线上读的是 `_render_prior_context()` + 本轮消息。
不成对造，训出来的模型在三分之一的线上请求上都是 train/serve skew。

用法：``uv run python scripts/train/build_planner_queries.py --limit 1200``
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app.agent.llm import get_llm  # noqa: E402

CARDS_PATH = PROJECT_ROOT / "data" / "rag" / "category_cards.jsonl"
ANCHORS_PATH = PROJECT_ROOT / "data" / "train" / "planner_anchors.jsonl"
OUT_DIR = PROJECT_ROOT / "data" / "train"

BATCH = 4  # 一次让 LLM 造 4 个会话：再多措辞会趋同，再少浪费 prompt 里的品类卡上下文
CONCURRENCY = 12
REQ_TIMEOUT = 150  # 秒。没有它，挂起的请求会占死并发槽让 gather 永不返回（M21 实测教训）

# ── 约束维度矩阵：(取值, 权重)。权重对齐 planner_anchors.jsonl 实测画像 ──────────────
DIMS: dict[str, list[tuple[str, int]]] = {
    # 币种缺省（"预算300"）是老 bug 源头：模型每轮猜的币种都不同 → 预算内空召回。
    # 必须占足量，让 RL 学会把它交给规则解析器的口径。
    "budget": [("none", 50), ("usd", 22), ("bare", 16), ("cny", 8), ("hard_limit", 4)],
    "exclude": [("none", 75), ("material", 12), ("color", 6), ("type", 4), ("brand", 3)],
    "scene": [
        ("none", 45),
        ("travel", 10),
        ("commute", 10),
        ("gift", 10),
        ("sport", 8),
        ("camping", 7),
        ("office", 5),
        ("home", 5),
    ],
    "audience": [("self", 70), ("boyfriend", 8), ("girlfriend", 8), ("kid", 7), ("parent", 7)],
    "soft": [
        ("none", 40),
        ("durable", 15),
        ("value", 15),
        ("niche", 12),
        ("premium", 10),
        ("light", 8),
    ],
    "platform": [("none", 82), ("cross_compare", 12), ("specific", 6)],
    "landed": [("none", 88), ("need_landed", 12)],  # 到手价：带收货国，触发四层解析
    "spec": [("none", 78), ("numeric", 22)],  # 16寸/256GB/12小时/M码——已知 bad case 族
    # explicit=直说品类；scene_only=只给场景不给品类（"给新家配齐厨房好物"，域漂移高发）；
    # bundle=一套齐（多槽位）。后两者是 planner 最容易判错的形态，刻意超配。
    "clarity": [("explicit", 68), ("scene_only", 20), ("bundle", 12)],
}

# ── 追问轮类型：值是「这一轮要做什么」，措辞交给 LLM ─────────────────────────────
FOLLOWUPS: list[tuple[str, int]] = [
    ("add_constraint", 30),  # "只要防水的" / "要红轴的"
    ("retract_constraint", 18),  # "算了，塑料的也行"——撤回是已修但脆弱的老病灶
    ("change_budget", 16),  # "预算提到600吧"
    ("switch_category", 14),  # "保温杯不买了，看看跑步鞋吧"——retrieval 判定的靶子
    ("ask_landed", 12),  # "他们的到手价是多少？"
    ("finalize", 10),  # "就这样，给我最终推荐吧"
]

FOLLOWUP_RATIO = 0.34  # 实测线上占比，见 build_planner_anchors.py

# 可关闭的维度 → 它的「关闭值」。clarity 不在内：它是句子形态不是约束，永远有值。
OPTIONAL_DIMS = {
    "budget": "none",
    "exclude": "none",
    "scene": "none",
    "audience": "self",
    "soft": "none",
    "platform": "none",
    "landed": "none",
    "spec": "none",
}

# 单条 query 同时挂几个约束。**首轮试跑就是栽在没有这个**——8 个维度独立抽样，
# 期望活跃数 ~3.4 个，造出「必须5美元以内，包关税运费寄美国，办公用，要轻的」这种
# 需求文档腔，而真实锚的中位长度只有 17 字。约束数才是长度的真正驱动量。
ACTIVE_K: list[tuple[int, int]] = [(1, 34), (2, 36), (3, 22), (4, 8)]

# 每个维度的**目标边际概率**（多少比例的 query 带这个约束）。
#
# 第二版才想明白：活跃维度总数守恒（= ACTIVE_K 的期望 2.04），所以边际概率不能各自
# 随便定——它们的**和必须等于 2.04**，否则就是在互相抢配额。第一版「独立抽样再做减法」
# 就是不懂这笔账：减法只减不加，把 budget 从 50% 稀释到 33%、exclude 从 25% 稀释到 14.5%。
# 现在改成按这张表加权无放回抽 k 个维度置活跃，边际就精确可控了。
#
# budget / exclude 锚定真实实测（50.0% / 24.5%），其余按经验分配剩下的 1.29 配额。
DIM_TARGET: dict[str, float] = {
    "budget": 0.50,
    "exclude": 0.25,
    "soft": 0.30,
    "scene": 0.25,
    "audience": 0.20,
    "spec": 0.18,
    "platform": 0.20,
    "landed": 0.16,
}
SPEC_OK_RATIO = 0.263  # 247 个品类里只有 26.3% 真的存在常见数值规格（实测）

# 数值规格只对「真的存在常见数值规格」的品类开放。不设这道闸就会造出
# 「毛绒玩具16寸」「护眼用品16档」「发饰长度15cm」——真人不会这么搜，
# 而且这类样本的 must_have 锚是无意义的，会直接污染 R_retrieval 的 reward。
_SPEC_OK = re.compile(
    r"laptop|computer|monitor|tablet|phone|storage|drive|memory|camera|tv|television|"
    r"headphone|speaker|watch|luggage|backpack|bag|handbag|clothing|shoes|apparel|"
    r"bottle|cookware|mattress|bedding|furniture|tool|battery|printer|keyboard|console"
)


def _pick(rng: random.Random, choices: list[tuple[str, int]]) -> str:
    return rng.choices([c for c, _ in choices], weights=[w for _, w in choices], k=1)[0]


def _pick_int(rng: random.Random, choices: list[tuple[int, int]]) -> int:
    return rng.choices([c for c, _ in choices], weights=[w for _, w in choices], k=1)[0]


def _solve_weights(iters: int = 40, n: int = 3000) -> dict[str, float]:
    """反解抽样权重，让实际边际收敛到 DIM_TARGET。

    为什么需要这一步：加权**无放回**抽样的边际不等于权重——高权重维度在第一次抽取里
    就被拿走，后续抽取轮不到它，实际边际系统性低于目标（实测 budget 45% vs 目标 50%、
    spec 12% vs 18%），低权重维度则被抬高。这是抽样方式的固有偏移，不是权重没调好。

    所以权重不手调，用蒙特卡洛迭代反解：measured 偏低就把权重按比例调大，跑几十轮收敛。
    好处是 DIM_TARGET 保持**声明式**——以后想改配额，改目标值即可，不用再手工试权重。
    """
    w = dict(DIM_TARGET)
    rng = random.Random(20260810)  # 固定 seed：权重求解必须可复现，否则每次跑出的数据集分布都不同
    for _ in range(iters):
        hit = dict.fromkeys(w, 0)
        for _ in range(n):
            spec_ok = rng.random() < SPEC_OK_RATIO
            cur = {d: (0.0 if d == "spec" and not spec_ok else v) for d, v in w.items()}
            pool = [d for d, v in cur.items() if v > 0]
            for _ in range(min(_pick_int(rng, ACTIVE_K), len(pool))):
                c = rng.choices(pool, weights=[cur[d] for d in pool], k=1)[0]
                hit[c] += 1
                pool.remove(c)
        for d in w:
            measured = hit[d] / n
            # 加一层阻尼（0.5 次幂）防迭代震荡；measured 为 0 时不更新，避免除零
            if measured > 0:
                w[d] *= (DIM_TARGET[d] / measured) ** 0.5
    return w


_WEIGHTS = _solve_weights()


def _draw_dims(category: str, rng: random.Random) -> dict[str, str]:
    """按 DIM_TARGET 无放回抽 k 个维度置活跃，再抽各自取值。

    数值规格的品类闸在**权重层**做，不在事后砍：不适配的品类权重直接置 0，适配的品类
    按 1/26.3% 反向补偿。事后砍会把 spec 的全局边际从 18% 稀释到 3.2%（第一版实测），
    而它正对着「16 寸笔记本包」那一族已知 bad case，是刻意要超配的维度。
    """
    weights = dict(_WEIGHTS)
    if not _SPEC_OK.search(category.lower()):
        weights["spec"] = 0.0  # 不适配的品类彻底不抽；适配品类的补偿已在 _solve_weights 里解出

    k = _pick_int(rng, ACTIVE_K)
    pool = [d for d, w in weights.items() if w > 0]
    active: list[str] = []
    for _ in range(min(k, len(pool))):
        chosen = rng.choices(pool, weights=[weights[d] for d in pool], k=1)[0]
        active.append(chosen)
        pool.remove(chosen)  # 无放回：同一维度不会被选两次

    dims = dict(OPTIONAL_DIMS)  # 先全关，再逐个打开被选中的
    for d in active:
        # 活跃维度内部按 DIMS 的相对权重抽取值，但排除「关闭值」——它已经被选为活跃了
        options = [(v, w) for v, w in DIMS[d] if v != OPTIONAL_DIMS[d]]
        dims[d] = _pick(rng, options)
    dims["clarity"] = _pick(rng, DIMS["clarity"])  # 形态维度独立于约束预算
    return dims


def _load_cards() -> dict[str, dict]:
    """按品类聚合卡片：bestseller 给真实商品、price_range 给价位、attribute 给典型属性。"""
    cards: dict[str, dict] = {}
    for line in CARDS_PATH.open(encoding="utf-8"):
        if not line.strip():
            continue
        d = json.loads(line)
        cat = d["category"]
        c = cards.setdefault(cat, {"category": cat, "aliases": [], "evidence": [], "attrs": []})
        # 中文别名是造中文 query 的唯一来源（品类名本身全是英文）
        c["aliases"] = list(
            dict.fromkeys(
                c["aliases"] + [a for a in d.get("aliases", []) if re.search(r"[一-鿿]", a)]
            )
        )
        if d.get("card_type") in ("bestseller", "price_range"):
            c["evidence"] += d.get("raw_evidence", [])[:5]
        elif d.get("card_type") in ("attribute", "attribute_schema"):
            c["attrs"].append(str(d.get("summary", ""))[:200])
    for c in cards.values():
        prices = [float(m.group(1)) for e in c["evidence"] if (m := re.search(r"\$([\d.]+)", e))]
        # 中位价当预算锚：LLM 拿到的是"这个品类真实卖多少钱"，造出的预算才 grounded
        c["median_price"] = round(sorted(prices)[len(prices) // 2], 2) if prices else None
    return cards


DIM_HINT = {
    "budget": {
        "none": "不提预算",
        "usd": "明确写美元预算",
        "cny": "明确写人民币预算",
        "bare": "只说数字不说币种（如「预算300」）",
        "hard_limit": "强调硬上限（如「必须5美元以内」）",
    },
    "exclude": {
        "none": "无排除项",
        "material": "排除某材质",
        "color": "排除某颜色",
        "type": "排除某子类型",
        "brand": "排除某品牌或「别太大众的牌子」",
    },
    "clarity": {
        "explicit": "直接说出品类名",
        "scene_only": "只说场景/用途，不出现品类词",
        "bundle": "要一套/一整套（多件搭配）",
    },
    "platform": {"none": "", "cross_compare": "要求跨平台比价", "specific": "指定某个平台"},
    "landed": {"none": "", "need_landed": "要求算上关税运费的到手价，并说明寄到哪个国家"},
    "spec": {"none": "", "numeric": "带一个数值规格（如 16 寸 / 256GB / 保温12小时 / M码）"},
}

PROMPT = """\
你在为一个跨境电商购物 Agent 造训练数据。请针对下面这个商品品类，造 {n} 个**中文购物意图会话**。

品类：{category}（中文可称：{aliases}）
该品类真实价位中位数：${price}
典型属性参考：{attrs}

每个会话的第 1 轮必须严格满足指定的约束维度组合，后续轮（如果有）按指定类型追问。

会话规格：
{specs}

硬性要求：
- **口语、短**。第 1 轮 12~28 字，追问轮 4~12 字。这是真人在对话框里打的字，不是需求文档。
- 预算金额要贴合该品类真实价位（参考上面的中位数，可上下浮动，但别离谱）。
- 追问轮只写增量意思，**不要重复第 1 轮已说过的内容**（真人不会重复）。
- 不要出现「我需要一款」「请帮我推荐一款符合以下要求的」这种模板腔。
- 品类词用中文口语说法（参考上面的中文别名），不要用英文品类名。
- **指定维度与该品类明显不搭时（例如给娃娃配件配「露营」场景），直接忽略那个维度**。
  宁可少一个约束，也不要造出真人不会说的话——这批数据的价值全在贴近真实分布。

严格输出 JSON 数组，{n} 个元素，每个形如：
{{"turns": ["第1轮", "追问轮（没有则省略）"]}}
不要输出任何解释文字。"""


def _spec_text(idx: int, dims: dict[str, str], followups: list[str]) -> str:
    """把维度组合翻译成 LLM 看得懂的一行规格。"""
    parts = [
        DIM_HINT["clarity"][dims["clarity"]],
        DIM_HINT["budget"][dims["budget"]],
        DIM_HINT["exclude"][dims["exclude"]],
    ]
    for k in ("platform", "landed", "spec"):
        if hint := DIM_HINT[k][dims[k]]:
            parts.append(hint)
    if dims["scene"] != "none":
        parts.append(f"场景是{dims['scene']}")
    if dims["audience"] != "self":
        parts.append(f"买给{dims['audience']}")
    if dims["soft"] != "none":
        parts.append(f"带软偏好：{dims['soft']}")
    line = f"会话{idx}：" + "；".join(p for p in parts if p)
    if followups:
        line += "；后续追问轮类型依次为：" + "、".join(followups)
    else:
        line += "；单轮，无追问"
    return line


def _plan_sessions(cards: dict[str, dict], limit: int, rng: random.Random) -> list[dict]:
    """确定性地排好每个会话的品类与维度组合——LLM 只负责措辞，不负责分布。"""
    cats = sorted(cards)
    rng.shuffle(cats)
    sessions = []
    for i in range(limit):
        cat = cats[i % len(cats)]  # 轮转保证 247 个品类均匀覆盖，不靠随机
        dims = _draw_dims(cat, rng)
        n_follow = 0
        if rng.random() < FOLLOWUP_RATIO:
            n_follow = rng.choices([1, 2], weights=[75, 25], k=1)[0]
        sessions.append(
            {
                "id": f"pq_{i:05d}",
                "category": cat,
                "dims": dims,
                "followups": [_pick(rng, FOLLOWUPS) for _ in range(n_follow)],
            }
        )
    return sessions


def _parse(text: str, expect: int) -> list[dict] | None:
    m = re.search(r"\[.*]", text, re.S)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(arr, list) or len(arr) != expect:
        return None
    if not all(
        isinstance(x, dict) and isinstance(x.get("turns"), list) and x["turns"] for x in arr
    ):
        return None
    return arr


async def generate(sessions: list[dict], cards: dict[str, dict], sink) -> int:
    """同品类的会话凑一批（共享品类卡上下文），逐批落盘。

    逐批落盘是 M21 的血泪教训：没有增量落盘时，任何一个挂起点都会让整跑的产出归零。
    """
    llm, sem = get_llm(), asyncio.Semaphore(CONCURRENCY)
    by_cat: dict[str, list[dict]] = {}
    for s in sessions:
        by_cat.setdefault(s["category"], []).append(s)
    batches = [g[i : i + BATCH] for g in by_cat.values() for i in range(0, len(g), BATCH)]
    done = [0]

    async def one(batch: list[dict]) -> None:
        card = cards[batch[0]["category"]]
        specs = "\n".join(_spec_text(i + 1, s["dims"], s["followups"]) for i, s in enumerate(batch))
        prompt = PROMPT.format(
            n=len(batch),
            category=card["category"],
            aliases="、".join(card["aliases"][:6]) or card["category"],
            price=card["median_price"] if card["median_price"] is not None else "未知",
            attrs=(card["attrs"][0][:180] if card["attrs"] else "无"),
            specs=specs,
        )
        async with sem:
            for _ in range(2):  # 只重试一次：解析失败多半是这批规格太拧巴，重试第二次也白搭
                try:
                    resp = await asyncio.wait_for(llm.ainvoke(prompt), timeout=REQ_TIMEOUT)
                except (TimeoutError, Exception):
                    continue
                arr = _parse(str(resp.content), len(batch))
                if arr:
                    sink([{**s, "turns": a["turns"]} for s, a in zip(batch, arr, strict=True)])
                    done[0] += len(batch)
                    if done[0] % 200 < BATCH:
                        print(f"  已生成 {done[0]}/{len(sessions)} 个会话", flush=True)
                    return

    await asyncio.gather(*(one(b) for b in batches))
    return done[0]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1200, help="造多少个会话")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="planner_queries.jsonl")
    args = ap.parse_args()

    if not ANCHORS_PATH.exists():
        raise SystemExit("先跑 build_planner_anchors.py——合成分布要对齐真实锚")
    cards = _load_cards()
    rng = random.Random(args.seed)
    sessions = _plan_sessions(cards, args.limit, rng)
    n_follow = sum(1 for s in sessions if s["followups"])
    pct_follow = n_follow / len(sessions)
    print(
        f"品类 {len(cards)} 个，排定 {len(sessions)} 个会话"
        f"（含追问轮 {n_follow} 个 = {pct_follow:.1%}）"
    )

    out = OUT_DIR / args.out
    fh = out.open("w", encoding="utf-8")

    def sink(rows: list[dict]) -> None:
        for r in rows:
            turns = [str(t).strip() for t in r["turns"] if str(t).strip()]
            # LLM 常把追问轮吞掉或多吐一轮；规格是确定性的，以规格为准截断/丢弃
            want = 1 + len(r["followups"])
            if len(turns) < want:
                r["followups"] = r["followups"][: len(turns) - 1]
            turns = turns[:want]
            fh.write(
                json.dumps(
                    {
                        "id": r["id"],
                        "category": r["category"],
                        "dims": r["dims"],
                        "turns": [
                            {
                                "turn": i,
                                "text": t,
                                "followup_type": (r["followups"][i - 1] if i else None),
                            }
                            for i, t in enumerate(turns)
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        fh.flush()

    n = await generate(sessions, cards, sink)
    fh.close()
    print(f"完成 {n}/{len(sessions)} 个会话 → {out.relative_to(PROJECT_ROOT)}")
    print("下一步：planner_quality_gate.py 过质量门（长度/分布/去重/维度覆盖）")


if __name__ == "__main__":
    asyncio.run(main())
