"""S0-1 源③：按**已知 bad case 族**定向造对抗样本。

源②（`build_planner_queries.py`）按维度矩阵铺的是**平均分布**——它保证覆盖面，但对
「模型偏偏在这类 query 上翻车」的长尾无能为力。这批数据的作用相反：**刻意超配历史翻车点**，
让 GRPO 的 `R_retrieval` 有足够梯度打在真正疼的地方。ROADMAP M23 S0-1 定的占比不低于 20%。

族的来源不是拍脑袋，全是本仓库真实发生过、且**修法被实测证伪过至少一次**的 bad case：

| 族 | 历史 | 为什么 prompt 修不好 |
| --- | --- | --- |
| `accessory_flood` | 「搜手机出配件」 | 根因是数据不是算法：整机与配件在源数据里同类目（TCL 标成 Nintendo DS）。黑名单闸 + 品类过滤两方案均被实测证伪 |
| `usage_confusion` | 篮球包 vs 通勤包 | 展示门只挡跨品类垃圾，挡不住同品类用途混淆（cross-encoder 0.97 交叠） |
| `numeric_spec` | 16 寸笔记本包 | 数值是 embedding 处理不了的算子，只能靠 planner 把它抽成硬约束走专道 |
| `domain_drift` | 「正式场合戴的手表」 | planner 域漂移，prompt 治不死（已记档） |
| `pollution_backfill` | 手表 badcase 污染 | 召回池被同品牌其他品类灌满，需要补搜闸 |
| `cross_lingual` | 中文口语 → 英文商品库 | 小众品类的中文说法在库里没有字面对应 |
| `gender_age` | 性别/品类冲突红线 | P0 红线，判错直接一票否决 |
| `retract` | 撤回约束 3/3 全挂 | 已修（P_t 单写者重构），但机制脆弱，要有回归样本守着 |

每族给 2~4 条**人工种子**（真实 bad case 的原话或其变体），LLM 只做「换品类的同构扩写」，
不许自由发挥——扩写跑偏就失去了"对着痛点"的意义。

用法：``uv run python scripts/train/build_planner_adversarial.py --limit 300``
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
OUT_DIR = PROJECT_ROOT / "data" / "train"
CONCURRENCY = 12
REQ_TIMEOUT = 150

# family → (配额权重, 适用品类正则, 种子模板, 扩写指令)
FAMILIES: dict[str, dict] = {
    "accessory_flood": {
        "weight": 18,
        "cat": re.compile(r"phone|camera|laptop|computer|console|printer|headphone|tablet|tv|watch"),
        "seeds": ["想买手机，安卓系统，拍照好的", "我要买 HP 打印机的墨盒，对比几个平台哪个划算", "想买第3代的苹果无线耳机，预算200美元"],
        "hint": "用户要的是**整机/正品本体**，但这个品类的配件（壳、膜、线、耗材、支架）在库里数量碾压本体。"
                "造的 query 要像真人一样只说品类不强调「整机」——正因为不强调，模型才容易召回一堆配件。",
    },
    "usage_confusion": {
        "weight": 14,
        "cat": re.compile(r"bag|backpack|handbag|luggage|shoes|clothing|apparel|bottle|chair"),
        "seeds": ["想买个通勤双肩包，预算 80 美元，要能装 16 寸笔记本", "帮我找一个便宜耐用的旅行背包，防水尼龙材质，不要塑料的，预算 50 美元"],
        "hint": "同一品类下**用途完全不同**的两拨货（篮球包/通勤包/登山包）。query 要把用途说清楚，"
                "但不出现能直接字面匹配的词——考的是模型把用途抽成软偏好而不是丢掉。",
    },
    "numeric_spec": {
        "weight": 14,
        "cat": re.compile(r"laptop|computer|monitor|storage|drive|memory|bottle|clothing|shoes|tv|battery|luggage|backpack"),
        "seeds": ["要能装17寸以上笔记本的双肩包，预算60美元", "想要一个256GB的U盘，传文件用，预算40美元", "最好能保温12小时以上", "想买一件男士纯棉短袖T恤，要M码的，预算30美元"],
        "hint": "必须带一个**可比较的数值规格**（尺寸/容量/时长/码数/功率），且数值要是该品类真实存在的档位。"
                "这类约束 embedding 抓不住，只能靠 planner 抽成硬约束。",
    },
    "domain_drift": {
        "weight": 14,
        "cat": re.compile(r".*"),
        "seeds": ["给我推荐一款正式场合适合戴的手表", "给新家配齐厨房好物，预算800美元", "新生入学一套，预算 1500", "想把全套护肤品换成成分党友好的，预算500美元，敏感肌能用的"],
        "hint": "**不直接说品类**，只给场合/身份/人生阶段，让模型自己推品类域。这正是 planner 判错域的高发形态。",
    },
    "pollution_backfill": {
        "weight": 10,
        "cat": re.compile(r"watch|phone|shoes|bag|camera|headphone"),
        "seeds": ["给我推荐几款相机，要求日本品牌，主要拍人像照片，预算 10000", "帮我找点便宜的大牌平替包，越像越好"],
        "hint": "带**品牌或价位锚**的 query——召回池容易被同品牌的其它品类灌满（搜某牌相机返回该牌背带）。",
    },
    "cross_lingual": {
        "weight": 10,
        "cat": re.compile(r".*"),
        "seeds": ["我要一套床上三件套，纯棉的，预算 1000，要有东方元素（如刺绣、花鸟纹样）", "想要更明显的东方元素（如刺绣、花鸟纹样）"],
        "hint": "用**地道中文说法**（含中式风格词、方言化叫法、网络流行叫法），商品库里全是英文标题，"
                "字面对不上，只能靠向量跨语言对齐。",
    },
    "gender_age": {
        "weight": 10,
        "cat": re.compile(r"men|women|girls|boys|baby|kids|maternity|clothing|shoes|toys"),
        "seeds": ["想买条男士运动短裤，便宜点能跑步穿的", "帮我挑几件夏天穿的男士短袖，预算 40 美元以内", "送闺蜜的伴手礼，预算100美元，要可爱有质感"],
        "hint": "**性别或年龄段是硬约束**（男士/女款/童装/孕妇/老人）。判错即 P0 红线一票否决，"
                "所以要造得容易判错——比如「给男朋友买」但品类偏女性向。",
    },
    "retract": {
        "weight": 10,
        "cat": re.compile(r".*"),
        "seeds": ["算了，塑料的也行，给我便宜实惠的", "想了想白色其实也行", "算了，皮革的也行", "预算提到600吧"],
        "hint": "**必须是两轮**：第 1 轮立一个硬约束，第 2 轮明确撤回或放宽它。考的是 P_t 能否把旧约束真正拿掉。",
    },
}

PROMPT = """你在为跨境电商购物 Agent 造**对抗性**训练数据——专门针对模型已知会翻车的一类 query。

这一类叫「{family}」，特征是：
{hint}

真实翻车样例（照着它们的**形态**写，不是照抄内容）：
{seeds}

请针对下面这个品类，造 {n} 个同构的中文购物 query：
品类：{category}（中文可称：{aliases}）
该品类真实价位中位数：${price}

硬性要求：
- 口语、短（12~28 字），像真人在对话框里打的字。
- 必须**保持这一类的翻车特征**——这是这批数据唯一的价值，写成普通 query 就白造了。
- 涉及金额时贴合上面的真实价位。
{multi_turn}
严格输出 JSON 数组，{n} 个元素，每个形如 {{"turns": ["第1轮"{turn_hint}]}}，不要输出解释文字。"""


def _load_cards() -> dict[str, dict]:
    """只取造 query 必需的三样：中文别名、价位锚、品类名。"""
    cards: dict[str, dict] = {}
    for line in CARDS_PATH.open(encoding="utf-8"):
        if not line.strip():
            continue
        d = json.loads(line)
        c = cards.setdefault(d["category"], {"category": d["category"], "aliases": [], "prices": []})
        c["aliases"] = list(dict.fromkeys(c["aliases"] + [a for a in d.get("aliases", []) if re.search(r"[一-鿿]", a)]))
        c["prices"] += [float(m.group(1)) for e in d.get("raw_evidence", []) if (m := re.search(r"\$([\d.]+)", e))]
    for c in cards.values():
        c["median_price"] = round(sorted(c["prices"])[len(c["prices"]) // 2], 2) if c["prices"] else None
    return cards


def _plan(cards: dict[str, dict], limit: int, rng: random.Random) -> list[tuple[str, str, int]]:
    """排 (family, category, n) 三元组：每族按权重分配额，族内在**适用品类**上轮转。

    适用品类是族定义里的正则——「配件淹没」只在手机/相机/打印机这类有海量配件的品类上成立，
    在「园艺工具」上造这种 query 是无中生有。
    """
    total_w = sum(f["weight"] for f in FAMILIES.values())
    plans: list[tuple[str, str, int]] = []
    for name, fam in FAMILIES.items():
        quota = max(1, round(limit * fam["weight"] / total_w))
        cats = [c for c in sorted(cards) if fam["cat"].search(c.lower())]
        if not cats:
            print(f"  [warn] {name} 没有适用品类，跳过")
            continue
        rng.shuffle(cats)
        per = 3  # 每次请求造 3 条：同族同品类再多就开始重复措辞
        for i in range(0, quota, per):
            plans.append((name, cats[(i // per) % len(cats)], min(per, quota - i)))
    rng.shuffle(plans)  # 打散：同族请求扎堆会让并发全打在一个 prompt 形态上
    return plans


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
    if not all(isinstance(x, dict) and isinstance(x.get("turns"), list) and x["turns"] for x in arr):
        return None
    return arr


async def generate(plans: list[tuple[str, str, int]], cards: dict[str, dict], sink) -> int:
    llm, sem = get_llm(), asyncio.Semaphore(CONCURRENCY)
    done = [0]

    async def one(family: str, category: str, n: int) -> None:
        fam, card = FAMILIES[family], cards[category]
        two_turn = family == "retract"
        prompt = PROMPT.format(
            family=family, hint=fam["hint"],
            seeds="\n".join(f"- {s}" for s in fam["seeds"]),
            n=n, category=category,
            aliases="、".join(card["aliases"][:6]) or category,
            price=card["median_price"] if card["median_price"] is not None else "未知",
            multi_turn="- **每条必须两轮**：第 2 轮是简短追问（4~12 字）。\n" if two_turn else "",
            turn_hint=', "第2轮"' if two_turn else "",
        )
        async with sem:
            for _ in range(2):
                try:
                    resp = await asyncio.wait_for(llm.ainvoke(prompt), timeout=REQ_TIMEOUT)
                except (TimeoutError, Exception):
                    continue
                if arr := _parse(str(resp.content), n):
                    sink(family, category, arr)
                    done[0] += n
                    if done[0] % 60 < n:
                        print(f"  已生成 {done[0]} 条", flush=True)
                    return

    await asyncio.gather(*(one(f, c, n) for f, c, n in plans))
    return done[0]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="planner_adversarial.jsonl")
    args = ap.parse_args()

    cards = _load_cards()
    rng = random.Random(args.seed)
    plans = _plan(cards, args.limit, rng)
    print(f"{len(FAMILIES)} 个 bad case 族，排定 {len(plans)} 次请求 / 约 {sum(p[2] for p in plans)} 条")

    out = OUT_DIR / args.out
    fh = out.open("w", encoding="utf-8")
    state = {"i": 0}

    def sink(family: str, category: str, arr: list[dict]) -> None:
        for a in arr:
            turns = [str(t).strip() for t in a["turns"] if str(t).strip()]
            if not turns:
                continue
            fh.write(json.dumps({
                "id": f"adv_{state['i']:05d}", "category": category, "family": family,
                "source": "adversarial",
                "turns": [
                    {"turn": i, "text": t, "followup_type": ("retract_constraint" if i and family == "retract" else ("add_constraint" if i else None))}
                    for i, t in enumerate(turns)
                ],
            }, ensure_ascii=False) + "\n")
            state["i"] += 1
        fh.flush()

    n = await generate(plans, cards, sink)
    fh.close()
    print(f"完成 {n} 条 → {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    asyncio.run(main())
