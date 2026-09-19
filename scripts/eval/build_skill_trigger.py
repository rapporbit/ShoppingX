"""生成 SKILL 触发标注集（阶段 S4）。

**量的是什么**：skill 正文改成「模型自觉调 ``Skill(skill=…)`` 取」之后（阶段 S2 推翻预注入），
「这轮该读哪份 skill」不再是一个纯函数的注入判定，而是**模型第一次调用时的一个决策**。所以这
条验收必须真跑模型，只看第一批工具调用里有没有那条 ``Skill``——见 ``run_skill_trigger.py``。

**标注集为什么内联在脚本里**：同 ``build_eval_queries.py``——``/data/*`` 被 gitignore，而人工
策划的 query 没法用算法复现，清单只有写在脚本里才进得了版本库、才 review 得了。

**每条的字段：**
- ``id``：稳定标识。
- ``skill``：期望模型这一轮读的 skill 目录名；``None`` = 负例（这轮不该读任何 skill）。
- ``bucket``：对应 ``prompt/prompts.yml`` 意图分流表的哪一行（正例）/ 哪条「不读」规则（负例）。
- ``query``：用户原话。
- ``why``：这条为什么该（不该）触发——写给将来看 bad case 的人，不进模型。

**覆盖口径**：``skills/`` 下 7 份 SKILL.md 里的 6 份各出 2 条正例，``image-shopping`` **刻意不
覆盖**——它的判据是 ``run_agent(image_paths=…)`` 而仓库内没有可复现的图源（``uploaded/`` 被
gitignore），与其拿一张跑不起来的路径凑数，不如在报告里如实标「未覆盖」。

用法：
    uv run python scripts/eval/build_skill_trigger.py
"""

from __future__ import annotations

import json
from pathlib import Path

DATASET_PATH = Path("data/eval/skill_trigger.jsonl")

#: 刻意不覆盖的 skill 与理由（原样写进产物头部，报告照抄，别让「没测」看起来像「测过没事」）。
UNCOVERED: dict[str, str] = {
    "image-shopping": "判据是 image_paths，仓库内无可复现图源（uploaded/ 被 gitignore）",
}

CASES: list[dict] = [
    # ── search-discovery：多约束 / 送礼 / 新说法 ──
    {
        "id": "sd01_multi_constraint",
        "skill": "search-discovery",
        "bucket": "几个约束叠着的需求",
        "query": "预算80美元，给宿舍买个能放得下A4纸的收纳箱，别塑料味重的，最好耐脏",
        "why": "预算 + 尺寸 + 材质排除 + 软偏好四个约束叠着，分流表第一行",
    },
    {
        "id": "sd02_gift",
        "skill": "search-discovery",
        "bucket": "买来送人",
        "query": "给男朋友挑个生日礼物，100美元以内，他平时喜欢露营",
        "why": "送礼一节（收礼人事实三处来源、稳妥/惊喜/便宜三件）只写在 search-discovery 里",
    },
    {
        "id": "sd03_trend_word",
        "skill": "search-discovery",
        "bucket": "含新说法潮流词（intent_grounding=web）",
        "query": "最近很火的那种多巴胺配色的托特包，预算50美元",
        "why": "潮流词要先意图翻译成库内词汇，翻译口径 S1 已从 system prompt 搬进这份 skill",
    },
    # ── purchase-research：还没有候选先讲怎么挑 / 评价某款 ──
    {
        "id": "pr01_how_to_choose",
        "skill": "purchase-research",
        "bucket": "还没有候选，先问「这个品类怎么挑」",
        "query": "机械键盘到底该怎么挑？我完全不懂，先给我讲讲看什么",
        "why": "宽问两步、网页内容只讲品类标准、收尾走 present_guide，全在这份 skill",
    },
    {
        "id": "pr02_evaluate",
        "skill": "purchase-research",
        "bucket": "评价某款值不值（evaluate）",
        "query": "罗技 MX Master 3S 这个鼠标值得买吗",
        "why": "点名对象的口碑评测，evaluate 分支 S1 已从 system prompt 搬进这份 skill",
    },
    # ── bundle-planning：多槽位 ──
    {
        "id": "bp01_kitchen_set",
        "skill": "bundle-planning",
        "bucket": "多槽位「一套齐」（bundle_slots ≥2）",
        "query": "下个月搬进新公寓，厨房想一次配齐：电水壶、平底锅、一套餐具，总共150美元",
        "why": "三个槽位 + 总预算，预算按槽取整分摊与一槽一条 item_search 的口径在这份 skill",
    },
    {
        "id": "bp02_parallel_categories",
        "skill": "bundle-planning",
        "bucket": "多槽位「多类并列」",
        "query": "帮我配一套入门健身的装备，瑜伽垫和哑铃都要，预算200美元",
        "why": "多类并列同样走槽位（每槽一件、改一槽只动一槽）",
    },
    # ── order-care：下单 / 查单 / 取消 / 售后 ──
    {
        "id": "oc01_query_order",
        "skill": "order-care",
        "bucket": "查单",
        "query": "我上周买的那个订单现在到哪了",
        "why": "状态必须来自本会话 query_order（不准凭空说），这条纪律写在 order-care",
    },
    {
        "id": "oc02_cancel",
        "skill": "order-care",
        "bucket": "取消",
        "query": "把我刚下的那单取消了吧，不想要了",
        "why": "取消的强制顺序（先 query_order）与「不得说已取消」的说法口径在 order-care",
    },
    # ── memory-personalization：记忆本身成为话题 ──
    {
        "id": "mp01_remember",
        "skill": "memory-personalization",
        "bucket": "记住 X",
        "query": "记住我不吃坚果，以后别给我推荐带坚果的零食",
        "why": "key 怎么起、写哪一条，口径在 memory-personalization（M4 新建）",
    },
    {
        "id": "mp02_what_do_you_know",
        "skill": "memory-personalization",
        "bucket": "你都记得我什么",
        "query": "你现在都记得我些什么？",
        "why": "翻记忆与「不能回『已删除』」的遗忘口径同在这份 skill",
    },
    # ── cross-border-duty：关税 / 运费 / 到手价 ──
    {
        "id": "cd01_duty_estimate",
        "skill": "cross-border-duty",
        "bucket": "关税口径",
        "query": "从美国买个300美元的包寄到德国，关税大概要交多少",
        "why": "税费口径与免税额分档只写在 cross-border-duty",
    },
    {
        "id": "cd02_landed_cost",
        "skill": "cross-border-duty",
        "bucket": "到手价口径",
        "query": "寄到日本的话，一副200美元的耳机到手价要算上多少运费和税",
        "why": "到手价 = 货价 + 运费 + 税的算法与收货国解析在这份 skill",
    },
    # ── 负例：这轮不该读任何 skill ──
    {
        "id": "neg01_single_item",
        "skill": None,
        "bucket": "单品直搜",
        "query": "搜一下 Anker 65W 氮化镓充电器",
        "why": "点名了要什么、一次检索就能答，分流表明写这种不读 skill",
    },
    {
        "id": "neg02_single_item_budget",
        "skill": None,
        "bucket": "单品直搜",
        "query": "我想买个 Kindle Paperwhite，150美元以内的",
        "why": "仍是点名单品，一个预算不构成「几个约束叠着」",
    },
    {
        "id": "neg03_chitchat",
        "skill": None,
        "bucket": "闲聊",
        "query": "你是谁做出来的？都能帮我干点什么",
        "why": "非购物意图直接兜底收尾",
    },
    {
        "id": "neg04_off_intent",
        "skill": None,
        "bucket": "非购物",
        "query": "帮我写一首关于秋天的短诗",
        "why": "与商品无关，走 chat_fallback",
    },
    {
        "id": "neg05_category_price_band",
        "skill": None,
        "bucket": "品类行情快捷路径",
        "borderline": True,
        "query": "现在无线鼠标大概都什么价位区间",
        "why": (
            "问的是库内供给行情 → category_insight 直接答。**与 purchase-research 的边界例**："
            "它离「这个品类怎么挑」只差一层意思，判失败时先怀疑尺子、再怀疑模型"
        ),
    },
]


def main() -> None:
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    ids = [c["id"] for c in CASES]
    dups = {i for i in ids if ids.count(i) > 1}
    if dups:
        raise SystemExit(f"标注集存在重复 id：{sorted(dups)}")

    # 覆盖自检：skills/ 下每份 skill 要么有正例，要么在 UNCOVERED 里写明为什么不测。
    on_disk = {p.name for p in Path("skills").iterdir() if (p / "SKILL.md").is_file()}
    covered = {c["skill"] for c in CASES if c["skill"]}
    missing = on_disk - covered - set(UNCOVERED)
    if missing:
        raise SystemExit(f"这些 skill 既没有正例也没登记在 UNCOVERED：{sorted(missing)}")
    unknown = covered - on_disk
    if unknown:
        raise SystemExit(f"标注集引用了不存在的 skill：{sorted(unknown)}")

    with DATASET_PATH.open("w", encoding="utf-8") as f:
        for c in CASES:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    pos = sum(1 for c in CASES if c["skill"])
    neg = len(CASES) - pos
    print(f"SKILL 触发标注集写入 {DATASET_PATH}（正例 {pos} / 负例 {neg}）")
    print(
        "按 skill："
        + "，".join(f"{s} {sum(1 for c in CASES if c['skill'] == s)}" for s in sorted(covered))
    )
    print("未覆盖：" + "，".join(f"{k}（{v}）" for k, v in UNCOVERED.items()))


if __name__ == "__main__":
    main()
