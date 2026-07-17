"""偏好域 ``PrefDomain`` —— 一条偏好「管哪个品类」的封闭枚举。

**为什么必须是封闭枚举（而不是自由文本）。** 域的用途是**限定偏好的作用范围**：「买鞋时不喜欢
皮革」不该在买沙发、买表带时也把含 leather 的商品全杀掉。要做到这一点，读取端必须能判断
「这条偏好的域」和「本轮在买什么」是不是同一个域——自由文本做不到（``shoes`` / ``footwear`` /
``跑鞋`` / ``running shoes`` 对不上），而语义匹配要在 item_picker 这条热路径上加一次 embedding
网络往返，为一个字段匹配付这个延迟不值。

收敛成枚举后，**读取端就是一次字符串相等比较**：匹配成本被挪到写入侧（curator 后处理、偏好页面
解析，都不在用户感知的延迟里）与 planner（本来就在跑 LLM，多要一个字段零额外调用）。

**两个逃生舱的语义天差地别，别搞反：**

- :data:`DOMAIN_OTHER` —— 「判不出是哪个域」。**只在本轮生效**，是保守档。LLM 漏判时落这里。
- :data:`DOMAIN_GLOBAL` —— 「跨品类底线」（安全 / 过敏 / 伦理，如「我素食，任何动物皮革都不要」）。
  **全局生效**，是激进档，必须由 LLM 明确判断或用户显式指定，**绝不能是漏填的默认值**。

选项越多，LLM 漏判越多，所以默认值的失效方向就越重要：漏填落 ``other``（最坏是这条偏好本轮
没生效，用户再说一遍即可），而不是 ``global``（最坏是它在所有品类里静默杀商品，用户只会觉得
「这破 Agent 老是搜不出东西」，且**归因不了**）。

划分粒度参考 ``app.recall.duty`` 的品类税率表——那是本仓库已经在用的一套真实品类切分。但**不与它
耦合**：duty 吃的是商品自带的原始 category 文本（各平台体系不同、脏），本模块是**意图侧**的域，
两者不是一回事，强行共用一份枚举只会让两边都别扭。
"""

from __future__ import annotations

from typing import Literal, get_args

PrefDomain = Literal[
    # 穿戴
    "apparel",  # 服饰（上衣 / 裤装 / 外套 / 内衣）
    "footwear",  # 鞋履
    "bags",  # 箱包（背包 / 手袋 / 行李箱）
    "jewelry_watches",  # 首饰 / 腕表
    # 电子
    "electronics",  # 消费电子（耳机 / 音箱 / 影音 / 相机）
    "computers",  # 电脑及配件（笔记本 / 键鼠 / 显示器）
    "phones",  # 手机及配件
    # 家居
    "home_kitchen",  # 家居日用 / 厨房用具
    "furniture",  # 家具
    "garden",  # 园艺 / 户外庭院
    # 个护 / 健康 / 食品
    "beauty",  # 美妆个护
    "health",  # 保健 / 医疗器械
    "food",  # 食品饮料
    # 兴趣 / 其他实体
    "sports",  # 运动户外装备
    "toys_baby",  # 玩具 / 母婴
    "books_media",  # 图书 / 影音媒体
    "auto",  # 汽车用品
    "pet",  # 宠物用品
    "office",  # 办公文具
    "tools",  # 工具 / 五金
    # 逃生舱（语义见模块 docstring，别搞反）
    "other",  # 判不出具体域 → 只在本轮生效（保守）
    "global",  # 跨品类底线 → 全局生效（激进，须明确判定）
]

#: 判不出域时的**保守**默认：只在本轮生效。LLM 漏填一律落这里。
DOMAIN_OTHER: PrefDomain = "other"
#: 跨品类底线（安全 / 过敏 / 伦理）：全局生效。**不是**漏填的默认值。
DOMAIN_GLOBAL: PrefDomain = "global"

ALL_DOMAINS: tuple[PrefDomain, ...] = get_args(PrefDomain)

#: 域 → 中文标签。两处用：拼进 LLM 的 prompt（让它知道每个域装什么），以及前端偏好页面展示。
DOMAIN_LABELS: dict[PrefDomain, str] = {
    "apparel": "服饰",
    "footwear": "鞋履",
    "bags": "箱包",
    "jewelry_watches": "首饰腕表",
    "electronics": "消费电子",
    "computers": "电脑及配件",
    "phones": "手机及配件",
    "home_kitchen": "家居厨房",
    "furniture": "家具",
    "garden": "园艺户外",
    "beauty": "美妆个护",
    "health": "保健医疗",
    "food": "食品饮料",
    "sports": "运动户外",
    "toys_baby": "玩具母婴",
    "books_media": "图书影音",
    "auto": "汽车用品",
    "pet": "宠物用品",
    "office": "办公文具",
    "tools": "工具五金",
    "other": "判不出具体品类（只在本轮生效）",
    "global": "跨品类底线：安全 / 过敏 / 伦理（全局生效，慎用）",
}


# 域 → **高精度品类核心词**（zh + en）。用途是给「用户原文词面」一票确定性的域判定，
# 反证 LLM 的 domains / category 漂移（手表 query 被判成 apparel 这类「合法但错」）。
#
# 收词纪律：**宁漏勿错**。漏 = 无反证证据、一切照旧（中性）；错 = 反证本身反转（比漂移更糟）。
# 所以只收「出现即几乎必然在买该品类」的名词：歧义词一律不收（"dress" 会命中 dress watch、
# "ring" 会命中 phone ring）。词表覆盖不求全——它是反证信号，不是分类器。
_T = tuple[str, ...]
DOMAIN_TERMS: dict[PrefDomain, _T] = {
    "apparel": (
        *("shirt", "jacket", "hoodie", "sweater", "jeans"),
        *("衬衫", "外套", "卫衣", "毛衣", "牛仔裤", "连衣裙"),
    ),
    "footwear": ("shoes", "sneakers", "boots", "sandals", "跑鞋", "球鞋", "靴子", "凉鞋", "拖鞋"),
    "bags": (
        *("backpack", "handbag", "suitcase", "luggage", "tote"),
        *("背包", "手提包", "行李箱", "书包", "钱包"),
    ),
    "jewelry_watches": (
        *("watch", "watches", "necklace", "bracelet", "earrings"),
        *("手表", "腕表", "项链", "手链", "耳环", "首饰"),
    ),
    "electronics": (
        *("headphones", "earbuds", "speaker", "camera", "projector"),
        *("耳机", "音箱", "相机", "投影仪"),
    ),
    "computers": ("laptop", "keyboard", "monitor", "笔记本电脑", "键盘", "显示器", "鼠标"),
    "phones": ("smartphone", "phone", "iphone", "手机"),
    "home_kitchen": ("cookware", "blender", "kettle", "厨具", "锅具", "餐具", "保温杯", "水壶"),
    "furniture": (
        *("sofa", "couch", "desk", "bookshelf", "mattress"),
        *("沙发", "书桌", "椅子", "床垫", "书架", "衣柜"),
    ),
    "garden": ("gardening", "planter", "园艺", "庭院", "花盆"),
    "beauty": (
        *("lipstick", "shampoo", "skincare", "perfume", "sunscreen"),
        *("口红", "洗发水", "护肤", "香水", "防晒霜"),
    ),
    "health": ("vitamin", "supplement", "维生素", "保健品", "血压计", "体温计"),
    "food": ("snacks", "coffee beans", "零食", "咖啡豆", "茶叶"),
    "sports": (
        *("yoga mat", "dumbbell", "tent", "sleeping bag"),
        *("瑜伽垫", "哑铃", "帐篷", "睡袋", "护膝"),
    ),
    "toys_baby": ("lego", "stroller", "diaper", "玩具", "婴儿车", "尿布", "积木", "奶瓶"),
    "books_media": ("novel", "textbook", "小说", "图书", "教材", "绘本"),
    "auto": ("tire", "dash cam", "轮胎", "行车记录仪", "车载"),
    "pet": ("dog food", "cat litter", "leash", "猫粮", "狗粮", "猫砂", "宠物"),
    "office": ("stationery", "printer", "文具", "打印机", "订书机"),
    "tools": ("screwdriver", "wrench", "electric drill", "螺丝刀", "扳手", "电钻", "五金"),
}


def infer_domains_from_text(text: str) -> set[PrefDomain]:
    """文本词面 → 品类域集合（确定性投票，宁漏勿错）。

    命中口径**必须**复用 :func:`app.utils.terms.term_hits`（词边界 + 否定修饰）——全链路
    「命中怎么算」一个口径；裸 ``in`` 会让 "watch" 命中 "watching"、词表精度纪律作废。
    返回空集 = 词表覆盖不到（跨语言表述 / 未收词），**不是**「不属于任何域」——消费方
    据此把空集当「无证据、维持现状」处理，绝不能当反证用。
    """
    if not text:
        return set()
    from app.utils.terms import term_hits

    lowered = text.lower()
    return {
        d for d, terms in DOMAIN_TERMS.items() if any(term_hits(t, lowered) for t in terms)
    }


def reconcile_domains(domains: list[PrefDomain], evidence_text: str) -> list[PrefDomain]:
    """LLM 判的域 ∪ 词面证据判的域——**并入不替换**（域反证的执行动作）。

    并入的失效方向是「多注入一个域的偏好」（软性加减分，可见可纠）；替换的失效方向是
    「把 LLM 判对的域丢了」（域隔离静默失效）。词面证据为空（词表覆盖不到）时原样返回。
    """
    evidence = infer_domains_from_text(evidence_text)
    missing = sorted(evidence - set(domains))
    return [*domains, *missing] if missing else domains


def domain_menu() -> str:
    """渲染成一段「域 = 说明」清单，拼进 planner / curator / parser 的 prompt 供 LLM 选。

    单一事实来源：枚举改了，三处 prompt 自动跟着变，不会出现「prompt 里还留着已删的域」。
    """
    return "\n".join(f"- {d}：{DOMAIN_LABELS[d]}" for d in ALL_DOMAINS)
