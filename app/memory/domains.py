"""品类域词表 —— 运行时只剩一个用途：``item_picker`` 的品类门锚核验。

``infer_domains_from_text`` 把「用户原文」和 planner 的 ``category`` 各投一票域，两边有词表证据
却互不相交时，说明 category 这个锚不可信，本轮不按它执法（见 ``app/tools/item_picker.py``）。

**枚举与中文标签冻结给 planner 训练腿**（``app/eval/planner_reward.py`` 与 ``scripts/train/``
的 golden / SFT 构建）：训练数据和 reward 按这套域打过分，改了就无法按原口径复现。planner
运行时已不再输出域（2026-09-25 换成 ``topic_switch``），长期记忆也早在 M4 就不按域过滤了——
别再把它接回运行时的偏好作用范围判定。
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
    # 训练 golden 里的两个非品类值：other = 归不进具体域；global = 旧的跨品类底线标记
    "other",
    "global",
]

ALL_DOMAINS: tuple[PrefDomain, ...] = get_args(PrefDomain)

#: 域 → 中文标签。只给训练腿拼 golden / SFT 的 prompt 菜单用。
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
# 反证 planner 的 category 漂移（手表 query 被判成 apparel 这类「合法但错」）。
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
    return {d for d, terms in DOMAIN_TERMS.items() if any(term_hits(t, lowered) for t in terms)}

