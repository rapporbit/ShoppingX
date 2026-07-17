"""生成 Agent 级 Rubric 评测集（购物意图 query），ROADMAP M11 / refdocs 08。

**与 ``build_category_golden.py`` 的区别：** 召回金标集能确定性构造（品类→自家卡片），
而 Agent 级评测 query 是**人工策划的测试资产**——没有算法能「生成」一条有代表性的购物
意图。所以这里的做法是：把策划好的 query 清单内联在本脚本里（可跟踪、可 review），
``data/eval/queries.jsonl`` 作为可复现产物落盘（``/data/*`` 被 gitignore，靠本脚本复现）。

**每条 query 的字段（Rubric 动态生成，这些只作分桶/对照锚点，不是硬 ground truth）：**
- ``id``：稳定标识，bad case 回溯用。
- ``bucket``：能力/意图分桶，按桶聚合分数、定位「哪类场景在掉分」。
- ``intent``：``shopping`` / ``chitchat`` / ``refuse``——决定该走哪个终结工具。
- ``query``：自然语言购物意图（用户母语中文；商品库多语言，部分 query 专测跨语言召回）。
- ``constraints``：结构化硬/软约束，喂给 Rubric 当 P0 红线校验的锚点（预算/材质/人群/黑名单）。
- ``expected_path``：**参考**工具路径（P1 执行规范对照用，非硬性——模型有合理自由度）。
- ``probe``：这条探测什么 + 主要关联哪档（P0 一票否决 / P1 扣分 / P2 打分）。

覆盖口径：九大工具 + dispatch_tool 全路径；P0（预算/性别品类冲突/安全违禁/到手价/黑名单）、
P1（收尾/澄清/不死循环/fork 合理性）、P2（需求覆盖/场景洞察/决策建议）各档；含跨语言召回与
长期记忆注入两类专项。约 18 条，刻意「少而代表」——评测要跑得起、改完能快速回归对照。

用法：
    uv run python scripts/eval/build_eval_queries.py
"""

from __future__ import annotations

import json
from pathlib import Path

QUERIES_PATH = Path("data/eval/queries.jsonl")

# 价格锚点（USD，clean 商品集实测）：p25≈12.9 / 中位≈30.6 / p75≈229。低预算 query 卡在
# p25 以下逼模型硬守预算；高客单 query（手机/箱包/键盘）天然存在超预算诱惑，专测 P0 价格红线。
QUERIES: list[dict] = [
    # ── 多约束精挑：planner → item_search → item_picker → shopping_summary ──
    {
        "id": "q01_travel_set",
        "bucket": "多约束精挑",
        "intent": "shopping",
        "query": "想买便宜又抗造的旅行收纳三件套，预算300美元，不要塑料的，喜欢小众牌子",
        "constraints": {
            "budget_usd": 300,
            "exclude_materials": ["塑料"],
            "category": "旅行收纳/箱包",
            "soft": ["抗造耐用", "小众品牌"],
        },
        "expected_path": ["planner", "item_search", "item_picker", "shopping_summary"],
        "probe": "P0 主体不超 300 / 不含塑料材质；P1 给购买理由并收尾；P2 覆盖『抗造+小众』软偏好",
    },
    {
        "id": "q16_gift_multi_constraint",
        "bucket": "多约束精挑",
        "intent": "shopping",
        "query": "送闺蜜的伴手礼，预算100美元，要可爱有质感，别太大众的牌子，不要塑料感的",
        "constraints": {
            "budget_usd": 100,
            "exclude_materials": ["塑料"],
            "audience": "女性",
            "soft": ["可爱", "质感", "小众"],
        },
        "expected_path": ["planner", "item_search", "item_picker", "shopping_summary"],
        "probe": "P0 预算/材质红线；P2 软偏好覆盖 + 『伴手礼』场景洞察力",
    },
    {
        "id": "q18_skincare_sensitive",
        "bucket": "多约束精挑",
        "intent": "shopping",
        "query": "想把全套护肤品换成成分党友好的，预算500美元，敏感肌能用的",
        "constraints": {
            "budget_usd": 500,
            "category": "美妆护肤",
            "soft": ["成分党友好", "敏感肌可用"],
        },
        "expected_path": [
            "planner",
            "category_insight",
            "item_search",
            "item_picker",
            "shopping_summary",
        ],
        "probe": "P1 不死循环、按预算收尾；P2 成分/敏感肌洞察；兼测跨语言美妆召回（belleza 等）",
    },
    # ── P0 红线陷阱：预算 / 性别品类冲突 / 安全违禁 / 到手价 ──
    {
        "id": "q02_budget_trap_earbuds",
        "bucket": "红线-预算",
        "intent": "shopping",
        "query": "想要一副降噪蓝牙耳机，预算最多80美元，音质好点",
        "constraints": {"budget_usd": 80, "category": "蓝牙耳机"},
        "expected_path": ["item_search", "item_picker", "shopping_summary"],
        "probe": "P0 主体单价 > 80 即 fail——库里有高价耳机，专测是否硬卡预算而非『推个好的』",
    },
    {
        "id": "q03_gender_mismatch_gift",
        "bucket": "红线-性别品类冲突",
        "intent": "shopping",
        "query": "给男朋友挑生日礼物，预算150美元，他喜欢运动健身",
        "constraints": {"budget_usd": 150, "audience": "男性", "category": "运动健身"},
        "expected_path": ["planner", "item_search", "item_picker", "shopping_summary"],
        "probe": "P0 推女士用品/化妆品即 fail（库内 SHEIN 女装占比高，易误召回）；P2 礼物场景洞察",
    },
    {
        "id": "q04_lookalike_explicit",
        "bucket": "明确意图检索",
        "intent": "shopping",
        "query": "帮我找点便宜的大牌平替包，越像越好",
        "constraints": {"category": "包袋", "soft": ["平价", "大牌风/平替"]},
        "expected_path": ["item_search", "item_picker", "shopping_summary"],
        "probe": "用户明确要『大牌平替』属正常需求，应正常检索给清单；不得因『仿/平替』判合规红线",
    },
    {
        "id": "q14_phone_case_low_budget",
        "bucket": "红线-预算",
        "intent": "shopping",
        "query": "买个手机壳，必须5美元以内，越便宜越好",
        "constraints": {"budget_usd": 5, "category": "手机壳"},
        "expected_path": ["item_search", "item_picker", "shopping_summary"],
        "probe": "P0 主体 > 5 即 fail——预算压在 p25（12.9）以下，测极限硬约束守得住吗",
    },
    # ── 跨平台比价 / 到手价：price_compare + shipping_calc + fork ──
    {
        "id": "q05_price_compare_samsung",
        "bucket": "跨平台比价",
        "intent": "shopping",
        "query": "同款 Samsung 手机帮我跨平台比个价，哪个到手最便宜",
        "constraints": {"brand": "Samsung", "category": "手机"},
        "expected_path": [
            "dispatch_tool",
            "item_search",
            "price_compare",
            "shipping_calc",
            "shopping_summary",
        ],
        "probe": "P1 必须调 price_compare 且汇率归一、跨平台用 fork 并行；P2 比价结论清晰可执行",
    },
    {
        "id": "q06_landed_cost_luggage",
        "bucket": "到手价",
        "intent": "shopping",
        "query": "从海外买个结实的行李箱，算上关税运费别超过200美元，帮我看到手价",
        "constraints": {"budget_usd": 200, "budget_kind": "landed", "category": "箱包/行李箱"},
        "expected_path": ["item_search", "shipping_calc", "item_picker", "shopping_summary"],
        "probe": "P1 必须调 shipping_calc 算 landed cost；P0 到手价（含税运）> 200 即 fail",
    },
    {
        "id": "q17_compare_hp_ink",
        "bucket": "跨平台比价",
        "intent": "shopping",
        "query": "我要买 HP 打印机的墨盒，对比几个平台哪个划算",
        "constraints": {"brand": "HP", "category": "打印机配件"},
        "expected_path": ["item_search", "price_compare", "shopping_summary"],
        "probe": "P1 调 price_compare；P2 品牌精确匹配 HP（库内有 HP），不串到杂牌",
    },
    # ── 品类洞察 / 外部事实：category_insight(RAG) + web_search ──
    {
        "id": "q07_category_insight_keyboard",
        "bucket": "品类洞察",
        "intent": "shopping",
        "query": "我对机械键盘不太懂，这个品类一般看哪些参数？有没有爆款",
        "constraints": {"category": "机械键盘"},
        "expected_path": ["category_insight", "item_search", "shopping_summary"],
        "probe": "P1 应调 category_insight 走 RAG 品类知识；P2 给出典型属性/爆款的洞察力",
    },
    {
        "id": "q08_web_search_review",
        "bucket": "外部事实",
        "intent": "shopping",
        "query": "最近有什么测评推荐的平价机械键盘？想入一把",
        "constraints": {"category": "机械键盘"},
        "expected_path": ["web_search", "item_search", "shopping_summary"],
        "probe": "P1 外部测评/推荐应走 web_search 取证，不得凭空编造；P2 引用具体测评来源",
    },
    # ── 跨语言召回：库内含西语/印尼语商品（ropa de hombre / deportes / kecantikan）──
    {
        "id": "q11_cross_lang_shorts",
        "bucket": "跨语言召回",
        "intent": "shopping",
        "query": "想买条男士运动短裤，便宜点能跑步穿的",
        "constraints": {"audience": "男性", "category": "运动服饰", "soft": ["便宜", "跑步可穿"]},
        "expected_path": ["item_search", "item_picker", "shopping_summary"],
        "probe": "P2 召回质量——中文 query 要能跨语言匹配西语商品（ropa de hombre / deportes）",
    },
    # ── 长期记忆注入：无显式约束，靠 <user_long_term_preferences> 兜 ──
    {
        "id": "q12_memory_injection",
        "bucket": "记忆注入",
        "intent": "shopping",
        "query": "还是按我之前说的偏好，再帮我推荐两件家居好物",
        "constraints": {"category": "家居"},
        "expected_path": ["item_search", "item_picker", "shopping_summary"],
        "probe": "P1 应读取并尊重已沉淀偏好；P0 违背已知黑名单/排除项即 fail（需先写入测试偏好）",
    },
    # ── 全链路 fork：满足 fork 三件事，走完搜→比价→到手价→精挑→收尾 ──
    {
        "id": "q15_full_chain_kitchen",
        "bucket": "全链路fork",
        "intent": "shopping",
        "query": "给新家配齐厨房好物，预算800美元，跨平台帮我搜、比价、算到手价，最后给我清单",
        "constraints": {"budget_usd": 800, "category": "厨房家居"},
        "expected_path": [
            "planner",
            "dispatch_tool",
            "item_search",
            "price_compare",
            "shipping_calc",
            "item_picker",
            "shopping_summary",
        ],
        "probe": "满链路；P1 fork 触发合理且收尾完整；P0 合计不超 800；P2 组合搭配策略",
    },
    # ── P1 专项：信息不足该澄清、非购物该兜底 ──
    {
        "id": "q13_underspecified_clarify",
        "bucket": "澄清",
        "intent": "shopping",
        "query": "随便给我推荐点东西吧",
        "constraints": {},
        "expected_path": ["chat_fallback"],
        "probe": "P1 信息严重不足应先澄清意图，不得乱搜/硬凑清单/死循环",
    },
    {
        "id": "q09_chitchat_capability",
        "bucket": "闲聊兜底",
        "intent": "chitchat",
        "query": "你都能帮我干啥呀？",
        "constraints": {},
        "expected_path": ["chat_fallback"],
        "probe": "P0/P1 必须用 chat_fallback 终结，不得误触检索工具或死循环",
    },
    {
        "id": "q10_chitchat_weather",
        "bucket": "闲聊兜底",
        "intent": "chitchat",
        "query": "今天北京天气怎么样",
        "constraints": {},
        "expected_path": ["chat_fallback"],
        "probe": "非购物意图，chat_fallback 收尾并诚实说明能力边界，不编造天气",
    },
    {
        # 2026-07-14 线上死锁回归（thread 5511f63f）：追问轮换品类，planner 误判 reuse +
        # 阶段闸拦死 item_search，模型 27 轮打转到用户取消。回归断言的是「换品类后必须正常
        # 收尾且清单是新品类」——不锁死内部走哪条路（逃生门 / 补搜 / planner 判对都算过）。
        "id": "q19_category_switch_from_reuse",
        "bucket": "多轮换品类",
        "intent": "shopping",
        "turns": [
            "我想买一套户外防水冲锋衣，不要太鲜艳的颜色，可以户外和都市通勤穿",
            "换个方向，我想要深色长袖衬衫式的上衣，带胸前口袋的，不要防水冲锋衣类型的了",
        ],
        "query": "换个方向，我想要深色长袖衬衫式的上衣，带胸前口袋的，不要防水冲锋衣类型的了",
        "constraints": {
            "category": "长袖衬衫式上衣",
            "exclude_categories": ["防水冲锋衣", "rain jacket"],
            "soft": ["深色", "带胸前口袋"],
        },
        "expected_path": ["planner", "item_search", "item_picker", "shopping_summary"],
        "probe": (
            "P0 必须给出最终清单（不许卡死 / 空手收尾），且清单主体是衬衫式上衣而非上一轮的"
            "防水冲锋衣；P1 如实说明与上一轮需求的衔接；P2 覆盖『深色+胸前口袋』软偏好"
        ),
    },
    {
        # 2026-07-14 相机 bad case 回归（threads c39e36f8/475717e0）：旧 harness 首搜召回全是
        # 配件、重试被阶段白名单拦死 → 四轮空手；预算制放行重试后才捞到真机身。同轮还暴露
        # curator 瞎换汇（10000 CNY 拍成 $1000 覆盖 planner 的 $1400，已删其产出权）。
        "id": "q20_camera_accessory_flood",
        "bucket": "配件淹没品类",
        "intent": "shopping",
        "query": "给我推荐几款相机，要求日本品牌，主要拍人像照片，预算 10000",
        "constraints": {
            "category": "相机",
            "exclude_categories": ["配件", "镜头盖", "相机包", "三脚架", "胶卷"],
            "budget": "10000 CNY（≈ $1400，不许按错误汇率缩水）",
            "soft": ["日本品牌", "适合人像"],
        },
        "expected_path": ["planner", "item_search", "item_picker", "shopping_summary"],
        "probe": (
            "P0 必须给出最终清单且主推是真相机机身/套机（召回被配件淹没时允许重试检索，"
            "不许空手收尾、不许拿配件或胶卷充当主推）；P1 预算按人民币如实折算（约 $1400 量级，"
            "不得悄悄缩成 $1000）、配件如出现在清单须明确标注不是相机；P2 给出人像适配理由"
            "（对焦/画幅/镜头焦段等）"
        ),
    },
    {
        # 「一套齐」组成已列明：planner 拆槽（evidence 全有）→ 槽位批 fork → 组合优选。
        # 不该触发 ask_user（用户逐一点名了组成，没什么可确认的）。
        "id": "q21_bundle_listed_slots",
        "bucket": "套装组合",
        "intent": "shopping",
        "query": "旅行三件套：行李箱、旅行收纳袋、洗漱包，总预算 300，不要塑料的，喜欢耐用的",
        "constraints": {
            "budget": "300 CNY（总预算，约束的是三件合计，不是单件）",
            "exclude_materials": ["塑料"],
            "slots": ["行李箱", "旅行收纳袋", "洗漱包"],
            "soft": ["耐用"],
        },
        "expected_path": ["planner", "parallel_dispatch_tool", "item_picker", "shopping_summary"],
        "probe": (
            "P0 清单必须跨品类凑齐三个点名槽位各一件（不许只出单品类清单）、三件合计不超总预算、"
            "不含塑料材质、不编造；P1 组成已列明不该反问用户、讲清预算怎么分（哪槽花钱哪槽省）；"
            "P2 覆盖『耐用』软偏好并给每槽选购理由"
        ),
    },
    {
        # 「一套齐」开放式组成：「新生入学一套」没有标准答案，planner 拆的槽全是推断
        # （evidence 空）→ 应先 ask_user 列槽请用户增删；无回复则按必备槽继续并如实交代口径。
        "id": "q22_bundle_open_composition",
        "bucket": "套装组合",
        "intent": "shopping",
        "query": "新生入学一套，预算 1500",
        "constraints": {
            "budget": "1500 CNY（总预算）",
            "composition": "开放式（组成系统推断，无标准答案）",
        },
        "expected_path": [
            "planner",
            "ask_user",
            "parallel_dispatch_tool",
            "item_picker",
            "shopping_summary",
        ],
        "probe": (
            "P0 清单总价不超总预算、至少覆盖 2 个不同子品类、没检索/没找到的槽位如实交代"
            "（不许拿别的商品冒充）；P1 组成系推断应先 ask_user 让用户确认增删（用户未回复时"
            "按建议必备项继续并说明），不许既不问也不说明就自作主张；P2 讲清预算分配与剩余"
        ),
    },
    {
        # 2026-07-15 背包 bad case 回归（thread 63093a85）：reuse 追问轮只问到手价，模型跳过
        # item_picker 直接收尾 → summary 的清单通道（get_last_picks，只认本轮定稿）为空，
        # 收尾 LLM 照「没找到」模板编出与候选池自相矛盾的答案——price_compare 明明刚算完
        # 12 件到手价。修复 = phase_check 底线 3（本轮未精挑拒收尾）。同型：q05 定点查价首轮、
        # gcjp 相机（c39e36f8）与英国文学（7400ce43）。
        "id": "q23_reuse_landed_cost_followup",
        "bucket": "多轮追问到手价",
        "intent": "shopping",
        "turns": [
            "通勤背包，能装 16 寸笔记本，防泼水，预算 400 以内",
            "他们的到手价是多少？",
        ],
        "query": "他们的到手价是多少？",
        "constraints": {
            "category": "通勤背包",
            "budget": "400 CNY",
            "must": ["16 寸笔记本适配", "防泼水"],
        },
        "expected_path": ["planner", "price_compare", "item_picker", "shopping_summary"],
        "probe": (
            "P0 必须给出带到手价（含税运口径）的商品清单，绝不许答『没找到符合条件的商品』"
            "——上一轮已有候选且本轮比价已算出到手价，空清单即与自身候选池矛盾；"
            "P1 到手价须交代收货国口径；P2 清单延续上一轮的背包候选而非重新检索一批新的"
        ),
    },
    # ── 约束类型学巡检（app/utils/terms.py CONSTRAINT_LANES 的 gap 类，每类一条）──
    # 目的不是「必须全对」，是**把裸奔面量出来**：这些约束现状走 topic 语义道 = 对算子双盲
    # （embedding 对数字/枚举失明、字面匹不中变体）。跑分暴露哪类真坏、坏得多严重，
    # 补专道的优先级按证据排——不等下一个生产 badcase 来定。
    {
        "id": "q24_enum_size",
        "bucket": "约束类型学巡检",
        "intent": "shopping",
        "query": "想买一件男士纯棉短袖T恤，要M码的，预算30美元",
        "constraints": {"category": "T恤", "must": ["M码", "纯棉"], "budget": "30 USD"},
        "expected_path": ["planner", "item_search", "item_picker", "shopping_summary"],
        "probe": (
            "P0 清单是成人男士 T 恤且不超预算；P1 尺码是互斥枚举——主推标题若明写 XL/XXL "
            "而无 M 可选即冲突（探测 enum_size 无专道的双盲）；P2 纯棉偏好有覆盖"
        ),
    },
    {
        "id": "q25_count_pack",
        "bucket": "约束类型学巡检",
        "intent": "shopping",
        "query": "买两个装的不锈钢保温杯，预算50美元",
        "constraints": {"category": "保温杯", "must": ["两个装", "不锈钢"], "budget": "50 USD"},
        "expected_path": ["planner", "item_search", "item_picker", "shopping_summary"],
        "probe": (
            "P0 清单是保温杯且不超预算；P1 装量约束——主推明写 1-pack/单只而清单不说明时"
            "即漏（探测 count_pack 无专道）；P2 不锈钢材质有覆盖"
        ),
    },
    {
        "id": "q26_numeric_range",
        "bucket": "约束类型学巡检",
        "intent": "shopping",
        "query": "要一个能装17寸以上笔记本的双肩包，预算60美元",
        "constraints": {"category": "双肩包", "must": ["17寸以上适配"], "budget": "60 USD"},
        "expected_path": ["planner", "item_search", "item_picker", "shopping_summary"],
        "probe": (
            "P0 清单是双肩包且不超预算；P1 范围算子——主推标题明写只装 14/15.6 寸即冲突"
            "（探测 numeric_range：spec 专道只有等值±容差，无比较算子）；P2 适配性有说明"
        ),
    },
    {
        "id": "q27_storage_unit",
        "bucket": "约束类型学巡检",
        "intent": "shopping",
        "query": "想要一个256GB的U盘，传文件用，预算40美元",
        "constraints": {"category": "U盘", "must": ["256GB"], "budget": "40 USD"},
        "expected_path": ["planner", "item_search", "item_picker", "shopping_summary"],
        "probe": (
            "P0 清单是 U 盘/闪存盘且不超预算；P1 容量档位——主推明写 64GB/128GB 即冲突"
            "（探测 numeric_spec 单位表缺 GB/TB：长度/升有专道，存储没有）；P2 传输速度等有提示"
        ),
    },
    {
        "id": "q28_generation",
        "bucket": "约束类型学巡检",
        "intent": "shopping",
        "query": "想买第3代的苹果无线耳机，预算200美元",
        "constraints": {"category": "无线耳机", "must": ["第3代", "苹果"], "budget": "200 USD"},
        "expected_path": ["planner", "item_search", "item_picker", "shopping_summary"],
        "probe": (
            "P0 清单是苹果无线耳机且不超预算；P1 代际是互斥枚举——主推明写 2nd generation "
            "而不说明时即漏（探测 generation 无专道）；P2 代际差异（降噪/续航）有说明"
        ),
    },
    # ── 记忆专项（P_t 单写者+id 增量重构验收，docs/plans/P_t重构-单写者id增量-执行计划.md §1.3）──
    # 每条钉一个历史 bug 形态/不变量。改造前预期 M2/M3/M4 挂；改造后要求全绿——「挂转绿」
    # 就是重构的直接证据。多轮 case 只对最后一轮打分（run_rubric 的 turns 语义）。
    {
        "id": "m1_constraint_persistence",
        "bucket": "记忆专项",
        "intent": "shopping",
        "turns": [
            "想买一个厨房砧板，不要塑料的，预算100",
            "最好是深色系的",
        ],
        "query": "最好是深色系的",
        "constraints": {
            "category": "砧板",
            "exclude_materials": ["塑料"],
            "budget": "100 CNY",
            "soft": ["深色系"],
        },
        "expected_path": ["planner", "item_picker", "shopping_summary"],
        "probe": (
            "P0 清单是砧板、不超预算且不含塑料材质——T1 的硬排除在 T2 只补颜色偏好后必须"
            "仍然生效（探 I1 存续：约束不因后续轮未重述而静默消失）；P2 深色偏好有覆盖"
        ),
    },
    {
        "id": "m2_explicit_retract",
        "bucket": "记忆专项",
        "intent": "shopping",
        "turns": [
            "想买厨房收纳盒，不要塑料的",
            "要大一点的，能放下调料瓶",
            "算了，塑料的也行，给我便宜实惠的",
        ],
        "query": "算了，塑料的也行，给我便宜实惠的",
        "constraints": {
            "category": "厨房收纳盒",
            "retracted": ["不要塑料（T3 已明确撤回，不得继续生效）"],
            "soft": ["大容量", "便宜实惠"],
        },
        "expected_path": ["planner", "item_picker", "shopping_summary"],
        "probe": (
            "P0 撤回必须精确生效：不许再声称「已为你排除塑料」、不许因塑料排除杀空候选池"
            "（探 I2 撤回）；P1 塑料款重新可选——清单允许含塑料，全非塑料时须另有理由而非"
            "沿用旧排除；P2 覆盖大容量与实惠"
        ),
    },
    {
        "id": "m3_soft_to_hard_upgrade",
        "bucket": "记忆专项",
        "intent": "shopping",
        "turns": [
            "想买一条通勤半身裙，尽量别太花哨",
            "还是说死吧：绝对不要花哨的，素色最好",
        ],
        "query": "还是说死吧：绝对不要花哨的，素色最好",
        "constraints": {
            "category": "半身裙",
            "exclude": ["花哨（T2 已由软偏好升级为硬排除）"],
            "soft": ["素色", "通勤"],
        },
        "expected_path": ["planner", "item_picker", "shopping_summary"],
        "probe": (
            "P0 清单全为素色/低调款，「花哨」按硬排除执行（探归并：软偏好改口升级为硬约束后"
            "只剩一条硬的，不许新旧两条并存打架、不许仍按软偏好放行花哨款）；P2 通勤场景理由"
        ),
    },
    {
        "id": "m4_topic_switch_epoch",
        "bucket": "记忆专项",
        "intent": "shopping",
        "turns": [
            "想买不锈钢保温杯，不要粉色的，预算100",
            "保温杯不买了，看看跑步鞋吧，预算500",
        ],
        "query": "保温杯不买了，看看跑步鞋吧，预算500",
        "constraints": {
            "category": "跑步鞋",
            "budget": "500 CNY（新意图口径，不是旧的 100）",
            "stale": ["不锈钢/不要粉色/预算100 均属上一意图，不得压制本轮"],
        },
        "expected_path": ["planner", "item_search", "item_picker", "shopping_summary"],
        "probe": (
            "P0 清单是跑步鞋而非保温杯，预算按新的 500 口径——不许仍按旧 100 过滤把鞋杀光"
            "（探 I3 换题清代：旧意图约束不跨 epoch 压制新检索）；P1 不把保温杯的约束"
            "（不锈钢、不要粉色）当作对跑步鞋生效或挂在嘴上"
        ),
    },
    {
        "id": "m5_budget_release",
        "bucket": "记忆专项",
        "intent": "shopping",
        "turns": [
            "推荐几款机械键盘，预算50美元",
            "不限预算了，直接上最好的",
        ],
        "query": "不限预算了，直接上最好的",
        "constraints": {
            "category": "机械键盘",
            "budget": "已放开（clear_budget，旧 50 USD 上限不得继续过滤）",
        },
        "expected_path": ["planner", "item_picker", "shopping_summary"],
        "probe": (
            "P0 预算放开必须生效：主推可以且应当出现超 50 美元的高端款，不许仍按 50 美元"
            "过滤、不许声称受预算限制（探 clear_budget）；P2 讲清「最好」好在哪（轴体/做工/"
            "无线方案等），并如实标价"
        ),
    },
]


def main() -> None:
    QUERIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    # id 唯一性自检——手维护清单最容易复制粘贴撞 id，撞了评测汇总会静默覆盖。
    ids = [q["id"] for q in QUERIES]
    dups = {i for i in ids if ids.count(i) > 1}
    if dups:
        raise SystemExit(f"评测集存在重复 id：{sorted(dups)}")

    with QUERIES_PATH.open("w", encoding="utf-8") as f:
        for q in QUERIES:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    buckets: dict[str, int] = {}
    for q in QUERIES:
        buckets[q["bucket"]] = buckets.get(q["bucket"], 0) + 1
    print(f"评测集写入 {QUERIES_PATH}（{len(QUERIES)} 条 query）")
    print("分桶分布：" + "，".join(f"{b} {n}" for b, n in sorted(buckets.items())))


if __name__ == "__main__":
    main()
