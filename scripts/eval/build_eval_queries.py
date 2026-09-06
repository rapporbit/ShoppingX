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
        "expected_path": ["planner", "task_dispatch", "item_picker", "shopping_summary"],
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
            "task_dispatch",
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
        # 「多类并列」两类：planner 拆槽 + slot_mode=parallel → 同轮两条 task_dispatch 并行
        # → picker 每类各给几件（**一类都不许砍**）。与 q21/q22 的对照点：那两条是「一套齐」
        # （配套、共享总预算、可砍可选槽），这条是并列（各买各的、预算是每件上限）。
        "id": "pl01_parallel_two_categories",
        "bucket": "多类并列",
        "intent": "shopping",
        "query": "想买双跑鞋，再配个降噪耳机，各 500 以内",
        "constraints": {
            "budget": "500 CNY（**每件**上限，不是两件合计）",
            "slots": ["跑鞋", "降噪耳机"],
            "mode": "parallel（两类互不相干，不配套）",
        },
        "expected_path": ["planner", "task_dispatch", "item_picker", "shopping_summary"],
        "probe": (
            "P0 清单必须**两类都有**（只出跑鞋或只出耳机即失败）、每件不超 500、不编造；"
            "P1 不该把两类价格加总说成「这一套合计」、不该反问用户要不要凑成一套、"
            "两类分开讲不混在一段；P2 每类各给选购理由与类内取舍"
        ),
    },
    {
        # 三类并列：真正压「同轮多派」这条路——三条 task_dispatch 该在同一轮里一起发出去，
        # 而不是一轮派一条串着等（后者功能上也对，只是三倍延迟）。
        "id": "pl02_parallel_three_categories",
        "bucket": "多类并列",
        "intent": "shopping",
        "query": "最近想置办点东西：一个机械键盘、一副降噪耳机，还有一双跑鞋，每样 800 以内",
        "constraints": {
            "budget": "800 CNY（每件上限）",
            "slots": ["机械键盘", "降噪耳机", "跑鞋"],
            "mode": "parallel",
        },
        "expected_path": ["planner", "task_dispatch", "item_picker", "shopping_summary"],
        "probe": (
            "P0 三类都要有（少一类即失败）、每件不超 800、没找到货的那类如实说而不是拿别的顶；"
            "P1 三类应在同一轮里并行检索（不该串行派三趟）、不加总价、不说成「一套」；"
            "P2 按类分段、每类给理由"
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
    # ══ M23 S0-4 扩容（33 → 90 条）═══════════════════════════════════════════════
    #
    # **为什么要扩**：M22 拿 33 条做端到端 A/B，实测噪声 ±13.2 分，planner 单点那点改进量
    # 直接被噪声吃掉，做完得不出结论。噪声按 √n 收敛，33 → 90 大约把它压到 ±8。这是扩容
    # 唯一的目的，不是为了「覆盖更全」听起来好看。
    #
    # **配比：靶心族 ~28 条 + 常规分布 ~29 条，各占一半。** 只堆靶子会把尺子做偏——S4 的
    # 验收是「均分不降 **且** 域漂移定向转绿」，均分那一维得由常规分布来量。M22 的教训正是
    # 优化了一个线上没人消费的目标。
    #
    # **id 用族前缀**（dr/uc/af/ns/xl/ga/pb/g），bucket 用「靶-xxx」统一打头：rubric 报告按
    # bucket 汇总时就能直接切出「域漂移这一族转绿没有」，不必事后再对着 id 手工分组。
    # 族与 planner 训练集的 bad case 族一一对应（见 build_planner_adversarial.py）。
    # ── 靶-域漂移：planner 按**使用场景**归域而不是按商品本体，域一错长期偏好就全不生效 ──
    {
        "id": "dr01_formal_watch",
        "bucket": "靶-域漂移",
        "intent": "shopping",
        "query": "想买块正式场合戴的手表，低调有质感，预算2000",
        "constraints": {
            "category": "腕表",
            "domain": "jewelry_watches（**不是** apparel/furniture——线上真实误判过）",
            "budget": "2000 CNY",
            "soft": ["低调", "有质感"],
        },
        "expected_path": ["planner", "item_search", "item_picker", "shopping_summary"],
        "probe": (
            "P0 清单必须是手表本体，不许出现西装/正装配饰/家居摆件（探域漂移：「正式场合」"
            "这个场景词不得把品类带离腕表）；P2 讲清「低调有质感」体现在哪（表盘尺寸/材质/"
            "机芯），别只堆参数"
        ),
    },
    {
        "id": "dr02_running_watch",
        "bucket": "靶-域漂移",
        "intent": "shopping",
        "query": "跑步用的运动手表，能测心率，预算1500",
        "constraints": {
            "category": "运动手表",
            "domain": "jewelry_watches 或 electronics（按本体归；**不是** sports 装备）",
            "budget": "1500 CNY",
            "must_have": ["心率监测"],
        },
        "expected_path": ["planner", "item_search", "item_picker", "shopping_summary"],
        "probe": (
            "P0 清单是可穿戴手表而非跑鞋/护具/运动配件（探「跑步」场景词的漂移力）；"
            "P1 心率功能须逐条落实到商品，说不出就别声称；P2 续航/防水等跑步真实关切"
        ),
    },
    {
        "id": "dr03_study_desk_lamp",
        "bucket": "靶-域漂移",
        "intent": "shopping",
        "query": "给孩子书房配个护眼台灯，预算300",
        "constraints": {
            "category": "台灯",
            "domain": "home_kitchen / furniture（**不是** toys_baby——「孩子」是受众不是品类）",
            "budget": "300 CNY",
        },
        "expected_path": ["planner", "item_search", "item_picker", "shopping_summary"],
        "probe": (
            "P0 清单是台灯，不许出现儿童玩具/学习桌椅/文具（探受众词「孩子」把域带偏 "
            "toys_baby）；P2 护眼相关维度（无频闪/显色指数/色温）讲到点上"
        ),
    },
    {
        "id": "dr04_camping_cookware",
        "bucket": "靶-域漂移",
        "intent": "shopping",
        "query": "露营做饭用的锅具，轻便耐用，预算400",
        "constraints": {
            "category": "户外锅具",
            "domain": "home_kitchen（本体是炊具；sports 可并存但不能只有它）",
            "budget": "400 CNY",
            "soft": ["轻便", "耐用"],
        },
        "expected_path": ["planner", "item_search", "item_picker", "shopping_summary"],
        "probe": (
            "P0 清单是锅具炊具，不许滑成帐篷/睡袋/营地灯等露营装备（探场景词吞掉本体）；"
            "P2 轻便与耐用是一对矛盾诉求，要在理由里正面权衡，别两个形容词都贴上完事"
        ),
    },
    {
        "id": "dr05_pet_carrier",
        "bucket": "靶-域漂移",
        "intent": "shopping",
        "query": "带猫出门用的航空箱，透气结实，预算500",
        "constraints": {
            "category": "宠物航空箱",
            "domain": "pet（**不是** bags——「箱」字面易把域带到箱包）",
            "budget": "500 CNY",
            "soft": ["透气", "结实"],
        },
        "expected_path": ["planner", "item_search", "item_picker", "shopping_summary"],
        "probe": (
            "P0 清单是宠物出行箱笼，不许出现行李箱/旅行包（探字面词「箱」的域牵引）；"
            "P1 透气与结实要落到具体结构（栅门/承重/锁扣）"
        ),
    },
    # ── 靶-用途混淆：同品类不同用途，cross-encoder 分不开（实测篮球包 vs 通勤包 0.97 交叠）──
    {
        "id": "uc01_basketball_backpack",
        "bucket": "靶-用途混淆",
        "intent": "shopping",
        "query": "打篮球用的双肩包，能装下球和球鞋，预算300",
        "constraints": {
            "category": "运动背包",
            "budget": "300 CNY",
            "must_have": ["可装篮球", "独立鞋仓"],
        },
        "expected_path": ["planner", "item_search", "item_picker", "shopping_summary"],
        "probe": (
            "P0 清单须是运动/球类背包，不许清一色通勤电脑包（探同品类用途混淆——展示门只挡"
            "跨品类垃圾，挡不住这个）；P1 「能装球和鞋」要逐条对上商品的容量/鞋仓说明，"
            "对不上就如实说没有，别硬圆"
        ),
    },
    {
        "id": "uc02_commute_vs_hiking_shoes",
        "bucket": "靶-用途混淆",
        "intent": "shopping",
        "query": "城市通勤穿的休闲鞋，不要登山鞋那种厚底的，预算400",
        "constraints": {
            "category": "休闲鞋",
            "budget": "400 CNY",
            "exclude": ["登山鞋", "厚底"],
        },
        "expected_path": ["planner", "item_search", "item_picker", "shopping_summary"],
        "probe": (
            "P0 不许出现登山/徒步鞋（用户明说排除，属硬约束）；P1 「厚底」这个排除要真执行到"
            "商品层，不能只在话术里承认；P2 通勤场景的搭配/久走舒适度"
        ),
    },
    {
        "id": "uc03_office_vs_gaming_chair",
        "bucket": "靶-用途混淆",
        "intent": "shopping",
        "query": "久坐办公用的人体工学椅，不要电竞椅那种花哨的，预算1200",
        "constraints": {
            "category": "办公椅",
            "budget": "1200 CNY",
            "exclude": ["电竞椅", "花哨配色"],
            "soft": ["久坐舒适"],
        },
        "expected_path": ["planner", "item_search", "item_picker", "shopping_summary"],
        "probe": (
            "P0 清单无电竞椅（赛车座造型/大面积撞色）；P1 久坐相关支撑维度（腰托/坐深/"
            "椅背角度）要落到商品；P2 别把「人体工学」当形容词空喊"
        ),
    },
    {
        "id": "uc04_gym_water_bottle",
        "bucket": "靶-用途混淆",
        "intent": "shopping",
        "query": "健身房用的运动水壶，大容量单手能开，预算150",
        "constraints": {
            "category": "运动水壶",
            "budget": "150 CNY",
            "must_have": ["大容量", "单手开合"],
        },
        "expected_path": ["planner", "item_search", "item_picker", "shopping_summary"],
        "probe": (
            "P0 清单是运动水壶而非保温杯/办公杯（探同品类用途混淆）；P1 容量数字要给出来，"
            "「单手开」对不上的商品别硬说；P2 材质与清洗便利"
        ),
    },
    # ── 靶-配件淹没：根因在数据不在算法（整机与配件在源库同类目），只能靠 planner 的检索词
    #    把本体点出来。黑名单闸 / 品类过滤两个方案都被实测证伪过，别再往那边修。
    {
        "id": "af01_phone_body",
        "bucket": "靶-配件淹没",
        "intent": "shopping",
        "query": "想买台安卓手机，拍照好点的，预算3000",
        "constraints": {
            "category": "智能手机（整机）",
            "budget": "3000 CNY",
            "not_accessory": ["手机壳", "钢化膜", "数据线", "支架"],
        },
        "expected_path": ["planner", "item_search", "item_picker", "shopping_summary"],
        "probe": (
            "P0 清单主体必须是手机整机，配件占比不得过半（探配件淹没）；P1 拍照维度落到"
            "具体参数或评价；P2 如实说明数据源里整机偏少时的取舍，别假装挑得很满"
        ),
    },
    {
        "id": "af02_printer_body",
        "bucket": "靶-配件淹没",
        "intent": "shopping",
        "query": "家用打印机推荐几款，能打照片的，预算1500",
        "constraints": {
            "category": "打印机（整机）",
            "budget": "1500 CNY",
            "not_accessory": ["墨盒", "硒鼓", "打印纸"],
        },
        "expected_path": ["planner", "item_search", "item_picker", "shopping_summary"],
        "probe": "P0 清单是打印机本体不是耗材；P1 照片打印能力要有依据；P2 耗材成本可作加分洞察",
    },
    {
        "id": "af03_console_body",
        "bucket": "靶-配件淹没",
        "intent": "shopping",
        "query": "给孩子买台游戏主机，预算2500",
        "constraints": {
            "category": "游戏主机（整机）",
            "budget": "2500 CNY",
            "not_accessory": ["手柄", "游戏卡带", "收纳包", "保护壳"],
        },
        "expected_path": ["planner", "item_search", "item_picker", "shopping_summary"],
        "probe": "P0 主推是主机本体（手柄/卡带只能作补充）；P2 适龄与内容生态的考量",
    },
    {
        "id": "af04_camera_body_vs_lens",
        "bucket": "靶-配件淹没",
        "intent": "shopping",
        "query": "想入门微单相机，拍风景为主，预算6000",
        "constraints": {
            "category": "微单相机（机身或套机）",
            "budget": "6000 CNY",
            "not_accessory": ["相机包", "UV 镜", "清洁套装", "电池手柄"],
        },
        "expected_path": ["planner", "item_search", "item_picker", "shopping_summary"],
        "probe": "P0 清单是相机本体/套机，不许被配件灌满；P1 是否含镜头要讲明；P2 入门友好度",
    },
    # ── 靶-数值规格：数值是 embedding 处理不了的算子，必须由 planner 抽成硬约束走专道 ──
    {
        "id": "ns01_laptop_bag_16",
        "bucket": "靶-数值规格",
        "intent": "shopping",
        "query": "要能装16寸笔记本的电脑包，预算400",
        "constraints": {"category": "电脑包", "budget": "400 CNY", "numeric": "≥16 寸"},
        "expected_path": ["planner", "item_search", "item_picker", "shopping_summary"],
        "probe": (
            "P0 主推须明确支持 16 寸（标称 13/14 寸的不许当主推）；P1 尺寸对不上的要说明"
            "而不是含糊带过（探数值规格专道）"
        ),
    },
    {
        "id": "ns02_monitor_27",
        "bucket": "靶-数值规格",
        "intent": "shopping",
        "query": "27寸2K显示器，办公用，预算1500",
        "constraints": {
            "category": "显示器",
            "budget": "1500 CNY",
            "numeric": ["27 寸", "2K 分辨率"],
        },
        "expected_path": ["planner", "item_search", "item_picker", "shopping_summary"],
        "probe": "P0 尺寸与分辨率两个数值都要命中，缺一即算未满足；P2 办公场景的接口/支架考量",
    },
    {
        "id": "ns03_thermos_12h",
        "bucket": "靶-数值规格",
        "intent": "shopping",
        "query": "保温12小时以上的杯子，500毫升左右，预算200",
        "constraints": {
            "category": "保温杯",
            "budget": "200 CNY",
            "numeric": ["保温 ≥12h", "约 500ml"],
        },
        "expected_path": ["planner", "item_search", "item_picker", "shopping_summary"],
        "probe": "P0 保温时长与容量都要落到商品；P1 商品没标时长就如实说未标，不许编",
    },
    {
        "id": "ns04_ssd_1tb",
        "bucket": "靶-数值规格",
        "intent": "shopping",
        "query": "1TB的移动固态硬盘，传视频用，预算800",
        "constraints": {"category": "移动固态硬盘", "budget": "800 CNY", "numeric": "1TB"},
        "expected_path": ["planner", "item_search", "item_picker", "shopping_summary"],
        "probe": "P0 容量须为 1TB（512GB / 2TB 不算命中）；P2 读写速度与接口对「传视频」的相关性",
    },
    # ── 靶-跨语言：线上是中文口语，库里商品标题基本是英文。planner 的检索词给不出英文，
    #    这条链就断在召回（M21/M22 反复验证过：中文词打英文库，R@20 掉一大截）。
    {
        "id": "xl01_zh_only_niche",
        "bucket": "靶-跨语言",
        "intent": "shopping",
        "query": "想买个懒人沙发，能躺能靠的那种，预算600",
        "constraints": {"category": "懒人沙发/豆袋沙发", "budget": "600 CNY"},
        "expected_path": ["planner", "item_search", "item_picker", "shopping_summary"],
        "probe": (
            "P0 清单确为懒人沙发/豆袋类而非普通沙发（探中文口语品类词能否翻成英文检索词，"
            "如 bean bag chair / floor sofa）；P2 材质填充与清洗"
        ),
    },
    {
        "id": "xl02_zh_kitchen_gadget",
        "bucket": "靶-跨语言",
        "intent": "shopping",
        "query": "厨房用的沥水篮，洗菜方便的，预算80",
        "constraints": {"category": "沥水篮", "budget": "80 CNY"},
        "expected_path": ["planner", "item_search", "item_picker", "shopping_summary"],
        "probe": "P0 清单是沥水/洗菜篮（英文 colander / strainer），不许滑成收纳筐；P2 材质与尺寸",
    },
    {
        "id": "xl03_zh_slang_category",
        "bucket": "靶-跨语言",
        "intent": "shopping",
        "query": "想买个飞盘，公园玩的那种，预算120",
        "constraints": {"category": "飞盘", "budget": "120 CNY"},
        "expected_path": ["planner", "item_search", "item_picker", "shopping_summary"],
        "probe": "P0 清单是飞盘（frisbee / flying disc）而非其他户外玩具；P1 材质与安全性",
    },
    {
        "id": "xl04_zh_apparel_detail",
        "bucket": "靶-跨语言",
        "intent": "shopping",
        "query": "想要件冲锋衣，三合一可拆内胆的，预算900",
        "constraints": {
            "category": "冲锋衣",
            "budget": "900 CNY",
            "must_have": ["三合一/可拆内胆"],
        },
        "expected_path": ["planner", "item_search", "item_picker", "shopping_summary"],
        "probe": (
            "P0 清单是冲锋衣（英文 3-in-1 jacket / hardshell），不许给普通夹克；"
            "P1 「可拆内胆」这个结构特征要对上商品描述"
        ),
    },
    # ── 靶-性别年龄：受众词是**过滤条件**不是品类词。判错的两个方向都要探：拿受众当品类
    #    （「给女朋友」→ 搜女装）、和把受众整个丢掉（男士剃须刀出女士脱毛仪）。
    {
        "id": "ga01_mens_shaver",
        "bucket": "靶-性别年龄",
        "intent": "shopping",
        "query": "给我爸买个电动剃须刀，好用耐用，预算500",
        "constraints": {
            "category": "电动剃须刀",
            "audience": "男性/长辈",
            "budget": "500 CNY",
        },
        "expected_path": ["planner", "item_search", "item_picker", "shopping_summary"],
        "probe": (
            "P0 清单是男士电动剃须刀，不许混入女士脱毛仪/理发器（探受众约束是否真执行）；"
            "P2 「给长辈」的操作简便与清洗维护"
        ),
    },
    {
        "id": "ga02_kids_scooter",
        "bucket": "靶-性别年龄",
        "intent": "shopping",
        "query": "给5岁女儿买个滑板车，安全点的，预算400",
        "constraints": {
            "category": "儿童滑板车",
            "audience": "5 岁儿童",
            "budget": "400 CNY",
            "soft": ["安全"],
        },
        "expected_path": ["planner", "item_search", "item_picker", "shopping_summary"],
        "probe": (
            "P0 清单是**儿童**滑板车，不许给成人款/电动款（探年龄约束）；P1 安全性要落到"
            "具体设计（刹车/宽轮/限速），不能空喊"
        ),
    },
    {
        "id": "ga03_girlfriend_gift_not_apparel",
        "bucket": "靶-性别年龄",
        "intent": "shopping",
        "query": "给女朋友买个降噪耳机，预算1200",
        "constraints": {
            "category": "降噪耳机",
            "audience": "女性（受众，**不是**品类）",
            "budget": "1200 CNY",
        },
        "expected_path": ["planner", "item_search", "item_picker", "shopping_summary"],
        "probe": (
            "P0 清单是降噪耳机，绝不许因为「女朋友」滑向女装/首饰（探受众词被当成品类词——"
            "对抗集里 gender_age 族的核心错法）；P2 佩戴舒适度/配色可作加分"
        ),
    },
    # ── 靶-污染补搜：召回池被同品牌其他品类灌满，命中数为 0 时要触发补搜而不是硬凑一份清单 ──
    {
        "id": "pb01_brand_pollution_watch",
        "bucket": "靶-污染补搜",
        "intent": "shopping",
        "query": "卡西欧的手表，电子屏那种，预算800",
        "constraints": {
            "category": "电子表",
            "brand": "卡西欧",
            "budget": "800 CNY",
        },
        "expected_path": ["planner", "item_search", "item_picker", "shopping_summary"],
        "probe": (
            "P0 清单是该品牌的**手表**，不许被同品牌计算器/键盘/乐器灌满（探污染补搜闸："
            "品类命中为 0 时应补搜，而不是把品牌命中的杂项当结果）；P1 品牌对不上就如实说"
        ),
    },
    {
        "id": "pb02_brand_pollution_shoes",
        "bucket": "靶-污染补搜",
        "intent": "shopping",
        "query": "耐克的跑步鞋，缓震好的，预算700",
        "constraints": {"category": "跑鞋", "brand": "耐克", "budget": "700 CNY"},
        "expected_path": ["planner", "item_search", "item_picker", "shopping_summary"],
        "probe": "P0 清单是该品牌跑鞋，不许被同品牌服饰/背包/袜子填满；P2 缓震技术要说到点上",
    },
    {
        "id": "pb03_empty_then_backfill",
        "bucket": "靶-污染补搜",
        "intent": "shopping",
        "query": "想买个能烤红薯的小烤箱，预算300",
        "constraints": {"category": "小烤箱", "budget": "300 CNY"},
        "expected_path": ["planner", "item_search", "item_picker", "shopping_summary"],
        "probe": (
            "P0 清单是烤箱本体；若库内「烤红薯」这个说法直接召回为空，须走补搜/换词再检索，"
            "不许把空结果或跑题商品直接端出来（探 must_have_hits==0 的补搜闸）"
        ),
    },
    # ── 靶-预算币种：planner 的老病灶。用户不写币种时系统按 CNY 兜底，模型若自己猜成 USD，
    #    预算就悄悄放大 7 倍（反向则缩水）。三条分别探：裸数字、明示外币、追问轮抄上文重折。
    {
        "id": "bc01_bare_number_budget",
        "bucket": "靶-预算币种",
        "intent": "shopping",
        "query": "推荐个蓝牙音箱，预算200",
        "constraints": {
            "category": "蓝牙音箱",
            "budget": "200 CNY（未明示币种，按默认口径；**不得**当成 200 美元）",
        },
        "expected_path": ["planner", "item_search", "item_picker", "shopping_summary"],
        "probe": (
            "P0 清单主体折合人民币不超 200；P1 回复须注明「按人民币理解」这类口径说明"
            "（currency_assumed 为真时的告知义务），不许闷头按美元选货"
        ),
    },
    {
        "id": "bc02_explicit_foreign_currency",
        "bucket": "靶-预算币种",
        "intent": "shopping",
        "query": "买个机械键盘，预算60欧元以内",
        "constraints": {
            "category": "机械键盘",
            "budget": "60 EUR（明示币种，须按欧元折算，不得当人民币）",
        },
        "expected_path": ["planner", "item_search", "item_picker", "shopping_summary"],
        "probe": "P0 主体折合不超 60 欧元；P1 报价要给出归一后的可比口径，别中欧混着说",
    },
    {
        "id": "bc03_followup_budget_not_refolded",
        "bucket": "靶-预算币种",
        "intent": "shopping",
        "turns": [
            "想要个双肩包，预算80美元",
            "不要皮革的",
        ],
        "query": "不要皮革的",
        "constraints": {
            "category": "双肩包",
            "budget": "80 USD（**沿用上一轮**，本轮没提钱就不该重折）",
            "exclude": ["皮革"],
        },
        "expected_path": ["planner", "item_picker", "shopping_summary"],
        "probe": (
            "P0 预算仍按 80 美元执行、清单无皮革款；P1 不许把上一轮渲染出的「≤ $80」当成本轮"
            "新预算再按人民币折一次（实测过的预算缩水 7 倍 bug，探 budget_amount_grounded）"
        ),
    },
    # ══ 常规分布（27 条）：量「均分不降」这一维 ═══════════════════════════════════
    # 靶心族只能回答「这一族转绿没有」，回答不了「整体有没有被改坏」。这批按**能力面**铺开，
    # 尤其补上 evaluate 这个缺口——planner 判 tasks 有五档，种子集此前几乎只压 recommend 一档，
    # 等于 planner 的任务判定能力有四档没在尺子上。
    {
        "id": "g01_dorm_desk_setup",
        "bucket": "多约束精挑",
        "intent": "shopping",
        "query": "宿舍桌面想弄整洁点，预算200，别买一堆塑料收纳",
        "constraints": {"budget": "200 CNY", "exclude": ["塑料收纳"], "soft": ["整洁", "省空间"]},
        "expected_path": ["planner", "item_search", "item_picker", "shopping_summary"],
        "probe": "P0 无塑料收纳件、总价可控；P1 场景推断合理（桌面/宿舍尺度）；P2 组合搭配的思路",
    },
    {
        "id": "g02_office_keyboard_quiet",
        "bucket": "多约束精挑",
        "intent": "shopping",
        "query": "办公室用的键盘，要安静不吵同事，预算300，不要机械轴那种响的",
        "constraints": {"budget": "300 CNY", "exclude": ["响的机械轴"], "soft": ["静音"]},
        "expected_path": ["planner", "item_search", "item_picker", "shopping_summary"],
        "probe": "P0 主推静音键盘（薄膜/静电容/静音轴），不许推青轴类；P2 办公场景的手感权衡",
    },
    {
        "id": "g03_baby_shower_gift",
        "bucket": "多约束精挑",
        "intent": "shopping",
        "query": "同事生小孩想送个礼，预算300，实用点别太花哨",
        "constraints": {"budget": "300 CNY", "audience": "新生儿家庭", "soft": ["实用", "不花哨"]},
        "expected_path": ["planner", "item_search", "item_picker", "shopping_summary"],
        "probe": "P0 送礼场景合适、安全无风险品类；P2 「实用」要给出真实使用频次的判断",
    },
    {
        "id": "g04_winter_commute_jacket",
        "bucket": "多约束精挑",
        "intent": "shopping",
        "query": "冬天骑车通勤穿的外套，要挡风保暖，预算600，不要羽绒的",
        "constraints": {"budget": "600 CNY", "exclude": ["羽绒"], "soft": ["挡风", "保暖"]},
        "expected_path": ["planner", "item_search", "item_picker", "shopping_summary"],
        "probe": "P0 清单无羽绒服；P1 挡风保暖落到面料/结构；P2 骑行场景（袖长/下摆）的考量",
    },
    {
        "id": "g05_compare_switch_lite",
        "bucket": "跨平台比价",
        "intent": "shopping",
        "query": "Switch Lite 各个平台哪个便宜，帮我比一下",
        "constraints": {"category": "游戏掌机", "task": "price_compare"},
        "expected_path": ["planner", "item_search", "price_compare", "shopping_summary"],
        "probe": "P0 给出多平台可比报价且币种归一；P1 比的是同款而非不同型号；P2 差价成因",
    },
    {
        "id": "g06_compare_airpods",
        "bucket": "跨平台比价",
        "intent": "shopping",
        "query": "苹果无线耳机哪个平台划算，我要正品",
        "constraints": {"category": "无线耳机", "task": "price_compare", "hard": ["正品"]},
        "expected_path": ["planner", "item_search", "price_compare", "shopping_summary"],
        "probe": "P0 不推明显山寨/平替（用户明说正品）；P1 报价可比；P2 保修与渠道差异",
    },
    {
        "id": "g07_compare_coffee_machine",
        "bucket": "跨平台比价",
        "intent": "shopping",
        "query": "这几个平台的胶囊咖啡机价格差多少？预算1000以内",
        "constraints": {"category": "胶囊咖啡机", "budget": "1000 CNY", "task": "price_compare"},
        "expected_path": ["planner", "item_search", "price_compare", "shopping_summary"],
        "probe": "P0 预算内且多平台对比；P2 胶囊耗材成本这类长期开销的洞察",
    },
    # ── evaluate：用户点名一件东西问「值不值」。planner 该判 [evaluate] 而不是 [recommend]，
    #    下游也不该无脑再搜一堆别的塞给他。此前种子集这一档是空的。
    {
        "id": "g08_evaluate_named_item",
        "bucket": "单品评价",
        "intent": "shopping",
        "query": "罗技 MX Master 3S 这个鼠标值得买吗？我主要写代码",
        "constraints": {"task": "evaluate", "target": "罗技 MX Master 3S", "scene": "编程办公"},
        "expected_path": ["planner", "item_search", "shopping_summary"],
        "probe": (
            "P0 回答的是「这一款值不值」，不许答非所问地甩一份别的鼠标清单（探 tasks 判定）；"
            "P1 结论要有依据（参数/评价/价位）；P2 结合「写代码」这个具体场景讲取舍"
        ),
    },
    {
        "id": "g09_evaluate_two_items",
        "bucket": "单品评价",
        "intent": "shopping",
        "query": "小米空气炸锅和美的的比，哪个更值？",
        "constraints": {"task": "evaluate", "targets": ["小米空气炸锅", "美的空气炸锅"]},
        "expected_path": ["planner", "item_search", "shopping_summary"],
        "probe": "P0 对这两者给出正面比较与结论；P1 不许只列参数不下判断；P2 差异点讲到关键处",
    },
    {
        "id": "g10_evaluate_worth_upgrade",
        "bucket": "单品评价",
        "intent": "shopping",
        "query": "我现在用的是两年前的千元机，换新款有必要吗",
        "constraints": {"task": "evaluate", "scene": "换机决策"},
        "expected_path": ["planner", "shopping_summary"],
        "probe": (
            "P0 给出「要不要换」的判断而不是直接推销手机；P1 缺关键信息（用途/痛点）时"
            "可以先澄清；P2 换与不换的条件讲清楚"
        ),
    },
    {
        "id": "g11_landed_cost_jp",
        "bucket": "到手价",
        "intent": "shopping",
        "query": "这个咖啡壶寄到日本要多少钱？含税含运",
        "constraints": {"task": "landed_cost", "dest": "JP"},
        "expected_path": ["planner", "item_search", "shipping_calc", "shopping_summary"],
        "probe": "P0 给出到手价拆解（货值+运费+税）与收货国口径；P1 估算须标明是估算",
    },
    {
        "id": "g12_landed_cost_budget_cap",
        "bucket": "到手价",
        "intent": "shopping",
        "query": "买双跑鞋寄到美国，到手不超过120美元",
        "constraints": {"task": "landed_cost", "dest": "US", "budget": "120 USD（到手价口径）"},
        "expected_path": [
            "planner",
            "item_search",
            "shipping_calc",
            "item_picker",
            "shopping_summary",
        ],
        "probe": "P0 **到手价**不超 120 美元（不是货值不超）；P1 拆解清楚；P2 关税规则的说明",
    },
    {
        "id": "g13_landed_cost_no_country",
        "bucket": "到手价",
        "intent": "shopping",
        "query": "算下这个包的到手价",
        "constraints": {"task": "landed_cost", "dest": "未明示（走四层解析或澄清）"},
        "expected_path": ["planner", "shipping_calc", "shopping_summary"],
        "probe": (
            "P0 收货国未明示时要么按默认口径并**讲明是假设**，要么问一句；不许闷头按某国算"
            "还不说（探 dest_country_assumed 的告知义务）"
        ),
    },
    {
        "id": "g14_category_intel_mattress",
        "bucket": "品类洞察",
        "intent": "shopping",
        "query": "买床垫一般看什么？我腰不好",
        "constraints": {"task": "category_intel", "scene": "腰部支撑"},
        "expected_path": ["planner", "category_insight", "shopping_summary"],
        "probe": (
            "P0 回答的是选购维度而不是直接甩清单（探 category_intel 单跑，别自作主张拖成"
            "全流程）；P2 「腰不好」要影响维度排序（支撑性优先于软硬偏好）"
        ),
    },
    {
        "id": "g15_category_intel_robot_vacuum",
        "bucket": "品类洞察",
        "intent": "shopping",
        "query": "扫地机器人现在主流都什么价位？功能差在哪",
        "constraints": {"task": "category_intel"},
        "expected_path": ["planner", "category_insight", "shopping_summary"],
        "probe": "P0 给价位段与功能分档；P1 数据薄时要自报口径，不许编一个精确市场分布",
    },
    {
        "id": "g16_web_fact_trend",
        "bucket": "外部事实",
        "intent": "shopping",
        "query": "今年比较火的那种露营椅是什么样的？想买一把",
        "constraints": {"task": ["category_intel", "recommend"], "grounding": "web（时效性说法）"},
        "expected_path": ["planner", "web_search", "item_search", "shopping_summary"],
        "probe": (
            "P0 先把「今年火的」翻译成具体品类特征再检索（探 intent_grounding=web）；"
            "P1 别硬编潮流说法"
        ),
    },
    {
        "id": "g17_web_fact_niche_term",
        "bucket": "外部事实",
        "intent": "shopping",
        "query": "想买个痛包，能装谷子的那种，预算200",
        "constraints": {
            "category": "痛包（二次元展示包）",
            "budget": "200 CNY",
            "grounding": "web（圈层新说法）",
        },
        "expected_path": ["planner", "web_search", "item_search", "shopping_summary"],
        "probe": (
            "P0 要真的理解「痛包/谷子」是什么再搜（探新说法的 web 接地）；"
            "P1 不懂就先查，不许望文生义搜成医疗护具"
        ),
    },
    {
        "id": "g18_clarify_vague_gift",
        "bucket": "澄清",
        "intent": "shopping",
        "query": "想给朋友买个礼物",
        "constraints": {"missing": ["预算", "对象偏好", "场合"]},
        "expected_path": ["planner", "ask_user"],
        "probe": (
            "P0 应先澄清关键缺口（预算/对象）而不是硬猜着推一堆；P1 问题要可点选、一次问到"
            "点子上，不许连问五个问题把人问烦"
        ),
    },
    {
        "id": "g19_clarify_conflicting",
        "bucket": "澄清",
        "intent": "shopping",
        "query": "要个便宜的高端相机",
        "constraints": {"conflict": ["便宜", "高端"]},
        "expected_path": ["planner", "ask_user"],
        "probe": "P0 识别出诉求内部冲突并澄清预算档位；P1 不许闷头按某一头理解就开搜",
    },
    {
        "id": "g20_chitchat_refuse_medical",
        "bucket": "闲聊兜底",
        "intent": "chitchat",
        "query": "我最近老失眠，你说我该吃点什么药？",
        "constraints": {"safety": "医疗建议红线"},
        "expected_path": ["chat_fallback"],
        "probe": (
            "P0 不得给出用药建议（安全红线），应建议就医；P1 可自然过渡到助眠**用品**这类"
            "能力范围内的事，但不许硬转成推销"
        ),
    },
    {
        "id": "g21_chitchat_out_of_scope",
        "bucket": "闲聊兜底",
        "intent": "chitchat",
        "query": "帮我写一封辞职信吧",
        "constraints": {"scope": "非购物意图"},
        "expected_path": ["chat_fallback"],
        "probe": "P0 说明能力边界并收尾，不许开检索；P1 语气自然、别机械地念免责声明",
    },
    {
        "id": "g22_bundle_camping",
        "bucket": "套装组合",
        "intent": "shopping",
        "query": "第一次露营，装备帮我配一套，预算2000",
        "constraints": {"bundle": True, "budget": "2000 CNY（总预算）", "audience": "新手"},
        "expected_path": ["planner", "item_search", "item_picker", "shopping_summary"],
        "probe": (
            "P0 按槽位给出成套方案且**总价**不超 2000（不是每件不超）；P1 必需项与可选项"
            "要分清；P2 新手友好度（易上手/收纳）"
        ),
    },
    {
        "id": "g23_bundle_home_office",
        "bucket": "套装组合",
        "intent": "shopping",
        "query": "在家办公的桌面一套，显示器支架和键鼠都要，预算1500",
        "constraints": {
            "bundle": True,
            "budget": "1500 CNY（总预算）",
            "slots_named": ["显示器支架", "键盘", "鼠标"],
        },
        "expected_path": ["planner", "item_search", "item_picker", "shopping_summary"],
        "probe": "P0 用户点名的三个槽位都要有且总价达标；P1 没点名的槽位若自行添加须说明理由",
    },
    # ── 多轮：追问轮占线上请求的三分之一，尺子上不能只有单轮 ──
    {
        "id": "g24_refine_tighten",
        "bucket": "记忆专项",
        "intent": "shopping",
        "turns": ["推荐几个双肩包，预算400", "只要黑色的"],
        "query": "只要黑色的",
        "constraints": {"category": "双肩包", "budget": "400 CNY（不变）", "add": ["黑色"]},
        "expected_path": ["planner", "item_picker", "shopping_summary"],
        "probe": (
            "P0 在既有候选上收紧（retrieval=reuse），清单全黑且仍在预算内；"
            "P1 不必重新检索，也不许把预算约束弄丢"
        ),
    },
    {
        "id": "g25_refine_loosen",
        "bucket": "记忆专项",
        "intent": "shopping",
        "turns": ["找个保温杯，预算100", "预算提到300吧，要好点的"],
        "query": "预算提到300吧，要好点的",
        "constraints": {"category": "保温杯", "budget": "300 CNY（放宽，须重搜）"},
        "expected_path": ["planner", "item_search", "item_picker", "shopping_summary"],
        "probe": (
            "P0 新清单要真的出现 100~300 档的商品（探 augment：放宽后旧池子不含新放开的"
            "那部分，必须重搜）；P1 不许只在旧候选里挑最贵的几个充数"
        ),
    },
    {
        "id": "g26_multi_constraint_accumulate",
        "bucket": "记忆专项",
        "intent": "shopping",
        "turns": ["想买跑鞋，预算500", "不要白色的", "要透气的"],
        "query": "要透气的",
        "constraints": {
            "category": "跑鞋",
            "budget": "500 CNY",
            "accumulated": ["不要白色", "透气"],
        },
        "expected_path": ["planner", "item_picker", "shopping_summary"],
        "probe": (
            "P0 三轮约束**全部**同时生效（预算 + 非白 + 透气）——探约束跨轮存续，"
            "少一条都算失败；P1 不许要求用户重述前面说过的条件"
        ),
    },
    {
        "id": "g27_followup_ask_reason",
        "bucket": "记忆专项",
        "intent": "shopping",
        "turns": ["推荐两个机械键盘，预算400", "第二个为什么推荐给我？"],
        "query": "第二个为什么推荐给我？",
        "constraints": {"task": "evaluate", "target": "上一轮清单的第 2 件"},
        "expected_path": ["planner", "shopping_summary"],
        "probe": (
            "P0 答的是**上一轮那件**的推荐理由，不许重新搜一批新的（探指代解析 + "
            "retrieval=reuse）；P1 理由要落到该商品的具体属性"
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
