"""可调参数注册表：后台管理页面的**单一事实来源**。

后端校验、前端表单渲染、参数说明与危险标注全部从这里生成——前端不硬编码任何参数名或范围，
新增一个可调参数只改这一个文件。

**为什么参数在代码里仍是模块级常量、而不是 ``param("X")`` 式的运行时函数**：
仓库既有测试大量用 ``monkeypatch.setattr(mod, "RELEVANCE_FLOOR", 0.45)`` 直接改模块属性来
构造场景。改成函数调用会让这些 monkeypatch 设到一个没人读的属性上——测试照样绿，但实际不再
约束任何东西（静默失效比崩掉更危险）。故保留常量形态，改为**可重新赋值**：各模块提供
``_load_params()`` 重新从 env 求值自己的常量，覆盖层改完 env 后按 ``reload`` 字段回调它。
副作用是热更新只对**新任务**生效（跑到一半的循环已读过旧值），这正是我们要的语义。

``default`` 必须与各模块代码里写的默认值逐字一致——两处漂移会让「恢复默认」把参数恢复成一个
从未生效过的值。``tests/test_admin_config.py::test_registry_default_matches_code`` 断言这一点。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

Kind = Literal["int", "float", "str", "bool"]
Group = Literal["model", "retrieval", "display", "scoring"]

# 分组的展示名与说明（前端按此渲染分区标题）。
GROUPS: dict[str, dict[str, str]] = {
    "model": {
        "label": "模型档位",
        "desc": "主/快/视觉/判官四档模型与温度。改完对新任务生效，进行中的任务不受影响。",
    },
    "retrieval": {
        "label": "检索召回",
        "desc": "item_search 一侧：召回多少、什么算相关、召不够怎么补。",
    },
    "display": {
        "label": "展示商品卡",
        "desc": "item_picker 输出侧：最终给用户看几张卡、什么样的候选不配上卡。",
    },
    "scoring": {
        "label": "精挑打分权重",
        "desc": (
            "item_picker 排序权重。这组直接决定推荐顺序，调错会把推荐做反——注意每项的标定说明。"
        ),
    },
}


@dataclass(frozen=True)
class Param:
    """一个可调参数的声明。

    ``key`` 即环境变量名（覆盖层写 ``os.environ[key]``，模块 ``_load_params()`` 再读回来）。
    ``reload`` 是改动后要回调 ``_load_params()`` 的模块路径——漏填会导致改了不生效。
    ``warning`` 非空时前端显示醒目告警：留给「标定证伪 / 未标定 / 调错会做反推荐」的参数。
    """

    key: str
    group: Group
    label: str
    kind: Kind
    default: Any
    reload: str
    help: str
    minimum: float | None = None
    maximum: float | None = None
    warning: str = ""
    # 空字符串是否是合法值（模型名类参数：空 = 回退到 LLM_MAIN / 关闭该档）。
    allow_empty: bool = False
    # 值住在 ``reload`` 那个模块的哪个常量里。填了它，页面显示的就是**真实生效值**——直接读那个
    # 常量，而不是拿 env 再推导一遍。
    #
    # **这不是锦上添花，是修一个真 bug**：``ITEM_SEARCH_MAX_TOP_K`` 的代码默认值是动态的（取
    # ``DEFAULT_TOP_K``），把召回条数改成 15 后模块里 MAX_TOP_K 确实变成了 15，但「读 env → 没配
    # → 回退注册表 default」的推导只会得出 10。页面于是显示 10，实际生效 15——UI 撒谎，且正是在
    # 「我刚改完、想确认它生效了没」的那一刻撒谎。
    #
    # 留空表示「值不住在模块常量里」（模型档位是在 get_*_llm() 内部直读 env 的），那种只能靠推导。
    const: str = ""
    # 密钥类参数（API key）。三条特殊规则，缺一条就是把密钥往外送：
    #
    # 1. **永不回显**：``current_value`` 一律返回空串，页面另收一个 ``masked``（``sk-…a1b2``）只用于
    #    确认「配没配、是不是那一把」。管理员是应用层角色、未必有服务器 shell，把明文 key 送进页面
    #    等于凭空扩大它的暴露面。
    # 2. **留空 = 不改**（不是「设为空」）：正因为不回显，页面上它天生就是空的；若把空当成新值，
    #    管理员随便改个别的参数点保存，key 就被抹了、全站 LLM 调用当场失效。
    # 3. **要清空走单项「恢复默认」**：删 override 行退回 .env 基线，语义明确、不与规则 2 打架。
    secret: bool = False


_LLM = "app.agent.llm"
_SEARCH = "app.tools.item_search"
_PICKER = "app.tools.item_picker"

PARAMS: tuple[Param, ...] = (
    # ---------------- 模型档位 ----------------
    # endpoint 排在模型名之前：换供应商时必须先把这两项指对，否则填了新模型名也只会拿旧 key 打向
    # 旧地址、当场失败。三档模型（主/快/判官）共用这一套；视觉档可单独指到别家（见下面 VISION_*）。
    Param(
        key="OPENAI_BASE_URL",
        group="model",
        label="API 地址（base URL）",
        kind="str",
        default="",
        reload=_LLM,
        help=(
            "OpenAI 兼容的 endpoint，主 / 快 / 判官三档共用。换供应商时先改这里，再改下面的模型名。"
        ),
        warning=(
            "改这里等于改「对话内容发到哪」——所有 query、用户偏好连同 API key 都会打向这个地址。"
            "只填你自己的供应商，别填来路不明的中转。"
        ),
    ),
    Param(
        key="OPENAI_API_KEY",
        group="model",
        label="API key",
        kind="str",
        default="",
        reload=_LLM,
        secret=True,
        help=(
            "配合上面的 base URL 使用。**页面永不回显**，只给前几位与后四位供核对；留空表示不改动，"
            "要清空请点这一项的「恢复默认」退回 .env 里的值。"
        ),
    ),
    Param(
        key="VISION_BASE_URL",
        group="model",
        label="视觉 API 地址",
        kind="str",
        default="",
        reload=_LLM,
        allow_empty=True,
        help="只在视觉模型要指到**另一家**供应商时填。留空 = 复用上面的 base URL（常见情形）。",
    ),
    Param(
        key="VISION_API_KEY",
        group="model",
        label="视觉 API key",
        kind="str",
        default="",
        reload=_LLM,
        secret=True,
        # 「留空」在这里有两个互斥的含义，别混：**没配置**它 = 复用主 key（llm.py 的 `or` 回退）；
        # 而**页面上的空输入框** = 不改动（密钥不回显，空是它的天然初始态）。想从「已配置」回到
        # 「复用主 key」，走「恢复」，不是把框清空。
        help="没配置时复用主 API key。页面留空 = 不改；要清掉它改回复用主 key，点右上角「恢复」。",
    ),
    Param(
        key="LLM_MAIN",
        group="model",
        label="主模型",
        kind="str",
        default="",
        reload=_LLM,
        help=(
            "主 loop 与子 Agent 共享的模型（同质 fork 的硬约束：主子必须同款）。改后清 lru_cache 重"
            "建实例。"
        ),
        warning="这是唯一必配项，没有代码默认值——填错模型名会让所有任务在第一次调用时就失败。",
    ),
    Param(
        key="LLM_FAST",
        group="model",
        label="快档模型",
        kind="str",
        default="",
        reload=_LLM,
        allow_empty=True,
        help="子 Agent 执行 + shopping_summary 文案用。留空 = 与主模型同款（推荐）。",
        warning=(
            "实测教训：换更弱的小模型（qwen-turbo 等）在子任务上反而更慢且降级。正确做法是留空用主"
            "模型同款、只靠下面的开关关掉推理。"
        ),
    ),
    Param(
        key="LLM_FAST_REASONING",
        group="model",
        label="快档保留推理",
        kind="bool",
        default=False,
        reload=_LLM,
        help=(
            "关（默认）= 经 enable_thinking:false 省掉思考解码，能力不变但显著更快。仅 hybrid 模型"
            "（Qwen/DeepSeek）有效。"
        ),
    ),
    Param(
        key="LLM_VISION",
        group="model",
        label="视觉模型",
        kind="str",
        default="",
        reload=_LLM,
        allow_empty=True,
        help=(
            "image_understand 专用（主 loop 永不直接见图）。留空 = 整条图片理解腿优雅降级关闭，不报"
            "错。"
        ),
    ),
    Param(
        key="LLM_JUDGE",
        group="model",
        label="判官模型",
        kind="str",
        default="",
        reload=_LLM,
        allow_empty=True,
        help="Rubric 离线评测用，不参与线上链路。留空 = 回退主模型。",
    ),
    Param(
        key="LLM_TEMPERATURE",
        group="model",
        label="主模型温度",
        kind="float",
        default=0.3,
        minimum=0.0,
        maximum=2.0,
        reload=_LLM,
        help="主/子 Agent 共享。调高会让工具调用与收尾判断变得不稳定。",
    ),
    Param(
        key="LLM_VISION_TEMPERATURE",
        group="model",
        label="视觉模型温度",
        kind="float",
        default=0.1,
        minimum=0.0,
        maximum=2.0,
        reload=_LLM,
        help="压到 0.1：看图是读事实不是做创作，同一张图两次调用该给同一个品类。",
    ),
    Param(
        key="LLM_JUDGE_TEMPERATURE",
        group="model",
        label="判官模型温度",
        kind="float",
        default=0.0,
        minimum=0.0,
        maximum=2.0,
        reload=_LLM,
        help="0 保证评分可复现。评测是尺子，尺子不能自己抖。",
    ),
    Param(
        key="LLM_REQUEST_TIMEOUT",
        group="model",
        label="单请求超时（秒）",
        kind="float",
        default=60.0,
        minimum=5.0,
        maximum=600.0,
        reload=_LLM,
        help="一次卡死的 API 调用拖垮整条任务的根因防线。跑 Rubric 评测时需调到 300。",
    ),
    Param(
        key="LLM_MAX_RETRIES",
        group="model",
        label="请求重试次数",
        kind="int",
        default=2,
        minimum=0,
        maximum=5,
        reload=_LLM,
        help="超时或连接焊死后换连接重试的次数。",
    ),
    # ---------------- 检索召回 ----------------
    Param(
        key="ITEM_SEARCH_TOP_K",
        group="retrieval",
        label="单次召回条数",
        kind="int",
        default=10,
        minimum=1,
        maximum=50,
        reload=_SEARCH,
        const="DEFAULT_TOP_K",
        help=(
            "跨平台 fork 时每个子 Agent 都要吃一份这么大的候选 JSON（20 条 ≈ 3.7K token 且必然缓存 "
            "miss）。收到 10 是 token审计的结论。"
        ),
    ),
    Param(
        key="ITEM_SEARCH_MAX_TOP_K",
        group="retrieval",
        label="召回条数硬封顶",
        kind="int",
        default=10,
        minimum=1,
        maximum=50,
        reload=_SEARCH,
        const="MAX_TOP_K",
        help=(
            "模型传的 top_k 只当上界候选，真正生效的是 min(top_k, 本值)。默认与「单次召回条数」同值"
            "。"
        ),
        warning=(
            "这是机制闸不是建议：实测模型会无视默认值自己传 top_k=20，改 prompt 说服不了它。放大前"
            "先想清楚 token 代价。"
        ),
    ),
    Param(
        key="ITEM_SEARCH_SINGLE_POOL_K",
        group="retrieval",
        label="单平台召回池",
        kind="int",
        default=30,
        minimum=1,
        maximum=100,
        reload=_SEARCH,
        const="SINGLE_PLATFORM_POOL_K",
        help=(
            "单平台没有「平台数 × top_k」的乘数，若同样只召 10 条则池子≈展示上限、精排几乎无筛除空"
            "间。放大的只是进登记表的池子，不进模型上下文。"
        ),
    ),
    Param(
        key="ITEM_SEARCH_RENDER_CAP",
        group="retrieval",
        label="渲染进上下文条数",
        kind="int",
        default=5,
        minimum=1,
        maximum=30,
        reload=_SEARCH,
        const="RENDER_CAP",
        help=(
            "召回池可以大（30 供 picker 精排），但渲染给模型只取头部这么多条。作用是让主 loop 判断"
            "本轮检索质量（要不要换词/补搜），不是给模型精挑。"
        ),
    ),
    Param(
        key="RELEVANCE_FLOOR",
        group="retrieval",
        label="相关性下限",
        kind="float",
        default=0.45,
        minimum=0.0,
        maximum=1.0,
        reload=_SEARCH,
        const="RELEVANCE_FLOOR",
        help="滤掉低于此余弦相似度的召回，让 total_recall 反映「相关」召回数而非「最近的垃圾」。",
        warning=(
            "实测局限：BGE-M3 给任何真实英文 query 都打 ≥0.48（连库里根本没有的挖掘机/处方药也 0.48"
            "-0.53），三段分布严重重叠。0.45实际只挡乱码级无关，挡不住「品类缺货」——那是数据稀疏问"
            "题，调高阈值治不了，只会开始误杀真候选。"
        ),
    ),
    Param(
        key="CATEGORY_MATCH_FLOOR",
        group="retrieval",
        label="品类一致性下限",
        kind="float",
        default=0.5,
        minimum=0.0,
        maximum=1.0,
        reload=_SEARCH,
        const="CATEGORY_MATCH_FLOOR",
        help=(
            "定点型号调查时挡「配件标题里带宿主型号」的假阳性（如「给 XM5 用的耳机壳」型号对得上但"
            "不是耳机本体）。"
        ),
        warning="诚实标注：阈值未跑线上真实 embedding 校准，是个保守初值。",
    ),
    Param(
        key="ITEM_SEARCH_RETRY_MIN_HITS",
        group="retrieval",
        label="触发自动补搜的条数",
        kind="int",
        default=3,
        minimum=0,
        maximum=20,
        reload=_SEARCH,
        const="RETRY_MIN_HITS",
        help=(
            "召回少于这么多条就自动摘掉评分门槛重搜一次。取 3 而非 0：只召回一两条时模型照样会自己"
            "发起重搜，那轮 Think 的解码开销正是要省掉的。设 0 = 关闭自动补搜。"
        ),
    ),
    Param(
        key="ITEM_SEARCH_EXCLUDE_BUFFER",
        group="retrieval",
        label="硬排除补偿条数",
        kind="int",
        default=10,
        minimum=0,
        maximum=50,
        reload=_SEARCH,
        const="EXCLUDE_FETCH_BUFFER",
        help=(
            "记忆黑名单在召回阶段就过滤，多召这么多条补偿被杀的名额——否则「不要皮革」的用户搜公文包"
            "，10 条里 8 条皮的，杀完只剩 2 条还无处补货。"
        ),
    ),
    # ---------------- 展示商品卡 ----------------
    Param(
        key="PICK_DISPLAY_CAP",
        group="display",
        label="商品卡展示上限",
        kind="int",
        default=8,
        minimum=1,
        maximum=30,
        reload=_PICKER,
        const="PICK_DISPLAY_CAP",
        help=(
            "语义是「合适的都给，但有上限」，不是「固定取 N 件」。登记表仍是全量，只收紧渲染给模型"
            "的件数。"
        ),
        warning=(
            "20→8 是 token 审计的结论：一次吐 20 件 ×（长标题+理由句）≈2,700tokens，是整条链最大的"
            "单条工具结果，且此后每步解码都要重读。往回调大要有心理准备。"
        ),
    ),
    Param(
        key="PICK_REL_SHOW_RATIO",
        group="display",
        label="展示相对门比例",
        kind="float",
        default=0.35,
        minimum=0.0,
        maximum=1.0,
        reload=_PICKER,
        const="PICK_REL_SHOW_RATIO",
        help=(
            "以「池内最高品类相关分 × 本比例」为门，低于门的判品类不符、宁缺毋滥不凑数展示。相对而"
            "非绝对：peak 低时门也低、不硬造空。保底至少留最高分 1 件。设 0 = 关闭。"
        ),
        warning=(
            "0.35 是保守初值、未消融标定（误杀代价高于放垃圾故取低值）。另：此门只挡得住跨品类垃圾"
            "，挡不住同品类的用途混淆（篮球包对旅行包 cross-encoder 打0.97）。"
        ),
    ),
    # ---------------- 精挑打分权重 ----------------
    Param(
        key="PICK_W_MATCH_HARD",
        group="scoring",
        label="硬约束命中加分（关键词）",
        kind="float",
        default=2.0,
        minimum=0.0,
        maximum=10.0,
        reload=_PICKER,
        const="_W_MATCH_HARD",
        help=(
            "正向硬约束（如「必须金属」）字面命中的加分，比软偏好重。刻意不做二值淘汰：库无可靠材质"
            "字段，keep-only 会误杀「是金属但标题没写金属」的候选。"
        ),
    ),
    Param(
        key="PICK_W_MATCH_HARD_SEM",
        group="scoring",
        label="硬约束命中加分（语义）",
        kind="float",
        default=1.0,
        minimum=0.0,
        maximum=10.0,
        reload=_PICKER,
        const="_W_MATCH_HARD_SEM",
        help="同上，但走 embedding 语义相似度而非字面命中。",
    ),
    Param(
        key="PICK_W_MATCH_SEM",
        group="scoring",
        label="软偏好语义加分",
        kind="float",
        default=0.7,
        minimum=0.0,
        maximum=10.0,
        reload=_PICKER,
        const="_W_MATCH_SEM",
        help=(
            "对正向软意图算候选语义相似度加分，补「精致高级感」这类没写字面词但语义近的近邻。设 0 ="
            " 关闭该路（省编码、零额外延迟）。"
        ),
        warning=(
            "0.7 由消融标定得来（scripts/ablation_semantic_pick.py：NDCG@10 在 0.5→0.8 单调升、峰约"
            " 0.8，取 0.7 为不过拟合n=5 噪声的保守值）。这是少数几个真跑过标定的值，别随手改。"
        ),
    ),
    Param(
        key="PICK_W_ATTEN_SEM",
        group="scoring",
        label="软避讳语义减分",
        kind="float",
        default=0.7,
        minimum=0.0,
        maximum=10.0,
        reload=_PICKER,
        const="_W_ATTEN_SEM",
        help=(
            "对负向软意图算语义相似度减分（如「不要塑料感」）。作独立减分项、不进正向 query 向量，"
            "躲开 embedding 否定语义弱导致的反向召回。"
        ),
    ),
    Param(
        key="PICK_W_AFFINITY",
        group="scoring",
        label="行为亲和加分",
        kind="float",
        default=0.2,
        minimum=0.0,
        maximum=10.0,
        reload=_PICKER,
        const="_W_AFFINITY",
        help=(
            "从收藏行为聚合出的属性词命中加分。刻意低于软偏好（1.0）：这是推断出的取向，比用户亲口"
            "说的弱一档。设 0 = 关闭。"
        ),
        warning=(
            "0.2 是调出来的、不是标定的。初版 0.5 被真实链路 A/B 打脸：收藏 2 件亚麻 → 7 件亚麻款霸"
            "占前 7，一件 3.8 分的压过 4.6分的。往上调很容易让弱证据重新变成主导排序因子。"
        ),
    ),
    Param(
        key="PICK_W_SPEC_CONFLICT",
        group="scoring",
        label="数值规格冲突减分",
        kind="float",
        default=2.0,
        minimum=0.0,
        maximum=10.0,
        reload=_PICKER,
        const="_W_SPEC_CONFLICT",
        help=(
            "标题明确标了另一档位（14 inch 对 16 寸要求）时沉底。与「标题没写」（不奖不罚）完全两回"
            "事。不硬淘汰以容解析噪声。"
        ),
    ),
    Param(
        key="PICK_RERANK_FLOOR",
        group="scoring",
        label="品类门阈值（普通轮）",
        kind="float",
        default=0.2,
        minimum=0.0,
        maximum=1.0,
        reload=_PICKER,
        const="_RERANK_FLOOR",
        help=(
            "cross-encoder 相关分低于此值判跨品类混入，降权沉底但不剔除（定稿场景：炊具池混进背包，"
            "实测分离 0.97 vs 0.006）。"
        ),
    ),
    Param(
        key="PICK_W_RERANK_MISS",
        group="scoring",
        label="品类门未过减分",
        kind="float",
        default=2.0,
        minimum=0.0,
        maximum=10.0,
        reload=_PICKER,
        const="_W_RERANK_MISS",
        help="低于上面阈值时的降权量级。",
    ),
    Param(
        key="PICK_SLOT_RERANK_FLOOR",
        group="scoring",
        label="槽位逐出门（套装轮）",
        kind="float",
        default=0.0,
        minimum=0.0,
        maximum=1.0,
        reload=_PICKER,
        const="_SLOT_RERANK_FLOOR",
        help="套装轮把低于此分的候选逐出槽位。0 = 关闭（当前默认）。",
        warning=(
            "默认关是标定结论、不是没来得及调：badcase 4c0ac682 用真实数据证伪了绝对阈值——品类内配"
            "件蹭词场景分数完全交叠（贴纸 0.30~0.60 vs 真笔袋0.055），任何阈值要么放垃圾要么杀真品"
            "。开这个门之前请先重跑标定。"
        ),
    ),
    Param(
        key="PICK_W_SLOT_RERANK",
        group="scoring",
        label="槽内排序加分（套装轮）",
        kind="float",
        default=1.0,
        minimum=0.0,
        maximum=10.0,
        reload=_PICKER,
        const="_W_SLOT_RERANK",
        help="套装轮把相关分作槽内排序信号，真品在场时把蹭词垃圾压下去。",
    ),
)

# key → Param 的索引（覆盖层校验用）。
BY_KEY: dict[str, Param] = {p.key: p for p in PARAMS}

# 需要回调的模块路径全集（去重后按 PARAMS 声明序，保证 MAX_TOP_K 这类依赖同模块前序常量的参数
# 在同一次 _load_params() 里按源码顺序重新求值）。
RELOAD_MODULES: tuple[str, ...] = tuple(dict.fromkeys(p.reload for p in PARAMS))
