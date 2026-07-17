"""执行层哨兵文案：工具被闸拦下时**回给模型**的固定信号。

**为什么是哨兵而不是摘工具**：所有禁用逻辑统一在执行层（pre_tool_call Hook）处理，从不从模型
可见工具表中摘除工具——tools 列表每轮保持完整不变，保住 prompt cache 前缀稳定。模型调了被禁
工具 → 收到哨兵消息 → 下一轮自然转向可用工具。（refdocs 17-4 §2.3 主张「隐藏 > 拒绝」，本项目
出于前缀缓存治理反其道而行，是自觉的取舍。）

拒绝的理由对模型必须**真实**，不能张冠李戴——所以 fork 耗尽分并行/串行两条文案，检索耗尽分
「软收敛」与「硬挡」两档。
"""

from __future__ import annotations


def converge_directive(count: int) -> str:
    """刚越检索预算（soft）：工具照常执行，但结果尾部追加强制收敛指令。"""
    return (
        f"[强制收敛] 已累计商品检索 {count} 次。停止任何新的 item_search / web_search。"
        "立即基于现有候选完成用户要求的剩余步骤（用户要了比价 / 到手价才 price_compare / "
        "shipping_calc），再 item_picker 精挑，并**以调用 shopping_summary 结束**——"
        "不要只用文字说「我来收尾」。"
    )


def reuse_backfill_note(count: int, cap: int) -> str:
    """复用轮补搜（soft）：工具照常执行，结果尾部注明这是小预算、搜完即收敛。

    与 :func:`converge_directive` 分开写是为了拒绝理由真实：预算小不是因为「搜太多了」，
    而是因为 planner 判了本轮复用旧候选。
    """
    return (
        f"[复用轮补搜] planner 判定本轮复用上一轮候选，补搜预算仅 {cap} 次（已用 {count} 次）。"
        "请把本次召回与既有候选合流后立即 item_picker 精挑并收尾，不要再检索；"
        "若精挑后候选仍不够用，机制会自动授权补搜。"
    )


def reuse_retrieval_exhausted(count: int, cap: int) -> str:
    """复用轮补搜预算耗尽（block）：工具**不执行**，直接回哨兵。"""
    return (
        f"[复用轮检索预算耗尽] planner 判定本轮复用上一轮候选，补搜预算 {cap} 次已用完"
        f"（累计 {count} 次），本次未执行。立即基于现有候选调 item_picker 精挑、"
        "shopping_summary 收尾；若精挑后候选确实不够用，机制会自动授权补搜，无需你重试检索。"
    )


def retrieval_exhausted(count: int) -> str:
    """再越检索预算（block）：工具**不执行**，直接回哨兵。"""
    return (
        f"[检索预算耗尽] 已累计商品检索 {count} 次，本次未执行，禁止再检索。"
        "立即基于现有候选调 item_picker 精挑、再 shopping_summary 收尾"
        "（**用户要了比价 / 到手价才**先 price_compare / shipping_calc，没要就跳过）。"
    )


def sub_search_budget_note(used: int, cap: int) -> str:
    """「预算可见」批注：子 item_search 结果尾部，告诉模型本平台还能搜几次 + 找够用就停。

    这是「找够用 not 找最好」的可见化——模型据此规划，而非被摘掉工具后措手不及。
    """
    if used < cap:
        return (
            f"[检索预算] 本平台已搜 {used}/{cap} 次。目标是**找够用**不是找最优——"
            "召回里有合适候选就立刻收敛返回，不要为追更优反复重搜。"
        )
    return (
        f"[检索预算] 本平台检索已用满（{used}/{cap}）。请立即基于已得候选收敛返回，"
        "item_search 已不再可用。"
    )


# item_picker 出口定点收尾提示：精选刚就绪、模型正要决定下一步的那一刻注入，比 system prompt
# 开头的全局纪律有效得多（紧贴决策点、每次必触发）。治的是「口头收尾」顽疾——模型常在 item_picker
# 后输出『精挑完成，现在生成清单🎉』就停，却没真调 shopping_summary（实测 q03/q15 复现，有方差、
# 纯 prompt 压不死）。
#
# 提示里**只留 user_intent 一个参数**：曾经这里写的是「把 picks 原样传给它的 picks 参数」——可
# picks 早已改成 InjectedToolArg（模型侧根本不可见、传不了），签名换了、这条护栏没跟着换。于是
# 模型收到一条指向不存在参数的指令，只能自己揣摩着填当时还存在的 item_ids，而它揣摩的方式是
# LLM 的天性：「再精选几件最好的」——picker 挑了 10 件，收尾只进去 4 件（实测）。item_ids 现已
# 从签名整个摘除（清单恒为 picker 定稿的全部 picks），模型传无可传、也就没得砍——这条哨兵
# 提到的参数必须与真实签名保持同步，否则就是在复刻它自己治过的病。
SUMMARY_NUDGE = (
    "[系统提示] 精选清单已就绪。你的下一个动作**必须是真的调用 shopping_summary 工具**——"
    "只把用户原始意图传给 user_intent，清单即 item_picker 精选的全部商品（已按推荐度排好序）。"
    "**不要**输出「精挑完成，现在生成清单」这类纯文字就停下：说要收尾＝立刻调 shopping_summary，"
    "清单内容由它产出，不是你手敲。"
)

# 硬挡哨兵：跨平台并行 fork 跑过一轮后再发起 fork 一律回这条——逼主 loop 用现有候选收尾。
FORK_EXHAUSTED_PARALLEL = (
    "[禁止再 fork] 你已并行 fork 过一轮跨平台检索。本任务不允许为「再找找更好的」对同品类"
    "重复 fork 拓宽。立即用已汇集的候选完成用户要求的剩余步骤（按需 price_compare / "
    "shipping_calc）→ item_picker 精挑，并**以调用 shopping_summary 结束**。"
)

# 硬挡哨兵：纯串行 dispatch_tool（未跑过并行轮）超过 max_serial 次——跟跨平台 fork 无关，
# 文案不能沿用 FORK_EXHAUSTED_PARALLEL（那句话对这个场景是假的历史陈述）。
FORK_EXHAUSTED_SERIAL = (
    "[禁止再 fork] 你已连续派发多个独立子任务（dispatch_tool），已达本次上限。"
    "请立即基于已收到的各子任务结果完成剩余步骤（按需合流 price_compare / shipping_calc → "
    "item_picker 精挑），并**以调用 shopping_summary 结束**，不要再派发新的子任务。"
)

# 深度闸哨兵：子 Agent（depth≥1）调聚合/终结工具时回这条——它没有跨平台/跨商品全局视图。
SUB_AGGREGATION_DENIED = (
    "[子任务无权收尾] 你是被派发的独立子任务（单平台检索或单个商品调查），没有跨平台 / 跨商品"
    "合流后的全局视图，price_compare / shipping_calc / item_picker / shopping_summary 需要"
    "主流程合流所有子任务结果后统一做。请直接把你收敛到的结果 JSON 返回给主流程，由它收尾。"
)

# 上下文闸哨兵：子调 planner/category_insight 时回这条——平台无关、主流程已做、结果在 demands。
SUB_CONTEXT_DENIED = (
    "[子任务无需此工具] planner（意图拆解）与 category_insight（品类常识）是平台无关的，"
    "主流程已跑过一次、结果就写在你收到的 demands 里。请直接据 demands 的预算/关键词/硬约束/"
    "软偏好/品类常识做 item_search，不要重复拆解意图或查品类。"
)

# web_search 拦截哨兵：购物流程中已有候选时拦截（不是「找更好」的渠道）。
WEBSEARCH_DENIED = (
    "[web_search 未执行] 购物流程中已有候选商品，web_search 不是「找更好」的渠道。"
    "当前已有 item_search 候选，请基于现有候选继续（比价 / 精挑 / 收尾），不要用它「找更好」。"
    "（评价 / 行情类任务的口碑查询另有小配额；被拦说明本轮任务不含评价诉求或配额已用完。）"
)

# 子 Agent 本平台 item_search 次数用满。
SUB_SEARCH_EXHAUSTED = (
    "[本平台检索已用满] 你在本平台的 item_search 次数已达上限。请立即基于已召回的候选收敛，"
    "把候选 JSON 返回给主流程，不要再尝试 item_search（找够用即可，不必追最优）。"
)

# 主 loop 跑过并行 fork 后再直调 item_search。
MAIN_POSTFORK_SEARCH_DENIED = (
    "[检索阶段已结束] 跨平台并行检索已完成，候选已汇集。请基于现有候选完成用户要求的剩余步骤"
    "（按需比价 / 算到手价 → item_picker 精挑 → 收尾），不要再直接 item_search「再找找更好」。"
)

# token 预算硬线哨兵：成本放大器工具一律拦截，只留收尾链。
BUDGET_HARD_DENIED = (
    "[token 预算已超限] 本次工具未执行。请立即基于现有候选走收尾"
    "（按需 price_compare / shipping_calc → item_picker 精挑），"
    "并**以调用 shopping_summary 结束**。"
)

# 终结硬停哨兵（over-loop 治理）：主 loop 本轮已调过终结工具收尾，之后再调任何工具一律拦下，
# 逼模型直接输出面向用户的收尾文案——断掉「调完 shopping_summary 又 item_search / 再 picker」
# 的打转尾巴。
TERMINAL_REACHED_DENIED = (
    "[本轮已收尾] 你已调用过终结性工具完成本轮收尾，请**不要再调用任何工具**，"
    "直接输出面向用户的最终收尾文案即可（商品卡已由终结工具产出，无需重复检索 / 精挑）。"
)

# 主 loop 没调终结工具就想用纯文字收尾时，post_reflect 追加这条并重发一次模型。
TERMINAL_TOOL_NUDGE = (
    "[系统提示] 你刚才没有调用任何工具就准备结束对话，但本轮还没有调用过终结性工具"
    "（shopping_summary 或 chat_fallback）。**必须真的调用工具才能收尾**，不能只用文字回答就"
    "停下：有商品卡可给就调 shopping_summary；纯文字 / 非购物场景就调 chat_fallback。"
)

# token 预算软线提示原先住在这里（``BUDGET_SOFT_HINT``），已被 ``model_router.MINIMAL_HINT`` 取代：
# 那一档不只提醒，还换模型 + 收工具（见 hooks/context_compress.py 的 budget_router）。不留兼容层。


# L1 工具白名单哨兵（refdocs 16-6 §2.1）：模型调了一个根本不存在的工具名。文案刻意不列出可用
# 工具清单——那等于把工具表塞进一条错误消息里，既浪费 token，也让「诱导模型枚举内部工具」变得
# 廉价。工具表本就在每轮请求的 tools 参数里，模型看得见。
TOOL_NOT_ALLOWED = (
    "[工具不存在] `{tool}` 不是本系统的工具，未执行。请从你可用的工具列表里选一个，"
    "或基于现有信息继续。"
)


# 工具级熔断哨兵：某个工具连续失败到阈值 → 断路器 OPEN，后续调用快速失败不再真执行。
def tool_breaker_open(tool_name: str) -> str:
    return (
        f"[工具暂时不可用] {tool_name} 连续失败多次，已被熔断保护暂时停用，本次未执行。"
        "请改用其它工具或基于现有信息继续，不要重试该工具。"
    )


# 全部内部控制文案的方括号前缀——output_guard（hooks/session_hooks.py）据此清洗模型鹦鹉学舌
# 抄进最终回复的控制行。**新增哨兵时前缀必须同步登记在这里**，否则清洗漏网（曾漏
# [阶段推进] 等一整批，而它恰是每条正常链路必然出现的通告）。哨兵与清洗表共用这一份，杜绝漂移。
INTERNAL_MARKERS: tuple[str, ...] = (
    "[Harness 拒绝]",
    "[Harness 提示]",
    "[系统提示]",
    "[强制收敛]",
    "[强制收尾]",
    "[漂移纠正]",
    "[漂移提醒]",
    "[偏好丢失]",
    "[检索预算]",
    "[检索预算耗尽]",
    "[预算提醒]",
    "[token 预算已超限]",
    "[本轮已收尾]",
    "[格式问题]",
    "[顺序问题]",
    "[相关性问题]",
    "[阶段推进]",
    "[阶段回退]",
    "[禁止再 fork]",
    "[子任务无权收尾]",
    "[子任务无需此工具]",
    "[web_search 未执行]",
    "[本平台检索已用满]",
    "[检索阶段已结束]",
    "[工具不存在]",
    "[工具暂时不可用]",
    "[dispatch_tool 拒绝]",
    "[dispatch_tool 超时]",
    "[dispatch_tool 错误]",
    "[…工具结果过长已截断",
)
