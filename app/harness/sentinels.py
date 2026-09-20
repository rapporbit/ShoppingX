"""执行层哨兵文案：工具被闸拦下时**回给模型**的固定信号。

**为什么是哨兵而不是摘工具**：所有禁用逻辑统一在执行层（pre_tool_call Hook）处理，从不从模型
可见工具表中摘除工具——tools 列表每轮保持完整不变，保住 prompt cache 前缀稳定。模型调了被禁
工具 → 收到哨兵消息 → 下一轮自然转向可用工具。（refdocs 17-4 §2.3 主张「隐藏 > 拒绝」，本项目
出于前缀缓存治理反其道而行，是自觉的取舍。）

拒绝的理由对模型必须**真实**，不能张冠李戴——所以检索耗尽分「软收敛」与「硬挡」两档。
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


def retrieval_exhausted(count: int) -> str:
    """再越检索预算（block）：工具**不执行**，直接回哨兵。"""
    return (
        f"[检索预算耗尽] 已累计商品检索 {count} 次，本次未执行，禁止再检索。"
        "立即基于现有候选调 item_picker 精挑、再 shopping_summary 收尾"
        "（**用户要了比价 / 到手价才**先 price_compare / shipping_calc，没要就跳过）。"
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
    "收尾文案写进 summary、每件一条理由写进 reasons、用户原话传 user_intent；清单即 item_picker "
    "精选的全部商品（已按推荐度排好序），你不选件。"
    "**不要**输出「精挑完成，现在生成清单」这类纯文字就停下：说要收尾＝立刻调 shopping_summary，"
    "清单内容由它产出，不是你手敲。"
)

# 这里曾有派发耗尽 / 子搜上限 / postfork 直搜三条哨兵（``[禁止再派发]`` / ``[本平台检索已用满]`` /
# ``[检索阶段已结束]``），A4 删子 Agent 时随各自的闸一起删除。

# web_search 拦截哨兵：购物流程中已有候选时拦截（不是「找更好」的渠道）。
WEBSEARCH_DENIED = (
    "[web_search 未执行] 购物流程中已有候选商品，web_search 不是「找更好」的渠道。"
    "当前已有 item_search 候选，请基于现有候选继续（比价 / 精挑 / 收尾），不要用它「找更好」。"
    "（评价 / 行情类任务的口碑查询另有小配额；被拦说明本轮任务不含评价诉求或配额已用完。）"
)


# research 配额哨兵：会话级搜索条数用尽 / 本次要的条数超过剩余额度时回这条。
# 明确报出「剩几条、单次要几条」——模型据此可以少给几个 target 再试一次（这是真出路），
# 而不是收到一句笼统的「不许调」后换个措辞硬撞同一道闸。
def research_quota_denied(planned: int, remaining: int, quota: int) -> str:
    if remaining <= 0:
        return (
            f"[research 未执行] 本次会话的公网研究额度已用完（每会话最多 {quota} 条搜索）。"
            "请基于已查到的资料与现有候选继续，需要补充事实时如实说明「这部分没有查到」，"
            "不要换措辞重试。"
        )
    return (
        f"[research 未执行] 本次要研究 {planned} 个对象（＝{planned} 条搜索），"
        f"但本会话只剩 {remaining} 条公网研究额度。请只保留最关键的 {remaining} 个对象重调一次，"
        "其余对象基于已有资料判断或如实说明不确定。"
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
    "（shopping_summary / present_guide / chat_fallback）。**必须真的调用工具才能收尾**，不能只用"
    "文字回答就停下：有商品卡可给就调 shopping_summary；讲的是选购标准就调 present_guide"
    "（分节写进 sections）；其余纯文字 / 非购物场景调 chat_fallback。"
)

# token 预算软线提示原先住在这里（``BUDGET_SOFT_HINT``），已被 ``model_router.MINIMAL_HINT`` 取代：
# 那一档不只提醒，还换模型 + 收工具（见 hooks/budget.py 的 budget_router）。不留兼容层。


# 这里曾有 TOOL_NOT_ALLOWED（refdocs 16-6 §2.1 的 L1 白名单拒绝文案）。工具名不在 Toolkit 里
# 时框架自己就不会执行，这条文案 301 会话一次都没发出去过；tool_whitelist 2026-09-10 降为
# 告警后没了消费者，已删。


# 工具级熔断哨兵：某个工具连续失败到阈值 → 断路器 OPEN，后续调用快速失败不再真执行。
def tool_breaker_open(tool_name: str) -> str:
    return (
        f"[工具暂时不可用] {tool_name} 连续失败多次，已被熔断保护暂时停用，本次未执行。"
        "请改用其它工具或基于现有信息继续，不要重试该工具。"
    )


# 主检索依赖（Qdrant / embedding）不可用：重试无意义那一档，见 app/utils/dependency.py。
# 与 tool_breaker_open 的分工：那条说的是「这个工具先别用了，换个工具继续干活」，这条说的是
# 「货源本身断了，本轮没法交付商品，如实收尾」——后者必须点名 chat_fallback，否则模型会退而
# 求其次去 web_search 拼几个商品名给用户，那是拿搜索结果冒充库存，踩 P0。
def dependency_down_notice(dependency: str, tool_name: str) -> str:
    return (
        f"[依赖暂时不可用] 商品检索依赖（{dependency}）当前连不上，{tool_name} 本次没有拿到任何"
        "数据。**这不是检索词的问题，换个词重试同样拿不到**，请不要再调用检索类工具。"
        "请直接调用 chat_fallback，如实告诉用户「商品检索服务暂时不可用，请稍后重试」——"
        "不要用 web_search 的结果或你自己的记忆拼凑商品清单，那不是我们库里的货。"
    )


# 全部内部控制文案的方括号前缀——output_guard（hooks/session_hooks.py）据此清洗模型鹦鹉学舌
# 抄进最终回复的控制行。**新增哨兵时前缀必须同步登记在这里**，否则清洗漏网（曾漏
# [阶段推进] 等一整批，而它恰是每条正常链路必然出现的通告）。哨兵与清洗表共用这一份，杜绝漂移。
INTERNAL_MARKERS: tuple[str, ...] = (
    "[Harness 拒绝]",
    "[Harness 提示]",
    "[系统提示]",
    "[系统已自动执行]",
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
    "[未执行取消]",
    "[格式问题]",
    "[顺序问题]",
    "[相关性问题]",
    "[阶段推进]",
    "[阶段回退]",
    "[web_search 未执行]",
    "[research 未执行]",
    "[工具不存在]",
    "[工具暂时不可用]",
    "[依赖暂时不可用]",
    "[…工具结果过长已截断",
)


# 交易顺序闸哨兵：没查就取消时回这条。
CANCEL_WITHOUT_QUERY = (
    "[未执行取消] 取消订单前必须先 query_order 查到那张单，确认它确实存在、属于该用户、"
    "且状态是 CONFIRMED。请先调 query_order，拿到真实订单号与状态后再取消——取消是写操作，"
    "取消错了改的是库里的真实状态。"
)
