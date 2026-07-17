"""shopping_summary —— 终结性工具：给最终清单 + 选购理由 + 沉淀新偏好。

主链路的收尾。拿到精挑后的候选，生成面向用户的购物清单（每件配选购理由，对应其硬约束 /
软偏好），并识别本轮可沉淀的新偏好（如用户明确「不要塑料」「喜欢小众」），输出供长期记忆
落库（M7 接 Store；M4 先只返回，不落库）。

**终结性**（在 ``TERMINAL_TOOLS`` 里）：一旦调用即收尾，主 loop 不再继续。这是对治
「Agent 不收尾死循环」最直接的一手——给模型一个明确的「话讲完了」的出口。

用 LLM 生成清单文案与偏好识别（``with_structured_output`` 约束结构）。

**返回形态（M9 合龙关键）**：用 ``response_format="content_and_artifact"`` ——给模型/前端看的
是可读的清单文案（``content``），完整结构化输出作为 ``artifact`` 挂在 ToolMessage 上，
前端 M10 直接拿 artifact 渲染商品卡；``run_agent`` 收尾也从中取回结构化清单落产物文件。
（偏好沉淀已不在此:改由会话结束后独立的记忆管家扫全轮对话统一判定，见 ``app/memory/curator.py``。）
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from typing import Annotated

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.tools import InjectedToolArg, tool
from pydantic import BaseModel, Field, model_validator

from app.agent.llm import get_fast_llm
from app.agent.prompts import get_shopping_summary_prompt
from app.agent.token_budget import charge_tool_llm_usage
from app.api import monitor
from app.api.context import get_dest_country, is_dest_country_assumed
from app.tools._args import drop_none_values
from app.tools._bundle import (
    drop_pick_from_report,
    get_bundle_report,
    refresh_report_prices,
    render_allocation,
    slot_display,
)
from app.tools._candidates import (
    enrich,
    get_last_picks,
    hydrate,
    picker_finalized,
    register_updates,
    registry_snapshot,
)
from app.tools.schemas import ItemCandidate
from app.tools.shipping_calc import cost_item


class SummaryItem(BaseModel):
    """清单里的一件商品 + 选购理由。

    价格两个字段各说各的、**互不兜底**：``landed_usd`` 只在本轮真跑过 shipping_calc 时才有值，
    ``price_usd`` 是平台货价（未含税运）。曾经的写法是 landed 缺失就拿货价顶上，结果没算过运费的
    货价被前端原样标成「到手价（含税运）」——跨境单里这个数能差出一大截，等于骗用户。
    """

    item_id: str
    platform: str
    title: str
    landed_usd: float | None = None
    price_usd: float | None = None
    reason: str = ""
    # 套装槽位名（「一套齐」轮才非空）：前端据此把卡片按槽分组渲染（组头 = 槽名 + 该槽花费）。
    # 走结构化字段而不是理由文案里的【槽名】前缀——前缀会被收尾 LLM 重写叙事句时丢掉（实测）。
    slot: str = ""
    # 商品图 URL / 商品页 URL：均不喂给 LLM、不由它生成（防 URL 幻觉），收尾按 item_id 从原候选
    # 回填真实值。image_url 供卡片显图，url 供点击跳转到该平台商品页。
    image_url: str = ""
    url: str = ""


class ShoppingSummaryOutput(BaseModel):
    """shopping_summary 的结构化返回（终结性）。

    **不含偏好沉淀**:偏好识别 / 落库已剥离给会话结束后独立运行的记忆管家
    （``app/memory/curator.py``）——summary 只管购物清单文案 + 商品卡，不再抽 new_preferences。
    """

    summary: str = Field(description="面向用户的购物清单文案（含每件的选购理由）")
    items: list[SummaryItem] = Field(default_factory=list)


# LLM 重写理由的件数上限。**这是延迟与文风之间的那颗螺丝**：
#
# 旧版让模型给**每一件**写理由，解码量随件数线性涨（实测 10 件 → 8.5s，主 loop 降快档后它成了
# 全链路单点最贵的一段）。可 picks 是**按分排好序**的——用户的注意力也是；第 8 件的理由写得多
# 漂亮，多半没人读。于是只给最前面这几件重写成叙事句，后面的用 item_picker 的确定性理由
# （``_card_reason``）。解码量因此**固定**：3 件和 10 件花一样的时间。
_LLM_REASON_TOP_N = 3

# 理由完整性兜底的启发式：快模型偶尔把某件的 reason 生成到一半就停（实测出现过 "评分4.1，标题"
# 这种半句话直接进了商品卡）。structured_output 拿不到可靠的 finish_reason，只能判「明显异常」：
# 近空（<6 字）或以连接词 / 语气助词 / 字段名收尾 = 半句被截。检出即退回确定性理由，不重调 LLM。
_DANGLING_TAIL = (
    "，",
    "、",
    "：",
    ":",
    "（",
    "(",
    "和",
    "与",
    "或",
    "的",
    "了",
    "标题",
    "评分",
    "—",
    "…",
)
_MIN_REASON_CHARS = 6


def _looks_truncated(reason: str) -> bool:
    r = reason.strip()
    return len(r) < _MIN_REASON_CHARS or r.endswith(_DANGLING_TAIL)


class SummaryReason(BaseModel):
    """一件候选的选购理由（按 item_id 对应回 picks）。"""

    item_id: str
    reason: str = ""


class _SummaryDraft(BaseModel):
    """喂给 LLM 的**精简产出契约**：一段收尾文案 + 前 N 件的叙事理由。

    模型**不产**商品卡的任何确定字段（title / platform / 到手价 / 图 / 链接）——那些 picks 里都有，
    让模型逐件重吐既费解码又平白引入「改写标题 / 生成到一半截断」的风险。它只写两样人写得比规则
    好的东西：① 把整批货与用户原话对上的那段话；② 最前面 :data:`_LLM_REASON_TOP_N` 件的理由。

    第 N+1 件之后一律用 ``item_picker`` 算好的确定性理由（``_card_reason``）——它同时也是前 N 件的
    **兜底**：模型漏写、写崩、超时，卡片照样有完整理由。
    """

    # 显式 null 归一为缺席（badcase cdee1d6d 同族，见 drop_none_values）。
    _null_is_absent = model_validator(mode="before")(staticmethod(drop_none_values))

    summary: str = Field(description="面向用户的收尾文案（推荐清单开场白 / 评价结论）")
    reasons: list[SummaryReason] = Field(
        default_factory=list,
        description=f"**只为最前面 {_LLM_REASON_TOP_N} 件**写选购理由，按 item_id 对应",
    )
    # off-intent 判定为什么放这里而不是加一条确定性规则：picker 是纯确定性打分，而库里配件与
    # 整机同 category、标题全含品类词（"Camera Case for Sony"），静态品类过滤 / 黑名单两方案均被
    # 实测证伪。本 LLM 调用**本来就在**逐件读标题写理由（相机 bad case 里它自发写出「这是配件
    # 不是相机本身」），判断已经做了、只差结构化后果——零额外调用，执行侧有确定性护栏
    # （摘除后主推区必须 ≥2 件，见 shopping_summary 内的守卫）。
    #
    # 套装轮同理且更甚：占了某槽却不是该槽品类本体的（贴纸占水杯槽，badcase 4c0ac682 里收尾
    # LLM 自发写出「虽然不算传统水杯」——判断已经有了，差的就是摘除出口），reranker 阈值门
    # 在「品类内配件蹭词」上被真实数据标定证伪（分数完全交叠），这里是唯一可行的判别器。
    off_intent: list[str] = Field(
        default_factory=list,
        description="与用户本轮要买的品类明显不符的 item_id（周边配件混入等）；套装轮里"
        "「占了某槽但明显不是该槽品类本体」的也算（如贴纸占了水杯槽）。拿不准不填",
    )


def _card_reason(src: ItemCandidate) -> str:
    """一件商品卡的选购理由——**确定性组装，不走 LLM**。

    优先复用 ``item_picker`` 的 ``pick_reason``（它本就是完整的一句话理由，见
    ``item_picker._build_reason``）；候选没经过 picker（模型直接拿检索结果收尾）时，用评分 /
    价格 / 平台兜一句完整话。

    两个用处：**第 4 件起的常规理由**，以及**前 3 件的兜底**（模型漏写 / 写崩 / 超时）。所以它
    必须永远返回一句完整的话——绝不返回空串，让商品卡带着空理由上桌。这句话是**用户正脸看到的**，
    措辞得是人话：不出现「检索」「候选」这类只有工程师会说的词。
    """
    if src.pick_reason.strip():
        return src.pick_reason.strip()
    bits: list[str] = []
    if src.rating is not None:
        bits.append(f"评分 {src.rating}")
    price = src.landed_usd if src.landed_usd is not None else src.price_usd
    if price is not None:
        label = "到手价" if src.landed_usd is not None else "售价"
        bits.append(f"{label}约 ${price:.2f}")
    if src.platform:
        bits.append(f"在 {src.platform.capitalize()} 上有售")
    return "，".join(bits) + "，可以作为备选。" if bits else "可以作为备选。"


logger = logging.getLogger("shoppingx.tools.shopping_summary")

# ── 收尾文案流式生成（杠杆3·感知延迟）──────────────────────────────────────────────
# 收尾链此前是「结构化调用全跑完才一次性下发」——用户在最后一步干等 8~12s。改为流式：边生成
# 边把 summary 字段的**累计文本**经 summary_delta 事件推给前端逐字渲染，reasons 等完整 JSON
# 到齐后照旧结构化校验。任何一环不支持（供应商不流式 / 字段顺序反了 / 局部 JSON 解不出）都
# 静默降级——增量少发或不发，最终产物与非流式路径完全一致，绝不影响正确性。

# 从累计的 tool-call args 缓冲里抠"summary"字符串值（可能尚未闭合）。
# `(?:[^"\\]|\\.)*` 只吞完整转义对——缓冲结尾悬垂的半个转义不会进组，json.loads 不会炸。
_PARTIAL_SUMMARY_RE = re.compile(r'"summary"\s*:\s*"((?:[^"\\]|\\.)*)')
# 增量推送节流：累计文本至少长出这么多字符才发一条（几百字文案 → 十几条事件）。
_DELTA_MIN_CHARS = 12


def _partial_summary(buf: str) -> str:
    """从流式累积的 JSON 参数缓冲里提取 summary 的当前文本；解不出返回空串（本 tick 跳过）。"""
    m = _PARTIAL_SUMMARY_RE.search(buf)
    if not m:
        return ""
    try:
        return json.loads('"' + m.group(1) + '"')  # 借 JSON 自己做反转义
    except ValueError:
        return ""  # 罕见：\uXXXX 恰好写到一半——下一个 chunk 补齐后自然恢复


def strip_item_ids(text: str, id_map: Mapping[str, str]) -> str:
    """把文案里的商品 ID 换成商品名——**确定性后处理，不指望模型自觉**。

    ID（ASIN 之类）是内部主键：用户不认、用不上，卡片上本就有商品名和链接，写进散文里只是把
    数据库主键怼到他脸上。prompt 里已经明令禁止，但实测模型**只是有时遵守**（同一句 query 跑三次，
    两次干净、一次写成「最推荐的是 **Travel Packing Cubes 3pc Set**（B098QCGR94）」）——这种
    「大多数时候对」的东西不能靠劝，只能在出口处兜死。

    **换名不是删字**：裸删会把句子撕烂（「看 B0XXX 和 B0YYY 这两件」→「看 和 这两件」）。按 ID
    在句中的三种位置分别处置，务求剩下的仍是一句通顺的话：

    - ``名字（B0XXX）``  → ``名字``          （名字已在，括号里的 ID 是纯冗余，整块删）
    - ``B0XXX（名字）``  → ``名字``          （ID 占了名字的位置，用括号里的名字顶上）
    - 裸 ``B0XXX``      → ``名字``          （拿 id_map 里的商品名替换）
    """
    ids = [i for i in id_map if i.strip()]
    if not ids:
        return text
    alt = "|".join(re.escape(i) for i in sorted(ids, key=len, reverse=True))
    # ① ID 后面紧跟括号说明（`B0XXX（3-Piece Packing Cubes）`）——留括号里的名字，丢 ID。
    text = re.sub(rf"(?:{alt})\s*[（(]\s*([^）)]+?)\s*[)）]", r"\1", text)
    # ② 括号里只有 ID（`Travel Cubes（B0XXX）`）——名字已在括号外，整块删掉。
    text = re.sub(rf"\s*[（(]\s*(?:{alt})\s*[)）]", "", text)
    # ③ 剩下的裸 ID——换成商品名（删了会留下「看 和 这两件」这种残句）。
    text = re.sub(rf"(?:{alt})", lambda m: id_map.get(m.group(0), "").strip(), text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


async def _stream_draft(
    messages: list[tuple[str, str]],
    usage_cb: UsageMetadataCallbackHandler,
    id_map: Mapping[str, str],
) -> _SummaryDraft:
    """流式路径：强制单工具调用，边收 args 分片边推 summary 增量，收齐后整体校验。

    ``stream_usage=True`` 让供应商在流尾带 usage（OpenAI 兼容口径的 stream_options），
    usage_cb 才有账可记；不支持的供应商会在这里抛错 → 上层降级到阻塞路径。

    增量推给前端前也过一遍 :func:`strip_item_ids`：否则模型写出的 ID 会先逐字渲染到用户眼前，
    等收尾的最终文案再把它换掉——闪一下的主键，用户照样看见了。
    """
    bound = get_fast_llm().bind_tools([_SummaryDraft], tool_choice=_SummaryDraft.__name__)
    buf = ""
    emitted = 0
    async for chunk in bound.astream(messages, config={"callbacks": [usage_cb]}, stream_usage=True):
        for tc in getattr(chunk, "tool_call_chunks", None) or []:
            buf += tc.get("args") or ""
        text = _partial_summary(buf)
        if len(text) >= emitted + _DELTA_MIN_CHARS:
            emitted = len(text)
            await monitor.report_summary_delta(strip_item_ids(text, id_map))
    return _SummaryDraft.model_validate(json.loads(buf))


async def _generate_draft(
    messages: list[tuple[str, str]], id_map: Mapping[str, str]
) -> _SummaryDraft:
    """先走流式（用户提前看到文案），失败降级为原阻塞结构化调用；usage 两条路都入账。"""
    usage_cb = UsageMetadataCallbackHandler()
    try:
        try:
            return await _stream_draft(messages, usage_cb, id_map)
        except Exception:
            # 降级代价是重调一次（前一次流着的 token 白花）——记 warning 让它可见，
            # 若某供应商长期走不了流式，该在配置层关掉而不是每轮白烧一遍。
            logger.warning("收尾文案流式生成失败，降级为阻塞结构化调用", exc_info=True)
            # method 钉死的理由见 planner.py（默认值随模型能力画像浮动，qwen 系会 400）。
            structured = get_fast_llm().with_structured_output(
                _SummaryDraft, method="function_calling"
            )
            result = await structured.ainvoke(messages, config={"callbacks": [usage_cb]})
            return (
                result
                if isinstance(result, _SummaryDraft)
                else _SummaryDraft.model_validate(result)
            )
    finally:
        # 放 finally：无论哪条路、成功失败，已消耗的 token 都入账（helper 自身绝不抛）。
        charge_tool_llm_usage(usage_cb.usage_metadata)


def _landed_note(picks: list[ItemCandidate]) -> str:
    """候选带到手价时，把**收货国口径**一并交代给写文案的模型（没到手价则返回空串）。

    候选 JSON 里只有 ``landed_usd`` 这个数，没有「寄往哪国」——模型于是写得出「到手 $30.99」，
    却写不出这是按寄往哪算的。可这个数**只在某个收货国下成立**（关税免征额 US $0 / CN $7 /
    AU $660，差两个数量级），而用户本轮多半压根没提收货国（是 planner 从他会话里说过的话 /
    长期收货地推的，甚至是系统默认值）。不交代口径，等于给他一个不敢信的数字。
    """
    if not any(c.landed_usd is not None for c in picks):
        return ""
    dest = get_dest_country().upper()
    note = f"到手价口径：候选里的 landed_usd 是**按寄往 {dest} 估算**的（含国际运费 + 关税）。"
    if is_dest_country_assumed():
        note += "收货国是系统默认值（用户从没说过）——文案里务必提醒他「实际收货地不同请告诉我」。"
    return note + "文案里必须说清这个口径。\n\n"


def _bundle_note(picks: list[ItemCandidate]) -> str:
    """套装轮把**预算分配表**一并交代给写文案的模型（非套装轮返回空串）。

    分配是 item_picker 组合优选算好的确定性事实（哪槽花了多少、砍了谁、缺了谁——见
    ``app.tools._bundle``），模型只负责把它讲成人话。同 ``_landed_note`` 的思路：候选 JSON 里
    只有单件的价格，看不出「这是在总预算里怎么分的」，不交代口径模型就只能挨件复述。

    注入前用**当前**有效价刷新分配表（``refresh_report_prices``）：组合定稿时多半还没算
    到手价，不刷新就会「分配表一套数、商品卡另一套数」。
    """
    report = get_bundle_report()
    if not report:
        return ""
    report = refresh_report_prices(report, picks)
    return (
        "这是一次「一套齐」套装推荐，预算分配如下（确定性计算结果，照实转述、不得改数）：\n"
        f"{render_allocation(report)}\n"
        "文案必须讲清：总价 vs 总预算、钱主要花在哪个槽、哪个槽选了省钱款；被放弃的可选槽 / "
        "没找到货的必备槽要如实交代，绝不拿别的商品顶替缺货的槽。"
        "「本轮未检索的槽」**不是缺货**——不要写成「没找到」，也不要建议用户单独再搜它。"
        "**逐槽核对**：某槽的入选商品若明显不是该槽品类本体（如贴纸/周边占了水杯槽），把它的 "
        "item_id 列进 off_intent、文案里如实说该槽没找到合适的——绝不把周边包装成正品推荐。\n\n"
    )


def _compact(picks: list[ItemCandidate]) -> str:
    """把候选压成喂给 LLM 的精简 JSON（只留生成清单要用的字段，省 token）。"""
    rows = [
        {
            "item_id": c.item_id,
            "platform": c.platform,
            "title": c.title,
            "brand": c.brand,
            "price_usd": c.price_usd,
            "landed_usd": c.landed_usd,
            "rating": c.rating,
            "pick_reason": c.pick_reason,
            "pref_matched": c.pref_matched,  # prompt 的「没一件命中偏好」判据（结构化，不靠措辞）
        }
        for c in picks
    ]
    return json.dumps(rows, ensure_ascii=False)


@tool(response_format="content_and_artifact")
async def shopping_summary(
    user_intent: str = "",
    picks: Annotated[list[ItemCandidate] | None, InjectedToolArg] = None,
) -> tuple[str, ShoppingSummaryOutput]:
    """终结性工具：基于精选候选生成最终购物清单 + 选购理由，并沉淀新偏好。

    何时调用：信息已足够、拿到精挑后的候选、要给用户最终答复时——调它即收尾，不要再检索。
    清单内容就是 item_picker 本轮精选的**全部** picks（已按推荐度排好序、已封顶件数），
    你不需要也无法指定件数——筛选是 picker 的职责。
    参数：
      - user_intent：用户本轮的原始意图（帮模型把理由对齐到需求）。
    """
    # 候选来源两档：
    #   ① picks（InjectedToolArg，模型侧不可见）—— 直接调用 / 单测注入现成候选，绕开登记表；
    #   ② 缺省 —— item_picker 本轮定稿的那批（get_last_picks，唯一的模型侧路径）。
    #
    # 曾有模型可传的 item_ids 参数，已摘除：只要参数在，模型就会「再精选一遍」——picker 挑了
    # 10 件、它只抄 4 件进来（实测），而它此刻没有打分依据，纯凭印象。政策上收尾前必过 picker
    # （<termination>：候选 >0 必先精挑；=0 空清单如实说），参数没有合法用途，摘掉即根治。
    if picks is None:
        picks = hydrate(get_last_picks())
        # 绊线（fail-loud，不改变行为）：机制上不该到达——phase_check 底线 3 已要求收尾前必过
        # 本轮 item_picker。真到了这里就是又出现了绕过精挑的收尾路径，此时收尾 LLM 会拿着空
        # 通道照「没找到」模板编出与候选池自相矛盾的答案（badcase 63093a85：price_compare 刚
        # 算完 12 件到手价，答复却称「没找到商品」）。大声报警让告警管道看见，别等用户发现。
        if not picks and not picker_finalized() and registry_snapshot():
            logger.error("收尾时 picks 为空但候选池非空：存在绕过本轮精挑的收尾路径")
            await monitor.report_error(
                "summary_empty_picks", "收尾时精选为空但候选池非空（疑似绕过 phase_check 底线）"
            )
    # 到手价口径收口（机制补算，零 LLM）：补搜进来的件没经过 price_compare/shipping_calc，
    # landed 为 None——不补算就会「裸价混进到手价总价」还声称含税（badcase 4c0ac682）。只在
    # 本轮确实有到手价口径（至少一件已有 landed）时补齐剩下的，纯裸价轮不无中生有。
    if any(c.landed_usd is not None for c in picks):
        dest = get_dest_country()
        backfilled = [c for c in picks if c.landed_usd is None and cost_item(c, dest)]
        if backfilled:
            register_updates(backfilled)
            logger.info("summary 到手价补算 %d 件（dest=%s）", len(backfilled), dest)
    await monitor.report_tool_start("shopping_summary", count=len(picks))
    try:
        # 文案生成用非推理快模型（perf/model-tiering）：这步只按既定 picks 组织文案，不需深推理。
        # **只让 LLM 产一段开场白**（_SummaryDraft）——title/platform/到手价/图/链接/每件理由全是
        # picks 里已知的确定字段，一律不经模型（省解码 + 杜绝改写/截断风险，见 _SummaryDraft）。
        # 于是这次调用的解码量不再随 picks 件数线性涨——10 件与 3 件花的时间基本一样。
        # 流式优先（summary_delta 逐字推给前端）、失败降级阻塞；usage 两条路都入账。
        top_ids = [c.item_id for c in picks[:_LLM_REASON_TOP_N]]
        id_map = {c.item_id: c.title for c in picks}
        draft = await _generate_draft(
            [
                ("system", get_shopping_summary_prompt()),
                (
                    "user",
                    f"用户意图：{user_intent}\n\n{_bundle_note(picks)}{_landed_note(picks)}"
                    f"精选候选（JSON，已按推荐度排序）：\n{_compact(picks)}"
                    f"\n\n**只为这 {len(top_ids)} 件写 reason**：{', '.join(top_ids)}",
                ),
            ],
            id_map,
        )
        # 文案与逐件理由都过一道 ID 剥离：prompt 禁了、模型仍会偶尔写（见 strip_item_ids）。
        draft.summary = strip_item_ids(draft.summary, id_map)
        reason_map = {r.item_id: strip_item_ids(r.reason, id_map) for r in draft.reasons}
        # off-intent 摘除（带确定性护栏）：draft LLM 判定「与本轮要买的品类明显不符」的商品从
        # 清单摘掉（相机清单里的支架 / 胶卷）。护栏：摘完主推区必须 ≥2 件，否则一件不摘——
        # LLM 判断只有在「清单仍然成立」的前提下才被执行，绝不让它把清单摘空。
        off = set(draft.off_intent) & {c.item_id for c in picks}
        if off:
            on_intent = [c for c in picks if c.item_id not in off]
            if len(on_intent) >= 2:
                logger.info("shopping_summary off-intent 摘除 %d 件：%s", len(off), sorted(off))
                # 套装轮：被摘的件若占着槽位，把分配报告同步成「该槽缺货」——报告是续聊轮
                # 与产物落盘的事实来源，不同步就是「卡片没有贴纸、分配表还挂着它」。
                for c in picks:
                    if c.item_id in off and c.slot:
                        drop_pick_from_report(c.item_id)
                picks = on_intent
            else:
                logger.info(
                    "off-intent 判定会把清单摘到只剩 %d 件，保守全保留（判定：%s）",
                    len(on_intent),
                    sorted(off),
                )
        # 商品卡逐字段**确定性组装**：item_id/platform/title/到手价/图/链接全取自 picks（登记表
        # hydrate 的全量候选，url/image 不穿过模型）。reason 两档：前 N 件用模型写的叙事句，其余
        # （以及模型漏写 / 写崩的那几件）用 item_picker 算好的确定性理由。顺序 = picks 顺序。
        items: list[SummaryItem] = []
        for c in picks:
            # url/image 优先从登记表按 item_id 取真实值（无 session / 单测直传 picks 时退回 c）。
            src = enrich(c.item_id) or c
            reason = reason_map.get(c.item_id, "") if c.item_id in top_ids else ""
            if _looks_truncated(reason):
                reason = _card_reason(src)
            items.append(
                SummaryItem(
                    item_id=c.item_id,
                    platform=c.platform,
                    title=c.title,
                    # 两个价格字段照实透传、不互相兜底：没跑 shipping_calc 就没有到手价，前端会
                    # 老实标「货价（未含税运）」，而不是把货价冒充成到手价（见 SummaryItem）。
                    landed_usd=c.landed_usd,
                    price_usd=c.price_usd,
                    reason=reason,
                    image_url=src.image_url,
                    url=src.url,
                    # 内部盖章是槽 id；出卡片映射成展示名（旧会话按名字盖的章原样透传）。
                    slot=slot_display(src.slot or c.slot),
                )
            )
        out = ShoppingSummaryOutput(summary=draft.summary, items=items)
    except Exception:
        # 模型调用失败也要补一条 end 事件，否则前端（M8）会看到工具「永远在跑」。
        await monitor.report_tool_end("shopping_summary", error=True)
        raise
    await monitor.report_tool_end("shopping_summary", items=len(out.items))
    # content（给模型/前端的可读文案）+ artifact（完整结构化输出，供 run_agent 写回偏好）。
    return out.summary, out
