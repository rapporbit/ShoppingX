"""会话级候选登记表 + 候选的「模型可见」紧凑序列化。

**为什么存在**：候选在工具间靠 LLM 把列表当参数一路透传（item_search → price_compare →
shipping_calc → item_picker → shopping_summary）。其中 ``url`` / ``image_url`` **只用于前端商品卡**
（显图 + 点击跳转），模型推理根本不用——但旧实现把它们随候选一路喂给模型（input），又让模型一路
回吐（output，正是解码瓶颈），还得在上下文里活到收尾才能回填卡片。

本模块把「候选必须整包穿过模型」这条耦合解开：
  - :func:`compact_candidates`：候选的**模型可见投影**——只留模型做决策真正要用的字段
    （``item_id`` / ``title`` / ``price_usd`` / ``rating`` / ``category``，加已填的到手价与
    入选理由），丢掉 url、召回分、恒 0 的评价数、与 ``price_usd`` 重复的原价币种、以及尚未
    填充的阶段字段。各工具 Output 的 ``__str__`` 用它，让模型看到紧凑 JSON。
  - **模型不必搬运全量字段**：下游工具只收 ``item_ids``，靠 :func:`hydrate` 从登记表捞回全量候选，
    故「投影给模型的」与「工具算用的」可以是两份——前者按 token 优化，后者按正确性优化。
  - :func:`register` / :func:`enrich`：item_search 把**全量**候选（含真实 url/image_url）按
    ``item_id`` 登记到会话作用域；shopping_summary 收尾时按 ``item_id`` 回填卡片的 url/image。url
    不再需要穿过模型——丢没丢、改没改都不影响卡片（item_id 被模型改写时仍降级为空，与原行为一致）。

会话作用域照 :mod:`app.agent.retrieval_budget`：按 ``session_dir`` 为键的模块级 dict（ContextVar 的
``set`` 不回传 fork 父，故用显式 session_dir 聚合，主 / 子共享同一会话条目）。``run_agent`` 收尾调
:func:`reset_candidates` 清理，防模块级 dict 无界增长。无 session 作用域（单测）时各函数静默降级。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path

from pydantic import ValidationError

from app.api.context import get_session_dir
from app.tools.schemas import ItemCandidate

logger = logging.getLogger("shoppingx.candidates")

# 喂模型时从候选里剔除的字段：长且模型不用，仅供前端卡片（按 item_id 旁路回填）。
_MODEL_HIDDEN_FIELDS = frozenset({"url", "image_url"})

# 模型**不该拿来做决策**的字段，一律不喂：
#   score —— 召回相似度。候选本来就按它降序排好了，序位已经把这个信息表达完了；把 0.71 这种
#     数字摆到模型眼前，只会诱导它拿一个连自己都解释不了的分数当理由。
#   reviews_count —— 本仓库数据集**没有这一列**，恒为 0。喂给模型等于告诉它「这商品零评价」，
#     是负向误导。真有评价数的那天再放回来（届时它是有效的质量信号）。
#   pref_matched —— 精挑的内部判据，只给 shopping_summary 的 _compact 显式带上；主 loop 的
#     模型不需要它做任何决策。
_MODEL_NOISE_FIELDS = frozenset({"score", "reviews_count", "pref_matched"})

# price_usd 有值时冗余的字段：price 是原币种报价、currency 是币种，两者合起来的信息
# price_usd 已经全包了（且下游一律按 USD 决策）。price_usd 缺失时这两个是唯一的价格信息，
# 必须留（见 compact_candidates 的兜底分支）。
_REDUNDANT_WHEN_USD = frozenset({"price", "currency"})

# session_dir(str) -> {item_id: ItemCandidate（全量，含真实 url/image_url）}
_REGISTRY: dict[str, dict[str, ItemCandidate]] = {}


def _key() -> str | None:
    sd = get_session_dir()
    return str(sd) if sd is not None else None


def compact_candidates(
    cands: Iterable[ItemCandidate],
    *,
    drop: Iterable[str] = (),
    title_chars: int | None = None,
) -> list[dict]:
    """候选的**模型可见**投影：只留模型做决策真正要用的字段。

    判据是「模型拿这个字段干什么」。下游工具（price_compare / shipping_calc / item_picker /
    shopping_summary）**只收 item_ids**，候选体是 ``InjectedToolArg``（模型看不见也传不了），
    工具内部用 :func:`hydrate` 从登记表捞回**全量**候选。所以模型手里这份候选的唯一用途是
    **决定把哪些 item_id 传下去** + 给用户讲人话——凡是不服务这两件事的字段，都是纯烧 token。

    四道过滤：
      1. ``url`` / ``image_url``：只喂前端卡片，走 item_id 旁路回填（见模块 docstring）。
      2. ``score`` / ``reviews_count``：模型不该拿来决策（见 :data:`_MODEL_NOISE_FIELDS`）。
      3. ``price`` / ``currency``：``price_usd`` 有值时冗余；缺失时留作兜底（唯一价格信息）。
      4. **尚未填充**的渐进字段（``None`` / 空串）：``ItemCandidate`` 是渐进填充设计，召回阶段
         ``shipping_usd`` / ``landed_usd`` 等还是 ``None``。「字段是 null」与「字段不在」对模型
         同义（这步还没做），但 ``"shipping_usd": null`` 要花真金白银的 token。填过的照常留。

    ``drop`` 供调用方按上下文再砍：单平台检索时 ``item_search`` 传 ``{"platform"}``——顶层已经
    写了一次平台名，每条再重复一遍纯冗余（``platform="all"`` 合流时则必须留）。

    ``title_chars`` 供**回显场景**（price_compare / item_picker 的返回）截短标题：这批候选的完整
    标题在上下文里的检索结果中已经出现过一遍，回显时电商长标题（配件标题实测 200+ 字符）就是
    纯重复——截成短 handle + item_id 足够模型对上号，下游取数一律按 id hydrate 全量。**首次呈现**
    （item_search）不要传：模型要靠完整标题判断召回质量（如「全是配件」）。

    实测（20 条真实候选）：9,531 → 5,366 字符，-44%。
    """
    exclude = set(_MODEL_HIDDEN_FIELDS) | set(_MODEL_NOISE_FIELDS) | set(drop)
    rows: list[dict] = []
    for c in cands:
        # price_usd 在 → 原价 / 币种冗余；不在 → 它俩是唯一的价格信息，必须留。
        per_item = exclude | _REDUNDANT_WHEN_USD if c.price_usd is not None else exclude
        row = c.model_dump(exclude=per_item)  # pydantic 的 IncEx 不收 frozenset，故上面都用 set
        title = row.get("title")
        if title_chars is not None and isinstance(title, str) and len(title) > title_chars:
            row["title"] = title[:title_chars].rstrip() + "…"
        rows.append({k: v for k, v in row.items() if v is not None and v != ""})
    return rows


def register(cands: Iterable[ItemCandidate]) -> None:
    """把全量候选（含真实 url/image_url）按 item_id 登记到当前会话；无 session 作用域时静默跳过。

    仅 item_search（候选源头）调用：它产出的候选必带真实 url/image_url。下游工具阶段传回的候选
    可能已被模型紧凑序列化丢了 url，故**只在新值带 url/image 或尚无记录时写入**，避免空值覆盖真值。
    """
    k = _key()
    if k is None:
        return
    bucket = _REGISTRY.setdefault(k, {})
    for c in cands:
        if c.item_id not in bucket or c.url or c.image_url:
            bucket[c.item_id] = c


def enrich(item_id: str) -> ItemCandidate | None:
    """按 item_id 取回登记的全量候选（含 url/image_url）；无 session 作用域或无记录返回 None。"""
    k = _key()
    if k is None:
        return None
    return _REGISTRY.get(k, {}).get(item_id)


def update_fields(item_id: str, **fields: object) -> None:
    """把下游渐进算出的标量字段 patch 回登记表里的全量候选（就地更新登记项）。

    ``register`` 只在源头（item_search）写入基础候选、且守着「不拿空值覆盖真值」；而 price_compare
    /shipping_calc/item_picker 后续算出的 ``price_usd`` / ``landed_usd`` / ``pick_reason`` 等标量
    要能落回登记表——否则下游改用 :func:`hydrate` 按 id 捞回时拿到的是没算过运费/没写理由的旧快照。
    只覆盖非 ``None`` 值（``None`` 表示该阶段没算，别抹掉已有真值）；无会话作用域或该 id
    未登记时静默跳过。
    """
    k = _key()
    if k is None:
        return
    c = _REGISTRY.get(k, {}).get(item_id)
    if c is None:
        return
    for f, v in fields.items():
        if v is not None:
            setattr(c, f, v)


def register_updates(cands: Iterable[ItemCandidate]) -> None:
    """把一批候选的渐进填充字段回写登记表（price_compare/shipping_calc/item_picker 各阶段收尾调）。

    只回写渐进阶段字段（比价/到手价/精挑），基础字段与 url/image 以登记表源头值为准，不被下游
    紧凑序列化后的空值覆盖。逐条走 :func:`update_fields`。
    """
    for c in cands:
        update_fields(
            c.item_id,
            price_usd=c.price_usd,
            shipping_usd=c.shipping_usd,
            duty_usd=c.duty_usd,
            landed_usd=c.landed_usd,
            weight_kg=c.weight_kg,
            pick_reason=c.pick_reason or None,
            pref_matched=c.pref_matched,  # None=没经过 picker（update_fields 会跳过）
            slot=c.slot or None,  # 组合优选归的槽（空串=非套装/没归上，别抹掉登记表已有的章）
        )


# 本轮 item_picker 定稿的那批 picks 的 id（按推荐度排序）。与 _REGISTRY 同一套会话作用域。
#
# **为什么要单独记一份**：收尾时 shopping_summary 需要「这一批精选是哪几件」，而登记表是个累积
# 容器（装着本轮所有召回、还含上一轮读回的旧候选），从中区分不出谁入选了。以前这件事靠模型把
# picker 返回的 id 逐个抄进 shopping_summary 的入参——纯搬运、没有决策价值，却让模型每次多解码
# 一长串 id；更糟的是它抄的时候会顺手「再精选一遍」（10 件只抄 4 件），而它此刻既没有分数也没有
# 理由，纯凭印象——筛选是 item_picker 的职责，那里有权重、有排序、有封顶。
_LAST_PICKS: dict[str, list[str]] = {}


def set_last_picks(cands: Iterable[ItemCandidate]) -> None:
    """记下本轮 item_picker 定稿的 picks（收尾据此免抄 id）。"""
    k = _key()
    if k is not None:
        _LAST_PICKS[k] = [c.item_id for c in cands]


def get_last_picks() -> list[str]:
    """本轮 item_picker 定稿的 picks id（按推荐度排序）；本轮没精挑过则为空。"""
    k = _key()
    return list(_LAST_PICKS.get(k, [])) if k is not None else []


def picker_finalized() -> bool:
    """本轮 item_picker 是否定稿过（定稿为空也算）——区分「挑过没挑上」与「压根没挑」。

    get_last_picks 对两种情形都返回 []，分不出来；而这两种情形对收尾语义完全不同：
    前者「没找到」是诚实结论，后者「没找到」是拿着空通道编答案（badcase 63093a85）。
    判据是 _LAST_PICKS 的键存在性——item_picker 定稿时无条件 set_last_picks（含空列表）。
    """
    k = _key()
    return k is not None and k in _LAST_PICKS


def hydrate(item_ids: Iterable[str]) -> list[ItemCandidate]:
    """按 id 列表从会话登记表捞回全量候选（含 url + 渐进填充字段），保序、去重、跳过未命中。

    下游工具（price_compare / shipping_calc / shopping_summary）不让模型把整包候选当参数重吐——
    只收 ``item_ids``，在此按 id hydrate 回全量候选。（``item_picker`` 更进一步：连 id 都不收，直接
    吃 :func:`registry_snapshot` 的全集——本轮该精挑哪批是 planner 定的，不是模型抄出来的。）
    无会话作用域或某 id 未登记时该条跳过（调用方需对空结果兜底）。
    """
    seen: set[str] = set()
    out: list[ItemCandidate] = []
    for i in item_ids:
        if i in seen:
            continue
        seen.add(i)
        c = enrich(i)
        if c is not None:
            out.append(c)
    return out


def reset_candidates() -> None:
    """清当前会话的登记条目（run_agent 收尾调，防模块级 dict 无界增长）。

    只清**内存**。跨轮复用所需的那份已由 :func:`persist_candidates` 落到会话目录，下一轮开局由
    :func:`load_candidates` 读回——内存池仍按轮清，避免长会话把模块级 dict 撑爆。
    """
    k = _key()
    if k is not None:
        _REGISTRY.pop(k, None)
        _LAST_PICKS.pop(k, None)


# 落盘上限：只留最近登记的这么多条。候选是「上一轮搜过什么」的工作记忆，不是持久数据集；
# 一轮通常 10~30 条，60 足够覆盖「上一轮全部候选」，又不至于让文件与开局读盘无界增长。
_PERSIST_LIMIT = 60
_CANDIDATES_FILE = "candidates.json"


def persist_candidates(session_dir: Path) -> None:
    """把当前会话的候选登记表落到 ``session_dir/candidates.json``（run_agent 收尾、reset 之前调）。

    **为什么要落盘**：候选池原本随 :func:`reset_candidates` 每轮清空，于是「防水的我才要」这类
    追问在下一轮完全拿不到上一轮的候选——模型只好从 planner 重跑整条检索链（实测多花约 90 秒）。
    落盘后下一轮 :func:`load_candidates` 读回，追问轮直接 ``item_picker(新条件)`` 就地过滤（候选取自
    登记表，模型不必也无法指定是哪几件）。换品类那轮由 planner 判 ``search`` 时清掉（见 planner）。

    进程重启 / 多 worker 下也能续（同一 session_dir 即同一份文件）。写失败只记日志——候选池是
    加速用的工作记忆，丢了最多退回重新检索，不该拖垮收尾。
    """
    k = _key()
    bucket = _REGISTRY.get(k, {}) if k is not None else {}
    if not bucket:
        return
    recent = list(bucket.values())[-_PERSIST_LIMIT:]
    try:
        (session_dir / _CANDIDATES_FILE).write_text(
            json.dumps([c.model_dump() for c in recent], ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("候选池落盘失败，下一轮将退回重新检索：%s", exc)


def load_candidates(session_dir: Path) -> list[ItemCandidate]:
    """从 ``session_dir/candidates.json`` 读回上一轮候选并灌进内存登记表（run_agent 开局调）。

    返回读回的候选（保序），供 :func:`render_prior_candidates` 拼进当轮 human——模型得先「看见」
    这些 item_id 才可能复用它们。文件不存在 / 损坏一律返回空并静默走「本轮重新检索」的老路。
    """
    path = session_dir / _CANDIDATES_FILE
    if not path.exists():
        return []
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
        cands = [ItemCandidate.model_validate(r) for r in rows]
    except (OSError, ValueError, ValidationError) as exc:
        logger.warning("候选池读回失败（本轮退回重新检索）：%s", exc)
        return []
    register(cands)
    return cands


def registry_snapshot() -> list[ItemCandidate]:
    """当前会话登记表里的全部候选（含跨轮读回的）。无会话作用域返回空。

    两个消费者：
      - planner 判 ``retrieval``：它得知道「手上还有没有上一轮的候选、都是些什么」，才判得了本轮是
        收紧（reuse）、放宽（augment）还是换品类（search）。
      - item_picker 的**候选全集**：判完 search 的那轮，planner 已把旧候选清掉（见 planner），故此
        刻登记表自身就是「本轮该精挑的全部候选」——picker 直接吃它，不必让模型抄 id。
    """
    k = _key()
    if k is None:
        return []
    return list(_REGISTRY.get(k, {}).values())


def render_prior_candidates(cands: Iterable[ItemCandidate], *, limit: int = 24) -> str:
    """把读回的上轮候选渲染成当轮 human 里的 ``<prior_candidates>`` 块；无候选返回空串。

    喂的是 :func:`compact_candidates` 的同一份**模型可见投影**（无 url / 无召回分），只截前
    ``limit`` 条——这块的用途仅是让模型知道「哪些 item_id 现成可用」，不是让它在这里做精挑
    （精挑仍归 item_picker，候选体在工具内 hydrate）。
    """
    rows = compact_candidates(list(cands)[:limit])
    if not rows:
        return ""
    return json.dumps(rows, ensure_ascii=False)
