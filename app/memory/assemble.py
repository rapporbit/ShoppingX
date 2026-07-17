"""记忆装配：本轮记忆如何影响检索与精挑的**唯一**出口。

改造前，记忆的消费散在四处——``item_search`` 自己读 like 词、``item_picker`` 自己读两路 dislike、
主 loop 又另拿一份文本注入 prompt。四处各自读 Store、各自判域，于是「模型看到的」和「机制执行的」
是**两套东西**：注入 prompt 的偏好块跨域全量（拼进 human 时 planner 还没跑、域还不存在），而机制
侧走的是域内子集。模型看见一条本轮不该生效的偏好，很自觉地把它转述进 ``item_picker`` 的自由文本
参数，硬淘汰就这么绕过域闸生效了——真实 bug，买旅行包时差点被「买跑鞋时不要皮革」误杀。

收敛成单一装配点后，这个不一致**在结构上不可能再发生**：喂给模型的文本（:func:`render`）和机制
执行的词表（:class:`MemoryBundle`）由同一份装配结果渲染。

**硬 / 软的分界是「谁授权的」，不是 LLM 的置信度**（Mmem 原则：LLM 只做识别，用户做授权，机制做
执行）。这条规则此前散在三处（persist 的写入闸、injector 的两个读取函数、P_t 的 terms 提取），
现在只写在这里一处：

  - 硬淘汰（``exclude``）：用户在偏好页面亲手勾的黑名单（``is_blocking``）+ 本轮亲口说的「不要 X」
  - 软减分（``penalty``）：curator 从对话里**推断**的一贯取向 + 本轮的弱表达（「尽量别太花哨」）
  - 正向（``search_terms`` / ``must``）：一律不做二值淘汰——数据没有可靠的材质 / 风格字段，
    keep-only 会误杀一大片。
  - 行为亲和（``affinity``）：从**收藏**聚合出的弱正向证据（零 LLM，见
    :mod:`app.memory.affinity`），只进 item_picker 的弱加分。它是唯一一路「用户做了什么」而非
    「用户说了什么」的记忆——也正因为是推断，档位压到最低：不淘汰、不进检索词、冲突时让位于显式表达。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.api.context import get_session_domains, get_session_pt
from app.memory.affinity import affinity_terms
from app.memory.injector import _in_scope, _terms_of  # 域闸与原子词提取：装配的两块基石
from app.memory.store import PreferenceStore, get_store
from app.utils.terms import normalize_terms, term_hits


class MemoryBundle(BaseModel):
    """本轮记忆的装配结果。下游工具**只读这个**，不再各自读 Store。"""

    search_terms: list[str] = Field(default_factory=list)  # → item_search 拼进检索词
    exclude: list[str] = Field(default_factory=list)  # → item_picker 命中即淘汰
    penalty: list[str] = Field(default_factory=list)  # → item_picker 命中减分
    must: list[str] = Field(default_factory=list)  # → item_picker 强加分（不淘汰）
    # 行为亲和（收藏聚合，见 app.memory.affinity）→ item_picker 弱加分。**不进 search_terms**：
    # 检索词决定「捞哪一池」，把一个用户从没提过的词（canvas）塞进 query 会把整池带偏，代价是
    # 全局的；而它作为打分项只在池内微调排序，代价是局部的。行为是弱证据，只配后者。
    affinity: list[str] = Field(default_factory=list)
    budget_usd: float | None = None

    # 仅**长期库**来源的那部分，供 report_memory_applied 上报（P_t 是本轮约束，不算「记忆生效」）。
    # 记忆最危险的失败是静默：一条偏好误杀了一批商品，用户看不到提示、归因不到记忆头上。
    memory_exclude: list[str] = Field(default_factory=list)
    memory_penalty: list[str] = Field(default_factory=list)


# **槽位型**偏好：值是客观事实、不是商品的属性，因此不该参与任何「匹标题」的通路。
#
# 唯一的成员是 location（「常用收货地：CN」，keywords=["CN"]）。它被 planner 的收货国四层解析
# 专门消费（resolve_dest_country_layered 第 3 层），而 bundle 这边只要碰它就会出事：like →
# search_terms → 检索词变成「旅行三件套 cn」，把一个国家码塞进商品语义检索，纯噪声。
#
# 判据是「这个词会出现在商品标题里吗」：材质 / 风格 / 品牌会，收货地不会。
_SLOT_CATEGORIES = frozenset({"location"})


def _merge(*groups: list[str]) -> list[str]:
    """小写、保序去重地并列表——词表是拿去和商品标题做子串匹配的。"""
    out: list[str] = []
    seen: set[str] = set()
    for g in groups:
        for t in g:
            low = t.strip().lower()
            if low and low not in seen:
                seen.add(low)
                out.append(low)
    return out


async def assemble(user_id: str, store: PreferenceStore | None = None) -> MemoryBundle:
    """装配本轮记忆：长期库（**域内**）+ 会话级 P_t（本轮）。

    域过滤只在这里做一次（``_in_scope``，**fail-closed**）：``domains`` 为空（planner 没跑 /
    闲聊轮 / 单测直调）时**只有 global 域的偏好生效**。空域意味着「不知道本轮在买什么」，正确
    的失效方向是保守——宁可让偏好本轮不生效（用户再说一遍即可），也不能让跨域偏好在一个未知
    品类的轮次里静默杀商品（论证见 :func:`app.memory.injector._in_scope`）。
    """
    pt = get_session_pt()
    lt_search: list[str] = []
    lt_exclude: list[str] = []
    lt_penalty: list[str] = []
    lt_dislike_any_domain: list[str] = []  # 全部负向（**不过域闸**），只用来压制行为亲和，见下

    if user_id:
        st = store or get_store()
        domains = get_session_domains()
        for e in await st.read(user_id):
            if e.category in _SLOT_CATEGORIES:
                continue  # 槽位型偏好不参与「匹标题」的任何一路，见 _SLOT_CATEGORIES
            if e.polarity == "dislike":
                # **抑制集合不过域闸**（收集在 _in_scope 之前，是刻意的）。域闸是给「杀伤力」设的：
                # 一条「买跑鞋时不要皮革」不该在买沙发时淘汰商品。但拿它来**压制我们自己推断出来的
                # 弱加分**，性质完全不同——失效方向是「少加一点分」，安全。
                #
                # 不这么做就有个真洞：域判不出的轮次里（闲聊、planner 没跑），用户明说的 dislike 被
                # 域闸挡在外面、进不了 penalty，而收藏聚合出的 affinity 不受域闸约束照常加分——净效果
                # 是「他说不要皮革，系统反而把皮革顶上去了」。弱证据压过强证据，正好反了。
                lt_dislike_any_domain.extend(_terms_of(e))
            if not _in_scope(e, domains):
                continue
            if e.polarity == "like":
                # 只取 keywords（原子词）：content 是整句（「喜欢小众设计的帆布包」），拼进检索词
                # 会把 query 语义带偏。没给 keywords 的 like 条目不参与检索词（安全方向）。
                lt_search.extend(e.keywords)
            elif e.is_blocking:  # 用户亲手勾的「绝不推荐」→ 唯一能让长期偏好硬淘汰的授权
                lt_exclude.extend(_terms_of(e))
            else:  # curator 推断的 dislike → 只减分。误判的代价是排序偏一点，不是再也搜不到
                lt_penalty.extend(_terms_of(e))

    pt_exclude = pt.dislike_terms() if pt else []
    pt_penalty = pt.soft_dislike_terms() if pt else []
    pt_must = pt.like_terms() if pt else []

    exclude = _merge(lt_exclude, pt_exclude)
    penalty = _merge(lt_penalty, pt_penalty)

    # 行为亲和：收藏聚合出的弱正向证据（零 LLM，见 app.memory.affinity）。
    #
    # **说过的话压过做过的事**：用户排斥过的词，即便在收藏里高频出现也一律不进亲和——否则会出现
    # 「他嘴上说不要皮革，可他收藏过三双皮鞋」这种一边减分一边加分的自相矛盾，净效果取决于两个权重
    # 谁大，不可解释。显式是强证据、行为是弱证据，冲突时强的赢（口味会变：过去爱皮革，这次不要了）。
    #
    # **压制判定必须归一 + 走 term_hits，不能拿原始词做 `in set()` 的精确相等**——初版就是这么写的，
    # 在最常见的真实形态下静默失效，两处叠加：
    #   ① curator 从中文对话抽的原子词原样落库（keywords=['皮革']），而亲和词是英文标题 token
    #      ('leather')。不归一，两边永远对不上（item_picker 对 mem.penalty 也是**归一后**才拿去匹
    #      标题的，压制这一路却漏了同一道工序）。
    #   ② 归一后 blocked 里是 'leather'，而亲和 token 可能是更长的变体 'genuine leather'——精确相等
    #      照样穿过去。得用「这条排斥词命中了这个亲和词吗」来判，即 term_hits。
    # 失效的后果是把这条纪律整个架空：用户明说「不要皮革」，系统反而给真皮款加分。
    blocked = normalize_terms(_merge(exclude, penalty, lt_dislike_any_domain))
    affinity = [
        t for t in await affinity_terms(user_id, store) if not any(term_hits(b, t) for b in blocked)
    ]

    return MemoryBundle(
        search_terms=_merge(lt_search),
        exclude=exclude,
        penalty=penalty,
        must=_merge(pt_must),
        affinity=_merge(affinity),
        budget_usd=pt.budget_usd if pt else None,
        memory_exclude=_merge(lt_exclude),
        memory_penalty=_merge(lt_penalty),
    )


async def blocking_exclude_terms(user_id: str, store: PreferenceStore | None = None) -> list[str]:
    """用户全部 blocking 黑名单的匹配词——**不做域过滤**，已归一成可匹英文标题的小写词表。

    给**无会话上下文**的展示面用（目前是 ``/api/similar`` 搜同款）：那里没有 planner、没有品类域，
    ``_in_scope`` 的域闸无从谈起；而 blocking 的语义是「这件商品用户永远不该看到」，展示通路不该
    豁免。只收 ``is_blocking``（用户在偏好页面亲手勾的，条目少且明确）；不带 curator 推断的软
    dislike——减分需要排序语境，同款列表没有。失效方向可接受：跨域黑名单在同款列表里多挡一件，
    近邻会补位；反过来放行一件拉黑商品，用户会直接质疑「绝不推荐」这个承诺。
    """
    if not user_id:
        return []
    st = store or get_store()
    terms: list[str] = []
    for e in await st.read(user_id):
        if e.is_blocking:
            terms.extend(_terms_of(e))
    return normalize_terms(_merge(terms))
