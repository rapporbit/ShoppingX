"""行为亲和 —— 从**收藏**这个隐式行为信号里聚合正向偏好词。零 LLM、零延迟、无幻觉。

**为什么要有它。** 在此之前，长期库里的每一条偏好都是 ``curator`` 拿 LLM 从对话文本里**猜**出来的
（「喜欢小众设计的帆布包」）。但用户嘴上说的和手上做的不是一回事——推荐系统几十年的共识是隐式行为
比自陈偏好可靠：点收藏是要付出成本的真实动作，不像随口一句「都行吧」那样含糊。这条通路把已经躺在
``favorites`` 表里、却一直只用来渲染抽屉的行为数据，接进了精挑的打分。

**三条纪律，都是从既有原则推出来的，不是新发明的：**

1. **只加分，绝不淘汰**（Mmem 授权分档）。收藏是正向证据：「你收藏过帆布包」推得出「帆布可能对
   味」，推不出「你排斥皮革」——没收藏过的东西绝大多数只是没见过。agent 侧的推断一律只能软。
2. **证据阈 ≥2**（:data:`AFFINITY_MIN_EVIDENCE`）。收藏一件不构成偏好，可能只是随手存个链接。同一
   属性被收藏两次以上才当信号。这是「频次是客观的，LLM 猜的置信度不是」的落地——本模块不产生任何
   猜出来的分数，只数数。
3. **词表限定**。属性词只从 :data:`app.utils.terms.TITLE_ATTRS` 抽（材质 / 功能 / 风格），抽不到宁可
   为空。绝不从标题里臆造属性。

**不做域隔离，且这是有意的（诚实标注失效方向）。** 收藏发生在 ``POST /api/favorites``，那里没有会话
上下文——没跑过 planner，也就没有品类域；要补域，得让前端回传 thread_id、再去翻那个会话的
``pt.json``，而存量收藏根本补不回来。而失效方向是可接受的：最坏情况是买旅行包时，因为你收藏过真皮鞋，
皮革款包上浮了几名——正向外溢的代价是排序偏一点，不是误杀。**跨域硬淘汰才是真 bug**（买旅行包时被
「买跑鞋时不要皮革」杀掉候选），而那条路在这里根本不通：本模块的产物只进加分项。真需要域了，给
``Favorite`` 加一列 ``domain`` 即可，接口不用动。
"""

from __future__ import annotations

from collections import Counter

from app.memory.store import PreferenceStore, get_store
from app.utils.env import env_int
from app.utils.terms import title_attr_tokens

# 一个属性被多少件收藏命中，才算「一贯取向」而非偶然。设 1 即退回「收藏一件就当偏好」（不建议）。
AFFINITY_MIN_EVIDENCE = env_int("AFFINITY_MIN_EVIDENCE", 2)
# 最多取几个亲和词（按证据数降序）。封顶是为了不让一个收藏了几百件的重度用户把打分项冲成一片噪声：
# 亲和词越多，商品之间的区分度反而越低（人人都命中三四个）。设 0 即关闭整条通路。
AFFINITY_MAX_TERMS = env_int("AFFINITY_MAX_TERMS", 5)


async def affinity_terms(user_id: str, store: PreferenceStore | None = None) -> list[str]:
    """聚合该用户的行为亲和词：收藏标题里出现 ≥ :data:`AFFINITY_MIN_EVIDENCE` 次的属性 token。

    返回按证据数降序的英文小写词表（可直接拿去匹英文商品标题），最多 :data:`AFFINITY_MAX_TERMS` 个。
    匿名用户 / 收藏不足 / 通路关闭时返回空——空是完全正常的稳态，不是错误。
    """
    if not user_id or AFFINITY_MAX_TERMS <= 0:
        return []

    favorites = await (store or get_store()).read_favorites(user_id)  # 读失败已在 store 内降级为空

    counter: Counter[str] = Counter()
    for fav in favorites:
        # **一件收藏只投一票**（set 去重）：标题里 "Cotton ... 100% Cotton" 写了两遍也只是一件商品、
        # 一个证据。证据数要数的是「几件收藏」，不是「这个词出现了几次」——后者数的是电商标题的
        # 关键词堆砌程度，跟用户喜欢什么没关系。
        for token in set(title_attr_tokens(fav.title)):
            counter[token] += 1

    ranked = [term for term, hits in counter.most_common() if hits >= AFFINITY_MIN_EVIDENCE]
    return ranked[:AFFINITY_MAX_TERMS]
