"""M23 S0-2：dev / test 三票分歧的**人工裁决**（train 的分歧不裁，见下）。

为什么只裁 dev / test：它们是尺子（dev 量线上分布）和靶子（test 按 bad case 族分层）。
尺子不准，S2/S3 做得再漂亮也验收不了。train 的 124 条分歧量大且边际收益低，训练时按
``status`` 降权 / 排除即可——那点样本换不来几小时人工。

**这份文件必须入库**：LLM 投票可以重跑（跑一次一个样），人工裁决不能。产物 jsonl 在
``data/``（gitignore）靠脚本复现，而「复现」要复现得出同一份裁决，就得让裁决进版本库。
本项目已有先例：评测种子集也是硬编码在 `scripts/eval/build_eval_queries.py` 里的。

**裁决口径**（按线上 planner 口径，不是按标注者的习惯）：
1. domains 一致、只是 category 粒度不同（「剪刀 / 手锯 / 工具」）→ 取**最具体且与句意相符**的
   那个。粒度粗不算错，但 category 要当检索锚用，具体的更有信息量。
2. **句意与库自带类目冲突时以句意为准**——golden 是给 planner 的，planner 看的是用户的话。
   实测多条合成句已从库类目漂开：pq_00229 库类目 automotive paint，句子却是「给小孩买、
   喷玩具车用」→ 判儿童玩具漆；pq_00557 库类目 beauty & personal care，句子是「运动后用、
   不要玻璃瓶」→ 判运动水壶。
3. **句中压根没有品类词的，弃权（None）而不是编一个**。adv_00159「必须铁质，承重300斤」、
   adv_00225「必须环保可降解，贵点不怕」——三票分别猜「椅子」「环保袋」，那是模型在补全，
   不是判定。golden 里编一个品类，等于拿标注者的想象去发 reward。
   ``None`` = 这一维不计分；空串 = 确实没有品类（闲聊）。两者语义不同，别混。

用法：被 ``build_planner_golden.py`` 自动加载（正常跑与 ``--report-only`` 都会应用）。
"""

from __future__ import annotations

# id → 要覆盖的字段。只写要改的那几个键，没写的字段保持投票结果。
# category / domains 写 None = 弃权（reward 跳过这一维）。
OVERRIDES: dict[str, dict] = {
    # ── dev（真实锚）─────────────────────────────────────────────────────────
    # 跨品类套装：三票分别只看到了书包或文具的一半。线上口径「旅行三件套 = bags + apparel」
    # 同理，一套入学装备至少覆盖箱包 + 办公文具。
    "anchor_0014#t0": {"category": "入学装备", "domains": ["bags", "office"]},
    # 「随便给我推荐点东西吧」——有购物意图但无任何品类线索，域落保守档 other。
    "anchor_0053#t0": {"category": "", "domains": ["other"]},
    # 「照这张参考图找类似的」——品类在图里，文本判不出（线上由 image_understand 先解析）。
    "anchor_0068#t0": {"category": "", "domains": ["other"]},
    # ── test（对抗族 + 合成）─────────────────────────────────────────────────
    # 句中无品类词，只有规格与材质约束 → 弃权，不替模型补全。
    "adv_00159#t0": {"category": None, "domains": None},
    "adv_00159#t1": {"category": None, "domains": None},
    "adv_00225#t0": {"category": None, "domains": None},
    "adv_00225#t1": {"category": None, "domains": None},
    # 平板维修配件：域按商品本体归 computers（三票里 2 票如此），品类取最具体的写法。
    "pq_00126#t0": {"category": "平板电脑维修配件", "domains": ["computers"]},
    # 「看360度视频」→ VR 头显。「消费电子」太泛，当检索锚等于没给。
    "pq_00204#t0": {"category": "VR 眼镜", "domains": ["electronics"]},
    "pq_00204#t1": {"category": "VR 眼镜", "domains": ["electronics"]},
    # 库类目是 automotive paint，但句子是「给小孩买、喷玩具车用」——以句意为准。
    "pq_00229#t0": {"category": "儿童玩具漆", "domains": ["toys_baby"]},
    # 「剪绳子用」→ 剪刀。第二票的「手锯」是错判（锯不剪绳），库类目 cutting tools 佐证。
    "pq_00240#t0": {"category": "剪刀", "domains": ["tools"]},
    "pq_00240#t1": {"category": "剪刀", "domains": ["tools"]},
    "pq_00240#t2": {"category": "剪刀", "domains": ["tools"]},
    # 「露营用、12V」只说了场景与电压，具体买什么判不出 → 品类弃权，域保留 electronics。
    "pq_00284#t0": {"category": None, "domains": ["electronics"]},
    "pq_00284#t1": {"category": None, "domains": ["electronics"]},
    # 「再换个品类试试」——明说要换，但没说换成什么。品类此刻确实不存在。
    "pq_00378#t2": {"category": None, "domains": None},
    # 「洗牙用的」→ 冲牙器（三票都滑向牙刷）。口腔个护归 beauty，与三票一致。
    "pq_00391#t0": {"category": "冲牙器", "domains": ["beauty"]},
    "pq_00391#t1": {"category": "冲牙器", "domains": ["beauty"]},
    "pq_00396#t0": {"category": "鱼饲料", "domains": ["pet"]},
    # 「运动后用、不要玻璃瓶」→ 运动水壶。库类目 beauty & personal care 与句意无关，不采信。
    "pq_00557#t0": {"category": "运动水壶", "domains": ["sports"]},
    "pq_00718#t1": {"category": None, "domains": None},  # 「换别的品类看看」同 pq_00378
    # 云游戏是**服务**不是实体商品，域清单里没有对得上的 → other（保守档，只在本轮生效）。
    "pq_00725#t0": {"category": "云游戏服务", "domains": ["other"]},
    "pq_00725#t1": {"category": "云游戏服务", "domains": ["other"]},
    # 「坐长途车时玩、续航8小时」+ 库类目 PSP → 掌上游戏机（第三票的「平板电脑」偏了）。
    "pq_00752#t0": {"category": "掌上游戏机", "domains": ["electronics"]},
    "pq_00752#t1": {"category": "掌上游戏机", "domains": ["electronics"]},
    # 「老旧系统」句意过泛，品类弃权；域取两票一致的 computers。
    "pq_00836#t0": {"category": None, "domains": ["computers"]},
    "pq_00989#t0": {"category": "儿童螺丝玩具", "domains": ["toys_baby"]},
    "pq_00989#t1": {"category": "儿童螺丝玩具", "domains": ["toys_baby"]},
}
