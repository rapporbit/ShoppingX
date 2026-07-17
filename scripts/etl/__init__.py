"""品类知识库离线 ETL：标准化 → 聚合抽卡 → 入库门禁（refdocs 13-1 §2）。

四路原始数据在真实工程里形态各异（销售榜 / 属性聚合 / 成交价分位 …），中间靠一段离线
ETL 收敛成统一的 :class:`~app.recall.category_kb.CategoryCard`。本项目用 ``data/rag`` 的真实
Amazon 商品数据，按品类聚合出三类卡片，是这条管线的**可复现最小落地**。
"""
