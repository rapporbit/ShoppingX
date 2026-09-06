"""交易域（批 1 / 7.2）：TradeAgent 的载体。

**边界说清楚**：这是一套 mock 交易域——有状态机、有幂等、有归属校验、有落库，但**没有支付、
没有物流、没有库存**（CLAUDE.md 第 1 节的不覆盖项）。`place()` 直接进 CONFIRMED，因为这里没有
「等支付回调」这个环节；库存不做，因为本仓的商品数据是离线快照，编一个库存数字只会让人误以为
它对得上真实平台。

分层照 DDD 的薄版本：`money` / `order` / `address` 是不依赖任何 I/O 的值对象与聚合，`ports`
定义仓储协议，`repository_sql` 是它的 SQLAlchemy 实现，`usecases` 编排。工具层（`app/tools/
create_order.py` 等）只做入参强转 + 调用例 + 包 ToolChunk。
"""
