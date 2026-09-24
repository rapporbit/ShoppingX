"""长期记忆与会话级状态。

- :mod:`app.memory.facts`：``MemoryFact``（key / value / category）+ 写入单门 ``validate_fact``
  （PII 过滤）+ tier-one 选取。长期记忆的**建模与规则**都在这里。
- :mod:`app.memory.fact_store`：``MemoryFactStore`` 六个方法 + 保留期包装，后端是
  :mod:`app.db` 的 SQLite（``memory_facts`` 表）。
- :mod:`app.memory.curator`：会话结束后独立跑的记忆管家——回合后抽取，与 ``save_memory``
  工具、偏好页 API 并列为三条写路径，三条都过 ``validate_fact`` 同一道门。
- :mod:`app.memory.store`：``HistoryEntry`` / ``FavoriteItem`` + ``UserDataStore``
  （用户级**行为数据**：行为历史与收藏。长期记忆不在这儿，旧的 ``preferences`` 表已随迁移
  ``0016_drop_preferences`` 删除）。
- :mod:`app.memory.injector`：行为历史的渲染与写入。
- :mod:`app.memory.session_state`：本轮生效约束 P_t（planner 每轮从前几轮原话整体重算，
  不落盘、不进长期库）。
- :mod:`app.memory.assemble`：P_t + 收藏亲和的装配（**不含长期记忆**——它只经模型上下文生效）。
- :mod:`app.memory.domains`：品类域词表，运行时只服务 item_picker 品类门锚核验；
  枚举冻结给 planner 训练腿。
- :mod:`app.memory.strategies`：成功策略库（18-4）——学的是 **Agent 的打法**而非用户的取向，
  全局无 user_id，与 curator 那条路并列、零共享状态（两张表、两个 Store）。
"""

from app.memory.fact_store import MemoryFactStore, get_fact_store
from app.memory.facts import MemoryCategory, MemoryFact, validate_fact
from app.memory.store import (
    FavoriteItem,
    HistoryEntry,
    UserDataStore,
    get_user_data_store,
)
from app.memory.strategies import (
    Strategy,
    StrategyStore,
    get_strategy_store,
    match_strategies,
    render_strategy_block,
)

__all__ = [
    "FavoriteItem",
    "HistoryEntry",
    "MemoryCategory",
    "MemoryFact",
    "MemoryFactStore",
    "Strategy",
    "StrategyStore",
    "UserDataStore",
    "get_fact_store",
    "get_strategy_store",
    "get_user_data_store",
    "match_strategies",
    "render_strategy_block",
    "validate_fact",
]
