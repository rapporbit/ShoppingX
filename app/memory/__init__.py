"""长期记忆与会话级状态。

- :mod:`app.memory.facts`：``MemoryFact``（key / value / category）+ 写入单门 ``validate_fact``
  （PII 过滤）+ tier-one 选取。长期记忆的**建模与规则**都在这里。
- :mod:`app.memory.fact_store`：``MemoryFactStore`` 六个方法 + 保留期包装，后端是
  :mod:`app.db` 的 SQLite（``memory_facts`` 表）。
- :mod:`app.memory.curator`：会话结束后独立跑的记忆管家——回合后抽取，与 ``save_memory``
  工具、偏好页 API 并列为三条写路径，三条都过 ``validate_fact`` 同一道门。
- :mod:`app.memory.store`：``HistoryEntry`` / ``FavoriteItem`` + ``PreferenceStore``
  （现在只管**行为历史与收藏**，长期记忆那腿已迁到 fact_store）。
- :mod:`app.memory.injector`：行为历史的渲染与写入。
- :mod:`app.memory.session_state`：会话级短期状态 P_t（本轮约束，随 session.json 的
  ``middle_context`` 落盘，不进长期库）。
- :mod:`app.memory.assemble`：P_t + 收藏亲和的装配（**不含长期记忆**——它只经模型上下文生效）。
- :mod:`app.memory.domains`：``PrefDomain`` 封闭枚举，现在只服务 planner 的品类判定与评测。
- :mod:`app.memory.strategies`：成功策略库（18-4）——学的是 **Agent 的打法**而非用户的取向，
  全局无 user_id，与 curator 那条路并列、零共享状态（两张表、两个 Store）。
"""

from app.memory.domains import PrefDomain
from app.memory.fact_store import MemoryFactStore, get_fact_store
from app.memory.facts import MemoryCategory, MemoryFact, validate_fact
from app.memory.store import (
    FavoriteItem,
    HistoryEntry,
    PreferenceStore,
    get_store,
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
    "PrefDomain",
    "PreferenceStore",
    "Strategy",
    "StrategyStore",
    "get_fact_store",
    "get_store",
    "get_strategy_store",
    "match_strategies",
    "render_strategy_block",
    "validate_fact",
]
