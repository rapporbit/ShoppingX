"""日志脱敏（refdocs 16-6 §5）：用户数据出站到日志系统之前先打码。

**为什么 Agent 的日志格外敏感。** 普通 Web 服务的日志是「谁在什么时候调了哪个接口」；Agent 的
日志里是**用户完整的购物意图**（「想买送给女朋友的生日礼物，预算 300」）和**长期偏好**（不吃某类
食物、住哪个城市）。这些进了集中式日志系统就等于进了另一个数据库，而那个数据库通常没有主库那样
的访问控制。

**接法：structlog processor，而不是「在每个 logger.info 前套一层」。** refdocs 给的是后者
（``logger.info(sanitize_for_log(data))``），那要求每个调用点都记得套——漏一处就漏一次。做成
processor 挂进 structlog 管道后，**所有**经 structlog 打出的日志自动过一遍，新代码不必知道它存在。
这与本项目「机制兜底优于人的自觉」一以贯之。

**诚实标注（两个边界）：**

1. 存量的 stdlib ``logging.getLogger`` 调用**不经过**这条管道（``app/observability/logging.py`` 的
   模块 docstring 已声明这是渐进迁移）。所以本模块只覆盖 structlog 侧；stdlib 侧的日志目前不打码。
   要全覆盖得把 stdlib 也桥接进 structlog 的 ProcessorFormatter，那是更大的改动面。
2. 脱敏是**单向有损**的。query 打码后就没法从日志复原用户原话去debug——所以 query 保留首尾、
   只糊中间，够定位「是哪一类 query 出的问题」，不够还原隐私。

user_id 用 sha256 而非 refdocs 的 md5：md5 用作标识符哈希没有安全问题，但审计工具会一律告警，
换个不惹麻烦的。截前 12 位——碰撞概率对「日志里区分用户」这个用途足够低。
"""

from __future__ import annotations

import hashlib
from typing import Any

# 需要脱敏的 key 与对应策略。key 命中即处理，不管它嵌在哪一层（只处理顶层，见 sanitize_for_log）。
_QUERY_KEYS = frozenset({"query", "user_query", "demands", "final_answer", "content"})
_ID_KEYS = frozenset({"user_id", "uid"})
_SECRET_KEYS = frozenset({"api_key", "apikey", "token", "secret", "password", "authorization"})
_PREFERENCE_KEYS = frozenset({"preferences", "long_term_preferences"})

# query 短于这个长度就整条打码——「买鞋」保留首尾等于没脱敏。
_MIN_MASKABLE = 15
_HEAD, _TAIL = 10, 5


def mask_text(text: str) -> str:
    """保留前 10 字 + 后 5 字，中间打码。短文本整条打码（首尾会把它暴露光）。"""
    if len(text) <= _MIN_MASKABLE:
        return "***"
    return f"{text[:_HEAD]}***{text[-_TAIL:]}"


def hash_id(value: str) -> str:
    """把 user_id 哈希成一个稳定的短标识——日志里仍能按用户聚合，但复原不出是谁。"""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def sanitize_for_log(data: dict[str, Any]) -> dict[str, Any]:
    """返回脱敏后的**副本**（不原地改——调用方的 dict 常常还要拿去做业务）。

    只处理顶层 key：日志事件本就是扁平的 kv，为嵌套结构做深度遍历会在热路径上引入不必要的开销，
    也容易把大对象整个走一遍。真有嵌套敏感数据，脱敏责任在打日志的人。
    """
    result = dict(data)
    for key, value in result.items():
        lower = key.lower()
        if lower in _SECRET_KEYS:
            result[key] = "******"
        elif lower in _ID_KEYS and isinstance(value, str) and value:
            result[key] = hash_id(value)
        elif lower in _QUERY_KEYS and isinstance(value, str) and value:
            result[key] = mask_text(value)
        elif lower in _PREFERENCE_KEYS and isinstance(value, list):
            result[key] = [f"偏好:*** ({len(value)} 条)"]
    return result


def sanitize_log_processor(
    _logger: Any, _method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """structlog processor：在渲染前把事件字典里的敏感字段打码。

    放在管道里 ``merge_contextvars`` **之后**——contextvars 注入的 ``user_id`` 也要一起哈希，
    否则每条日志的上下文字段里都躺着明文 user_id，脱敏形同虚设。
    """
    return sanitize_for_log(event_dict)
