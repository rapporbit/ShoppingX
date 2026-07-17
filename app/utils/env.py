"""环境变量读取小工具：缺失 / 非法时回退默认值，绝不抛。

多个模块都要从 ``.env`` 读「整型上限 / 布尔开关」这类配置（压缩参数、loop 上限……），
各写一份解析既重复又会悄悄漂移。统一收在这里：一处修（如未来要 trim、记日志、限下界），
处处生效。``llm.py`` 的 ``_env_float`` 因早于本模块、且只它用，暂未并入（保持改动面最小）。
"""

import os


def env_int(key: str, default: int) -> int:
    """读取整型环境变量，缺失 / 空 / 非法时回退默认值。"""
    raw = os.environ.get(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def env_bool(key: str, default: bool) -> bool:
    """读取布尔型环境变量（1/true/yes/on 视为真），缺失 / 空时回退默认值。"""
    raw = os.environ.get(key)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_float(key: str, default: float) -> float:
    """读取浮点型环境变量，缺失 / 空 / 非法时回退默认值。"""
    raw = os.environ.get(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def env_str(key: str, default: str = "") -> str:
    """读取字符串环境变量并 trim，缺失 / 全空白时回退默认值。

    trim 是重点：``ADMIN_USERNAMES=" zjl"`` 这种手写 .env 时极易混进的前导空格，不去掉就会让
    白名单比对静默失配——配了管理员却进不去，且毫无报错。
    """
    raw = os.environ.get(key)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip()
