"""参数覆盖层：把后台改的值写进 ``os.environ`` 并回调各模块 ``_load_params()`` 重新求值。

**为什么经 os.environ 中转**：全仓库既有的参数读取口径就是 env（``env_int("PICK_DISPLAY_CAP", 8)``
这样），覆盖层写 env 等于复用了这套现成口径——连没在注册表里、没挂 reload 的模块也能在下次读取
时拿到新值。代价是进程级全局副作用，但部署是 ``--workers 1`` 单进程，且这本就是配置语义，可接受。

**基线快照**：``apply`` 会写 ``os.environ``，写完就再也分不清某个值是 ``.env`` 配的还是后台改的。
故模块导入时（早于任何 apply）快照一次原始 env，作为「来源」判定与「恢复默认」的回退目标。
导入时机由 :func:`snapshot_baseline` 的调用点保证——server 启动即 import 本模块。
"""

from __future__ import annotations

import importlib
import logging
import os
from typing import Any

from app.config.registry import BY_KEY, PARAMS, RELOAD_MODULES, Param

logger = logging.getLogger(__name__)

# 原始 env 快照：key → 值（None 表示该 key 在 .env / 环境里压根没配）。
_BASELINE_ENV: dict[str, str | None] = {}

# 当前生效的后台覆盖：key → 规范化后的字符串值。
_OVERRIDES: dict[str, str] = {}


def snapshot_baseline() -> None:
    """快照原始 env（幂等；必须在任何 :func:`apply` 之前调用一次）。"""
    if _BASELINE_ENV:
        return
    for p in PARAMS:
        _BASELINE_ENV[p.key] = os.environ.get(p.key)


snapshot_baseline()


class ParamValidationError(ValueError):
    """参数值非法（未知 key / 类型不对 / 越界 / 空值不被允许）。"""


def normalize(key: str, raw: Any) -> str:
    """校验并把值规范化成 env 字符串形态。非法即抛 :class:`ParamValidationError`。

    这是**唯一**的校验入口——API 层不再自己判类型范围，避免两处规则漂移。
    """
    p = BY_KEY.get(key)
    if p is None:
        raise ParamValidationError(f"未知参数：{key}")

    if p.kind == "bool":
        if isinstance(raw, bool):
            return "true" if raw else "false"
        s = str(raw).strip().lower()
        if s in {"1", "true", "yes", "on"}:
            return "true"
        if s in {"0", "false", "no", "off"}:
            return "false"
        raise ParamValidationError(f"{p.label}：需要布尔值，收到 {raw!r}")

    if p.kind == "str":
        s = "" if raw is None else str(raw).strip()
        if not s and not p.allow_empty:
            raise ParamValidationError(f"{p.label}：不能为空")
        return s

    # int / float：先转数，再验范围。空字符串一律拒绝（数值参数没有「留空」语义）。
    s = str(raw).strip()
    if not s:
        raise ParamValidationError(f"{p.label}：不能为空")
    try:
        num: float | int = int(s) if p.kind == "int" else float(s)
    except ValueError:
        kind_cn = "整数" if p.kind == "int" else "数值"
        raise ParamValidationError(f"{p.label}：需要{kind_cn}，收到 {raw!r}") from None
    if p.minimum is not None and num < p.minimum:
        raise ParamValidationError(f"{p.label}：不能小于 {p.minimum}（收到 {num}）")
    if p.maximum is not None and num > p.maximum:
        raise ParamValidationError(f"{p.label}：不能大于 {p.maximum}（收到 {num}）")
    return str(num)


def _reload(modules: set[str]) -> None:
    """回调目标模块的 ``_load_params()`` 让新 env 生效。

    按 :data:`RELOAD_MODULES` 的声明序遍历（而非入参 set 的随机序）：同模块内常量可能相互依赖
    （``MAX_TOP_K`` 默认取 ``DEFAULT_TOP_K``），一次 ``_load_params()`` 内按源码顺序求值即保序。
    """
    for path in RELOAD_MODULES:
        if path not in modules:
            continue
        try:
            mod = importlib.import_module(path)
            loader = getattr(mod, "_load_params", None)
            if callable(loader):
                loader()
            else:  # 注册表写了 reload 但模块没实现 —— 改了不生效，必须吵。
                logger.error("模块 %s 缺少 _load_params()，参数改动不会生效", path)
        except Exception:
            logger.exception("重载模块参数失败：%s", path)


def apply(values: dict[str, Any]) -> dict[str, str]:
    """校验 → 写 env → 重载模块。**全量校验通过才写**，避免改一半留下半生效状态。

    返回规范化后的 key → value。热更新只对**新任务**生效：进行中的循环已读过旧值。
    """
    normalized = {k: normalize(k, v) for k, v in values.items()}
    for k, v in normalized.items():
        os.environ[k] = v
        _OVERRIDES[k] = v
    _reload({BY_KEY[k].reload for k in normalized})
    return normalized


def reset(keys: list[str] | None = None) -> None:
    """恢复默认：把 env 退回基线快照（即 ``.env`` 里的原值，或彻底删除让代码默认值生效）。

    ``keys=None`` 表示全部恢复。
    """
    targets = list(BY_KEY) if keys is None else keys
    unknown = [k for k in targets if k not in BY_KEY]
    if unknown:
        raise ParamValidationError(f"未知参数：{', '.join(unknown)}")
    for k in targets:
        _OVERRIDES.pop(k, None)
        baseline = _BASELINE_ENV.get(k)
        if baseline is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = baseline
    _reload({BY_KEY[k].reload for k in targets})


def source_of(key: str) -> str:
    """当前值的来源：``override``（后台改的）/ ``env``（.env 配的）/ ``default``（代码默认值）。"""
    if key in _OVERRIDES:
        return "override"
    return "env" if _BASELINE_ENV.get(key) is not None else "default"


def current_value(p: Param) -> Any:
    """参数**当前实际生效**的值（按 kind 还原成 JSON 友好的类型）。

    声明了 ``const`` 的直接读模块常量——那才是真正在跑的那个值。**不要图省事改成「读 env、没配
    就回退 default」**：那是重新推导，一旦某个参数的代码默认值是动态的（``MAX_TOP_K`` 取
    ``DEFAULT_TOP_K``），推导结果就会和真实生效值分道扬镳，页面显示 10 而实际跑着 15。
    """
    # 密钥永不回显。页面靠 masked_value() 那几位确认「配没配、是不是那一把」就够了。
    if p.secret:
        return ""

    if p.const:
        try:
            module = importlib.import_module(p.reload)
            return getattr(module, p.const)
        except (ImportError, AttributeError):
            logger.exception("读取 %s.%s 失败，回退按 env 推导", p.reload, p.const)

    raw = os.environ.get(p.key)
    if raw is None or raw.strip() == "":
        # 模型名类允许留空，此时如实回空串；其余回代码默认值。
        return "" if p.kind == "str" else p.default
    try:
        if p.kind == "int":
            return int(raw)
        if p.kind == "float":
            return float(raw)
        if p.kind == "bool":
            return raw.strip().lower() in {"1", "true", "yes", "on"}
    except ValueError:  # env 里被手写成非法值 —— 代码侧 env_x() 会回退默认，这里如实反映。
        return p.default
    return raw


def masked_value(p: Param) -> str:
    """密钥的展示用掩码：``sk-…a1b2``。非密钥参数返回空串。

    只够回答「配没配 / 是不是我以为的那一把」，不够拿去用。**短串一律不露内容**——一把 12 位的 key
    露前 3 后 4 就交出去了大半，而「露一半」在暴力破解面前和全露差不多。
    """
    if not p.secret:
        return ""
    raw = os.environ.get(p.key, "")
    if not raw:
        return ""
    if len(raw) < 12:
        return "已配置"
    return f"{raw[:3]}…{raw[-4:]}"


def active_overrides() -> dict[str, str]:
    """当前内存里的后台覆盖（供启动时与 DB 对账 / 调试）。"""
    return dict(_OVERRIDES)
