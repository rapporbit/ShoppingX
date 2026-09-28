"""一次 run 的状态总表 —— **一张表、一个 reset**。

这里原本是十几个各自为政的模块级 dict（``_SESSION_PT`` / ``_BUNDLE`` / ``_REGISTRY`` /
``_STATE`` / ``_CAPS`` …），键都是 ``session_dir``、寿命都是一次 ``run_agent``、收尾都要在
``orchestrator`` 的 ``finally`` 里挨个 reset。漏调一个 = 上一轮状态泄进下一轮，**而且不报错**
（下一轮的 planner 还没跑就先按上轮的收货国走，看起来一切正常）。

收法：表在这里、**状态类留在拥有它的模块里**。本模块不认识任何具体状态类（只当它是个可无参
构造的 ``cls``），所以不 import tools / harness / memory —— 这是不产生循环依赖的关键。

键的语义没变：**同一个 session_dir = 同一份状态**，跨多个 ``with thread_scope(同一目录)`` 块
照样读得到（模块级 dict 不像 ContextVar 随 task 结束回收，这正是当初不用 ContextVar 的理由：
同轮 batch 的工具各跑在自己的子 task 里，``set`` 不回传父 context）。

无 session 作用域（单测直调 / 离线脚本）时 :func:`run_slot` 返回 ``None``，各模块据此静默降级
——与原来「``_key()`` 为 None 就 return」的行为一致。
"""

from pathlib import Path
from typing import TypeVar

T = TypeVar("T")

# session_dir(str) → {状态类: 该类的本 run 实例}
_RUNS: dict[str, dict[type, object]] = {}


def _key() -> str | None:
    # 延迟 import：``app.api.context`` 自己的四格状态也存在本表里，模块级 import 会成环。
    from app.api.context import get_session_dir

    sd = get_session_dir()
    return str(sd) if sd is not None else None


def run_slot(cls: type[T]) -> T | None:
    """取本 run 里 ``cls`` 那一格，没有就按需 ``cls()`` 建一个并记住（故字段须全有默认值）。

    无 session 作用域返回 ``None``。**写入方**用它。
    """
    k = _key()
    if k is None:
        return None
    slots = _RUNS.setdefault(k, {})
    st = slots.get(cls)
    if st is None:
        st = cls()
        slots[cls] = st
    return st  # type: ignore[return-value]


def peek_run_slot(cls: type[T]) -> T | None:
    """只读地取那一格：**没建过就返回 ``None``，不建**。

    「这一格建没建过」本身是信号，不能靠按需新建抹平：``token_budget.run_snapshot`` 靠它区分
    「一个模型调用都没发生」（返回 None，收尾走退预扣）与「跑过但用量是 0」。纯读取方用它，
    顺带也不会让一次只读查询在表里留下空条目。
    """
    k = _key()
    if k is None:
        return None
    return _RUNS.get(k, {}).get(cls)  # type: ignore[return-value]


def clear_run_slot(cls: type) -> None:
    """只丢掉 ``cls`` 那一格 —— 各模块自己的 ``reset_xxx()`` 薄封装用。

    这些薄封装不只是给 tests 用：``planner`` 判换域时要单清槽表（旧套装与新需求无关），
    不能顺手把候选登记表和预算计数一起清了。
    """
    k = _key()
    if k is not None:
        _RUNS.get(k, {}).pop(cls, None)


def reset_run_state(session_dir: Path | str | None = None) -> None:
    """丢掉这一 run 的**全部**状态（``run_agent`` 开局 + 收尾各一次）。

    缺省清当前 session 作用域那一份；``session_dir`` 显式传参供作用域外的调用方（tests 的
    fixture 跑在任何 ``thread_scope`` 之外，``get_session_dir()`` 是 None）。
    """
    k = str(session_dir) if session_dir is not None else _key()
    if k is not None:
        _RUNS.pop(k, None)


def snapshot_run_state() -> dict[type, object]:
    """本 run 全部状态格的浅拷贝（表本身拷一份，格里的对象是同一批）——按步检查点用，调用方
    须当场序列化，见 :func:`app.agent.checkpoint.save`。无 session 作用域返回空表。
    """
    k = _key()
    return dict(_RUNS.get(k, {})) if k is not None else {}


def restore_run_state(slots: dict[type, object]) -> None:
    """用检查点里的状态格整张替换本 run 的表——worker 接管续跑时代替开局的 ``reset_run_state``。"""
    k = _key()
    if k is not None:
        _RUNS[k] = dict(slots)
