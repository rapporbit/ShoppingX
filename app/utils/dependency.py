"""主检索依赖不可用时的统一异常。

**为什么要单独一个类型。** 工具壳（:mod:`app.tools._shell`）把实现抛出的任何异常都转成同一种
``state=ERROR`` + ``[error] ...`` 文本，于是模型看到的「参数写错了」和「Qdrant 连不上」长得一模
一样，它按同一套反应处理——换个检索词再来一次。参数错重试有意义；依赖挂了重试只是把三次超时
叠成一次超长等待，最后照样收不了尾，用户等了一分钟还是得到一句「没找到」。

分级要的信息只有一条：**这次失败重试有没有用**。``DependencyDown`` 就是「没用」那一档。它一路
带到 :mod:`app.harness.adapter` 的 ERROR 分支，在那里给模型贴一句「别重试，直接如实告知用户」。

只有**主检索通路**（Qdrant 召回、query embedding）用它：这两个挂了 ``item_search`` 必然空手而归。
reranker / web_search / category_insight 不用——它们各自有本地兜底或静默降级，失败不影响交付。
"""

from __future__ import annotations

from contextvars import ContextVar


class DependencyDown(RuntimeError):
    """某个外部依赖当前不可用（连不上 / 超时 / 熔断中），重试无意义。

    Args:
        dependency: 依赖名，用于文案与可观测（``qdrant`` / ``embedding`` / ``opensearch``）。
        detail: 补充说明，会拼进消息给模型看，所以写「做什么失败了」而不是堆栈术语。
    """

    def __init__(self, dependency: str, detail: str = "") -> None:
        self.dependency = dependency
        self.detail = detail
        msg = f"{dependency} 暂时不可用" + (f"（{detail}）" if detail else "")
        super().__init__(msg)


#: 工具结果 metadata 里的错误分级键。``_shell`` 写、``adapter`` 读，两边共用这一个常量而不是
#: 各写各的字面量——分级一旦漏读就是静默失效（提示不贴，模型照旧重试三次）。
ERROR_CODE_KEY = "code"
DEPENDENCY_DOWN_CODE = "dependency_down"

#: 本轮 run 里是否撞上过依赖不可用。**为什么需要这个旗子**：``DependencyDown`` 在工具壳里就被
#: 吞成 ``state=ERROR`` 了（有意为之——抛出去会掐死整条 loop），所以收尾处看不到它。没有旗子的
#: 话，一次 Qdrant 维护会让这批 run 全记成 ``success``（Agent 确实如实收了尾）或 ``failed``，
#: 两种都不对：它是**设计内的降级**，不该进 SLO 成功率的分母。
#:
#: 存的是**可变盒子**而非裸 bool：工具跑在框架 ``create_task`` 出来的子 context 里，在那里
#: ``ContextVar.set`` 回不到父 context，收尾处读到的会永远是 False（静默失效，且测试里单调
#: 工具一样测不出来）。盒子的引用是继承下去的，往里写才跨得回来。
_dependency_down: ContextVar[dict[str, bool] | None] = ContextVar(
    "shoppingx_dependency_down", default=None
)


def mark_dependency_down() -> None:
    """标记本轮撞上过依赖不可用（由工具壳在捕获 :class:`DependencyDown` 时调用）。

    没开盒子（离线脚本 / 单测直调工具）时是空操作。
    """
    box = _dependency_down.get()
    if box is not None:
        box["seen"] = True


def dependency_down_seen() -> bool:
    """本轮是否撞上过依赖不可用。没开盒子时按「没撞上」算。"""
    box = _dependency_down.get()
    return bool(box and box.get("seen"))


def reset_dependency_down() -> None:
    """开一个新盒子（``run_agent`` 每轮开局调）。**必须在派生任何工具协程之前**。"""
    _dependency_down.set({"seen": False})
