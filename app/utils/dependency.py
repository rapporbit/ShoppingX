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
