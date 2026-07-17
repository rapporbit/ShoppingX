"""模型吐出的结构不合形态时的共享容错：字符串化 list（工具入参）与显式 null（结构化输出）。

**为什么存在**：换模型 = 换 tool-call 保真度。qwen3.5-flash（gcjp 会话 d0724e95，2026-07-16）
把 ``list[str]`` 参数编码成 JSON **字符串**（``'["塑料","plastic"]'``），Pydantic 校验直接拒，
item_picker 同参连挂 4 次——校验错误文案对这类模型没有指导性（它「认为」自己传的就是列表），
重试到被强制收尾。字符串形如 JSON 数组是**无歧义**的：loads 出来就是模型想传的东西，机制层
解回比指望模型自纠可靠（同 planner 结构化输出 raw_decode 的教训，5a27c27）。

**只解无歧义形态**：str 且 strip 后以 ``[`` 开头 ``]`` 结尾且 loads 出 list 才转；裸字符串
（``"塑料"``）不猜——包成单元素对「空格分隔多 id」这类输入就是错的，交回 Pydantic 照常报错，
由 middleware 的 error 路径 + LoopDetector 打转提示兜底。元素类型仍归 Pydantic 管
（loads 出 ``[1, 2]`` 照常被 ``list[str]`` 拒）。

给模型的 JSON Schema 不受影响（仍是 ``array``）：BeforeValidator 只活在校验期，不进
``model_json_schema()``——不会反向诱导模型传字符串。
"""

from __future__ import annotations

import json
from typing import Annotated

from pydantic import BeforeValidator


def coerce_stringified_list(v: object) -> object:
    """str 形如 JSON 数组 → loads 成 list；其余原样交回后续校验。"""
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = json.loads(s)
            except ValueError:
                return v
            if isinstance(parsed, list):
                return parsed
    return v


# 模型可见的 list[str] 工具参数一律用它声明：
#   可选参数 → ``xs: StrListArg | None = None``；必填 → ``xs: StrListArg``。
StrListArg = Annotated[list[str], BeforeValidator(coerce_stringified_list)]


def drop_none_values(data: object) -> object:
    """结构化输出 schema 的「显式 null 归一为缺席」：值为 None 的键整个丢掉，让默认值接管。

    同族第三形态（gcjp 会话 cdee1d6d，2026-07-17）：deepseek-v4-flash（关思考 +
    function_calling）把 PlanOutput 的 5 个 list 字段吐成显式 ``null``——Pydantic 的
    ``default_factory`` 只兜「字段缺席」，不兜「显式 null」，planner 连挂 2 次。对模型而言
    「没有这项」写成 null 和不写是同一个意思，机制层归一比指望校验错误文案教会它可靠。

    用法（挂在 ``with_structured_output`` 的**顶层** schema 上，一行）::

        _null_is_absent = model_validator(mode="before")(staticmethod(drop_none_values))

    只处理顶层字段：嵌套对象里的 null 未实测出现过，不预防（YAGNI）。丢键对
    ``Optional[...] = None`` 字段是语义无损的（丢了以后默认值还是 None）。
    """
    if isinstance(data, dict):
        return {k: v for k, v in data.items() if v is not None}
    return data
