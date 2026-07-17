"""P0 子类型分流（refdocs 18-2 §3）：不是所有 P0 都能自动修。

三条分支，只有第一条自动生成规则：

- ``LEAK``    —— 泄露内部信息（endpoint / 密钥 / 内部字段名 / 绝对路径）。确定性可修：抽真实泄露
                 的字面串 → learned 脱敏规则。**但先验证串真出现在用户文本**（见 p0_fixer）。
- ``BANNED``  —— 推荐违禁 / 仿品。形态上像可修（加黑名单词），但本项目没有「商品合规黑名单」这个
                 正确落点——把违禁词塞进管**注入**的 content_filter 是类别错误。故只上报，不自动改。
- ``JUDGMENT``—— 判断类红线（超预算 / 人群冲突 / 未展示总价明细）。要模型推理，任何正则都堵不住。
                 只上报，转 prompt 腿 / 人工。

分流靠 dimension 名 + rationale 的关键词命中——粗但够用：分错顶多把一条本可自动修的 LEAK 落到
人工桶（漏修，不是误修），代价不对称地安全。
"""

from __future__ import annotations

import re
from enum import Enum


class Route(Enum):
    LEAK = "leak"  # 泄露类 → 自动抽字面串产脱敏规则（过验证闸后）
    BANNED = "banned"  # 违禁 / 仿品 → 只上报
    JUDGMENT = "judgment"  # 判断类红线 → 只上报，转 prompt / 人工


# 命中即判 LEAK：判词提到「泄露 / 暴露」某个内部标识。
_LEAK_HINTS = re.compile(
    r"泄露|暴露|internal|端点|endpoint|密钥|api[_ ]?key|token|"
    r"内部字段|绝对路径|filesystem|item_id|工具名|工具名称|内网"
)
# 命中即判 BANNED：违禁 / 仿品 / 假货。
_BANNED_HINTS = re.compile(r"仿品|仿大牌|高仿|假货|违禁|盗版|counterfeit|山寨|冒牌")


def classify(dimension: str, rationale: str) -> Route:
    """按 dimension + rationale 关键词判 P0 子类型。BANNED 优先于 LEAK：一条判词可能同时提到
    「推荐仿品」和「轨迹暴露 item_id」，前者是真问题、后者多半是 judge 把轨迹当回答的假阳性。"""
    text = f"{dimension} {rationale}"
    if _BANNED_HINTS.search(text):
        return Route.BANNED
    if _LEAK_HINTS.search(text):
        return Route.LEAK
    return Route.JUDGMENT
