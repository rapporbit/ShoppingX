"""pre_tool_call 收尾资格底线：shopping_summary 的三道安全底线（空候选 / 未规划 / 未精挑）。

**阶段白名单禁令已撤**（harness 重构第三段）：本 Hook 原先按 ``PHASE_TOOLS`` 白名单拒绝一切
「当前阶段不可用」的工具，但阶段表守护的是效率不是安全，而它依据的上游判定（planner 意图、
阶段推进信号）是假设不是承诺——2026-07-14 线上死锁即 planner 把换品类误判成 reuse、阶段闸
拦死 item_search 27 轮直到用户取消。热修（连拒 2 次逃生）只是给错墙开门；本段把墙拆掉：

- **效率约束改走预算**：reuse 轮的检索由 ``tool_gates.charge_retrieval`` 的小预算兜
  （``REUSE_RETRIEVAL_BUDGET``，≥1 永不为 0）——第一次补搜天然放行，不用攒拒绝换逃生。
- **阶段机降级为遥测 + prompt 提示**：``phase_machine`` 只记「走到哪一步」（Langfuse /
  transition_notice 的指路文案靠它），不再执法。
- **本 Hook 只留安全底线**：没有候选、或本轮压根没规划/精挑过，就不许 shopping_summary
  交卷。这两条是精确事实判定（登记表空 / 阶段还在 PLANNING），属安全闸，永远硬拒。

底线 2（PLANNING 拒收尾）在候选池跨轮存活后是必需的：续聊轮一开局候选就被读回登记表，只看
「有没有货」的话，模型会跳过 planner 与 item_picker 直接交卷，本轮的硬约束（「只要防水的」）
压根没被机制执行过。

底线 3（本轮未精挑拒收尾）补的是底线 2 的旁路：reuse 轮 planner 一跑阶段就被直跳 COMPARING，
底线 2 随即失效——而 shopping_summary 的清单**只认本轮 item_picker 定稿**（get_last_picks，
轮内内存态），模型跳过精挑直接收尾时它拿到的是空清单，收尾 LLM 会照「没找到」模板输出与
候选池自相矛盾的答案（badcase 63093a85 背包 / q05 定点查价 / gcjp 相机与英国文学，四起实锤，
全是模型自主跳过、无一来自强制通路）。判据是精确事实（called_tools，工具成功执行后才记录、
每轮新建），逃生动作唯一且永远可行（底线 1 保证登记表非空，item_picker 对任何候选池都能跑）
——这与 2026-07-14 被拆掉的阶段白名单（判据是 planner 的意图**假设**，模型可能无法满足）有
本质区别。漂移检测的强制收尾通路（推 CONCLUDING，见 drift_detector）会被本底线多拦一轮：
被拒 → 补跑 picker → 再收尾，不死锁；真卡死有 watchdog 硬停兜底。

仅主 loop（depth 0）生效。子 Agent 由深度闸管权限（shopping_summary 本就不许子调）。
"""

from __future__ import annotations

import logging
from typing import Any

from app.agent.fork_guard import current_fork_depth
from app.harness.middleware import HookRejectSignal, harness_hook
from app.harness.phase_machine import Phase, get_phase_machine
from app.harness.signals import candidate_count

logger = logging.getLogger("shoppingx.harness.phase_check")


@harness_hook("pre_tool_call", name="phase_check", priority=20)
async def check_phase_permission(context: dict[str, Any]) -> dict[str, Any] | None:
    """shopping_summary 收尾资格底线。仅 depth 0 生效，其余工具一律放行。"""
    if current_fork_depth() >= 1:
        return None

    machine = get_phase_machine()
    if machine is None:
        return None

    if context.get("tool_name", "") != "shopping_summary":
        return None

    if not candidate_count():
        raise HookRejectSignal(
            "当前还没有任何候选商品，无法生成购物清单。"
            "请先检索到候选再调 shopping_summary；若本轮并非购物意图，请改调 chat_fallback。"
        )
    if machine.phase is Phase.PLANNING:
        raise HookRejectSignal(
            "还没有为本轮做过精挑，不能直接出清单。手上的候选是上一轮按上一轮条件搜的，"
            "请先调 planner 判断本轮意图，再用 item_picker 按本轮条件精挑，然后收尾。"
        )
    if "item_picker" not in context.get("called_tools", set()):
        raise HookRejectSignal(
            "本轮还没精挑过，不能直接出清单——收尾清单只认本轮 item_picker 的定稿，"
            "跳过精挑会产出与候选自相矛盾的空清单。请先调 item_picker（对已入池候选"
            "就地打分过滤，开销很小），再调 shopping_summary 收尾。"
        )
    return None
