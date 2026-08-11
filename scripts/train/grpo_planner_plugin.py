"""ms-swift 的 GRPO reward 插件：把 `rollout_env` 接到 swift 的训练循环上。

这一层**只做适配，不含任何评分逻辑**——分怎么打全在 `app/eval/planner_reward.py`，检索怎么跑
全在 `rollout_env.py`。理由和 S0-5 一样：reward 是决定梯度方向的东西，一旦在框架适配层里
掺私货，换个框架（verl / 自研 loop）就得重新验一遍它还准不准。

swift 侧约定（4.4.2）：`--external_plugins <本文件> --reward_funcs planner_reward`。
`__call__` 拿到的 `completions` 是本步所有 rollout 的文本，数据集其余列（`golden_json` /
`text`）以同长度 list 的形式落在 kwargs 里。

**为什么 reward 里 print 一行统计**：GRPO 最常见的哑火是「分数全一样」——组内没有区分度时
优势恒为 0，loss 看着在降其实什么都没学。把每步的均值 / 组内 σ / parse 失败率打出来，
训练日志里一眼能看出是不是在空转。
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rollout_env import Retriever, load_reward, score_batch  # noqa: E402
from swift.rewards import ORM, orms  # noqa: E402

LOG_EVERY = int(os.environ.get("PLANNER_REWARD_LOG_EVERY", "1"))


class PlannerReward(ORM):
    """R = 0.45 检索 + 0.30 字段 + 0.15 格式 + 0.10 经济性；parse 失败 -1.0。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.reward_mod = load_reward()
        self.retriever = None if os.environ.get("PLANNER_NO_RETRIEVAL") else Retriever()
        self.step = 0

    def __call__(self, completions: list[str], **kwargs) -> list[float]:
        n = len(completions)
        golden_raw = kwargs.get("golden_json") or ["{}"] * n
        texts = kwargs.get("text") or [""] * n
        golds = [json.loads(g) if isinstance(g, str) else (g or {}) for g in golden_raw]
        brs = score_batch(list(completions), golds, list(texts), self.retriever, self.reward_mod)
        rewards = [b.total for b in brs]

        self.step += 1
        if LOG_EVERY and self.step % LOG_EVERY == 0:
            fail = sum(r == self.reward_mod.PARSE_FAIL_REWARD for r in rewards) / max(n, 1)
            sigma = statistics.pstdev(rewards) if n > 1 else 0.0
            print(f"[planner_reward] step={self.step} n={n} mean={statistics.mean(rewards):.4f} "
                  f"σ={sigma:.4f} parse_fail={fail:.2%}", flush=True)
        return rewards


orms["planner_reward"] = PlannerReward
