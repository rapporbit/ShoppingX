"""M7 示例：长期记忆的「跨会话记住偏好」闭环（离线，不调 LLM）。

复刻 refdocs/06 那个真实体验问题：
  会话 1：用户说「不要塑料」→ 写进 Store（持久化）
  会话 1 结束：消息历史丢弃
  会话 2（新会话）：从 Store 读出「不要塑料」→ 注入 system prompt → Agent 记得

演示三件事：
  1) 写回：把 shopping_summary 风格的新偏好（含 dislike）落库。
  2) 注入：build_preference_block 读出偏好 → 填进 system prompt 的 <user_long_term_preferences>。
  3) 语义召回：read_relevant 只挑和本次 query 最相关的几条（用户偏好多时省 token）。

用本地文件 Store + 本地确定性编码器，整段离线可跑、可复现（落到临时目录，不污染 data/）。

运行：uv run python examples/07_memory.py
"""

import asyncio
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.prompts import get_system_prompt  # noqa: E402
from app.memory.injector import build_preference_block, persist_new_preferences  # noqa: E402
from app.memory.store import LocalFileStore, PreferenceEntry  # noqa: E402
from app.recall.towers import TowerClient  # noqa: E402

USER_ID = "user-abc123"


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = LocalFileStore(root=Path(tmp), tower=TowerClient(model=None, local_dim=256))

        # ---- 会话 1：识别并沉淀偏好（主链路里这步由 shopping_summary 收尾时产出）----
        print("=== 会话 1：用户搜「旅行收纳袋，不要塑料」===")
        new_prefs = [
            SimpleNamespace(content="不接受塑料材质", category="material", polarity="dislike"),
            SimpleNamespace(content="偏好小众设计的品牌", category="style", polarity="like"),
            SimpleNamespace(content="单件预算 100-300 元", category="budget", polarity="like"),
        ]
        written = await persist_new_preferences(
            USER_ID, new_prefs, source_session="sess-1", store=store
        )
        print(f"已沉淀 {len(written)} 条偏好到长期记忆。\n")

        # ---- 会话 1 结束，消息历史丢弃。会话 2 是全新会话 ----
        print("=== 会话 2（新会话）：用户搜「洗漱包」===")
        block = await build_preference_block(USER_ID, query="洗漱包", store=store)
        print("从 Store 读出并注入的偏好块：")
        print(block, "\n")

        prompt = get_system_prompt(long_term_preferences=block)
        injected = prompt.split("<user_long_term_preferences>")[1].split(
            "</user_long_term_preferences>"
        )[0]
        print("system prompt 的 <user_long_term_preferences> 段落实际内容：")
        print(injected.strip(), "\n")

        assert "不接受塑料材质" in prompt, "偏好应已注入 system prompt"
        print("✅ 用户没重复说「不要塑料」，但 Agent（通过注入的偏好）记得。\n")

        # ---- 语义召回：偏好多时只挑相关的 ----
        print("=== read_relevant：按 query 相关性挑偏好（英文便于看出语义匹配）===")
        for key, content, cat in [
            ("dislike:material:plastic-bags", "dislikes plastic toiletry bags", "material"),
            ("like:style:niche-minimalist", "loves niche minimalist brands", "style"),
            ("history:ebay:power-bank", "bought a power bank on ebay last time", "history"),
        ]:
            await store.write(USER_ID, PreferenceEntry.create(key, content, category=cat))
        relevant = await store.read_relevant(USER_ID, "plastic water bottle", top_k=2)
        print("query「plastic water bottle」最相关的 2 条：")
        for e in relevant:
            print(f"  - {e.content}")


if __name__ == "__main__":
    asyncio.run(main())
