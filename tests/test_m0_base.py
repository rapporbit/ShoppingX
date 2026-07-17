"""M0 工程底座冒烟测试：上下文 / 路径 / 提示词 / LLM 工厂可用。"""

from pathlib import Path

import pytest

from app.agent.prompts import get_system_prompt
from app.api.context import get_session_dir, get_thread_id
from app.utils.path_utils import OUTPUT_ROOT, safe_join
from app.utils.thread_ctx import thread_scope


def test_system_prompt_renders_with_xml_blocks() -> None:
    prompt = get_system_prompt()
    # XML 分块 + 终结约束都应渲染出来。
    for tag in ("<role>", "<workflow>", "<tool_policy>", "<termination>", "<constraints>"):
        assert tag in prompt
    # 纯静态化：运行时占位符不复存在，改为 <runtime_context> 静态说明。
    assert "{long_term_preferences}" not in prompt
    assert "{recent_history}" not in prompt
    assert "<runtime_context>" in prompt


def test_system_prompt_is_static_no_runtime_injection() -> None:
    # 长期偏好 / 行为历史 / P_t 不再进 system prompt（改拼当轮 human）——system 段跨轮字节稳定。
    assert get_system_prompt() == get_system_prompt()
    assert "（暂无沉淀偏好）" not in get_system_prompt()


def test_pref_field_rules_injected_into_both_memory_prompts() -> None:
    """curator 与 preference_parse 落同一张偏好表，字段规则共用一段——读取时注入。

    占位符漏在成品里 = 模型收到一行「<<PREF_FIELD_RULES>>」天书；共享块没进来 = 两个入口
    的落库规则失去唯一事实源（曾各抄一份、措辞已漂移）。
    """
    from app.agent.prompts import get_memory_curator_prompt, get_preference_parse_prompt

    for text in (get_memory_curator_prompt(), get_preference_parse_prompt()):
        assert "<<PREF_FIELD_RULES>>" not in text
        assert "固定写 `ship_to`" in text  # 共享块的标志性内容真进来了


def test_thread_scope_sets_and_restores() -> None:
    assert get_thread_id() is None
    with thread_scope("t-123", Path("/tmp/shoppingx-test")):
        assert get_thread_id() == "t-123"
        assert get_session_dir() == Path("/tmp/shoppingx-test")
    # 离开作用域自动还原。
    assert get_thread_id() is None
    assert get_session_dir() is None


def test_safe_join_blocks_traversal() -> None:
    assert safe_join(OUTPUT_ROOT, "t-1", "report.md").name == "report.md"
    with pytest.raises(ValueError):
        safe_join(OUTPUT_ROOT, "../../etc/passwd")


def test_llm_factory_importable() -> None:
    # 只验证可导入与签名存在；真正构造需真实 env，放到联调阶段。
    from app.agent.llm import get_fast_llm, get_judge_llm, get_llm

    assert callable(get_llm)
    assert callable(get_judge_llm)
    assert callable(get_fast_llm)
