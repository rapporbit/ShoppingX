"""上下文级飞轮的「P0 自动修规则」腿（refdocs 18-2 §2-4，无 GPU）。

evolve_p0.py 驱动：从 Rubric 评测报告读 P0-破 bad case → 分流（LEAK / BANNED / JUDGMENT）→
对确定性泄露类**验证真泄露后**抽字面串，沉淀成 learned 脱敏规则（app/security/learned_rules）。

**只有 LEAK 类自动生成规则**，且必须过「串真的出现在用户可见文本里」这道闸（拦 judge 假阳性）。
BANNED / JUDGMENT 类只分类上报，不自动改任何规则——判断类红线（超预算 / 人群冲突）规则堵不住，
违禁品类目前没有正确的规则落点（不塞进管注入的 content_filter），都留给人工 / prompt 腿。
"""
