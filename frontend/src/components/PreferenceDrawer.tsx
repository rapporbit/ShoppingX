import { useCallback, useEffect, useState } from "react";
import {
  addPreference,
  clearPreferences,
  deletePreference,
  deleteSessionConstraint,
  fetchPreferences,
  fetchSessionConstraints,
  updatePreference,
} from "../api";
import type { FactWrite, Preference, SessionSnapshot } from "../types";
import { CloseIcon, HeartIcon, RefreshIcon } from "./icons";
import { PreferenceEditor } from "./PreferenceEditor";
import { PreferenceItem } from "./PreferenceItem";

// 长期记忆管理页。三块：本次会话的临时约束 / 添加一条 / 已有记忆（可改可删，可全部清空）。
//
// **用户必须能看、能改、能删**——这是这套记忆设计里唯一不可省的一环：模型在后台自动学、
// 自动写，那就得有一个地方让人原样看到它记了什么，并且改得动。展示的字段与注入给模型的完全
// 一致（`[category] key: value`），不多也不少。
//
// 「我的资料」那块（收货地 / 预算上限表单）随 M4 删了：收货国这类事实现在和别的事实走同一条
// 路——用户在对话里说一句，模型用 save_memory 落成一条 key 为 default_ship_to 的记忆，
// 在这个列表里照样看得见、改得动。少一张表单，少一处「写了页面却不知道生效没有」的接缝。
type PreferenceDrawerProps = {
  userId: string;
  open: boolean;
  refreshKey: number;
  onClose: () => void;
  // 会话级 P_t（可见可纠）：约束抽取过 LLM 的手且无自愈性，抽错时唯一的兜底是用户看得见、
  // 点得掉。快照由 WS 的 session_constraints 事件实时推（hook 持有），打开面板时再主动拉一次
  // 兜底（断线重连 / 刚切会话时事件还没来）。无会话（threadId=null）不渲染该区。
  threadId: string | null;
  session: SessionSnapshot | null;
  onSessionChange: (s: SessionSnapshot | null) => void;
};

// 手填新条目的初值：category 落 preference（最轻的一档）。硬规则要用户自己选——
// constraint 每轮都会注入给模型，UI 不替他把一条随手填的东西升成硬规则。
const EMPTY_DRAFT: FactWrite = { key: "", value: "", category: "preference" };

function toDraft(p: Preference): FactWrite {
  return { key: p.key, value: p.value, category: p.category };
}

export function PreferenceDrawer({
  userId,
  open,
  refreshKey,
  onClose,
  threadId,
  session,
  onSessionChange,
}: PreferenceDrawerProps) {
  const [prefs, setPrefs] = useState<Preference[]>([]);
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  // 正在手填的那条（点「添加一条」时出现的空卡）。null = 没在填。
  const [draft, setDraft] = useState<FactWrite | null>(null);
  // 正在编辑的已有条目：记住**旧** key（改了 key 就是换一条，PUT 要用旧的去删）。
  const [editing, setEditing] = useState<{ key: string; draft: FactWrite } | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      setPrefs(await fetchPreferences(userId));
    } finally {
      setLoading(false);
    }
  }, [userId]);

  useEffect(() => {
    if (open) void load();
  }, [open, load, refreshKey]);

  // 打开面板时主动拉一次会话约束兜底（session_constraints 是瞬态事件，刚切会话 / 重连时
  // 本地快照可能还是空的）。拉不到只是该区显示为空，不报错。
  useEffect(() => {
    if (open && threadId) {
      void fetchSessionConstraints(threadId).then(onSessionChange);
    }
    // onSessionChange 是 hook 的 setState，引用稳定；refreshKey 变化时也重拉（与长期偏好同步）
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, threadId, refreshKey]);

  const removeSessionConstraint = async (cid: string) => {
    if (!threadId || !session) return;
    // 乐观删除即时响应；后端删完还会经 WS 推一次权威快照，两者同构不跳动。
    onSessionChange({ ...session, constraints: session.constraints.filter((c) => c.id !== cid) });
    await deleteSessionConstraint(threadId, cid);
  };

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  const run = async (fn: () => Promise<void>) => {
    setBusy(true);
    setError("");
    try {
      await fn();
    } catch (e) {
      setError(e instanceof Error ? e.message : "操作失败");
    } finally {
      setBusy(false);
    }
  };

  const commitDraft = () =>
    run(async () => {
      if (!draft) return;
      await addPreference(userId, draft);
      setDraft(null);
      await load();
    });

  const saveEdit = () =>
    run(async () => {
      if (!editing) return;
      await updatePreference(userId, editing.key, editing.draft);
      setEditing(null);
      await load();
    });

  const remove = async (key: string) => {
    setPrefs((cur) => cur.filter((p) => p.key !== key)); // 乐观删除，界面即时响应
    await deletePreference(userId, key);
  };

  // 清空是**不可撤销**的（没有 tombstone、也没有回收站），所以这一步要二次确认——
  // 与单条删除的口径不同：删一条错了再填一遍就是，清空全部则是几个月的积累一次性没了。
  const clearAll = () =>
    run(async () => {
      if (!window.confirm(`确定清空全部 ${prefs.length} 条长期记忆？此操作不可撤销。`)) return;
      await clearPreferences(userId);
      await load();
    });

  return (
    <>
      <div className={`drawer-scrim ${open ? "show" : ""}`} onClick={onClose} />
      <aside className={`drawer drawer-wide ${open ? "open" : ""}`} aria-hidden={!open}>
        <div className="drawer-head">
          <div className="drawer-title">
            <HeartIcon width={18} height={18} />
            记忆管理
          </div>
          <div className="drawer-tools">
            <button className="icon-btn" onClick={() => void load()} disabled={loading} title="刷新">
              <RefreshIcon width={16} height={16} className={loading ? "spin" : ""} />
            </button>
            <button className="icon-btn" onClick={onClose} title="关闭">
              <CloseIcon width={18} height={18} />
            </button>
          </div>
        </div>

        <div className="drawer-user">用户：{userId}</div>

        {threadId &&
          session &&
          (session.constraints.length > 0 || session.current_intent || session.category) && (
          <section className="pref-section">
            <div className="pref-section-title">
              本次会话 <span className="pref-count">{session.constraints.length}</span>
            </div>
            {/* 选购摘要：Agent 当前以为你要什么（意图 / 品类 / 预算 / 已定槽位）。它理解偏了，
                在对话里纠正一句即可；这里只负责让偏差看得见。 */}
            {(session.current_intent || session.category || session.budget_usd != null) && (
              <dl className="session-summary">
                {session.current_intent && (
                  <div>
                    <dt>当前需求</dt>
                    <dd>{session.current_intent}</dd>
                  </div>
                )}
                {session.category && (
                  <div>
                    <dt>品类</dt>
                    <dd>{session.category}</dd>
                  </div>
                )}
                {session.budget_usd != null && (
                  <div>
                    <dt>预算</dt>
                    <dd>${session.budget_usd.toFixed(0)}</dd>
                  </div>
                )}
                {Object.entries(session.slots ?? {}).map(([k, v]) => (
                  <div key={k}>
                    <dt>{k}</dt>
                    <dd>{v}</dd>
                  </div>
                ))}
              </dl>
            )}
            <div className="pref-section-hint">
              这次聊天里记下的临时约束（会话结束自动清）。记错了点 × 删掉，立刻不再生效。
            </div>
            <ul className="drawer-list">
              {session.constraints.map((c) => (
                <li
                  key={c.id}
                  className={`pref-item ${c.polarity === "dislike" ? "pref-dislike" : ""}`}
                >
                  <div>
                    <div>
                      {c.content}
                      <span className="pref-meta">
                        {" "}
                        ·{" "}
                        {c.polarity === "dislike"
                          ? c.blocking
                            ? "直接排除"
                            : "降低排序"
                          : "优先推荐"}
                      </span>
                    </div>
                    {c.source_quote && (
                      <div className="pref-meta">来自你说的「{c.source_quote}」</div>
                    )}
                  </div>
                  <button
                    className="pref-del"
                    title="删除这条会话约束"
                    onClick={() => void removeSessionConstraint(c.id)}
                  >
                    ×
                  </button>
                </li>
              ))}
            </ul>
          </section>
        )}

        <section className="pref-section">
          <div className="pref-section-title">添加一条</div>
          <div className="pref-section-hint">
            三个字段就是模型读到的全部：key 是身份（同 key 覆盖），内容写成过几个月单看也成立的
            句子，归类决定它有多常被读到。
          </div>
          {draft === null ? (
            <button className="pref-manual" onClick={() => setDraft({ ...EMPTY_DRAFT })}>
              + 添加一条
            </button>
          ) : (
            <PreferenceEditor
              draft={draft}
              onChange={setDraft}
              onSubmit={commitDraft}
              onCancel={() => setDraft(null)}
              submitLabel="添加"
              busy={busy}
            />
          )}
          {error && <div className="pref-error">{error}</div>}
        </section>

        <section className="pref-section">
          <div className="pref-section-title">
            长期记忆 <span className="pref-count">{prefs.length}</span>
            {prefs.length > 0 && (
              <button className="pref-clear" onClick={() => void clearAll()} disabled={busy}>
                全部清除
              </button>
            )}
          </div>
          {prefs.length === 0 ? (
            <div className="drawer-empty">
              {loading ? "加载中…" : "还没有记忆。上面手动添加，或者聊几轮让 Agent 自己学。"}
            </div>
          ) : (
            <ul className="drawer-list">
              {prefs.map((p) =>
                editing?.key === p.key ? (
                  <li key={p.key} className="pref-item-editing">
                    <PreferenceEditor
                      draft={editing.draft}
                      onChange={(draft) => setEditing({ key: editing.key, draft })}
                      onSubmit={saveEdit}
                      onCancel={() => setEditing(null)}
                      submitLabel="保存"
                      busy={busy}
                    />
                  </li>
                ) : (
                  <PreferenceItem
                    key={p.key}
                    pref={p}
                    onEdit={() => setEditing({ key: p.key, draft: toDraft(p) })}
                    onDelete={() => void remove(p.key)}
                  />
                ),
              )}
            </ul>
          )}
        </section>
      </aside>
    </>
  );
}
