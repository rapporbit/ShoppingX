import { useCallback, useEffect, useState } from "react";
import {
  addPreferences,
  deletePreference,
  deleteSessionConstraint,
  fetchPreferences,
  fetchSessionConstraints,
  parsePreference,
  updatePreference,
} from "../api";
import type { Preference, PrefDraft, SessionSnapshot } from "../types";
import { CloseIcon, HeartIcon, RefreshIcon } from "./icons";
import { PreferenceEditor } from "./PreferenceEditor";
import { PreferenceItem } from "./PreferenceItem";
import { ProfileForm } from "./ProfileForm";

// 长期偏好管理页。三块：我的资料（硬事实）/ 添加偏好 / 已有偏好（结构化列表，可改可删）。
//
// 添加走「一句话 → LLM 解析成结构化草稿 → 用户过目 / 修改 → 确认落库」：偏好是结构化实体
// （polarity / blocking / domain / keywords 各自驱动不同的检索行为），让 LLM 猜的字段直接进库，
// 用户既不知道写了什么、也没机会纠正。草稿卡这一步就是把结构摆到台面上——「绝不推荐」尤其如此，
// 它是唯一会把商品从结果里删掉的字段，必须由用户在这张卡上亲手勾。
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

// 资料块已单独展示这两条，列表里就别重复出现。
const PROFILE_KEYS = new Set(["like:location:global:ship_to", "like:budget:global:budget_max"]);

// 手填新条目的初值。domain 落 other（判不出品类的保守档，用户自己去下拉里选具体的）、
// blocking 落 false——硬淘汰权只在用户亲手勾选时授予，UI 不替他预勾（同 /parse 草稿）。
const EMPTY_DRAFT: PrefDraft = {
  content: "",
  category: "other",
  domain: "other",
  slug: "",
  polarity: "like",
  blocking: false,
  keywords: [],
};

function toDraft(p: Preference): PrefDraft {
  return {
    content: p.content,
    category: p.category,
    domain: p.domain,
    slug: p.slug,
    polarity: p.polarity,
    blocking: p.blocking,
    keywords: [...p.keywords],
  };
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
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  // 待确认的新条目草稿（来自 /parse 或「手动填写」的空卡）。非空时列表上方渲染成可编辑卡。
  const [drafts, setDrafts] = useState<PrefDraft[]>([]);
  // 正在编辑的已有条目：记住**旧** dedup_key（改字段会换 key，PUT 要用旧的去删）。
  const [editing, setEditing] = useState<{ key: string; draft: PrefDraft } | null>(null);

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

  const parse = () =>
    run(async () => {
      const parsed = await parsePreference(userId, text.trim());
      setDrafts((cur) => [...cur, ...parsed]);
      setText("");
    });

  const commitDrafts = () =>
    run(async () => {
      await addPreferences(userId, drafts);
      setDrafts([]);
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
    setPrefs((cur) => cur.filter((p) => p.dedup_key !== key)); // 乐观删除，界面即时响应
    await deletePreference(userId, key);
  };

  const listed = prefs.filter((p) => !PROFILE_KEYS.has(p.dedup_key));

  return (
    <>
      <div className={`drawer-scrim ${open ? "show" : ""}`} onClick={onClose} />
      <aside className={`drawer drawer-wide ${open ? "open" : ""}`} aria-hidden={!open}>
        <div className="drawer-head">
          <div className="drawer-title">
            <HeartIcon width={18} height={18} />
            偏好管理
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

        <ProfileForm userId={userId} prefs={prefs} onSaved={setPrefs} />

        {threadId && session && session.constraints.length > 0 && (
          <section className="pref-section">
            <div className="pref-section-title">
              本次会话 <span className="pref-count">{session.constraints.length}</span>
            </div>
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
          <div className="pref-section-title">添加偏好</div>
          <div className="pref-section-hint">
            写一句话，我拆成结构化条目给你过目；也可以直接手动填。
          </div>
          <div className="pref-add">
            <input
              value={text}
              placeholder="如「不要塑料的，尽量选小众品牌」"
              onChange={(e) => setText(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && text.trim() && void parse()}
            />
            <button onClick={() => void parse()} disabled={!text.trim() || busy}>
              {busy ? "…" : "解析"}
            </button>
            <button
              className="pref-manual"
              title="不用 LLM，直接手动填一条"
              onClick={() => setDrafts((cur) => [...cur, { ...EMPTY_DRAFT }])}
            >
              手动
            </button>
          </div>
          {error && <div className="pref-error">{error}</div>}

          {drafts.length > 0 && (
            <div className="pref-drafts">
              <div className="pref-drafts-title">待确认（{drafts.length}）—— 可改，确认后才写入</div>
              {drafts.map((d, i) => (
                <PreferenceEditor
                  key={i}
                  draft={d}
                  onChange={(next) => setDrafts((cur) => cur.map((x, j) => (j === i ? next : x)))}
                  onSubmit={commitDrafts}
                  onCancel={() => setDrafts((cur) => cur.filter((_, j) => j !== i))}
                  submitLabel={`确认添加 ${drafts.length} 条`}
                  busy={busy}
                />
              ))}
            </div>
          )}
        </section>

        <section className="pref-section">
          <div className="pref-section-title">
            已有偏好 <span className="pref-count">{listed.length}</span>
          </div>
          {listed.length === 0 ? (
            <div className="drawer-empty">
              {loading ? "加载中…" : "还没有偏好。上面手动添加，或者聊几轮让 Agent 自己学。"}
            </div>
          ) : (
            <ul className="drawer-list">
              {listed.map((p) =>
                editing?.key === p.dedup_key ? (
                  <li key={p.dedup_key} className="pref-item-editing">
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
                    key={p.dedup_key}
                    pref={p}
                    onEdit={() => setEditing({ key: p.dedup_key, draft: toDraft(p) })}
                    onDelete={() => void remove(p.dedup_key)}
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
