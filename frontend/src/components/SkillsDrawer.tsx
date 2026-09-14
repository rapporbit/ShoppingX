import { useCallback, useEffect, useState } from "react";
import { createSkill, deleteSkill, fetchMySkills, updateSkill } from "../api";
import type { UserSkill } from "../types";
import { CloseIcon, RefreshIcon } from "./icons";

// 我的 Skill 抽屉：买家自己写的「选购方案」（对标参考项目的个人 Skill）。
//
// 它与「长期偏好」的分工：偏好是**事实**（不要塑料 / 喜欢小众），每轮按品类域注入；Skill 是**打法**
// （先问容量再比可证实规格、到手价按收货国算），name + description 常驻 Agent 的 skill 目录，正文
// 只在 Agent 自判相关、或用户在输入框敲 / 显式选中时才读进来。它是 reference_only 的参考资料：
// 改不了工具权限，也盖不过用户当轮说的预算 / 禁忌。
type SkillsDrawerProps = {
  userId: string;
  open: boolean;
  onClose: () => void;
  onChanged: () => void; // 增删改后通知 App 重拉目录，让输入框 / 菜单立刻看到
};

type Draft = { name: string; description: string; body: string };
const EMPTY: Draft = { name: "", description: "", body: "" };

export function SkillsDrawer({ userId, open, onClose, onChanged }: SkillsDrawerProps) {
  const [items, setItems] = useState<UserSkill[]>([]);
  const [loading, setLoading] = useState(false);
  const [editing, setEditing] = useState<string | null>(null); // null=不在编辑，""=新建，其余=name
  const [draft, setDraft] = useState<Draft>(EMPTY);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setItems(await fetchMySkills(userId));
    setLoading(false);
  }, [userId]);

  useEffect(() => {
    if (open) void load();
  }, [open, load]);

  const startEdit = (s?: UserSkill) => {
    setError(null);
    setEditing(s ? s.name : "");
    setDraft(s ? { name: s.name, description: s.description, body: s.body } : EMPTY);
  };

  const save = async () => {
    setError(null);
    try {
      if (editing === "") await createSkill(userId, draft);
      else if (editing) await updateSkill(userId, editing, draft);
      setEditing(null);
      await load();
      onChanged();
    } catch (e) {
      setError(e instanceof Error ? e.message : "保存失败");
    }
  };

  const drop = async (name: string) => {
    await deleteSkill(userId, name);
    await load();
    onChanged();
  };

  return (
    <>
      <div className={`drawer-scrim ${open ? "show" : ""}`} onClick={onClose} />
      <aside className={`drawer drawer-wide ${open ? "open" : ""}`} aria-hidden={!open}>
        <div className="drawer-head">
          <div className="drawer-title">
            <span className="skill-glyph">/</span>
            我的 Skill
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
        <div className="fav-note">
          Skill 是你自己写的<b>选购打法</b>（先问什么、比什么、怎么算到手价）。输入框敲 <code>/</code> 可显式选用；
          不选时 Agent 也会按「用途」一句话自己判断要不要读。它只是参考资料，改不了你当轮说的预算和禁忌。
        </div>

        {editing !== null ? (
          <div className="skill-editor">
            <input
              className="skill-input"
              placeholder="标识（小写字母/数字/-，如 weekend-backpack）"
              value={draft.name}
              disabled={editing !== ""}
              onChange={(e) => setDraft({ ...draft, name: e.target.value })}
            />
            <input
              className="skill-input"
              placeholder="用途一句话（Agent 靠它判断何时用，写具体）"
              value={draft.description}
              onChange={(e) => setDraft({ ...draft, description: e.target.value })}
            />
            <textarea
              className="skill-body"
              placeholder="正文：步骤 / 取舍规则 / 到手价口径……（Markdown）"
              rows={10}
              value={draft.body}
              onChange={(e) => setDraft({ ...draft, body: e.target.value })}
            />
            {error && <div className="composer-image-error">{error}</div>}
            <div className="skill-editor-acts">
              <button className="ghost-btn" onClick={() => setEditing(null)}>取消</button>
              <button className="pref-save" onClick={() => void save()}>保存</button>
            </div>
          </div>
        ) : (
          <>
            <div className="skill-list-head">
              <button className="pref-save" onClick={() => startEdit()}>+ 新建 Skill</button>
            </div>
            {items.length === 0 ? (
              <div className="drawer-empty">还没有个人 Skill。写一份你的选购打法，下次敲 / 就能用。</div>
            ) : (
              <ul className="fav-list">
                {items.map((s) => (
                  <li key={s.name} className="fav-row skill-row" onClick={() => startEdit(s)}>
                    <div className="fav-main">
                      <div className="fav-title">
                        /{s.catalog_name}
                        <span className="fav-platform">v{s.version}</span>
                      </div>
                      <div className="fav-meta">{s.description}</div>
                    </div>
                    <div className="fav-acts">
                      <button
                        className="icon-btn"
                        onClick={(e) => {
                          e.stopPropagation();
                          void drop(s.name);
                        }}
                        title="删除"
                      >
                        <CloseIcon width={16} height={16} />
                      </button>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </>
        )}
      </aside>
    </>
  );
}
