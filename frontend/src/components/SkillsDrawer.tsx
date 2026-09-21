import { useCallback, useEffect, useMemo, useState } from "react";
import {
  createSkill,
  deleteSkill,
  fetchBuiltinSkill,
  fetchMySkills,
  fetchSkillCatalog,
  updateSkill,
} from "../api";
import type { SkillCatalogItem, UserSkill } from "../types";
import { safeMarkdown } from "./FinalAnswer";
import { CloseIcon, RefreshIcon } from "./icons";

// 我的 Skill 抽屉：买家自己写的「选购方案」。
//
// 它与「长期偏好」的分工：偏好是**事实**（不要塑料 / 喜欢小众），每轮按品类域注入；Skill 是**打法**
// （先问容量再比可证实规格、到手价按收货国算），name + description 常驻 Agent 的 skill 目录，正文
// 只在 Agent 自判相关、或用户在输入框敲 / 显式选中时才读进来。它是 reference_only 的参考资料：
// 改不了工具权限，也盖不过用户当轮说的预算 / 禁忌。
//
// 版面：左列 = 我的 + 系统内置两组列表；右列 = 选中那份的只读详情，或编辑器。单列（窄屏 / 抽屉）时
// 右列一打开就把左列收起来、给一个「返回列表」——否则编辑器会被挤到整条列表的下面。
type SkillsDrawerProps = {
  userId: string;
  open: boolean;
  onClose: () => void;
  onChanged: () => void; // 增删改后通知 App 重拉目录，让输入框 / 菜单立刻看到
};

type Draft = { name: string; description: string; body: string };
type Selected = { source: "user" | "builtin"; name: string };
const EMPTY: Draft = { name: "", description: "", body: "" };

export function SkillsDrawer({ userId, open, onClose, onChanged }: SkillsDrawerProps) {
  const [items, setItems] = useState<UserSkill[]>([]);
  // 系统内置 skill（skills/ 下的 SKILL.md）：只读。目录接口只给 name + description，正文点开时再取。
  const [builtins, setBuiltins] = useState<SkillCatalogItem[]>([]);
  // 内置正文缓存：undefined = 还没取 / 取着，null = 取失败，string = 正文。
  const [builtinBodies, setBuiltinBodies] = useState<Record<string, string | null>>({});
  const [loading, setLoading] = useState(false);
  const [selected, setSelected] = useState<Selected | null>(null);
  const [editing, setEditing] = useState<string | null>(null); // null=不在编辑，""=新建，其余=name
  const [draft, setDraft] = useState<Draft>(EMPTY);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    const [mine, catalog] = await Promise.all([fetchMySkills(userId), fetchSkillCatalog(userId)]);
    setItems(mine);
    setBuiltins(catalog.filter((c) => c.source === "builtin"));
    setLoading(false);
  }, [userId]);

  useEffect(() => {
    if (open) void load();
  }, [open, load]);

  const select = (next: Selected) => {
    setEditing(null);
    setError(null);
    setSelected(next);
    // 没取过或上次取失败（null）都再取一次：失败后再点这一条就是重试。
    if (next.source === "builtin" && builtinBodies[next.name] == null) {
      void fetchBuiltinSkill(next.name).then((s) =>
        setBuiltinBodies((prev) => ({ ...prev, [next.name]: s ? s.body : null })),
      );
    }
  };

  // 新建可以带初稿：从一份内置 skill「复制为我的」时，把它的用途和正文填进来，标识留给用户起。
  const startEdit = (s?: UserSkill, seed?: Draft) => {
    setError(null);
    setEditing(s ? s.name : "");
    setDraft(s ? { name: s.name, description: s.description, body: s.body } : (seed ?? EMPTY));
    if (!s) setSelected(null);
  };

  const closeRight = () => {
    setEditing(null);
    setSelected(null);
  };

  const save = async () => {
    setError(null);
    try {
      if (editing === "") {
        const created = await createSkill(userId, draft);
        setSelected({ source: "user", name: created.name });
      } else if (editing) await updateSkill(userId, editing, draft);
      setEditing(null);
      await load();
      onChanged();
    } catch (e) {
      setError(e instanceof Error ? e.message : "保存失败");
    }
  };

  const drop = async (name: string) => {
    await deleteSkill(userId, name);
    if (selected?.source === "user" && selected.name === name) setSelected(null);
    if (editing === name) setEditing(null);
    await load();
    onChanged();
  };

  // 右列只读详情要的东西，两种来源抹平成一个形状；mine 非空 = 个人 skill，可编辑。
  const detail = useMemo(() => {
    if (!selected) return null;
    if (selected.source === "user") {
      const s = items.find((i) => i.name === selected.name);
      return s
        ? { title: s.catalog_name, tag: `v${s.version}`, description: s.description, body: s.body as string | null | undefined, mine: s }
        : null;
    }
    const b = builtins.find((i) => i.name === selected.name);
    return b
      ? { title: b.name, tag: "内置", description: b.description, body: builtinBodies[b.name], mine: null }
      : null;
  }, [selected, items, builtins, builtinBodies]);

  const rightOpen = editing !== null || detail !== null;
  const isActive = (source: Selected["source"], name: string) =>
    selected?.source === source && selected.name === name;

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
          Skill 是<b>选购打法</b>（先问什么、比什么、怎么算到手价）。输入框敲 <code>/</code> 可显式选用；
          不选时 Agent 也会按「用途」一句话自己判断要不要读。它只是参考资料，改不了你当轮说的预算和禁忌。
          点任意一条看正文；「系统内置」只读，可以复制一份改成你自己的。
        </div>

        <div className={`skill-page-grid ${rightOpen ? "detail-open" : ""}`}>
          <div className="skill-list-col">
            <div className="skill-group-head">
              <span className="drawer-section-title">我的（{items.length}）</span>
              <button className="pref-save" onClick={() => startEdit()}>+ 新建 Skill</button>
            </div>
            {items.length === 0 ? (
              <div className="drawer-empty">还没有个人 Skill。写一份你的选购打法，下次敲 / 就能用。</div>
            ) : (
              <ul className="fav-list">
                {items.map((s) => (
                  <li
                    key={s.name}
                    className={`fav-row skill-row ${isActive("user", s.name) ? "active" : ""}`}
                    onClick={() => select({ source: "user", name: s.name })}
                  >
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
            {builtins.length > 0 && (
              <div className="drawer-section">
                <div className="skill-group-head">
                  <span className="drawer-section-title">系统内置（{builtins.length}，只读）</span>
                </div>
                <ul className="fav-list">
                  {builtins.map((b) => (
                    <li
                      key={b.name}
                      className={`fav-row skill-row ${isActive("builtin", b.name) ? "active" : ""}`}
                      onClick={() => select({ source: "builtin", name: b.name })}
                    >
                      <div className="fav-main">
                        <div className="fav-title">
                          /{b.name}
                          <span className="fav-platform">内置</span>
                        </div>
                        <div className="fav-meta">{b.description}</div>
                      </div>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
          <div className="skill-editor-col">
            {rightOpen && (
              <button className="ghost-btn skill-back" onClick={closeRight}>← 返回列表</button>
            )}
            {editing !== null ? (
              <div className="skill-editor">
                <div className="skill-detail-title">{editing === "" ? "新建 Skill" : `编辑 /my/${editing}`}</div>
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
                  rows={16}
                  value={draft.body}
                  onChange={(e) => setDraft({ ...draft, body: e.target.value })}
                />
                {error && <div className="composer-image-error">{error}</div>}
                <div className="skill-editor-acts">
                  <button className="ghost-btn" onClick={() => setEditing(null)}>取消</button>
                  <button className="pref-save" onClick={() => void save()}>保存</button>
                </div>
              </div>
            ) : detail ? (
              <div className="skill-detail">
                <div className="skill-detail-head">
                  <div className="skill-detail-title">
                    /{detail.title}
                    <span className="fav-platform">{detail.tag}</span>
                  </div>
                  {detail.mine ? (
                    <button className="pref-save" onClick={() => startEdit(detail.mine ?? undefined)}>编辑</button>
                  ) : (
                    <button
                      className="pref-save"
                      disabled={!detail.body}
                      onClick={() =>
                        startEdit(undefined, { name: "", description: detail.description, body: detail.body ?? "" })
                      }
                      title="以这份内置 Skill 为初稿，新建一份你自己的"
                    >
                      复制为我的
                    </button>
                  )}
                </div>
                <div className="skill-detail-desc">{detail.description}</div>
                {detail.body === undefined ? (
                  <div className="drawer-empty">正文读取中…</div>
                ) : detail.body === null ? (
                  <div className="drawer-empty">正文没取到，再点一次左边这一条重试。</div>
                ) : (
                  <div
                    className="markdown skill-detail-body"
                    dangerouslySetInnerHTML={{ __html: safeMarkdown(detail.body) }}
                  />
                )}
              </div>
            ) : (
              <div className="skill-page-empty">
                左边点一份 Skill 看正文，或者点「新建 Skill」。
                <br />
                写清先问什么、比什么、到手价怎么算，Agent 读它的时候就照这个打法走。
              </div>
            )}
          </div>
        </div>
      </aside>
    </>
  );
}
