import { Select } from "./ui/Select";
import type { FactWrite, MemoryCategory } from "../types";

// 一条长期记忆的编辑卡——手填新条目与修改已有条目共用。
//
// 只有三个字段，而且**和模型读到的完全一样**（注入块每行就是 `[category] key: value`）：
//   key       主题标识    —— 身份。同 key 覆盖写，所以「改主意」= 用原 key 写新值
//   value     内容        —— 写成几个月后单看也成立的句子
//   category  归类        —— 决定注入优先级：constraint 每轮必注入，其余按新鲜度补位
//
// 上一版这里有七八个字段（polarity / blocking / domain / slug / keywords），用户改了其中一个
// 却看不出行为会怎么变，而模型根本没见过这些字段。页面上看得见的，就该是模型读得到的。
//
// **没有「绝不推荐」这类开关了**：记忆不再有直接淘汰商品的通路，它只注入给模型，由模型写进
// 工具入参。category 决定的是「这条多重要」，不是「杀伤力多大」。

const CATEGORIES: MemoryCategory[] = ["constraint", "preference", "context"];
const CATEGORY_CN: Record<MemoryCategory, string> = {
  constraint: "硬规则（每轮必读）",
  preference: "取向",
  context: "背景",
};

type PreferenceEditorProps = {
  draft: FactWrite;
  onChange: (next: FactWrite) => void;
  onSubmit: () => void;
  onCancel: () => void;
  submitLabel: string;
  busy?: boolean;
};

export function PreferenceEditor({
  draft,
  onChange,
  onSubmit,
  onCancel,
  submitLabel,
  busy,
}: PreferenceEditorProps) {
  const set = (patch: Partial<FactWrite>) => onChange({ ...draft, ...patch });
  const ready = draft.key.trim() !== "" && draft.value.trim() !== "";

  return (
    <div className="pref-editor">
      <input
        className="pref-editor-content"
        value={draft.value}
        placeholder="内容，如「不接受塑料材质」"
        onChange={(e) => set({ value: e.target.value })}
        onKeyDown={(e) => e.key === "Enter" && ready && !busy && onSubmit()}
      />

      <div className="pref-editor-row">
        <input
          className="pref-editor-key"
          value={draft.key}
          placeholder="key，如 material_avoid"
          onChange={(e) => set({ key: e.target.value })}
        />
        <Select
          value={draft.category}
          onChange={(v) => set({ category: v as MemoryCategory })}
          options={CATEGORIES.map((c) => ({ value: c, label: CATEGORY_CN[c] }))}
          ariaLabel="归类"
        />
      </div>

      <div className="pref-editor-hint">
        同一个 key 只存一条：改主意时用**原 key** 写新值，旧的自动被覆盖。
      </div>

      <div className="pref-editor-actions">
        <button onClick={onSubmit} disabled={!ready || busy}>
          {busy ? "…" : submitLabel}
        </button>
        <button className="pref-editor-cancel" onClick={onCancel}>
          取消
        </button>
      </div>
    </div>
  );
}
