import { DOMAIN_HINTS, DOMAIN_LABELS, DOMAIN_ORDER } from "../domains";
import type { PrefDomain, PrefDraft } from "../types";

// 一条偏好的**结构化编辑卡**——添加（LLM 解析出的草稿）与修改（已有条目）共用。
//
// 偏好不是一句话，是个实体：各字段驱动不同的检索行为，所以必须让用户看见并能改：
//   polarity  想要 / 不要      —— dislike 才会进黑名单；like 的 keywords 拼进检索词
//   blocking  绝不推荐         —— 命中即从结果里删掉。**只有这里能授予**：Agent 从对话里学到的
//                                 偏好在后端被强制 false，永远拿不到硬淘汰权
//   keywords  原子词           —— item_picker 真正拿去做字符串匹配的就是它，不是 content
//   domain    生效品类         —— 封闭枚举。global 是唯一跨品类生效的档位
//   category  归类             —— 封闭枚举，和 slug 一起构成去重身份
// 只让用户看 content 一句话，等于把真正起作用的部分藏起来了。
//
// blocking 不做成默认勾选、也不在 parse 草稿里替用户预勾：后端 LLM 的判定纪律是「拿不准一律
// false」，UI 不该把这个纪律绕过去。但也不加二次确认弹窗——写入是廉价可逆的（不对就取消勾选）。

const CATEGORIES = ["material", "style", "brand", "budget", "color", "size", "location", "other"];
const CATEGORY_CN: Record<string, string> = {
  material: "材质",
  style: "风格",
  brand: "品牌",
  budget: "预算",
  color: "颜色",
  size: "尺寸",
  location: "收货地",
  other: "其它",
};

type PreferenceEditorProps = {
  draft: PrefDraft;
  onChange: (next: PrefDraft) => void;
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
  const set = (patch: Partial<PrefDraft>) => onChange({ ...draft, ...patch });

  return (
    <div className="pref-editor">
      <input
        className="pref-editor-content"
        value={draft.content}
        placeholder="偏好内容，如「不接受塑料材质」"
        onChange={(e) => set({ content: e.target.value })}
      />

      <div className="pref-editor-row">
        <select
          value={draft.polarity}
          // 转成「想要」时把 blocking 清掉：硬淘汰只对 dislike 生效（后端 dislike_exclude_terms
          // 先按 polarity 过滤），留着一个不生效的 true 只会误导用户。
          onChange={(e) => {
            const polarity = e.target.value as PrefDraft["polarity"];
            set({ polarity, blocking: polarity === "dislike" && draft.blocking });
          }}
        >
          <option value="like">❤️ 想要</option>
          <option value="dislike">🚫 不要</option>
        </select>
        <select value={draft.category} onChange={(e) => set({ category: e.target.value })}>
          {CATEGORIES.map((c) => (
            <option key={c} value={c}>
              {CATEGORY_CN[c] ?? c}
            </option>
          ))}
        </select>
        <select
          className="pref-editor-domain"
          value={draft.domain}
          onChange={(e) => set({ domain: e.target.value as PrefDomain })}
        >
          {DOMAIN_ORDER.map((d) => (
            <option key={d} value={d}>
              {DOMAIN_LABELS[d]}
              {DOMAIN_HINTS[d] ? `（${DOMAIN_HINTS[d]}）` : ""}
            </option>
          ))}
        </select>
      </div>

      {/* 硬淘汰授权。分量配得上后果：勾上意味着这类商品**再也不会出现在结果里**，而且被删掉了
          什么你是看不见的。所以它只能由用户亲手勾——Agent 学到的偏好在后端拿不到这个权限。 */}
      {draft.polarity === "dislike" && (
        <label className={`pref-editor-block ${draft.blocking ? "on" : ""}`}>
          <input
            type="checkbox"
            checked={draft.blocking}
            onChange={(e) => set({ blocking: e.target.checked })}
          />
          <span className="pref-editor-block-text">
            <strong>绝不推荐这类商品</strong>
            <em>
              命中关键词的商品会被直接从结果里删掉，你不会看到被删了哪些。只是「不太喜欢、尽量
              避免」就别勾——不勾时它仍会压低这类商品的排序。
            </em>
          </span>
        </label>
      )}

      <div className="pref-editor-row">
        <label
          className="pref-editor-label"
          title="item_picker 真正拿去做匹配的原子词；不要的偏好务必填，中英文都给"
        >
          关键词
        </label>
        <input
          value={draft.keywords.join("，")}
          placeholder="塑料，plastic（逗号分隔）"
          onChange={(e) =>
            set({
              keywords: e.target.value
                .split(/[，,]/)
                .map((k) => k.trim())
                .filter(Boolean),
            })
          }
        />
        <label className="pref-editor-label" title="去重身份：改它等于换成另一条偏好">
          标识
        </label>
        <input
          className="pref-editor-slug"
          value={draft.slug}
          placeholder="plastic"
          onChange={(e) => set({ slug: e.target.value.trim() })}
        />
      </div>

      <div className="pref-editor-actions">
        <button
          className="pref-save"
          onClick={onSubmit}
          disabled={busy || !draft.content.trim() || !draft.slug.trim()}
        >
          {busy ? "保存中…" : submitLabel}
        </button>
        <button className="pref-cancel" onClick={onCancel} disabled={busy}>
          取消
        </button>
      </div>
    </div>
  );
}
