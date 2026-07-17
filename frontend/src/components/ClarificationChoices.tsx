import { useMemo, useState } from "react";
import { CheckIcon } from "./icons";

// ask_user 带 options 时，在展示区内嵌的可点选卡片——用户点鼠标作答，不复用底部聊天框。
// 回传给后端的仍是一段自然语言文本（契约不变），由这里据勾选拼出。
type Props = {
  question: string;
  options: string[];
  multiSelect: boolean;
  preselected?: string[] | null;
  // 已提交则禁用（回放历史里那张已作答的卡片保持只读，不再能点）。
  disabled?: boolean;
  onSubmit: (text: string) => void;
};

export function ClarificationChoices({
  question,
  options,
  multiSelect,
  preselected,
  disabled,
  onSubmit,
}: Props) {
  const initial = useMemo(
    () => new Set(multiSelect ? (preselected ?? []) : []),
    [multiSelect, preselected],
  );
  const [checked, setChecked] = useState<Set<string>>(initial);
  const [extra, setExtra] = useState("");

  // 单选：点一项即作答（该项标签直接回传）。
  const pickSingle = (opt: string) => {
    if (disabled) return;
    onSubmit(opt);
  };

  const toggle = (opt: string) => {
    if (disabled) return;
    setChecked((prev) => {
      const next = new Set(prev);
      if (next.has(opt)) next.delete(opt);
      else next.add(opt);
      return next;
    });
  };

  // 多选确认：把勾选项 + 补充框拼成一句让 Agent 能读懂的自然语言。
  const confirmMulti = () => {
    if (disabled) return;
    const kept = options.filter((o) => checked.has(o));
    const add = extra.trim();
    const parts: string[] = [];
    parts.push(kept.length ? `这套就要这些：${kept.join("、")}` : "这套先不要下面列的任何一件");
    if (add) parts.push(`另外还想加：${add}`);
    onSubmit(parts.join("；"));
  };

  return (
    <div className={`clarify-card ${disabled ? "is-done" : ""}`}>
      <div className="clarify-card-head">
        <span className="clarify-card-icon">?</span>
        <span className="clarify-card-q">{question}</span>
      </div>

      {multiSelect ? (
        <>
          <div className="clarify-checklist">
            {options.map((opt) => {
              const on = checked.has(opt);
              return (
                <button
                  key={opt}
                  type="button"
                  className={`clarify-check ${on ? "on" : ""}`}
                  onClick={() => toggle(opt)}
                  disabled={disabled}
                >
                  <span className="clarify-check-box">{on && <CheckIcon width={13} height={13} />}</span>
                  <span className="clarify-check-label">{opt}</span>
                </button>
              );
            })}
          </div>
          {!disabled && (
            <input
              className="clarify-extra"
              placeholder="想加清单里没有的？（可选，如：再加个床垫）"
              value={extra}
              onChange={(e) => setExtra(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") confirmMulti();
              }}
            />
          )}
          <div className="clarify-actions">
            <button
              type="button"
              className="clarify-confirm"
              onClick={confirmMulti}
              disabled={disabled}
            >
              {disabled ? "已确认" : "确认这套组成"}
            </button>
          </div>
        </>
      ) : (
        <div className="clarify-options">
          {options.map((opt) => (
            <button
              key={opt}
              type="button"
              className="clarify-option"
              onClick={() => pickSingle(opt)}
              disabled={disabled}
            >
              {opt}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
