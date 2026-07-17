import { useEffect } from "react";
import { PLATFORM_OPTIONS } from "../settings";
import { CloseIcon, GlobeIcon } from "./icons";

// 设置抽屉（与长期偏好抽屉同风格，从右侧滑出）：目前只有「检索平台」一项。
// 默认只勾 Amazon —— 召回库里其余平台近乎空，默认跨 5 平台 fork 等于派 4 个必然空手而归的子
// Agent。勾上第二个平台才真正触发跨平台并行检索与比价（更全，但更慢、更贵）。
// 受控组件：平台状态由 App 持有（顶栏要实时显示启用个数），落盘在 App 的 onToggle 里做。
type SettingsDrawerProps = {
  open: boolean;
  platforms: string[];
  onToggle: (id: string) => void;
  onClose: () => void;
};

export function SettingsDrawer({ open, platforms, onToggle, onClose }: SettingsDrawerProps) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  const multi = platforms.length > 1;

  return (
    <>
      <div className={`drawer-scrim ${open ? "show" : ""}`} onClick={onClose} />
      <aside className={`drawer ${open ? "open" : ""}`} aria-hidden={!open}>
        <div className="drawer-head">
          <div className="drawer-title">
            <GlobeIcon width={18} height={18} />
            设置
          </div>
          <div className="drawer-tools">
            <button className="icon-btn" onClick={onClose} title="关闭">
              <CloseIcon width={18} height={18} />
            </button>
          </div>
        </div>

        <div className="drawer-section">
          <div className="drawer-section-title">检索平台</div>
          <p className="drawer-hint">
            默认只搜 Amazon（库里商品最全）。勾选 2 个及以上平台才会真正跨平台并行检索、比价——
            结果更全，但每轮更慢、token 开销更大。
          </p>

          <ul className="platform-list">
            {PLATFORM_OPTIONS.map((opt) => {
              const checked = platforms.includes(opt.id);
              return (
                <li key={opt.id} className={`platform-item ${checked ? "on" : ""}`}>
                  <label>
                    <input
                      type="checkbox"
                      checked={checked}
                      onChange={() => onToggle(opt.id)}
                    />
                    <span className="platform-name">{opt.label}</span>
                    <span className="platform-note">{opt.note}</span>
                  </label>
                </li>
              );
            })}
          </ul>

          <div className="drawer-foot-note">
            当前：{multi ? `跨 ${platforms.length} 个平台比价` : "单平台模式（不 fork 子 Agent）"}
            。设置立即生效，下一轮提问按此执行。
          </div>
        </div>
      </aside>
    </>
  );
}
