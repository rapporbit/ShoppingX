import * as Switch from "@radix-ui/react-switch";
import { Globe } from "lucide-react";
import { PLATFORM_OPTIONS } from "../settings";
import { Sheet } from "./ui/Sheet";

// 设置面板（右侧滑出，外壳走 ui/Sheet）：目前只有「检索平台」一项。
// 默认只勾 Amazon —— 召回库里其余平台近乎空，默认跨 5 平台等于同轮多发 4 条必然空手而归的
// item_search。勾上第二个平台才真正触发跨平台并行检索与比价（更全，但更慢、更贵）。
// 受控组件：平台状态由 App 持有（顶栏要实时显示启用个数），落盘在 App 的 onToggle 里做。
type SettingsDrawerProps = {
  open: boolean;
  platforms: string[];
  onToggle: (id: string) => void;
  onClose: () => void;
};

export function SettingsDrawer({ open, platforms, onToggle, onClose }: SettingsDrawerProps) {
  const multi = platforms.length > 1;

  return (
    <Sheet open={open} title="设置" icon={<Globe size={18} strokeWidth={1.75} />} onClose={onClose}>
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
                  <span className="platform-name">{opt.label}</span>
                  <span className="platform-note">{opt.note}</span>
                  <Switch.Root
                    className="switch"
                    checked={checked}
                    onCheckedChange={() => onToggle(opt.id)}
                    aria-label={`启用 ${opt.label}`}
                  >
                    <Switch.Thumb className="switch-thumb" />
                  </Switch.Root>
                </label>
              </li>
            );
          })}
        </ul>

        <div className="drawer-foot-note">
          当前：
          {multi ? `跨 ${platforms.length} 个平台比价` : "单平台模式（只搜 Amazon）"}
          。设置立即生效，下一轮提问按此执行。
        </div>
      </div>
    </Sheet>
  );
}
