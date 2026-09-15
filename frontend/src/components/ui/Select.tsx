import * as RadixSelect from "@radix-ui/react-select";
import { Check, ChevronDown } from "lucide-react";

// 下拉选择：替换原生 <select>。原生的弹出面板由系统画、跟页面配色和圆角完全脱节；
// Radix 版本键盘 / 打字跳选 / 视口避让都齐，面板样式走自家变量。
// 接口刻意收窄成「值 + 选项列表」——现有三处用法只需要这个。
export type SelectOption = { value: string; label: string; disabled?: boolean };

// Radix 不允许 Item 的 value 是空串（空串被它当成「未选」）。「未设置」这类选项在业务里就是 ""，
// 内外两层各说各的：对外仍是 ""，对内换成哨兵。
const EMPTY = "__empty__";
const toInner = (v: string) => (v === "" ? EMPTY : v);
const toOuter = (v: string) => (v === EMPTY ? "" : v);

export function Select({
  value,
  onChange,
  options,
  placeholder,
  disabled,
  ariaLabel,
  className,
}: {
  value: string;
  onChange: (value: string) => void;
  options: SelectOption[];
  placeholder?: string;
  disabled?: boolean;
  ariaLabel?: string;
  className?: string;
}) {
  return (
    <RadixSelect.Root value={toInner(value)} onValueChange={(v) => onChange(toOuter(v))} disabled={disabled}>
      <RadixSelect.Trigger className={`select-trigger ${className ?? ""}`} aria-label={ariaLabel}>
        <RadixSelect.Value placeholder={placeholder} />
        <RadixSelect.Icon className="select-icon">
          <ChevronDown size={15} strokeWidth={1.75} />
        </RadixSelect.Icon>
      </RadixSelect.Trigger>
      <RadixSelect.Portal>
        <RadixSelect.Content className="select-content" position="popper" sideOffset={4} collisionPadding={8}>
          <RadixSelect.Viewport className="select-viewport">
            {options.map((o) => (
              <RadixSelect.Item key={o.value} value={toInner(o.value)} disabled={o.disabled} className="select-item">
                <RadixSelect.ItemText>{o.label}</RadixSelect.ItemText>
                <RadixSelect.ItemIndicator className="select-check">
                  <Check size={14} strokeWidth={2} />
                </RadixSelect.ItemIndicator>
              </RadixSelect.Item>
            ))}
          </RadixSelect.Viewport>
        </RadixSelect.Content>
      </RadixSelect.Portal>
    </RadixSelect.Root>
  );
}
