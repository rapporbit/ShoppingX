import type { ReactNode } from "react";
import * as RadixTooltip from "@radix-ui/react-tooltip";

// 提示气泡：替换原生 title。原生 title 要等 ~1s 才出现、样式不可控、触屏上根本不显示；
// Radix 的版本 400ms 出、键盘聚焦也出、位置自动避让视口。
// 只包一层 Provider 在 main.tsx，这里每个 Tooltip 直接用。
export function Tooltip({
  content,
  side = "bottom",
  children,
}: {
  content: ReactNode;
  side?: "top" | "bottom" | "left" | "right";
  children: ReactNode;
}) {
  if (!content) return <>{children}</>;
  return (
    <RadixTooltip.Root>
      <RadixTooltip.Trigger asChild>{children}</RadixTooltip.Trigger>
      <RadixTooltip.Portal>
        <RadixTooltip.Content className="tip" side={side} sideOffset={6} collisionPadding={8}>
          {content}
          <RadixTooltip.Arrow className="tip-arrow" width={10} height={5} />
        </RadixTooltip.Content>
      </RadixTooltip.Portal>
    </RadixTooltip.Root>
  );
}

export const TooltipProvider = RadixTooltip.Provider;
