import type { ReactNode } from "react";
import * as Dialog from "@radix-ui/react-dialog";
import { AnimatePresence, motion } from "motion/react";
import { X } from "lucide-react";

// 右侧滑出面板（设置 / 搜同款 / 后台管理共用）。四个「我的东西」面板已经是整页视图，不走这里。
// 底座 Radix Dialog：焦点圈养、Esc、点遮罩关、滚动锁；进出场交 motion（弹簧滑入，退场略快）。
// 关闭即卸载内容——这几个面板的内容都是打开那一刻现拉的，不需要常驻。
const overlayMotion = {
  initial: { opacity: 0 },
  animate: { opacity: 1 },
  exit: { opacity: 0 },
  transition: { duration: 0.2 },
};

const panelMotion = {
  initial: { x: "104%" },
  animate: { x: 0 },
  exit: { x: "104%", transition: { duration: 0.22, ease: [0.4, 0, 1, 1] } },
  transition: { type: "spring", stiffness: 420, damping: 40, mass: 0.9 },
} as const;

export function Sheet({
  open,
  title,
  icon,
  tools,
  className,
  onClose,
  children,
}: {
  open: boolean;
  title: string;
  icon?: ReactNode;
  // 标题右侧的附加按钮（刷新等），关闭键由这里统一放最右。
  tools?: ReactNode;
  className?: string;
  onClose: () => void;
  children: ReactNode;
}) {
  return (
    <Dialog.Root
      open={open}
      onOpenChange={(next) => {
        if (!next) onClose();
      }}
    >
      <AnimatePresence>
        {open && (
          <Dialog.Portal forceMount>
            <Dialog.Overlay asChild forceMount>
              <motion.div className="sheet-scrim" {...overlayMotion} />
            </Dialog.Overlay>
            <Dialog.Content
              asChild
              forceMount
              aria-describedby={undefined}
              onOpenAutoFocus={(e) => e.preventDefault()}
            >
              <motion.aside className={`sheet ${className ?? ""}`} {...panelMotion}>
                <div className="drawer-head">
                  <Dialog.Title className="drawer-title">
                    {icon}
                    {title}
                  </Dialog.Title>
                  <div className="drawer-tools">
                    {tools}
                    <Dialog.Close asChild>
                      <button className="icon-btn" aria-label="关闭">
                        <X size={18} strokeWidth={1.75} />
                      </button>
                    </Dialog.Close>
                  </div>
                </div>
                {children}
              </motion.aside>
            </Dialog.Content>
          </Dialog.Portal>
        )}
      </AnimatePresence>
    </Dialog.Root>
  );
}
