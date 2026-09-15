import type { ReactNode } from "react";
import * as Dialog from "@radix-ui/react-dialog";
import { AnimatePresence, motion } from "motion/react";
import { X } from "lucide-react";

// 居中弹窗（商品详情 / 对比 / 下单意向表单共用）。与右侧抽屉（.drawer）的分工：抽屉是「另一份
// 清单」（收藏 / 订单 / 偏好），弹窗是「针对眼前这一件（几件）商品的动作」，看完就关。
//
// 底座换成 Radix Dialog：焦点圈养、Esc / 点遮罩关闭、滚动锁、aria 全由它管，不再手写 keydown。
// 进出场交给 motion：Radix 关闭即卸载，所以用 forceMount + AnimatePresence 让退场动画放完再拆 DOM。
// 遮罩与内容在 Radix 里是兄弟节点（都直接挂 body），CSS 里 .modal 自己定位居中，不再依赖遮罩的 flex。
const overlayMotion = {
  initial: { opacity: 0 },
  animate: { opacity: 1 },
  exit: { opacity: 0 },
  transition: { duration: 0.18 },
};

const panelMotion = {
  initial: { opacity: 0, y: 14, scale: 0.98 },
  animate: { opacity: 1, y: 0, scale: 1 },
  exit: { opacity: 0, y: 8, scale: 0.985, transition: { duration: 0.14 } },
  transition: { type: "spring", stiffness: 480, damping: 36, mass: 0.7 },
} as const;

export function Modal({
  open,
  title,
  wide,
  onClose,
  children,
}: {
  open: boolean;
  title: string;
  wide?: boolean;
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
              <motion.div className="modal-scrim" {...overlayMotion} />
            </Dialog.Overlay>
            {/* 不自动聚焦第一个可点元素（那是关闭键，一打开就顶着个焦点环）；焦点落在面板本身，Esc / Tab 照常。 */}
            <Dialog.Content
              asChild
              forceMount
              aria-describedby={undefined}
              onOpenAutoFocus={(e) => e.preventDefault()}
            >
              <motion.div className={`modal ${wide ? "modal-wide" : ""}`} {...panelMotion}>
                <div className="modal-head">
                  <Dialog.Title className="modal-title">{title}</Dialog.Title>
                  <Dialog.Close asChild>
                    <button className="icon-btn" aria-label="关闭">
                      <X size={18} strokeWidth={1.75} />
                    </button>
                  </Dialog.Close>
                </div>
                <div className="modal-body">{children}</div>
              </motion.div>
            </Dialog.Content>
          </Dialog.Portal>
        )}
      </AnimatePresence>
    </Dialog.Root>
  );
}
