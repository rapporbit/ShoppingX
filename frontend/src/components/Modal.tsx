import { useEffect, type ReactNode } from "react";
import { CloseIcon } from "./icons";

// 居中弹窗（商品详情 / 对比 / 下单意向表单共用）。与右侧抽屉（.drawer）的分工：抽屉是「另一份
// 清单」（收藏 / 订单 / 偏好），弹窗是「针对眼前这一件（几件）商品的动作」，看完就关。
// 常驻 DOM 与否由调用方决定：这里 open=false 直接不渲染，弹窗内容每次打开都是新的。
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
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;
  return (
    <div className="modal-scrim" onClick={onClose}>
      <div
        className={`modal ${wide ? "modal-wide" : ""}`}
        role="dialog"
        aria-modal="true"
        aria-label={title}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-head">
          <div className="modal-title">{title}</div>
          <button className="icon-btn" onClick={onClose} title="关闭">
            <CloseIcon width={18} height={18} />
          </button>
        </div>
        <div className="modal-body">{children}</div>
      </div>
    </div>
  );
}
