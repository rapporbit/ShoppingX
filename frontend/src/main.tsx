import React from "react";
import ReactDOM from "react-dom/client";
import { MotionConfig } from "motion/react";
import { Toaster } from "sonner";
import App from "./App";
import { TooltipProvider } from "./components/ui/Tooltip";
import "./styles.css";

// Toaster 是全站唯一的轻提示出口（收藏 / 复制 / 请求失败这类「说一声就够」的反馈）。
// 单色系：不用 sonner 的默认彩色主题，靠 toastOptions.className 接自家变量。
ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    {/* reducedMotion="user"：系统开了「减弱动态效果」时，motion 的位移 / 缩放动画自动退化成淡入淡出。 */}
    <MotionConfig reducedMotion="user">
      <TooltipProvider delayDuration={350} skipDelayDuration={200}>
        <App />
        <Toaster
          position="top-center"
          offset={16}
          gap={8}
          duration={2400}
          visibleToasts={3}
          toastOptions={{ className: "toast", unstyled: true }}
        />
      </TooltipProvider>
    </MotionConfig>
  </React.StrictMode>,
);
