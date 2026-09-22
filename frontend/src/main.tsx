import React from "react";
import ReactDOM from "react-dom/client";
import { MotionConfig } from "motion/react";
import { Toaster } from "sonner";
import App from "./App";
import { TooltipProvider } from "./components/ui/Tooltip";
// 字体随构建打包（不走外部 CDN）。中文衬线按 unicode-range 切片，浏览器只下用到的那几片。
import "@fontsource-variable/inter/wght.css";
import "@fontsource-variable/newsreader/wght.css";
import "./styles.css";

// 中文衬线两个字重各 101 片切片 = 202 条 @font-face，约 200KB 的声明。静态 import 会并进主 CSS，
// 而主 CSS 阻塞首屏渲染；动态 import 让 Vite 拆成独立的 CSS 包、异步加载。字体本来就是
// font-display: swap，晚到的这一小会儿先用系统衬线顶着。
void import("@fontsource/noto-serif-sc/400.css");
void import("@fontsource/noto-serif-sc/500.css");

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
