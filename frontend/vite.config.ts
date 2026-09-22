import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// 开发期前端跑在 :5173，后端跑在 :8000。把 /api 与 /ws 反向代理到后端，
// 这样前端代码里全用同源相对路径（/api/...、ws://<本机>/ws/...），不必硬编码后端地址、
// 也绕过浏览器跨域。生产部署可换成 nginx 同款代理或把前端构建产物交给后端托管。
export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        // 三方库单独成包：它们几乎不变，业务代码每次部署都变。不拆的话一个 190KB(gzip) 的整包每次部署
        // 全量失效；拆开后回访用户只重下业务那一包（约 40KB）。react 与 radix 互相引用，必须放同一包。
        manualChunks(id) {
          if (!id.includes("node_modules") || id.includes("@fontsource")) return;
          if (/[\\/]node_modules[\\/](motion|framer-motion|motion-dom|motion-utils)[\\/]/.test(id)) return "vendor-motion";
          return "vendor";
        },
      },
    },
  },
  server: {
    // 钉死 IPv4：不写 host 时 vite 绑 localhost，macOS 上 Node 解析成 ::1，127.0.0.1 就连不上。
    host: "127.0.0.1",
    port: 5173,
    proxy: {
      "/api": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/ws": { target: "ws://127.0.0.1:8000", ws: true },
    },
  },
});
