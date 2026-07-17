import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// 开发期前端跑在 :5173，后端跑在 :8000。把 /api 与 /ws 反向代理到后端，
// 这样前端代码里全用同源相对路径（/api/...、ws://<本机>/ws/...），不必硬编码后端地址、
// 也绕过浏览器跨域。生产部署可换成 nginx 同款代理或把前端构建产物交给后端托管。
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/ws": { target: "ws://127.0.0.1:8000", ws: true },
    },
  },
});
