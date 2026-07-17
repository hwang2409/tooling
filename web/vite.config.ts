import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dev server and the built-asset preview both forward the versioned API
// (including the WebSocket stream upgrade) to the local mitm-inspector app
// server, so the browser sees one loopback origin.
const apiTarget = process.env.MITM_INSPECTOR_API_TARGET ?? "http://127.0.0.1:8000";
const apiProxy = {
  "/api/v1": {
    target: apiTarget,
    changeOrigin: false,
    ws: true,
  },
} as const;

export default defineConfig({
  plugins: [react()],
  server: { proxy: apiProxy },
  preview: { proxy: apiProxy },
});
