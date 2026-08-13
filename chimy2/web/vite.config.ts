import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Static deployment: relative asset paths so the bundle works when served from any
// subdirectory (GitHub Pages, plain S3, or a webroot pointed at dist/).
export default defineConfig({
  plugins: [react()],
  base: "./",
  build: {
    target: "es2022",
    sourcemap: false,
    assetsInlineLimit: 0,
    reportCompressedSize: false,
    rollupOptions: {
      output: {
        entryFileNames: "assets/[name]-[hash].js",
        chunkFileNames: "assets/[name]-[hash].js",
        assetFileNames: "assets/[name]-[hash][extname]",
      },
    },
  },
  server: {
    port: 5173,
    strictPort: true,
  },
});
