import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev: Vite on 127.0.0.1:5173 proxies /api to ksmech-serve on 127.0.0.1:8000 (no CORS needed).
// Build: dist/ is served by `ksmech-serve` itself (static mount after the /api routers).
// public/vendor/ (RDKit_minimal.{js,wasm}, 3Dmol-min.js) is filled by scripts/vendor.mjs.
export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    proxy: { "/api": "http://127.0.0.1:8000" },
  },
  preview: { host: "127.0.0.1", port: 4173, strictPort: true },
  build: { outDir: "dist", emptyOutDir: true, chunkSizeWarningLimit: 2048 },
});
