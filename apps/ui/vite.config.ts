import { defineConfig, searchForWorkspaceRoot } from "vite";
import react from "@vitejs/plugin-react";
// @ts-ignore TS7016: a plain .mjs module (no declarations), shared with the node test runner
import { VIZ_DIR, vizResolve } from "./scripts/viz-alias.mjs";

// Dev: Vite on 127.0.0.1:5173 proxies /api to ksmech-serve on 127.0.0.1:8000 (no CORS needed).
// Build: dist/ is served by `ksmech-serve` itself (static mount after the /api routers).
// public/vendor/ (RDKit_minimal.{js,wasm}, 3Dmol-min.js) is filled by scripts/vendor.mjs.
// @ksmech/viz is the source package packages/viz, resolved by alias (scripts/viz-alias.mjs);
// the dev server may read it (fs.allow) although it lies outside apps/ui.
export default defineConfig({
  plugins: [react()],
  resolve: vizResolve(),
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    proxy: { "/api": "http://127.0.0.1:8000" },
    fs: { allow: [searchForWorkspaceRoot(process.cwd()), VIZ_DIR] },
  },
  preview: { host: "127.0.0.1", port: 4173, strictPort: true },
  build: { outDir: "dist", emptyOutDir: true, chunkSizeWarningLimit: 2048 },
});
