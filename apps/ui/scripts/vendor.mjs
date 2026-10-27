// Copy the classic-script builds of RDKit.js and 3Dmol.js into public/vendor/, where
// index.html loads them with plain <script> tags (no ESM bundling of the wasm loader).
// Runs before `npm run dev` and `npm run build`; exits 1 with a clear message on a missing file.
import { copyFileSync, existsSync, mkdirSync, statSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const modules = join(root, "node_modules");
const out = join(root, "public", "vendor");

const copies = [
  { from: ["@rdkit/rdkit/dist/RDKit_minimal.js"], to: "RDKit_minimal.js" },
  { from: ["@rdkit/rdkit/dist/RDKit_minimal.wasm"], to: "RDKit_minimal.wasm" },
  // 3Dmol-min.js is the UMD build that sets window.$3Dmol; the unminified one is the fallback.
  { from: ["3dmol/build/3Dmol-min.js", "3dmol/build/3Dmol.js"], to: "3Dmol-min.js" },
];

function fail(message) {
  console.error(`vendor: ${message}`);
  console.error("vendor: run `PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1 npm --prefix apps/ui ci` first");
  process.exit(1);
}

mkdirSync(out, { recursive: true });
for (const { from, to } of copies) {
  const source = from.map((f) => join(modules, f)).find((f) => existsSync(f));
  if (!source) fail(`none of ${from.map((f) => `node_modules/${f}`).join(", ")} exists`);
  const size = statSync(source).size;
  if (size === 0) fail(`${relative(root, source)} is empty`);
  copyFileSync(source, join(out, to));
  console.log(`vendor: ${relative(root, source)} -> public/vendor/${to} (${size} bytes)`);
}
