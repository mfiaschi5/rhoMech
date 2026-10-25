// The webServer command of the e2e smoke test (design §13.3); run with cwd apps/ui.
//
// 1. Reset the data root apps/ui/.e2e-data (gitignored, outside test-results/; .e2e-data/port-<p>
//    for a KSMECH_E2E_PORT other than 8765, so a parallel run has its own) unless
//    KSMECH_E2E_KEEP=1.
// 2. Build dist/ when dist/index.html is missing or older than the newest build input (BUILD_INPUTS:
//    src/**, scripts/**, index.html, vite.config.ts, tsconfig.json, package.json,
//    package-lock.json), logging the file that triggered it; exit non-zero when the build fails.
//    KSMECH_E2E_REUSE_DIST=1 skips the staleness check (quick reruns) but never a missing dist/.
// 3. Choose the chemistry: KSMECH_E2E_CHEMISTRY, default "auto". "auto" runs
//    `ksmech-serve probe-cache --smiles "C[O-]" --smiles CCl --smiles CCO --smiles <camphor>`
//    against the repo's shared token cache: exit 0 -> "cached" (both smoke reactants and the
//    workbench tests' ethanol and camphor have real GFN2 tokens that load; cached mode never
//    computes a new molecule),
//    anything else -> "synthetic". "real" is accepted only when set explicitly (user
//    acceptance runs).
// 4. Write <dataRoot>/chemistry.txt and print "e2e chemistry: <c>".
// 5. Spawn `uv run --no-sync --project ../.. ksmech-serve --port <port> --chemistry <c>` (port:
//    KSMECH_E2E_PORT, default 8765) with one
//    compute worker, no auto index, no stub predictor and one BLAS/OpenMP thread; forward
//    SIGINT/SIGTERM to it and exit with its code.
// The steps run only when this file is the node entry point; importing it (tests) gives
// buildPlan() and newestInput() without side effects.
import { spawn, spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readdirSync, realpathSync, rmSync, statSync, writeFileSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const uiDir = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const repoRoot = resolve(uiDir, "..", "..");
const PORT = String(Number.parseInt(process.env.KSMECH_E2E_PORT ?? "", 10) || 8765);
const dataRoot = PORT === "8765" ? join(uiDir, ".e2e-data") : join(uiDir, ".e2e-data", `port-${PORT}`);
const CHEMISTRIES = ["auto", "cached", "synthetic", "real"];
// The smoke test's reactants, the workbench test's ethanol and the projection test's camphor
// (e2e/workbench.spec.ts).
const PROBE_SMILES = ["C[O-]", "CCl", "CCO", "CC1(C)[C@@H]2CC[C@@]1(C)C(=O)C2"];
// What `npm run build` (tsc --noEmit && vite build, after scripts/vendor.mjs) reads besides node_modules.
const BUILD_INPUTS = ["src", "scripts", "index.html", "vite.config.ts", "tsconfig.json", "package.json", "package-lock.json"];

const serveEnv = {
  ...process.env,
  KSMECH_DATA_ROOT: dataRoot,
  KSMECH_SHARED_CACHE_ROOT: repoRoot,
  KSMECH_SERVE_WORKERS: "1",
  KSMECH_SERVE_AUTO_INDEX: "0",
  KSMECH_SERVE_STUB_PREDICTOR: "0",
  OMP_NUM_THREADS: "1",
  OPENBLAS_NUM_THREADS: "1",
  MKL_NUM_THREADS: "1",
};
const uv = (args) => ["run", "--no-sync", "--project", repoRoot, "ksmech-serve", ...args];

function log(message) {
  console.log(`[e2e/serve] ${message}`);
}

function resetDataRoot() {
  if (process.env.KSMECH_E2E_KEEP === "1") {
    log(`keeping ${dataRoot} (KSMECH_E2E_KEEP=1)`);
  } else {
    rmSync(dataRoot, { recursive: true, force: true });
  }
  mkdirSync(dataRoot, { recursive: true });
}

/** The newest file under `path` (a file, or a directory walked recursively) as {path, mtimeMs};
 *  null when nothing exists there. */
function newestFile(path) {
  if (!existsSync(path)) return null;
  const st = statSync(path);
  if (!st.isDirectory()) return { path, mtimeMs: st.mtimeMs };
  let newest = null;
  for (const name of readdirSync(path)) {
    const f = newestFile(join(path, name));
    if (f && (!newest || f.mtimeMs > newest.mtimeMs)) newest = f;
  }
  return newest;
}

/** The newest of the BUILD_INPUTS of the app in `dir`, or null. */
export function newestInput(dir = uiDir) {
  let newest = null;
  for (const input of BUILD_INPUTS) {
    const f = newestFile(join(dir, input));
    if (f && (!newest || f.mtimeMs > newest.mtimeMs)) newest = f;
  }
  return newest;
}

/** Whether dist/ of the app in `dir` must be built, and why: {build, reason}. */
export function buildPlan(dir = uiDir, env = process.env) {
  const index = join(dir, "dist", "index.html");
  if (!existsSync(index)) return { build: true, reason: "dist/index.html is missing" };
  if (env.KSMECH_E2E_REUSE_DIST === "1") {
    return { build: false, reason: "KSMECH_E2E_REUSE_DIST=1: dist/ reused without a staleness check" };
  }
  const built = statSync(index).mtimeMs;
  const newest = newestInput(dir);
  if (newest && newest.mtimeMs > built) {
    return { build: true, reason: `${relative(dir, newest.path)} is newer than dist/index.html` };
  }
  return { build: false, reason: "dist/index.html is newer than every build input" };
}

function ensureBuild() {
  const plan = buildPlan();
  log(plan.build ? `${plan.reason}: npm run build` : `${plan.reason}: no rebuild`);
  if (!plan.build) return;
  const r = spawnSync("npm", ["run", "build"], { cwd: uiDir, stdio: "inherit" });
  if (r.status !== 0) {
    console.error(`[e2e/serve] npm run build failed (exit ${r.status ?? r.signal})`);
    process.exit(r.status || 1);
  }
}

function chooseChemistry() {
  const asked = process.env.KSMECH_E2E_CHEMISTRY ?? "auto";
  if (!CHEMISTRIES.includes(asked)) {
    console.error(`[e2e/serve] KSMECH_E2E_CHEMISTRY must be one of ${CHEMISTRIES.join(", ")}`);
    process.exit(2);
  }
  if (asked !== "auto") return asked;
  const probe = spawnSync("uv", uv(["probe-cache", ...PROBE_SMILES.flatMap((smi) => ["--smiles", smi])]), {
    cwd: uiDir,
    env: serveEnv,
    encoding: "utf8",
  });
  const out = `${probe.stdout ?? ""}${probe.stderr ?? ""}`.trim();
  if (out) log(`probe-cache (exit ${probe.status}):\n${out}`);
  return probe.status === 0 ? "cached" : "synthetic";
}

function main() {
  resetDataRoot();
  ensureBuild();
  const chemistry = chooseChemistry();
  writeFileSync(join(dataRoot, "chemistry.txt"), `${chemistry}\n`);
  console.log(`e2e chemistry: ${chemistry}`);

  const child = spawn("uv", uv(["--port", PORT, "--chemistry", chemistry]), {
    cwd: uiDir,
    env: serveEnv,
    stdio: "inherit",
  });
  for (const signal of ["SIGINT", "SIGTERM"]) {
    process.on(signal, () => {
      if (child.exitCode === null) child.kill(signal);
    });
  }
  child.on("error", (err) => {
    console.error(`[e2e/serve] cannot start ksmech-serve: ${err.message}`);
    process.exit(1);
  });
  child.on("exit", (code, signal) => {
    log(`ksmech-serve exited (${code ?? signal})`);
    process.exit(code ?? (signal ? 1 : 0));
  });
}

/** True when node runs this file (`node e2e/serve.mjs`), false when it is imported. */
function isEntryPoint() {
  try {
    return !!process.argv[1] && realpathSync(process.argv[1]) === realpathSync(fileURLToPath(import.meta.url));
  } catch {
    return false;
  }
}

if (isEntryPoint()) main();
