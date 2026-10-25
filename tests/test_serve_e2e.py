"""The Playwright smoke test of the web app (design 13.3) and unit tests of its harness.

- `test_playwright_smoke` runs `npm --prefix apps/ui run e2e`. Skipped unless KSMECH_E2E=1, and
  skipped with the reason when no system Chrome/Chromium exists (Playwright browsers are never
  downloaded). The harness builds the frontend when dist/ is missing or stale, starts
  `ksmech-serve --port 8765` (one compute worker, cached or synthetic chemistry, never xTB) and
  drives the browser.
- Without KSMECH_E2E: the browser lookup, e2e/serve.mjs's build plan (through node), and the UI
  probe. The probe bundles PROBE_ENTRY with vite (SSR build into a tmp dir, one thread), which
  imports modules of apps/ui/src, renders components with react-dom/server and returns JSON for
  the asserts. Node tests skip with the reason when node or apps/ui/node_modules is missing.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.test_serve_fixtures import no_xtb  # noqa: F401 (fixture)

pytestmark = pytest.mark.serve

REPO = Path(__file__).resolve().parents[1]
UI = REPO / "apps" / "ui"
SKIP_REASON = "set KSMECH_E2E=1 to run Playwright with a system Chrome"
TIMEOUT_S = 1500
NODE_TIMEOUT_S = 300
THREADS = dict.fromkeys(("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"), "1")
# rolldown (vite 8) runs rayon and tokio pools, libuv its own: one thread each
NODE_THREADS = THREADS | dict.fromkeys(
    ("RAYON_NUM_THREADS", "TOKIO_WORKER_THREADS", "UV_THREADPOOL_SIZE"), "1"
)
# the same list and text as apps/ui/e2e/browser.ts (test_browser_lookup_matches_browser_ts)
BROWSER_CANDIDATES = ("/usr/bin/google-chrome", "/usr/bin/chromium-browser", "/usr/bin/chromium")
NO_BROWSER_REASON = (
    "no system Chrome/Chromium at /usr/bin/google-chrome, /usr/bin/chromium-browser or "
    "/usr/bin/chromium; this repo never downloads Playwright browsers"
)
JSON_MARK = "KSMECH_JSON "


def _system_browser(candidates: tuple[str, ...] = BROWSER_CANDIDATES) -> str | None:
    """The first path of `candidates` that exists, as e2e/browser.ts picks it; None if none."""
    return next((c for c in candidates if Path(c).exists()), None)


@pytest.mark.e2e
@pytest.mark.skipif(os.environ.get("KSMECH_E2E") != "1", reason=SKIP_REASON)
def test_playwright_smoke():
    npm = shutil.which("npm")
    if npm is None:
        pytest.skip("npm is not on PATH")
    if not (UI / "node_modules" / "@playwright" / "test").is_dir():
        pytest.skip("apps/ui/node_modules is missing: PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1 npm ci")
    if _system_browser() is None:
        pytest.skip(NO_BROWSER_REASON)
    env = os.environ | THREADS | {"PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD": "1"}
    proc = subprocess.run(
        [npm, "--prefix", str(UI), "run", "e2e"],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
        check=False,
    )
    output = proc.stdout + proc.stderr
    print(output)
    if proc.returncode == 0 and " passed" not in output and " skipped" in output:
        pytest.skip(f"Playwright skipped every test (see the [e2e] lines):\n{output[-2000:]}")
    assert proc.returncode == 0, output[-6000:]
    assert " passed" in output


# ---- the browser lookup


def test_system_browser_none_when_no_candidate_exists(tmp_path):
    missing = (str(tmp_path / "google-chrome"), str(tmp_path / "chromium"))
    assert _system_browser(missing) is None
    assert _system_browser(()) is None


def test_system_browser_returns_the_first_existing(tmp_path):
    chromium = tmp_path / "chromium"
    chromium.write_text("#!/bin/sh\n")
    other = tmp_path / "chromium-browser"
    other.write_text("#!/bin/sh\n")
    candidates = (str(tmp_path / "google-chrome"), str(chromium), str(other))
    assert _system_browser(candidates) == str(chromium)


def test_browser_lookup_matches_browser_ts():
    ts = (UI / "e2e" / "browser.ts").read_text(encoding="utf-8")
    listed = re.search(r"BROWSER_CANDIDATES = \[(.*?)\] as const", ts, re.S)
    assert listed, "BROWSER_CANDIDATES not found in e2e/browser.ts"
    assert tuple(re.findall(r'"([^"]+)"', listed.group(1))) == BROWSER_CANDIDATES
    reason = re.search(r"NO_BROWSER_REASON =\s*((?:\"[^\"]*\"\s*\+?\s*)+);", ts)
    assert reason, "NO_BROWSER_REASON not found in e2e/browser.ts"
    assert "".join(re.findall(r'"([^"]*)"', reason.group(1))) == NO_BROWSER_REASON


# ---- node helpers


def _node(script: str, env: dict[str, str] | None = None) -> str:
    """Run the ES module `script` with node in apps/ui (bare imports resolve from its
    node_modules), single-threaded; returns stdout and fails with the output on a non-zero exit."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not on PATH")
    if not (UI / "node_modules" / "vite").is_dir():
        pytest.skip("apps/ui/node_modules is missing: PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1 npm ci")
    proc = subprocess.run(
        [node, "--input-type=module", "-e", script],
        cwd=UI,
        env=os.environ | NODE_THREADS | (env or {}),
        capture_output=True,
        text=True,
        timeout=NODE_TIMEOUT_S,
        check=False,
    )
    assert proc.returncode == 0, (
        f"node exited {proc.returncode}:\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}"
    )
    return proc.stdout


def _json_out(stdout: str):
    """The value of the last JSON_MARK line of `stdout`."""
    lines = [ln for ln in stdout.splitlines() if ln.startswith(JSON_MARK)]
    assert lines, f"no {JSON_MARK.strip()} line in:\n{stdout[-4000:]}"
    return json.loads(lines[-1][len(JSON_MARK) :])


# ---- e2e/serve.mjs: rebuild a missing or stale dist/

T0 = 1_700_000_000
INPUTS = (
    "src/api/client.ts",
    "src/features/curate/ExportReport.tsx",
    "scripts/vendor.mjs",
    "index.html",
    "vite.config.ts",
    "tsconfig.json",
    "package.json",
    "package-lock.json",
)


def _build_plan(ui_dir: Path, env: dict[str, str] | None = None) -> dict:
    serve = (UI / "e2e" / "serve.mjs").as_uri()
    script = (
        f"import {{ buildPlan }} from {json.dumps(serve)};\n"
        f"const plan = buildPlan({json.dumps(str(ui_dir))}, {json.dumps(env or {})});\n"
        f"console.log({json.dumps(JSON_MARK)} + JSON.stringify(plan));\n"
    )
    return _json_out(_node(script))


def _touch(path: Path, t: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("x\n")
    os.utime(path, (t, t))


def _fake_ui(root: Path) -> Path:
    for rel in INPUTS:
        _touch(root / rel, T0)
    return root


def test_serve_mjs_builds_a_missing_dist(tmp_path):
    ui = _fake_ui(tmp_path)
    assert _build_plan(ui) == {"build": True, "reason": "dist/index.html is missing"}
    # KSMECH_E2E_REUSE_DIST skips the staleness check, never a missing dist/
    assert _build_plan(ui, {"KSMECH_E2E_REUSE_DIST": "1"})["build"] is True


def test_serve_mjs_rebuilds_a_stale_dist(tmp_path):
    ui = _fake_ui(tmp_path)
    _touch(ui / "dist" / "index.html", T0 + 100)
    current = _build_plan(ui)
    assert current == {"build": False, "reason": "dist/index.html is newer than every build input"}
    for rel in ("src/features/curate/ExportReport.tsx", "package-lock.json", "scripts/vendor.mjs"):
        _touch(ui / rel, T0 + 200)
        plan = _build_plan(ui)
        assert plan == {"build": True, "reason": f"{rel} is newer than dist/index.html"}
        reused = _build_plan(ui, {"KSMECH_E2E_REUSE_DIST": "1"})
        assert reused["build"] is False and "KSMECH_E2E_REUSE_DIST=1" in reused["reason"]
        _touch(ui / rel, T0)
    _touch(ui / "src" / "features" / "new" / "Page.tsx", T0 + 300)  # a new file deep in src/
    assert _build_plan(ui)["reason"] == "src/features/new/Page.tsx is newer than dist/index.html"


# ---- the UI probe: frontend logic without a browser

PROBE_RUNNER = """
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import react from "@vitejs/plugin-react";
import { build } from "vite";

const cfg = JSON.parse(process.env.KSMECH_UI_PROBE);
const modules = join(process.cwd(), "node_modules");
await build({
  configFile: false,
  root: process.cwd(),
  logLevel: "warn",
  publicDir: false,
  plugins: [react()],
  resolve: {
    alias: [
      { find: /^react$/, replacement: join(modules, "react", "index.js") },
      { find: /^react-dom\\/server$/, replacement: join(modules, "react-dom", "server.node.js") },
      { find: /^@playwright\\/test$/, replacement: cfg.playwrightStub },
    ],
  },
  ssr: { noExternal: true, target: "node" },
  build: {
    ssr: cfg.entry,
    outDir: cfg.outDir,
    emptyOutDir: true,
    copyPublicDir: false,
    minify: false,
    rolldownOptions: { output: { entryFileNames: "probe.mjs" } },
  },
});
const probe = await import(pathToFileURL(join(cfg.outDir, "probe.mjs")).href);
console.log(cfg.mark + JSON.stringify(await probe.run()));
"""

# @UI@ is apps/ui. React and react-dom/server resolve to apps/ui's copies through the runner's
# aliases, so the components and the renderer share one React.
PROBE_ENTRY = """
import { createElement as h } from "react";
import { renderToStaticMarkup as html } from "react-dom/server";
import { warnNoBrowser } from "@UI@/playwright.config.ts";
import { cached, invalidate, onInvalidate, refresh } from "@UI@/src/api/cache.ts";
import {
  ExportPanel,
  ExportReportView,
  ExportWithdrawWarning,
} from "@UI@/src/features/curate/ExportReport.tsx";
import { MechanismWarnings, StaleBanner } from "@UI@/src/features/curate/MechanismView.tsx";
import { curateRequest } from "@UI@/src/features/curate/PathwayCurate.tsx";
import { stylesheet } from "@UI@/src/features/predict/NetworkView.tsx";
import { parseHash } from "@UI@/src/router.ts";

const DARK: Record<string, string> = {
  "--fg": "#e3e8ef",
  "--panel": "#1a2029",
  "--turnover": "#3cc9c9",
};

async function cacheRetry() {
  const key = "/probe/fails-once";
  let loads = 0;
  const fail = () => {
    loads += 1;
    return Promise.reject(new Error("boom"));
  };
  await cached(key, fail).catch(() => undefined);
  await Promise.resolve();
  const notified: string[] = [];
  const off = onInvalidate((k) => notified.push(k));
  const droppedByInvalidate = invalidate((k) => k === key);
  refresh(key);
  off();
  const value = await cached(key, () => {
    loads += 1;
    return Promise.resolve(42);
  });
  return { droppedByInvalidate, notified, loads, value };
}

function networkStyles() {
  const g = globalThis as Record<string, unknown>;
  g.document = { documentElement: {} };
  g.getComputedStyle = () => ({ getPropertyValue: (name: string) => DARK[name] ?? "" });
  try {
    const sheet = stylesheet() as unknown as { selector: string; style: Record<string, unknown> }[];
    const style = (selector: string) => sheet.find((s) => s.selector === selector)?.style ?? null;
    return { edge: style("edge"), virtual: style("edge.turnover-virtual") };
  } finally {
    delete g.document;
    delete g.getComputedStyle;
  }
}

function mechanism() {
  const warnings = [
    "INDEPENDENCE_UNCHECKED: steps 0 and 1 of order group 0 are joined by a merge",
    "NOT_INDEPENDENT: steps 1 and 2 of order group 1 share map 3",
  ];
  const stale_reasons = ["spec_version v3+aaaa != v3+bbbb"];
  return {
    warnings: html(h(MechanismWarnings, { warnings })),
    noWarnings: html(h(MechanismWarnings, { warnings: [] })),
    staleNever: html(h(StaleBanner, { m: { stale_reasons, last_recheck: null } })),
    staleRechecked: html(
      h(StaleBanner, { m: { stale_reasons, last_recheck: "2026-10-21T10:00:00Z" } }),
    ),
    notStale: html(h(StaleBanner, { m: { stale_reasons: [], last_recheck: null } })),
  };
}

function curate() {
  const edgeIds = ["e0000000000000001", "e0000000000000002"];
  const body = curateRequest("j0123456789abcdef", edgeIds, "validate", "ok", [0, 1], "  ");
  edgeIds.push("e0000000000000003");
  return { body, withConditions: curateRequest("j0", ["e1"], "save", "", [0], " MeOH, 25 C ") };
}

function exportViews() {
  const report = {
    export_id: "x0123456789abcdef",
    created_at: "2026-10-21T10:00:00Z",
    spec_version: "v3+",
    paths: {},
    n_mechanisms: 0,
    n_steps_considered: 0,
    n_steps_written: 0,
    skipped: {},
    ingest_rejections: {},
    duplicates_pmechdb: 0,
    duplicates_in_export: 0,
    withdrawn: 0,
    changed_ids: 0,
    steps: [],
    sha256: "0",
  };
  const pmechdb = "PMECHDB_UNAVAILABLE: no PMechDB steps; no PMechDB de-duplication";
  const warned = { ...report, warnings: [pmechdb] };
  return {
    panel: html(h(ExportPanel, { route: parseHash("#/curate") })),
    withdrawValidated: html(h(ExportWithdrawWarning, { chosen: ["corrected"] })),
    withdrawNone: html(h(ExportWithdrawWarning, { chosen: ["validated", "corrected"] })),
    reportWarnings: html(h(ExportReportView, { report: warned })),
    reportPlain: html(h(ExportReportView, { report })),
  };
}

function playwright() {
  const printed: string[] = [];
  const none = warnNoBrowser(undefined, (m) => printed.push(m));
  const found = warnNoBrowser("/usr/bin/chromium-browser", (m) => printed.push(m));
  return { none, found, printed };
}

export async function run() {
  return {
    cache: await cacheRetry(),
    network: networkStyles(),
    mechanism: mechanism(),
    curate: curate(),
    export: exportViews(),
    playwright: playwright(),
  };
}
"""


@pytest.fixture(scope="module")
def ui_probe(tmp_path_factory) -> dict:
    """PROBE_ENTRY bundled (vite SSR build into a tmp dir, @playwright/test stubbed) and run
    once: the JSON its run() returns. Nothing is written under apps/ui."""
    out = tmp_path_factory.mktemp("ui_probe")
    entry = out / "probe_entry.ts"
    entry.write_text(PROBE_ENTRY.replace("@UI@", UI.as_posix()), encoding="utf-8")
    stub = out / "playwright_stub.mjs"
    stub.write_text("export const defineConfig = (config) => config;\n", encoding="utf-8")
    cfg = {
        "entry": str(entry),
        "outDir": str(out / "bundle"),
        "playwrightStub": str(stub),
        "mark": JSON_MARK,
    }
    return _json_out(_node(PROBE_RUNNER, {"KSMECH_UI_PROBE": json.dumps(cfg)}))


def test_cache_refresh_reloads_after_a_failed_load(ui_probe):
    # cached() drops a rejected entry, so invalidate() finds nothing and notifies nobody: the
    # PhysicsPanel "retry" did nothing. refresh() (useCached.reload) notifies the readers anyway.
    c = ui_probe["cache"]
    assert c["droppedByInvalidate"] == 0
    assert c["notified"] == ["/probe/fails-once"]
    assert c["loads"] == 2 and c["value"] == 42


def test_network_edge_labels_follow_the_theme(ui_probe):
    edge, virtual = ui_probe["network"]["edge"], ui_probe["network"]["virtual"]
    assert edge["color"] == "#e3e8ef"
    assert edge["text-background-color"] == "#1a2029"  # --panel, not a fixed white box
    assert virtual["color"] == "#3cc9c9"  # turnover edges keep their own colour


def test_mechanism_warnings_carry_the_bare_code(ui_probe):
    out = ui_probe["mechanism"]["warnings"]
    assert 'data-code="INDEPENDENCE_UNCHECKED"' in out
    assert 'data-code="NOT_INDEPENDENT"' in out
    assert "<code>INDEPENDENCE_UNCHECKED: steps 0 and 1 of order group 0 are joined" in out
    assert "a merge broke map continuity" in out
    assert out.count("a merge broke map continuity") == 1  # only for INDEPENDENCE_UNCHECKED
    assert ui_probe["mechanism"]["noWarnings"] == ""


def test_stale_banner_after_a_recheck(ui_probe):
    m = ui_probe["mechanism"]
    assert "Re-check it." in m["staleNever"] and "Last re-checked" not in m["staleNever"]
    assert 'data-rechecked="false"' in m["staleNever"]
    rechecked = m["staleRechecked"]
    assert "Re-check it." not in rechecked
    assert "Last re-checked " in rechecked
    assert "the stored versions are never rewritten, so see the re-check rows below." in rechecked
    assert 'data-rechecked="true"' in rechecked
    assert "spec_version v3+aaaa != v3+bbbb" in rechecked
    assert m["notStale"] == ""


def test_curate_posts_the_edge_chain_not_pathway_id(ui_probe):
    body = ui_probe["curate"]["body"]
    assert "pathway_id" not in body
    assert body == {
        "job_id": "j0123456789abcdef",
        "edge_ids": ["e0000000000000001", "e0000000000000002"],  # a copy of the chain
        "action": "validate",
        "notes": "ok",
        "order_groups": [0, 1],
        "conditions": None,
    }
    assert ui_probe["curate"]["withConditions"]["conditions"] == "MeOH, 25 C"


def test_export_panel_says_what_a_download_holds(ui_probe):
    e = ui_probe["export"]
    assert "built from the database on every download" in e["panel"]
    assert "Rejected mechanisms are never exported" in e["panel"]
    assert "not training data" in e["panel"]
    assert "export-withdraw-warning" not in e["panel"]  # both statuses are checked at first
    warn = e["withdrawValidated"]
    assert 'class="cu-hint-warn"' in warn and 'data-unchecked="validated"' in warn
    assert "validated mechanisms are left out of this download" in warn
    assert e["withdrawNone"] == ""


def test_export_report_shows_its_warnings(ui_probe):
    e = ui_probe["export"]
    assert 'data-testid="export-warnings"' in e["reportWarnings"]
    assert 'data-code="PMECHDB_UNAVAILABLE"' in e["reportWarnings"]
    assert "no PMechDB de-duplication" in e["reportWarnings"]
    assert "export-warnings" not in e["reportPlain"]


def test_playwright_config_prints_the_no_browser_reason(ui_probe):
    p = ui_probe["playwright"]
    assert p["none"] is True and p["found"] is False
    assert p["printed"] == [f"[e2e] skipping: {NO_BROWSER_REASON}"]
