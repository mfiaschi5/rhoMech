// @ts-nocheck -- run by `npm --prefix apps/ui test` (node --experimental-strip-types --test).
// The package boundary of @ksmech/viz: its components are props-only, so nothing under
// packages/viz may import the app (apps/ui: its API client, store, router, pages), import a
// package outside its peer dependencies, or talk to a server. The one link to the app is
// apiTypes.ts, a type-only re-export of the generated API schemas (erased at runtime). The one
// module that fetches is scene/load.ts, and it reads static scene files, never the API.
import test from "node:test";
import assert from "node:assert/strict";
import { readdirSync, readFileSync, statSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const SRC = dirname(fileURLToPath(import.meta.url));
const VIZ = resolve(SRC, "..");
const PEERS = ["react", "react-dom", "@rdkit/rdkit", "3dmol"];
const BRIDGE = { file: "apiTypes.ts", spec: "../../../apps/ui/src/api/types.gen" };
const FETCHER = "scene/load.ts";

function files(dir, out = []) {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) files(p, out);
    else if (/\.(ts|tsx|mjs|js)$/.test(name)) out.push(p);
  }
  return out;
}

// this file is left out: its own regular expressions read as import specifiers
const SOURCES = files(SRC)
  .filter((p) => p !== fileURLToPath(import.meta.url))
  .map((p) => ({ path: relative(SRC, p), text: readFileSync(p, "utf8"), test: p.endsWith(".test.ts") }));

/** The code without comments (string contents kept). */
const code = (text) => text.replace(/\/\*[\s\S]*?\*\//g, "").replace(/(^|[^:"'`])\/\/.*$/gm, "$1");

/** Every module specifier of a file: import / export ... from, side-effect imports, import(). */
function specifiers(text) {
  const out = [];
  for (const m of code(text).matchAll(/(?:\bfrom\s*|^\s*import\s*|\bimport\s*\(\s*)["']([^"']+)["']/gm)) out.push(m[1]);
  return out;
}

const packageOf = (spec) => (spec.startsWith("@") ? spec.split("/").slice(0, 2).join("/") : spec.split("/")[0]);

test("the package's files were found", () => {
  assert.ok(SOURCES.length > 50, `only ${SOURCES.length} files under ${SRC}`);
  assert.ok(SOURCES.some((s) => s.path === FETCHER));
});

test("relative imports stay inside packages/viz (apiTypes.ts is the one type-only bridge)", () => {
  const bad = [];
  for (const s of SOURCES) {
    for (const spec of specifiers(s.text).filter((x) => x.startsWith("."))) {
      const target = resolve(SRC, dirname(s.path), spec);
      if (target.startsWith(VIZ + "/")) continue;
      if (s.path === BRIDGE.file && spec === BRIDGE.spec) continue;
      bad.push(`${s.path}: ${spec}`);
    }
  }
  assert.deepEqual(bad, []);
  const bridge = SOURCES.find((s) => s.path === BRIDGE.file);
  assert.match(code(bridge.text).trim(), /^export type \* from "[^"]+";$/, "apiTypes.ts must stay one type-only re-export");
});

test("bare imports are the peer dependencies only (node: modules in tests)", () => {
  const bad = [];
  for (const s of SOURCES) {
    for (const spec of specifiers(s.text).filter((x) => !x.startsWith("."))) {
      if (PEERS.includes(packageOf(spec)) || (s.test && spec.startsWith("node:"))) continue;
      bad.push(`${s.path}: ${spec}`);
    }
  }
  assert.deepEqual(bad, []);
});

test("nothing names the app's API client, store, router or pages", () => {
  const app = /apps\/ui|src\/api\/(http|cache|hooks|poll|predict|pool|workbench|browse|curate|embedding|meta)\b|\/state\/store|\brouter\b|\bregistry\b|\/features\//;
  const bad = SOURCES.filter((s) => s.path !== BRIDGE.file && !s.test).flatMap((s) => specifiers(s.text).filter((spec) => app.test(spec)).map((spec) => `${s.path}: ${spec}`));
  assert.deepEqual(bad, []);
});

test("no network calls: fetch only in scene/load.ts, no XHR / WebSocket / EventSource / beacon, no /api URL", () => {
  const net = /\bfetch\s*\(|XMLHttpRequest|WebSocket|EventSource|sendBeacon/;
  const api = /["'`]\/api\b/;
  const bad = [];
  for (const s of SOURCES.filter((x) => !x.test)) {
    const c = code(s.text);
    if (net.test(c) && s.path !== FETCHER) bad.push(`${s.path}: a network call`);
    if (api.test(c)) bad.push(`${s.path}: an /api URL`);
  }
  assert.deepEqual(bad, []);
});
