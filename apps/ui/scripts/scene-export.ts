// The scene exporter: reads a running ksmech-serve's API and writes one scene file of
// @ksmech/viz (packages/viz/src/scene/types.ts) with its assets, for pages that draw the views
// without a server (the wiki, #/scene). Run by node 22 (type stripping, @ksmech/viz through
// scripts/viz-node-hooks.mjs):
//
//   npm run scene:export -- --type step --job J --edge E --out DIR [--url http://127.0.0.1:8000]
//       [--cubes all|dominant|none] [--cube-spacing 0.35] [--cube-padding 2.5] [--no-3d]
//   npm run scene:export -- --type molecule --smiles 'C=CC=O' --out DIR
//       [--orbitals homo,lumo] [--cube-stride 2]
//   npm run scene:export -- --type network --job J --out DIR
//   common: [--name NAME] [--title TEXT] [--token BEARER] (a public-profile server's role)
//
// It writes DIR/NAME.json, the assets under DIR/NAME/ (orbital cubes, by relative path with byte
// size and sha256) and upserts DIR/index.json ([{file, type, title}], what #/scene lists). The
// document is checked with validateScene before it is written. A 202 pending answer (the worker
// is computing) is retried every 2 s for up to 10 min; nothing here computes chemistry itself.
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { parseArgs } from "node:util";
import type * as T from "../src/api/types.gen";
import type { AssetRef, CubeAsset, MoleculeScene, NetworkScene, Scene, SceneDoc, SceneEdge, StepScene } from "@ksmech/viz/scene/types";
import { SCENE_KIND, SCENE_VERSION } from "@ksmech/viz/scene/types";
import { validateScene } from "@ksmech/viz/scene/validate";
import { dominantReading } from "@ksmech/viz/step/stepText";
// @ts-ignore TS5097: node --experimental-strip-types needs the extension
import { strideCube } from "./sceneCube.ts";

const EXPORTER = "ksmech scene-export 1";
const PENDING_TIMEOUT_MS = 600_000;
const POLL_MS = 2000;
const CUBE_MEDIA = "chemical/x-cube";

type Opts = Record<string, string | boolean | undefined>;
interface Api {
  base: string;
  token: string | null;
}
interface Out {
  dir: string;
  name: string;
  assets: string[];
  notes: string[];
  /** The predict job's provenance (step and network scenes). */
  predict: T.Provenance | null;
}

const log = (m: string) => console.log(`[scene-export] ${m}`);
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const str = (o: Opts, k: string, d = "") => (typeof o[k] === "string" ? (o[k] as string) : d);
const num = (o: Opts, k: string, d: number) => (typeof o[k] === "string" ? Number(o[k]) : d);

function pick<V extends object, K extends keyof V>(o: V, keys: readonly K[]): Pick<V, K> {
  return Object.fromEntries(keys.map((k) => [k, o[k]])) as Pick<V, K>;
}

async function call<R>(api: Api, path: string, init: { method?: string; query?: Record<string, string | number>; body?: unknown; text?: boolean } = {}): Promise<R> {
  const url = new URL(`/api${path}`, api.base);
  for (const [k, v] of Object.entries(init.query ?? {})) url.searchParams.set(k, String(v));
  const headers: Record<string, string> = { accept: init.text ? "text/plain" : "application/json" };
  if (init.body !== undefined) headers["content-type"] = "application/json";
  if (api.token) headers.authorization = `Bearer ${api.token}`;
  const deadline = Date.now() + PENDING_TIMEOUT_MS;
  for (;;) {
    const res = await fetch(url, { method: init.method ?? "GET", headers, body: init.body === undefined ? undefined : JSON.stringify(init.body) });
    if (res.status === 202 && (init.method ?? "GET") === "GET") {
      const p = (await res.json()) as { label?: string };
      if (Date.now() > deadline) throw new Error(`GET ${path}: still pending after ${PENDING_TIMEOUT_MS / 1000} s`);
      log(`pending: ${p.label ?? path}`);
      await sleep(POLL_MS);
      continue;
    }
    if (!res.ok) throw new Error(`${init.method ?? "GET"} ${url.pathname}${url.search}: HTTP ${res.status} ${(await res.text()).slice(0, 400)}`);
    return (init.text ? await res.text() : await res.json()) as R;
  }
}

function writeAsset(out: Out, file: string, text: string): AssetRef {
  const bytes = Buffer.from(text, "utf8");
  mkdirSync(join(out.dir, out.name), { recursive: true });
  writeFileSync(join(out.dir, out.name, file), bytes);
  out.assets.push(`${out.name}/${file}`);
  return { path: `${out.name}/${file}`, bytes: bytes.length, sha256: createHash("sha256").update(bytes).digest("hex"), media_type: CUBE_MEDIA };
}

/** A job's network once it is done (a running job is waited for). */
async function jobNetwork(api: Api, job: string): Promise<T.NetworkOut> {
  const deadline = Date.now() + PENDING_TIMEOUT_MS;
  for (;;) {
    const j = await call<T.JobOut>(api, `/jobs/${encodeURIComponent(job)}`);
    if (j.status === "done" && j.result_kind === "network" && j.result) return j.result as T.NetworkOut;
    if (j.status !== "queued" && j.status !== "running") throw new Error(`job ${job} is ${j.status} with ${j.result_kind ?? "no"} result: ${j.error?.message ?? "a predict job has a network"}`);
    if (Date.now() > deadline) throw new Error(`job ${job} still ${j.status}`);
    log(`job ${job} ${j.status}…`);
    await sleep(POLL_MS);
  }
}

const SCENE_EDGE_KEYS = ["id", "kind", "source_state_id", "product_state_id", "source_mapped_smiles", "product_mapped_smiles", "matrix", "anchor", "arrows", "readings", "score", "logp", "predictor", "rank", "product_smiles", "sigma_keys", "formal_charges"] as const satisfies readonly (keyof SceneEdge)[];

/** The donor and acceptor tokens whose cubes a step scene carries. */
function cubeTokens(edge: T.EdgeOut, which: string): number[] {
  if (which === "none" || !edge.readings.length) return [];
  const readings = which === "dominant" ? [edge.readings[dominantReading(edge.readings)]] : edge.readings;
  return [...new Set(readings.flatMap((r) => [r.donor_token, r.acceptor_token]))];
}

async function stepCubes(api: Api, out: Out, edge: T.EdgeOut, o: Opts): Promise<CubeAsset[]> {
  const query = { spacing: num(o, "cube-spacing", 0.35), padding: num(o, "cube-padding", 2.5) };
  const cubes: CubeAsset[] = [];
  for (const token of cubeTokens(edge, str(o, "cubes", "all"))) {
    log(`cube of token ${token} (spacing ${query.spacing} Å, padding ${query.padding} Å)`);
    const text = await call<string>(api, `/pool/${edge.source_state_id}/orbital/${token}`, { query, text: true });
    cubes.push({ index: token, asset: writeAsset(out, `orbital-${token}.cube`, text) });
  }
  if (cubes.length) out.notes.push(`orbital cubes of GET /pool/{state}/orbital/{token} at spacing ${query.spacing} Å, padding ${query.padding} Å`);
  return cubes;
}

const speciesText = (n: T.NodeOut | undefined) => (n ? n.species.map((s) => (s.count > 1 ? `${s.count} ${s.display_smiles}` : s.display_smiles)).join(" + ") : "");

async function stepScene(api: Api, out: Out, o: Opts): Promise<StepScene> {
  const job = str(o, "job");
  const net = await jobNetwork(api, job);
  out.predict = net.provenance;
  const edge = net.edges.find((e) => e.id === str(o, "edge"));
  if (!edge) throw new Error(`job ${job} has no edge ${str(o, "edge")} (edges: ${net.edges.map((e) => e.id).join(", ")})`);
  const depiction = (state: string) => call<T.DepictionOut>(api, `/pool/${state}/depiction`);
  const geometry = o["no-3d"] ? null : await call<T.GeometryOut>(api, `/pool/${edge.source_state_id}/geometry`);
  const cubes = geometry ? await stepCubes(api, out, edge, o) : [];
  const product = edge.product_state_id ? { depiction: await depiction(edge.product_state_id) } : null;
  out.notes.push("the predicted step only: alternatives with the same product need POST /step/check and are left out");
  const title = str(o, "title") || `${speciesText(net.nodes.find((n) => n.id === edge.source))} → ${edge.product_smiles.join(" + ")}`;
  return { type: "step", title, edge: pick(edge, SCENE_EDGE_KEYS), probabilities: net.scores_are_probabilities, source: { depiction: await depiction(edge.source_state_id), geometry, cubes }, product };
}

async function networkScene(api: Api, out: Out, o: Opts): Promise<NetworkScene> {
  const net = await jobNetwork(api, str(o, "job"));
  out.predict = net.provenance;
  const root = net.nodes.find((n) => n.id === net.root);
  const title = str(o, "title") || `${speciesText(root)}: ${net.nodes.length} states, ${net.edges.length} steps`;
  return { type: "network", title, network: net };
}

/** POST /molecule, then GET /molecule/{id} until it is ready. */
async function readyMolecule(api: Api, smiles: string): Promise<T.MoleculeOut> {
  let mol = await call<T.MoleculeOut>(api, "/molecule", { method: "POST", body: { smiles } });
  const deadline = Date.now() + PENDING_TIMEOUT_MS;
  while (mol.status === "running" || mol.status === "missing") {
    if (Date.now() > deadline) throw new Error(`molecule ${mol.molecule_id} still ${mol.status}`);
    log(`molecule ${mol.molecule_id} ${mol.status}…`);
    await sleep(POLL_MS);
    mol = await call<T.MoleculeOut>(api, `/molecule/${mol.molecule_id}`);
  }
  if (mol.status === "failed") throw new Error(`molecule ${smiles} failed: ${mol.error?.message ?? ""}`);
  return mol;
}

const FRONTIER: Record<string, string> = { homo: "homo", lumo: "lumo", "homo-1": "homo_minus_1", "lumo+1": "lumo_plus_1" };

/** The orbital indices of "--orbitals homo,lumo,7" in a canonical set. */
function orbitalIndices(rows: readonly T.OrbitalRowOut[], spec: string): number[] {
  return spec.split(",").filter(Boolean).map((t) => {
    const k = t.trim().toLowerCase();
    const row = /^\d+$/.test(k) ? rows.find((r) => r.index === Number(k)) : rows.find((r) => r.frontier === (FRONTIER[k] ?? k));
    if (!row) throw new Error(`--orbitals: no orbital "${t}" (frontier: ${rows.filter((r) => r.frontier).map((r) => r.frontier).join(", ")})`);
    return row.index;
  });
}

async function moleculeCubes(api: Api, out: Out, mid: string, rows: readonly T.OrbitalRowOut[], o: Opts): Promise<CubeAsset[]> {
  const stride = Math.max(1, Math.round(num(o, "cube-stride", 2)));
  const cubes: CubeAsset[] = [];
  for (const index of orbitalIndices(rows, str(o, "orbitals", "homo,lumo"))) {
    log(`cube of canonical MO ${index} (draft, stride ${stride})`);
    const text = await call<string>(api, `/molecule/${mid}/cube`, { query: { kind: "orbital", set: "canonical", index, quality: "draft" }, text: true });
    cubes.push({ index, asset: writeAsset(out, `mo-${index}.cube`, strideCube(text, stride)) });
  }
  if (cubes.length) out.notes.push(`canonical MO cubes of GET /molecule/{id}/cube (quality draft), every ${stride}. grid point kept`);
  return cubes;
}

async function moleculeScene(api: Api, out: Out, o: Opts): Promise<MoleculeScene> {
  const smiles = str(o, "smiles");
  if (!smiles) throw new Error("--type molecule needs --smiles");
  const mol = await readyMolecule(api, smiles);
  const geometry = await call<T.MoleculeGeometryOut>(api, `/molecule/${mol.molecule_id}/geometry`);
  const set = await call<T.OrbitalSetOut>(api, `/molecule/${mol.molecule_id}/orbitals`, { query: { set: "canonical" } });
  const cubes = await moleculeCubes(api, out, mol.molecule_id, set.orbitals, o);
  const orbitals = pick(set, ["set", "start", "source", "homo_index", "orbitals"] as const);
  return { type: "molecule", title: str(o, "title") || smiles, smiles, molecule: pick(mol, ["molecule_id", "key", "charge", "multiplicity", "chemistry", "method"] as const), geometry, orbitals, cubes, selected: set.homo_index };
}

const BUILDERS: Record<Scene["type"], (api: Api, out: Out, o: Opts) => Promise<Scene>> = { step: stepScene, network: networkScene, molecule: moleculeScene };

function defaultName(o: Opts): string {
  const t = str(o, "type");
  const slug = (s: string) => s.replace(/[^A-Za-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 40);
  return t === "step" ? `step-${str(o, "edge")}` : t === "network" ? `network-${str(o, "job")}` : `molecule-${slug(str(o, "smiles"))}`;
}

function upsertIndex(dir: string, entry: { file: string; type: string; title: string }): void {
  const path = join(dir, "index.json");
  const list = existsSync(path) ? (JSON.parse(readFileSync(path, "utf8")) as (typeof entry)[]) : [];
  const next = [...list.filter((e) => e.file !== entry.file), entry];
  writeFileSync(path, `${JSON.stringify(next, null, 2)}\n`);
}

const STRING_OPTIONS = ["url", "type", "job", "edge", "smiles", "out", "name", "title", "token", "cubes", "cube-spacing", "cube-padding", "cube-stride", "orbitals"];
const OPTIONS = { ...Object.fromEntries(STRING_OPTIONS.map((k) => [k, { type: "string" as const }])), "no-3d": { type: "boolean" as const } };

async function main(argv: string[]): Promise<void> {
  const { values: o } = parseArgs({ args: argv, options: OPTIONS });
  const build = BUILDERS[str(o, "type") as Scene["type"]];
  if (!build || !str(o, "out")) throw new Error("usage: --type step|molecule|network --out DIR (see the header of scripts/scene-export.ts)");
  const api: Api = { base: str(o, "url", "http://127.0.0.1:8000"), token: str(o, "token") || null };
  const out: Out = { dir: resolve(str(o, "out")), name: str(o, "name") || defaultName(o), assets: [], notes: [], predict: null };
  mkdirSync(out.dir, { recursive: true });
  const meta = await call<T.MetaOut>(api, "/meta");
  const scene = await build(api, out, o);
  const request = Object.fromEntries(Object.entries(o).filter(([k]) => !["out", "token", "url"].includes(k)));
  const doc: SceneDoc = { kind: SCENE_KIND, version: SCENE_VERSION, scene, provenance: { exporter: EXPORTER, exported_at: new Date().toISOString(), server: pick(meta, ["api_schema", "app_version", "spec_version", "physics_fingerprint", "chemistry", "qm_method"] as const), request, predict: out.predict, notes: out.notes } };
  const v = validateScene(doc);
  if (!v.ok) throw new Error(`the exported scene does not validate:\n${v.errors.join("\n")}`);
  const file = `${out.name}.json`;
  writeFileSync(join(out.dir, file), `${JSON.stringify(doc, null, 1)}\n`);
  upsertIndex(out.dir, { file, type: scene.type, title: scene.title ?? out.name });
  log(`wrote ${join(out.dir, file)}${out.assets.length ? ` and ${out.assets.join(", ")}` : ""}`);
}

main(process.argv.slice(2)).catch((err: unknown) => {
  console.error(`[scene-export] ${err instanceof Error ? err.message : String(err)}`);
  process.exit(1);
});
