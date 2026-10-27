// validateScene: a hand-written structural check of a scene document (scene/types.ts, the same
// shape as scene.schema.json) before any view reads it. It checks every field a view relies on
// (types, integer indices, the asset references, ids that must point into the document); API
// fields the views never read are let through as they are. The errors name the JSON path:
// "scene.edge.matrix.bonds[0]: expected an integer triple, got [1,"a",1]".
import type { SceneDoc } from "./types";
// @ts-ignore TS5097: node --experimental-strip-types needs the extension (Vite accepts it)
import { SCENE_KIND, SCENE_TYPES, SCENE_VERSION, STEP_SCENE_VARIANTS } from "./types.ts";

export type ValidationResult = { ok: true; doc: SceneDoc } | { ok: false; errors: string[] };

type Obj = Record<string, unknown>;
const isObj = (x: unknown): x is Obj => typeof x === "object" && x !== null && !Array.isArray(x);
const isInt = (x: unknown): x is number => Number.isInteger(x);
const isNum = (x: unknown): x is number => typeof x === "number" && Number.isFinite(x);
const isIntTuple = (x: unknown, n: number) => Array.isArray(x) && x.length === n && x.every(isInt);

function shown(x: unknown): string {
  const s = JSON.stringify(x);
  return s === undefined ? String(x) : s.length > 60 ? `${s.slice(0, 57)}...` : s;
}

/** A relative path inside the scene's folder: no scheme, no absolute path, no "..". */
export function isRelativeAssetPath(p: string): boolean {
  return p.length > 0 && !/^[a-z][a-z0-9+.-]*:/i.test(p) && !p.startsWith("/") && !p.includes("\\") && !p.split("/").some((s) => s === ".." || s === "");
}

class Checker {
  readonly errors: string[] = [];
  fail(path: string, what: string, x: unknown): void {
    this.errors.push(`${path}: expected ${what}, got ${shown(x)}`);
  }
  obj(path: string, x: unknown): Obj | null {
    if (isObj(x)) return x;
    this.fail(path, "an object", x);
    return null;
  }
  test(path: string, x: unknown, ok: boolean, what: string): boolean {
    if (!ok) this.fail(path, what, x);
    return ok;
  }
  str = (path: string, x: unknown) => this.test(path, x, typeof x === "string", "a string");
  int = (path: string, x: unknown) => this.test(path, x, isInt(x), "an integer");
  num = (path: string, x: unknown) => this.test(path, x, isNum(x), "a number");
  bool = (path: string, x: unknown) => this.test(path, x, typeof x === "boolean", "a boolean");
  numOrNull = (path: string, x: unknown) => this.test(path, x, x === null || isNum(x), "a number or null");
  oneOf = (path: string, x: unknown, values: readonly unknown[]) => this.test(path, x, values.includes(x), `one of ${values.join(", ")}`);
  /** `x` is an array; `each` checks its items. */
  arr(path: string, x: unknown, each?: (p: string, item: unknown) => void): unknown[] {
    if (!Array.isArray(x)) return this.fail(path, "an array", x), [];
    if (each) x.forEach((item, k) => each(`${path}[${k}]`, item));
    return x;
  }
  strs = (path: string, x: unknown) => this.arr(path, x, (p, s) => this.str(p, s));
  ints = (path: string, x: unknown) => this.arr(path, x, (p, i) => this.int(p, i));
}

function checkAsset(c: Checker, path: string, x: unknown): void {
  const a = c.obj(path, x);
  if (!a) return;
  if (c.str(`${path}.path`, a.path)) c.test(`${path}.path`, a.path, isRelativeAssetPath(a.path as string), "a relative path inside the scene's folder");
  c.test(`${path}.bytes`, a.bytes, isInt(a.bytes) && (a.bytes as number) >= 0, "a byte count");
  c.test(`${path}.sha256`, a.sha256, typeof a.sha256 === "string" && /^[0-9a-f]{64}$/.test(a.sha256), "a lowercase hex sha256");
  c.str(`${path}.media_type`, a.media_type);
}

function checkCubes(c: Checker, path: string, x: unknown, known: (i: number) => boolean, what: string): void {
  c.arr(path, x, (p, item) => {
    const cube = c.obj(p, item);
    if (!cube) return;
    if (c.int(`${p}.index`, cube.index)) c.test(`${p}.index`, cube.index, known(cube.index as number), what);
    checkAsset(c, `${p}.asset`, cube.asset);
  });
}

function checkOrbitals(c: Checker, path: string, x: unknown): Set<number> {
  const indices = new Set<number>();
  if (x === null) return indices;
  const o = c.obj(path, x);
  if (!o) return indices;
  c.str(`${path}.set`, o.set);
  c.test(`${path}.homo_index`, o.homo_index, o.homo_index === null || isInt(o.homo_index), "an integer or null");
  c.arr(`${path}.orbitals`, o.orbitals, (p, item) => {
    const r = c.obj(p, item);
    if (!r) return;
    if (c.int(`${p}.index`, r.index)) indices.add(r.index as number);
    c.oneOf(`${p}.kind`, r.kind, ["occupied", "virtual"]);
    c.num(`${p}.occupation`, r.occupation);
    c.num(`${p}.energy_ev`, r.energy_ev);
  });
  return indices;
}

function checkMolecule(c: Checker, s: Obj): void {
  c.str("scene.smiles", s.smiles);
  const m = c.obj("scene.molecule", s.molecule);
  if (m) c.str("scene.molecule.molecule_id", m.molecule_id);
  const g = c.obj("scene.geometry", s.geometry);
  if (g) {
    c.str("scene.geometry.molblock", g.molblock);
    c.arr("scene.geometry.atoms", g.atoms);
  }
  const indices = checkOrbitals(c, "scene.orbitals", s.orbitals);
  checkCubes(c, "scene.cubes", s.cubes, (i) => indices.has(i), "an orbital index of scene.orbitals");
  if (s.selected !== undefined && s.selected !== null) c.test("scene.selected", s.selected, isInt(s.selected) && indices.has(s.selected), "an orbital index of scene.orbitals");
}

function checkDepiction(c: Checker, path: string, x: unknown): void {
  const d = c.obj(path, x);
  if (!d) return;
  c.str(`${path}.ref`, d.ref);
  c.str(`${path}.molblock`, d.molblock);
  c.arr(`${path}.atoms`, d.atoms, (p, item) => {
    const a = c.obj(p, item);
    if (!a) return;
    for (const k of ["depiction_idx", "pool_atom", "map"]) c.int(`${p}.${k}`, a[k]);
    c.str(`${p}.element`, a.element);
    c.ints(`${p}.hidden_h_atoms`, a.hidden_h_atoms);
  });
}

function checkGeometry(c: Checker, path: string, x: unknown): void {
  if (x === null) return;
  const g = c.obj(path, x);
  if (!g) return;
  c.arr(`${path}.species`, g.species, (p, item) => {
    const s = c.obj(p, item);
    if (!s) return;
    c.str(`${p}.molblock`, s.molblock);
    c.test(`${p}.display_offset`, s.display_offset, Array.isArray(s.display_offset) && s.display_offset.length === 3 && s.display_offset.every(isNum), "an [x, y, z] offset");
  });
}

function checkMatrix(c: Checker, path: string, x: unknown): void {
  const m = c.obj(path, x);
  if (!m) return;
  c.arr(`${path}.bonds`, m.bonds, (p, b) => c.test(p, b, isIntTuple(b, 3), "an integer triple"));
  c.arr(`${path}.lone`, m.lone, (p, l) => c.test(p, l, isIntTuple(l, 2), "an integer pair"));
}

/** The readings' tokens (the cubes a step scene may carry). */
function checkEdge(c: Checker, path: string, x: unknown): Set<number> {
  const tokens = new Set<number>();
  const e = c.obj(path, x);
  if (!e) return tokens;
  for (const k of ["id", "source_mapped_smiles", "product_mapped_smiles", "predictor"]) c.str(`${path}.${k}`, e[k]);
  c.oneOf(`${path}.kind`, e.kind, ["predicted", "corrected"]);
  checkMatrix(c, `${path}.matrix`, e.matrix);
  c.arr(`${path}.arrows`, e.arrows, (p, item) => {
    const a = c.obj(p, item);
    if (!a) return;
    c.oneOf(`${p}.kind`, a.kind, ["lone_pair_to_bond", "bond_to_bond", "bond_to_lone_pair"]);
    c.ints(`${p}.source`, a.source);
    c.ints(`${p}.target`, a.target);
  });
  c.arr(`${path}.readings`, e.readings, (p, item) => {
    const r = c.obj(p, item);
    if (!r) return;
    for (const k of ["i", "j", "donor_token", "acceptor_token"]) c.int(`${p}.${k}`, r[k]);
    for (const k of ["g_atom", "g_rs", "term"]) c.num(`${p}.${k}`, r[k]);
    if (isInt(r.donor_token)) tokens.add(r.donor_token);
    if (isInt(r.acceptor_token)) tokens.add(r.acceptor_token);
  });
  c.arr(`${path}.formal_charges`, e.formal_charges, (p, f) => c.test(p, f, isIntTuple(f, 2), "an integer pair"));
  c.strs(`${path}.product_smiles`, e.product_smiles);
  c.strs(`${path}.sigma_keys`, e.sigma_keys);
  c.numOrNull(`${path}.score`, e.score);
  c.numOrNull(`${path}.logp`, e.logp);
  return tokens;
}

function checkStep(c: Checker, s: Obj): void {
  const tokens = checkEdge(c, "scene.edge", s.edge);
  c.bool("scene.probabilities", s.probabilities);
  const src = c.obj("scene.source", s.source);
  if (src) {
    checkDepiction(c, "scene.source.depiction", src.depiction);
    checkGeometry(c, "scene.source.geometry", src.geometry);
    checkCubes(c, "scene.source.cubes", src.cubes, (t) => tokens.has(t), "a donor or acceptor token of scene.edge.readings");
  }
  if (s.product !== null) {
    const p = c.obj("scene.product", s.product);
    if (p) checkDepiction(c, "scene.product.depiction", p.depiction);
  }
  if (s.variants !== undefined) c.arr("scene.variants", s.variants, (p, v) => c.oneOf(p, v, STEP_SCENE_VARIANTS));
}

function checkNetwork(c: Checker, s: Obj): void {
  const n = c.obj("scene.network", s.network);
  if (!n) return;
  const nodes = new Set<string>();
  c.arr("scene.network.nodes", n.nodes, (p, item) => {
    const node = c.obj(p, item);
    if (!node) return;
    if (c.str(`${p}.id`, node.id)) nodes.add(node.id as string);
    c.str(`${p}.display_smiles`, node.display_smiles);
    c.int(`${p}.depth`, node.depth);
    c.arr(`${p}.species`, node.species);
  });
  c.test("scene.network.root", n.root, typeof n.root === "string" && nodes.has(n.root), "the id of a node");
  const isNode = (x: unknown) => typeof x === "string" && nodes.has(x);
  c.arr("scene.network.edges", n.edges, (p, item) => {
    const e = c.obj(p, item);
    if (!e) return;
    c.test(`${p}.source`, e.source, isNode(e.source), "the id of a node");
    c.test(`${p}.target`, e.target, isNode(e.target), "the id of a node");
    checkEdge(c, p, e);
  });
  c.arr("scene.network.pathways", n.pathways, (p, item) => {
    const w = c.obj(p, item);
    if (w) for (const k of ["node_ids", "edge_ids"]) c.strs(`${p}.${k}`, w[k]);
  });
  c.bool("scene.network.scores_are_probabilities", n.scores_are_probabilities);
  if (s.variant !== undefined) c.oneOf("scene.variant", s.variant, ["layered", "explorer"]);
}

function checkProvenance(c: Checker, x: unknown): void {
  const p = c.obj("provenance", x);
  if (!p) return;
  c.str("provenance.exporter", p.exporter);
  c.str("provenance.exported_at", p.exported_at);
  c.obj("provenance.server", p.server);
  c.obj("provenance.request", p.request);
  if (p.predict !== undefined && p.predict !== null) c.obj("provenance.predict", p.predict);
  c.strs("provenance.notes", p.notes);
}

/** Check a parsed scene document; `doc` is the same value, typed, when it passes. */
export function validateScene(x: unknown): ValidationResult {
  const c = new Checker();
  const d = c.obj("(document)", x);
  if (!d) return { ok: false, errors: c.errors };
  c.test("kind", d.kind, d.kind === SCENE_KIND, `"${SCENE_KIND}"`);
  c.test("version", d.version, d.version === SCENE_VERSION, `version ${SCENE_VERSION}`);
  const s = c.obj("scene", d.scene);
  if (s && c.oneOf("scene.type", s.type, SCENE_TYPES)) {
    if (s.title !== undefined) c.str("scene.title", s.title);
    if (s.type === "molecule") checkMolecule(c, s);
    else if (s.type === "step") checkStep(c, s);
    else checkNetwork(c, s);
  }
  checkProvenance(c, d.provenance);
  return c.errors.length ? { ok: false, errors: c.errors } : { ok: true, doc: x as SceneDoc };
}
