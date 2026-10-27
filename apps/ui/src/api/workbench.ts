// The molecule workbench router (serve/workbench; capability "workbench", display only): drawn
// molecules, their geometry, properties, symmetry, orbitals and cubes, and the server's 2D depiction.
import { apiUrl, request, type Query, type RequestOptions, type TextWithHeaders } from "./http";
import type * as T from "./types.gen";

const molecule = (mid: string, rest = "") => `/molecule/${encodeURIComponent(mid)}${rest}`;

/** POST /molecule: register a drawn molecule (molblock) or a SMILES and featurize it. Answers
 *  within the server's interactive wait with status ready, or running (poll getMolecule). Only
 *  this answer carries the sketch mapping. */
export const createMolecule = (body: T.MoleculeIn, o?: RequestOptions) =>
  request<T.MoleculeOut>({ method: "POST", path: "/molecule", body }, o);

/** POST /depict2d: the server's 2D drawing of a sketch (RDKit ring templates, the perspective
 *  template library, or match3d: the computed structure as the 3D view shows it), in the
 *  molblock's atom and bond order. */
export const depict2d = (body: T.Depict2DIn, o?: RequestOptions) =>
  request<T.Depict2DOut>({ method: "POST", path: "/depict2d", body }, o);

/** GET /molecule/{id}: the status (never computes; mapping is always null here). */
export const getMolecule = (mid: string, o?: RequestOptions) => request<T.MoleculeOut>({ path: molecule(mid) }, o);

export const getMoleculeGeometry = (mid: string, o?: RequestOptions) =>
  request<T.MoleculeGeometryOut>({ path: molecule(mid, "/geometry") }, o);

export const getMoleculeProperties = (mid: string, o?: RequestOptions) =>
  request<T.MoleculePropertiesOut>({ path: molecule(mid, "/properties") }, o);

/** GET /molecule/{id}/symmetry?tolerance=: the Schoenflies point group of the record geometry
 *  (tolerance in Å, 0.01..0.5, default 0.1 on the server). */
export const getMoleculeSymmetry = (mid: string, tolerance: number | null, o?: RequestOptions) =>
  request<T.PointGroupOut>({ path: molecule(mid, "/symmetry"), query: { tolerance } }, o);

export interface OrbitalSetQuery {
  set?: string | null;
  start?: string | null;
}

export const getOrbitalSet = (mid: string, q: OrbitalSetQuery = {}, o?: RequestOptions) =>
  request<T.OrbitalSetOut>({ path: molecule(mid, "/orbitals"), query: { set: q.set, start: q.start } }, o);

export type CubeKind = "orbital" | "orbital_density" | "density" | "fukui_minus" | "fukui_plus" | "dual" | "esp";
export type CubeQuality = "draft" | "normal" | "fine";

export interface MoleculeCubeQuery {
  kind: CubeKind;
  set?: string | null;
  start?: string | null;
  index?: number | null;
  quality?: CubeQuality | null;
}

/** The metadata of a workbench cube (X-Cube-* headers); null where the header says "none". */
export interface CubeMeta {
  kind: string;
  units: string;
  integral: number | null;
  expected: number | null;
  spacing: number | null;
}

function headerNumber(h: Headers, name: string): number | null {
  const v = h.get(name);
  if (v === null || v === "none" || v === "") return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

export function cubeMeta(h: Headers): CubeMeta {
  return {
    kind: h.get("X-Cube-Kind") ?? "",
    units: h.get("X-Cube-Units") ?? "",
    integral: headerNumber(h, "X-Cube-Integral"),
    expected: headerNumber(h, "X-Cube-Expected"),
    spacing: headerNumber(h, "X-Cube-Spacing"),
  };
}

function cubeQuery(q: MoleculeCubeQuery, download = false): Query {
  const orbital = q.kind === "orbital" || q.kind === "orbital_density";
  return {
    kind: q.kind,
    set: orbital ? q.set : undefined,
    start: orbital ? q.start : undefined,
    index: orbital ? q.index : undefined,
    quality: q.quality,
    download: download ? true : undefined,
  };
}

/** GET /molecule/{id}/cube: the cube text and its X-Cube-* metadata (pending while computed). */
export async function getMoleculeCube(
  mid: string,
  q: MoleculeCubeQuery,
  o?: RequestOptions,
): Promise<{ text: string; headers: Headers; meta: CubeMeta }> {
  const res = await request<TextWithHeaders>({ path: molecule(mid, "/cube"), query: cubeQuery(q), as: "textWithHeaders" }, o);
  return { ...res, meta: cubeMeta(res.headers) };
}

/** The URL of a workbench cube (download: as an attachment), for links. */
export function moleculeCubeUrl(mid: string, q: MoleculeCubeQuery, download = true): string {
  return apiUrl(molecule(mid, "/cube"), cubeQuery(q, download));
}
