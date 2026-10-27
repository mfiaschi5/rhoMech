// The data of a step viewport as plain values (no fetching): the shown step (the edge's own R,
// arrows and readings), the two sides of a step (stepGeometry.ts: the drawn reactant pool and
// the product state's depiction renamed into source pool atoms and fitted onto the reactant),
// the container width, and a hover set shared by a drawing and its side panels. The app's hooks
// (apps/ui/src/features/predict/step/stepHooks.ts) load the depictions; a scene carries them.
import { useEffect, useState } from "react";
import type { DepictionOut } from "../apiTypes";
import { drawnOptions, poolStructure, touchedAtoms, type PoolStructure } from "../pool/poolStructure";
import { alignProduct, productIndexMap, sideFragments, sideOf, xyOf, type Side, type XY } from "./stepGeometry";
import type { ArrowReadingOut, EdgeOut, MatrixIO, OrbitalReadingOut } from "../v9types";
// @ts-ignore TS5097: see stepScene.ts
import { BOND_LENGTH } from "../sketch/model.ts";

/** The width of an element, followed with a ResizeObserver. */
export function useBoxWidth(): [(el: HTMLElement | null) => void, number] {
  const [el, setEl] = useState<HTMLElement | null>(null);
  const [w, setW] = useState(0);
  useEffect(() => {
    if (!el) return;
    setW(el.clientWidth);
    const ro = new ResizeObserver(() => setW(el.clientWidth));
    ro.observe(el);
    return () => ro.disconnect();
  }, [el]);
  return [setEl, w];
}

/** What a variant draws: the edge itself or alternative k (0-based; -1: the edge). */
export interface ShownStep {
  index: number;
  matrix: MatrixIO;
  arrows: readonly ArrowReadingOut[];
  readings: readonly OrbitalReadingOut[];
  chargesAfter: ReadonlyMap<number, number> | null;
  score: number | null;
  logp: number | null;
  /** The alternative's reading is still being checked (or could not be). */
  pending: boolean;
  note: string | null;
}

/** The fields of an edge the step views read. */
export type StepEdge = Pick<EdgeOut, "matrix" | "arrows" | "readings" | "formal_charges" | "score" | "logp" | "source_mapped_smiles" | "product_mapped_smiles">;

/** The edge's own step (index -1); `charges` is its formal_charges as a map. */
export function predictedStep(edge: StepEdge, charges: ReadonlyMap<number, number> = new Map(edge.formal_charges)): ShownStep {
  return { index: -1, matrix: edge.matrix, arrows: edge.arrows, readings: edge.readings, chargesAfter: charges, score: edge.score, logp: edge.logp, pending: false, note: null };
}

export interface StepSides {
  /** The drawn reactant pool (hidden H that R or the arrows touch drawn, touched atoms labelled). */
  structure: PoolStructure | null;
  source: Side | null;
  /** The product side in source pool atoms (null: no product depiction). */
  product: Side | null;
  /** Product positions fitted onto the reactant (model units). */
  aligned: XY | null;
  loading: boolean;
  error: string | null;
}

/** The product state's structure drawing the H the reactant drawing shows and labelling the
 *  atoms R touches, renamed into source pool atoms. */
function productSide(dep: DepictionOut, toSource: Map<number, number>, src: PoolStructure, touched: Set<number>): Side {
  const toProduct = new Map([...toSource].map(([k, i]) => [i, k] as const));
  const drawnH = src.drawn.filter((a) => a.parent !== null).map((a) => toProduct.get(a.index)).filter((k): k is number => k !== undefined);
  const label = [...touched].map((i) => toProduct.get(i)).filter((k): k is number => k !== undefined);
  return sideOf(poolStructure(dep, { show: new Set(drawnH), label: new Set(label) }), toSource);
}

/** The atoms R or its arrows touch, as a memo key ("1,4,5"). */
export const touchKeyOf = (step: Pick<ShownStep, "matrix" | "arrows">) => [...touchedAtoms(step.matrix, step.arrows)].sort((a, b) => a - b).join(",");

/** Both sides from the drawn reactant structure and the product state's depiction; `error` says
 *  why the product side is missing when its depiction cannot be read. */
export function sidesOf(structure: PoolStructure | null, productDep: DepictionOut | null | undefined, toSource: Map<number, number> | null, touchKey: string): Pick<StepSides, "structure" | "source" | "product" | "aligned"> & { error: string | null } {
  const source = structure ? sideOf(structure) : null;
  let product: Side | null = null;
  let error: string | null = null;
  if (structure && productDep && toSource) {
    try {
      product = productSide(productDep, toSource, structure, new Set(touchKey ? touchKey.split(",").map(Number) : []));
    } catch (err) {
      error = `cannot read the product depiction: ${String(err)}`;
    }
  }
  const aligned = source && product ? alignProduct(xyOf(source), xyOf(product), sideFragments(product), 1.4 * BOND_LENGTH) : null;
  return { structure, source, product, aligned, error };
}

/** The drawn reactant structure of a step from the source state's depiction (null when it
 *  cannot be read). */
export function drawnStructure(dep: DepictionOut, step: Pick<ShownStep, "matrix" | "arrows">): PoolStructure | null {
  try {
    return poolStructure(dep, drawnOptions(dep, step.matrix, step.arrows).options);
  } catch {
    return null;
  }
}

/** Both sides of a step from the two depictions, with nothing loading (a scene). */
export function stepSidesOf(sourceDep: DepictionOut, productDep: DepictionOut | null, edge: StepEdge, step: ShownStep): StepSides {
  const structure = drawnStructure(sourceDep, step);
  const toSource = productIndexMap(edge.source_mapped_smiles, edge.product_mapped_smiles);
  const sides = sidesOf(structure, productDep, toSource, touchKeyOf(step));
  return { ...sides, loading: false, error: sides.error ?? (structure ? null : "cannot read the source depiction") };
}

/** Atoms and bonds ("i-j") under the pointer, shared by a drawing and its side panel. */
export interface HoverSet {
  atoms: ReadonlySet<number>;
  bonds: ReadonlySet<string>;
}

export const NO_HOVER: HoverSet = { atoms: new Set(), bonds: new Set() };

export function hoverOf(atoms: readonly number[], bonds: readonly (readonly [number, number])[] = []): HoverSet {
  return { atoms: new Set(atoms), bonds: new Set(bonds.map(([i, j]) => (i < j ? `${i}-${j}` : `${j}-${i}`))) };
}
