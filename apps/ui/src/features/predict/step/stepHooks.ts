// React glue of the step viewport variants: the variant choice (remembered in localStorage),
// the two sides of a step (stepGeometry.ts: the drawn reactant pool and the product state's
// depiction renamed into source pool atoms and fitted onto the reactant), the shown step (the
// edge, or one of its alternatives with its arrows and readings from POST /step/check), the
// container width, and a hover set shared by a drawing and its side panels.
import { useEffect, useMemo, useState } from "react";
import { useDrawnStructure, usePool, useStepCheck } from "../hooks";
import { poolStructure, touchedAtoms, type PoolStructure } from "../poolStructure";
import { EMPTY_MATRIX } from "../rmatrix";
import { alignProduct, productIndexMap, sideFragments, sideOf, xyOf, type Side, type XY } from "./stepGeometry";
import type { ArrowReadingOut, EdgeOut, MatrixIO, OrbitalReadingOut } from "../v9types";
// @ts-ignore TS5097: see stepScene.ts
import { BOND_LENGTH } from "../../workbench/sketch/model.ts";

export const STEP_VARIANTS = ["textbook", "animated", "orbital", "embedding"] as const;
export type StepVariant = (typeof STEP_VARIANTS)[number];
export const STEP_VARIANT_KEY = "ksmech.predict.stepVariant";
export const STEP_VARIANT_TITLES: Record<StepVariant, string> = { textbook: "Textbook", animated: "Animated", orbital: "Orbitals", embedding: "Embedding" };

function readVariant(): StepVariant {
  try {
    const v = window.localStorage.getItem(STEP_VARIANT_KEY);
    return (STEP_VARIANTS as readonly string[]).includes(v ?? "") ? (v as StepVariant) : "textbook";
  } catch {
    return "textbook";
  }
}

export function useStepVariant(): [StepVariant, (v: StepVariant) => void] {
  const [v, setV] = useState<StepVariant>(readVariant);
  const set = (next: StepVariant) => {
    setV(next);
    try {
      window.localStorage.setItem(STEP_VARIANT_KEY, next);
    } catch {
      // storage blocked: the choice lasts for this page only
    }
  };
  return [v, set];
}

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

export function useShownStep(edge: EdgeOut, index: number): ShownStep {
  const alt = index >= 0 ? edge.alternatives[index] : undefined;
  const check = useStepCheck(alt ? edge.source_state_id : null, alt?.matrix ?? EMPTY_MATRIX);
  const charges = useMemo(() => new Map(edge.formal_charges), [edge.formal_charges]);
  if (!alt) return { index: -1, matrix: edge.matrix, arrows: edge.arrows, readings: edge.readings, chargesAfter: charges, score: edge.score, logp: edge.logp, pending: false, note: null };
  const res = check.current ? check.result : null;
  const note = check.error ? `no reading for this alternative: ${check.error.message}` : !res ? "reading the alternative…" : null;
  return { index, matrix: alt.matrix, arrows: res?.arrows ?? [], readings: res?.readings ?? [], chargesAfter: res ? new Map(res.formal_charges) : null, score: alt.score, logp: alt.logp, pending: !res, note };
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
function productSide(dep: Parameters<typeof poolStructure>[0], toSource: Map<number, number>, src: PoolStructure, touched: Set<number>): Side {
  const toProduct = new Map([...toSource].map(([k, i]) => [i, k] as const));
  const drawnH = src.drawn.filter((a) => a.parent !== null).map((a) => toProduct.get(a.index)).filter((k): k is number => k !== undefined);
  const label = [...touched].map((i) => toProduct.get(i)).filter((k): k is number => k !== undefined);
  return sideOf(poolStructure(dep, { show: new Set(drawnH), label: new Set(label) }), toSource);
}

export function useStepSides(edge: EdgeOut, step: ShownStep): StepSides {
  const pool = usePool(edge.source_state_id);
  const prodPool = usePool(edge.product_state_id || null);
  const structure = useDrawnStructure(pool, step.matrix, step.arrows);
  const toSource = useMemo(() => productIndexMap(edge.source_mapped_smiles, edge.product_mapped_smiles), [edge.source_mapped_smiles, edge.product_mapped_smiles]);
  const touchKey = [...touchedAtoms(step.matrix, step.arrows)].sort((a, b) => a - b).join(",");
  return useMemo(() => {
    const source = structure ? sideOf(structure) : null;
    let product: Side | null = null;
    let error: string | null = null;
    if (structure && prodPool.depiction && toSource) {
      try {
        product = productSide(prodPool.depiction, toSource, structure, new Set(touchKey ? touchKey.split(",").map(Number) : []));
      } catch (err) {
        error = `cannot read the product depiction: ${String(err)}`;
      }
    }
    const aligned = source && product ? alignProduct(xyOf(source), xyOf(product), sideFragments(product), 1.4 * BOND_LENGTH) : null;
    return { structure, source, product, aligned, loading: pool.loading || prodPool.loading, error: error ?? pool.error?.message ?? null };
  }, [structure, prodPool.depiction, toSource, touchKey, pool.loading, prodPool.loading, pool.error]);
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
