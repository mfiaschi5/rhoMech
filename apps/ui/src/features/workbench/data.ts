// Cached reads of a computed molecule (api/cache.ts keys are the request paths): geometry,
// properties and orbital sets, each pending-aware (202 while the worker computes).
import { getMoleculeGeometry, getMoleculeProperties, getOrbitalSet } from "../../api/client";
import { usePendingCached, type PendingCached } from "../../api/hooks";
import type { MoleculeGeometryOut, MoleculePropertiesOut, OrbitalSetOut } from "../../api/types.gen";
import { CANONICAL } from "./params";

export function useGeometry(id: string | null): PendingCached<MoleculeGeometryOut> {
  return usePendingCached(id ? `/molecule/${id}/geometry` : null, (o) => getMoleculeGeometry(id ?? "", o));
}

export function useProperties(id: string | null): PendingCached<MoleculePropertiesOut> {
  return usePendingCached(id ? `/molecule/${id}/properties` : null, (o) => getMoleculeProperties(id ?? "", o));
}

/** One orbital set; `start` is ignored for the canonical set. */
export function useOrbitalSet(id: string | null, set: string, start: string): PendingCached<OrbitalSetOut> {
  const st = set === CANONICAL ? null : start;
  const key = id ? `/molecule/${id}/orbitals?set=${set}&start=${st ?? ""}` : null;
  return usePendingCached(key, (o) => getOrbitalSet(id ?? "", { set, start: st }, o));
}
