// The workbench's feature store: the selected and hovered record atoms (QM record order, the
// order of the geometry, properties and orbital centres), shared by the 3D view, the tables and
// the sketch highlight. Selection is transient UI state, so it lives here and not in the hash.
import { createStore, useStore } from "../../state/store";

export interface WorkbenchSelection {
  /** Selected record atom indices, ascending. */
  atoms: readonly number[];
  /** The record atom under the pointer in a table (null: none). */
  hover: number | null;
}

const EMPTY: WorkbenchSelection = { atoms: [], hover: null };

export const selectionStore = createStore<WorkbenchSelection>(EMPTY);

const sorted = (xs: Iterable<number>) => [...new Set(xs)].sort((a, b) => a - b);

/** Select exactly `atoms` (a row click), or toggle them into the selection (`add`). */
export function selectAtoms(atoms: readonly number[], add = false): void {
  selectionStore.update((s) => {
    if (!add) return { ...s, atoms: sorted(atoms) };
    const cur = new Set(s.atoms);
    const allIn = atoms.every((a) => cur.has(a));
    for (const a of atoms) {
      if (allIn) cur.delete(a);
      else cur.add(a);
    }
    return { ...s, atoms: sorted(cur) };
  });
}

/** Toggle one atom (a click in the 3D view). */
export function toggleAtom(atom: number): void {
  selectAtoms([atom], true);
}

export function clearSelection(): void {
  selectionStore.set(EMPTY);
}

export function setHover(atom: number | null): void {
  selectionStore.update((s) => (s.hover === atom ? s : { ...s, hover: atom }));
}

export function useSelection(): WorkbenchSelection {
  return useStore(selectionStore);
}
