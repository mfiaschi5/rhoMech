// Page-local shared state of the Curate feature (no state library: src/state/store.ts).
//   curationRevision  bumped after every curation mutation (POST /curate, PATCH, a finished
//                     re-check or export), so the mechanism list and the latest export reload;
//   created           the MechanismOut a POST /curate returned, by id. Its `warnings` (e.g.
//                     INDEPENDENCE_UNCHECKED) are shown on the mechanism view even if a later
//                     GET does not repeat them.
import { createStore } from "../../state/store";
import type { MechanismOut } from "../../api/types.gen";

export const curationRevision = createStore<number>(0);

export function bumpRevision(): void {
  curationRevision.update((n) => n + 1);
}

const created = new Map<number, MechanismOut>();

export function rememberCreated(m: MechanismOut): void {
  created.set(m.id, m);
}

export function createdWarnings(id: number): string[] {
  return created.get(id)?.warnings ?? [];
}
