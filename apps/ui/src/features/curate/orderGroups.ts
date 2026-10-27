// Order groups (design §9.3, C15): one int per step, non-decreasing, starting at 0 and growing by
// at most 1. The UI builds them from "independent of previous step" toggles: an independent step
// shares its predecessor's group, any other step opens the next group. The server is the judge
// of independence (422 NOT_INDEPENDENT, warning INDEPENDENCE_UNCHECKED); independenceHints only
// previews its rule so the curator sees the problem before posting.

/** order_groups from the toggles (toggle i is "step i is independent of step i-1"; toggle 0 is ignored). */
export function groupsFromToggles(independent: readonly boolean[], n: number): number[] {
  const groups: number[] = [];
  for (let i = 0; i < n; i++) {
    if (i === 0) groups.push(0);
    else groups.push(independent[i] ? groups[i - 1] : groups[i - 1] + 1);
  }
  return groups;
}

/** The toggles that give `groups` (the inverse of groupsFromToggles on valid groups). */
export function togglesFromGroups(groups: readonly number[]): boolean[] {
  return groups.map((g, i) => i > 0 && g === groups[i - 1]);
}

/** Why `groups` is not a valid order-group list, or null when it is. */
export function groupsProblem(groups: readonly number[]): string | null {
  if (groups.length === 0) return null;
  if (groups[0] !== 0) return "the first step must be in group 0";
  for (let i = 1; i < groups.length; i++) {
    const step = groups[i] - groups[i - 1];
    if (step !== 0 && step !== 1) return `step ${i}: groups must be contiguous and non-decreasing`;
  }
  return null;
}

export function sameGroups(a: readonly number[], b: readonly number[]): boolean {
  return a.length === b.length && a.every((g, i) => g === b[i]);
}

/** The indices of the toggled steps, as the hash keeps them ("ind=1,3"). */
export function parseToggleParam(value: string | null, n: number): boolean[] {
  const on = new Set(
    (value ?? "")
      .split(",")
      .map((v) => Number.parseInt(v, 10))
      .filter((v) => Number.isInteger(v) && v > 0 && v < n),
  );
  return Array.from({ length: n }, (_, i) => on.has(i));
}

export function formatToggleParam(independent: readonly boolean[]): string | null {
  const on = independent.map((v, i) => (v && i > 0 ? i : -1)).filter((i) => i > 0);
  return on.length ? on.join(",") : null;
}

/** What independence needs to know about a step. */
export interface StepMaps {
  reactantStateId: string | null;
  productStateId: string | null;
  touchedMaps: readonly number[];
}

export interface IndependenceHint {
  step: number;
  other: number;
  code: "NOT_INDEPENDENT" | "INDEPENDENCE_UNCHECKED";
  maps: number[];
}

function groupMembers(groups: readonly number[]): number[][] {
  const out = new Map<number, number[]>();
  groups.forEach((g, i) => out.set(g, [...(out.get(g) ?? []), i]));
  return [...out.values()].filter((m) => m.length > 1);
}

/** The server's check (§9.3) run in the browser: within a group of continuous steps (each product
 *  state is the next reactant state) every pair must touch disjoint map sets; a pair whose maps
 *  are not continuous (a merge renumbered them) can only be accepted unchecked. */
export function independenceHints(groups: readonly number[], steps: readonly StepMaps[]): IndependenceHint[] {
  const hints: IndependenceHint[] = [];
  for (const members of groupMembers(groups)) {
    let continuous = true;
    for (let k = 1; k < members.length; k++) {
      const a = steps[members[k - 1]];
      const b = steps[members[k]];
      if (!a || !b || !a.productStateId || a.productStateId !== b.reactantStateId) {
        continuous = false;
        hints.push({ step: members[k], other: members[k - 1], code: "INDEPENDENCE_UNCHECKED", maps: [] });
      }
    }
    if (!continuous) continue;
    for (let x = 0; x < members.length; x++) {
      for (let y = x + 1; y < members.length; y++) {
        const seen = new Set(steps[members[x]].touchedMaps);
        const shared = steps[members[y]].touchedMaps.filter((m) => seen.has(m));
        if (shared.length) hints.push({ step: members[y], other: members[x], code: "NOT_INDEPENDENT", maps: shared });
      }
    }
  }
  return hints;
}
