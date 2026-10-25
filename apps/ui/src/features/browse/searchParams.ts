// The Browse search state, kept in the hash (design §11.2):
//   #/browse?q=&mode=exact|substructure&side=reactants|products|either&src=pmechdb,user_curated
//           &st=<status>&res=true|false&off=<offset>        (+ index=<job> while an index job runs)
//   #/browse/step/<ref>?<the same search params>              a step, with its search kept
// A present `q` (even empty) means "a search was submitted"; an empty query lists every step of
// the chosen sources. A missing `src` means "the default": both sources when the PMechDB index is
// ready, curated steps only otherwise (the pmechdb checkbox then starts unchecked).
import type { BrowseSource } from "../../api/client";
import type { Route } from "../../router";

export type Mode = "exact" | "substructure";
export type Side = "reactants" | "products" | "either";
export type Resonance = "" | "true" | "false";

export const SOURCES: readonly BrowseSource[] = ["pmechdb", "user_curated"];
export const SIDES: readonly Side[] = ["either", "reactants", "products"];
export const STEP_STATUSES = ["dataset", "predicted", "validated", "corrected", "rejected"] as const;
export const PAGE_SIZE = 50;

export interface SearchState {
  submitted: boolean;
  q: string;
  mode: Mode;
  side: Side;
  /** null: the default sources (they depend on the index status). */
  sources: BrowseSource[] | null;
  status: string;
  resonance: Resonance;
  offset: number;
}

const SEARCH_KEYS = ["q", "mode", "side", "src", "st", "res", "off"];

export function readSearch(params: URLSearchParams): SearchState {
  const side = params.get("side") as Side | null;
  const st = params.get("st") ?? "";
  const res = params.get("res");
  const off = Number.parseInt(params.get("off") ?? "0", 10);
  return {
    submitted: params.has("q"),
    q: params.get("q") ?? "",
    mode: params.get("mode") === "substructure" ? "substructure" : "exact",
    side: side && SIDES.includes(side) ? side : "either",
    sources: params.has("src") ? parseSources(params.get("src")) : null,
    status: (STEP_STATUSES as readonly string[]).includes(st) ? st : "",
    resonance: res === "true" || res === "false" ? res : "",
    offset: Number.isFinite(off) && off > 0 ? off : 0,
  };
}

function parseSources(value: string | null): BrowseSource[] {
  const want = new Set((value ?? "").split(","));
  return SOURCES.filter((s) => want.has(s));
}

/** The default sources for an index state (§11.8). */
export function defaultSources(indexState: string | undefined): BrowseSource[] {
  return indexState === "ready" ? [...SOURCES] : ["user_curated"];
}

/** A browse hash for `search` (and optionally a step), keeping the other params of `route`. */
export function browseHash(route: Route, search: SearchState | null, step?: string | null): string {
  const params = new URLSearchParams(route.params);
  if (search) {
    for (const k of SEARCH_KEYS) params.delete(k);
    params.set("q", search.q);
    params.set("mode", search.mode);
    params.set("side", search.side);
    if (search.sources) params.set("src", search.sources.join(","));
    if (search.status) params.set("st", search.status);
    if (search.resonance) params.set("res", search.resonance);
    if (search.offset > 0) params.set("off", String(search.offset));
  }
  const path = step ? `browse/step/${encodeURIComponent(step)}` : "browse";
  const query = params.toString();
  return `#/${path}${query ? `?${query}` : ""}`;
}

/** The key of one concrete search (the load reruns when it changes). */
export function searchKey(s: SearchState, sources: readonly BrowseSource[]): string {
  return JSON.stringify([s.q, s.mode, s.side, [...sources], s.status, s.resonance, s.offset]);
}
