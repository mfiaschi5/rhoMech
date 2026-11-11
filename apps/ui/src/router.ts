// Hash routing by hand (design §11.2). All selection state lives in the hash, so it survives
// reloads and Playwright can deep-link to it. Routes:
//   #/predict                                   #/predict?r=<smiles>&n=<count>&r=...&n=...
//   #/predict/job/<jobId>?pathway=&node=&edge=&arrow=&cand=<donor>,<dsite>,<acceptor>,<asite>
//   #/correct/<jobId>/edge/<edgeId>             #/correct/<jobId>/node/<nodeId>
//   #/curate   #/curate/<jobId>?pathway=<pid>   #/curate/m/<mechId>
//   #/browse?q=&mode=&side=&src=&st=            #/browse/step/<ref>
//   #/workbench?smiles=&m=&set=&start=&orb=&rep=&layers=&q=
//   #/wiki   #/wiki?q=<search>   #/wiki/<slug>
//   #/scene?src=scenes/<name>.json               (a scene file, no shell and no API call)
// The section names of ChemRho are aliases of the page ids (PAGE_ALIASES): #/chemptah is
// #/workbench, #/clotomech is #/predict, #/chemthot is #/browse (same parts and query).
import { useMemo, useSyncExternalStore } from "react";

// The molecule workbench while /api/meta reports the predict capability off (spec v8 port);
// was "predict", then "browse"
export const DEFAULT_PAGE = "workbench";

/** Section names that stand for a page id; parseHash reads them as that page. */
export const PAGE_ALIASES: Readonly<Record<string, string>> = {
  chemptah: "workbench",
  clotomech: "predict",
  chemthot: "browse",
};

export interface Route {
  /** First path segment ("workbench", "predict", "correct", "curate", "browse"); DEFAULT_PAGE when empty. */
  page: string;
  /** The remaining path segments, URI-decoded. */
  parts: string[];
  /** The query after "?", repeated keys kept (r / n of the predict prefill). */
  params: URLSearchParams;
  /** The raw hash, "#/..." ("" when the location has none). */
  hash: string;
}

/** A route to build: params values that are null / undefined / "" are dropped. */
export interface RouteSpec {
  page: string;
  parts?: (string | number)[];
  params?: Record<string, string | number | boolean | null | undefined | (string | number)[]>;
}

export function parseHash(hash: string = window.location.hash): Route {
  const body = hash.replace(/^#\/?/, "");
  const q = body.indexOf("?");
  const path = q < 0 ? body : body.slice(0, q);
  const query = q < 0 ? "" : body.slice(q + 1);
  const segments = path.split("/").filter((s) => s.length > 0).map(safeDecode);
  return {
    page: pageOf(segments[0]),
    parts: segments.slice(1),
    params: new URLSearchParams(query),
    hash,
  };
}

function pageOf(segment: string | undefined): string {
  if (!segment) return DEFAULT_PAGE;
  return PAGE_ALIASES[segment.toLowerCase()] ?? segment;
}

function safeDecode(s: string): string {
  try {
    return decodeURIComponent(s);
  } catch {
    return s;
  }
}

export function formatRoute(spec: RouteSpec): string {
  const path = [spec.page, ...(spec.parts ?? []).map((p) => encodeURIComponent(String(p)))];
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(spec.params ?? {})) {
    const values = Array.isArray(value) ? value : [value];
    for (const v of values) {
      if (v === null || v === undefined || v === "") continue;
      params.append(key, String(v));
    }
  }
  const query = params.toString();
  return `#/${path.join("/")}${query ? `?${query}` : ""}`;
}

/** Go to `to` ("#/..." or a RouteSpec). `replace` rewrites the current history entry. */
export function navigate(to: string | RouteSpec, opts: { replace?: boolean } = {}): void {
  const hash = typeof to === "string" ? (to.startsWith("#") ? to : `#${to}`) : formatRoute(to);
  if (hash === window.location.hash) return;
  if (opts.replace) {
    const url = `${window.location.pathname}${window.location.search}${hash}`;
    window.history.replaceState(window.history.state, "", url);
    window.dispatchEvent(new HashChangeEvent("hashchange"));
  } else {
    window.location.hash = hash;
  }
}

/** The current route with some query params set (null / "" removes one); for selection state. */
export function withParams(
  route: Route,
  updates: Record<string, string | number | null | undefined>,
): string {
  const params = new URLSearchParams(route.params);
  for (const [key, value] of Object.entries(updates)) {
    if (value === null || value === undefined || value === "") params.delete(key);
    else params.set(key, String(value));
  }
  const query = params.toString();
  const path = [route.page, ...route.parts.map(encodeURIComponent)].join("/");
  return `#/${path}${query ? `?${query}` : ""}`;
}

function subscribe(onChange: () => void): () => void {
  window.addEventListener("hashchange", onChange);
  window.addEventListener("popstate", onChange);
  return () => {
    window.removeEventListener("hashchange", onChange);
    window.removeEventListener("popstate", onChange);
  };
}

/** The current route; re-renders on every hash change. */
export function useHashRoute(): Route {
  const hash = useSyncExternalStore(
    subscribe,
    () => window.location.hash,
    () => "",
  );
  return useMemo(() => parseHash(hash), [hash]);
}

// ---- the #/predict?r=&n= prefill (Browse "predict from these reactants" links here)

export interface PrefillSpecies {
  smiles: string;
  count: number;
}

/** "#/predict?r=C[O-]&n=1&r=CCl&n=1"; counts default to 1. */
export function predictPrefillHash(species: PrefillSpecies[], target?: string | null): string {
  const t = target?.trim() ? { t: [target.trim()] } : {};
  return formatRoute({
    page: "predict",
    params: { r: species.map((s) => s.smiles), n: species.map((s) => s.count), ...t },
  });
}

/** The prefilled target product of a #/predict route ("" when there is none). */
export function parsePredictTarget(route: Route): string {
  return route.page === "predict" ? (route.params.get("t") ?? "").trim() : "";
}

/** The prefilled reactant rows of a #/predict route (empty when there is none). */
export function parsePredictPrefill(route: Route): PrefillSpecies[] {
  if (route.page !== "predict") return [];
  const smiles = route.params.getAll("r");
  const counts = route.params.getAll("n");
  return smiles
    .filter((s) => s.trim().length > 0)
    .map((s, i) => {
      const n = Number.parseInt(counts[i] ?? "1", 10);
      return { smiles: s.trim(), count: Number.isFinite(n) && n > 0 ? n : 1 };
    });
}

/** "#/predict/job/<jobId>" with optional selection params. */
export function jobHash(
  jobId: string,
  sel: { pathway?: string; node?: string; edge?: string; arrow?: number; cand?: string } = {},
): string {
  return formatRoute({ page: "predict", parts: ["job", jobId], params: sel });
}

/** "#/correct/<jobId>/edge/<edgeId>" or ".../node/<nodeId>". */
export function correctHash(jobId: string, target: { edge?: string; node?: string }): string {
  if (target.edge) return formatRoute({ page: "correct", parts: [jobId, "edge", target.edge] });
  return formatRoute({ page: "correct", parts: [jobId, "node", target.node ?? ""] });
}

/** "#/curate/<jobId>?pathway=<pid>". */
export function curateHash(jobId: string, pathwayId?: string): string {
  return formatRoute({ page: "curate", parts: [jobId], params: { pathway: pathwayId } });
}

/** "#/wiki/<slug>" (the index without a slug). */
export function wikiHash(slug?: string): string {
  return formatRoute({ page: "wiki", parts: slug ? [slug] : [] });
}

/** The `cand` param "<donor>,<dsite>,<acceptor>,<asite>" as numbers, or null. */
export function parseCand(value: string | null): [number, number, number, number] | null {
  if (!value) return null;
  const nums = value.split(",").map((v) => Number.parseInt(v, 10));
  if (nums.length !== 4 || nums.some((n) => !Number.isFinite(n))) return null;
  return [nums[0], nums[1], nums[2], nums[3]];
}
