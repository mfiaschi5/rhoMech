// The browse router (serve/browse): the PMechDB index and the step search.
import { request, type RequestOptions } from "./http";
import type * as T from "./types.gen";

export const getBrowseStatus = (o?: RequestOptions) =>
  request<T.IndexStatusOut>({ path: "/browse/status" }, o);

/** POST /browse/index: start (or rejoin) the PMechDB index job. */
export const buildBrowseIndex = (o?: RequestOptions) =>
  request<T.JobCreatedOut>({ method: "POST", path: "/browse/index" }, o);

export type BrowseSource = "pmechdb" | "user_curated";

export interface SearchQuery {
  q: string;
  mode?: "exact" | "substructure";
  side?: "reactants" | "products" | "either";
  sources?: readonly BrowseSource[];
  status?: string;
  resonance?: boolean;
  limit?: number;
  offset?: number;
}

export const searchSteps = (s: SearchQuery, o?: RequestOptions) =>
  request<T.SearchOut>(
    {
      path: "/browse/search",
      query: {
        q: s.q,
        mode: s.mode,
        side: s.side,
        source: s.sources,
        status: s.status,
        resonance: s.resonance,
        limit: s.limit,
        offset: s.offset,
      },
    },
    o,
  );

/** GET /browse/step/{ref}; ref is "pmechdb:<step_id>" or "curated:<mid>:<step_index>". */
export const getBrowseStep = (ref: string, o?: RequestOptions) =>
  request<T.StepViewOut>({ path: `/browse/step/${encodeURIComponent(ref)}` }, o);
