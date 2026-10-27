// The wiki router (serve/wiki): the page index with its search, and one page rendered by the
// server (Markdown -> HTML with raw HTML disabled); the types are the generated ones (types.gen.ts).
import { request, type RequestOptions } from "./http";
import type * as T from "./types.gen";

export type WikiPageSummary = T.WikiPageBriefOut;
export type WikiIndexOut = T.WikiIndexOut;
export type WikiTocEntry = T.WikiHeadingOut;
/** html: scene figures are <figure class="wiki-scene" data-scene-src="scenes/...">. */
export type WikiPageOut = T.WikiPageOut;

/** GET /wiki?q=: every page, or the pages matching `q` (title, summary, tags, text). */
export const getWikiIndex = (q?: string, o?: RequestOptions) =>
  request<WikiIndexOut>({ path: "/wiki", query: { q: q?.trim() ? q.trim() : undefined } }, o);

/** GET /wiki/{slug}; 404 WIKI_PAGE_NOT_FOUND. */
export const getWikiPage = (slug: string, o?: RequestOptions) =>
  request<WikiPageOut>({ path: `/wiki/${encodeURIComponent(slug)}` }, o);
