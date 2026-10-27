// The wiki router (serve/wiki): the page index with its search, and one page rendered by the
// server (Markdown -> HTML with raw HTML disabled). The types are written by hand to the
// backend's contract until types.gen.ts carries WikiIndexOut and WikiPageOut; then they become
// re-exports of the generated ones.
import { request, type RequestOptions } from "./http";

export interface WikiPageSummary {
  slug: string;
  title: string;
  summary: string;
  tags: string[];
}

export interface WikiIndexOut {
  pages: WikiPageSummary[];
}

export interface WikiTocEntry {
  level: number;
  /** The heading's id attribute in `html`. */
  id: string;
  text: string;
}

export interface WikiPageOut {
  slug: string;
  title: string;
  summary: string;
  tags: string[];
  /** Server-rendered; scene figures are <figure class="wiki-scene" data-scene-src="scenes/...">. */
  html: string;
  markdown: string;
  /** The scene files the page embeds. */
  scenes: string[];
  toc: WikiTocEntry[];
  updated: string;
}

/** GET /wiki?q=: every page, or the pages matching `q` (title, summary, tags, text). */
export const getWikiIndex = (q?: string, o?: RequestOptions) =>
  request<WikiIndexOut>({ path: "/wiki", query: { q: q?.trim() ? q.trim() : undefined } }, o);

/** GET /wiki/{slug}; 404 WIKI_PAGE_NOT_FOUND. */
export const getWikiPage = (slug: string, o?: RequestOptions) =>
  request<WikiPageOut>({ path: `/wiki/${encodeURIComponent(slug)}` }, o);
