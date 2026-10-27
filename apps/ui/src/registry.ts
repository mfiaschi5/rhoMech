// The feature seam (design §11.1). main.tsx imports every src/features/*/index.{ts,tsx} eagerly;
// each feature calls registerPage() for the pages it provides, so the shell never names a
// feature file and builds before any feature exists. Features never import each other's pages:
// they share src/api, src/chem, src/components, src/state, router.ts, this file and the
// visualizations of @ksmech/viz (packages/viz: the 2D renderer, molfile reader, 3D viewer, step
// and network views). One exception: predict reuses the workbench's sketcher (sketch/), never its
// page or its stores; correct reuses predict's modules (pool canvas, hooks), never its page.
// "scene" (features/scene) is an embed page: no nav, no shell, no API call.
//
//   // src/features/predict/index.tsx
//   import { registerPage } from "../../registry";
//   registerPage("predict", PredictPage);   // PredictPage: (props: PageProps) => JSX.Element
import { useSyncExternalStore, type ComponentType } from "react";
import type { Route } from "./router";

export type PageId = "workbench" | "predict" | "correct" | "curate" | "browse" | "login" | "users";
/** Pages outside the nav, drawn without the app shell and without /api/meta (#/scene). */
export type EmbedPageId = "scene";

export const PAGE_IDS: readonly PageId[] = ["workbench", "predict", "correct", "curate", "browse", "login", "users"];
/** The pages of the nav bar; "login" and "users" are reached from the header's user menu. */
export const NAV_PAGE_IDS: readonly PageId[] = ["workbench", "predict", "correct", "curate", "browse"];
export const EMBED_PAGE_IDS: readonly EmbedPageId[] = ["scene"];

export interface PageProps {
  route: Route;
}

const pages = new Map<PageId | EmbedPageId, ComponentType<PageProps>>();
const listeners = new Set<() => void>();
let version = 0;

export function isPageId(id: string): id is PageId {
  return (PAGE_IDS as readonly string[]).includes(id);
}

export function isEmbedPage(id: string): id is EmbedPageId {
  return (EMBED_PAGE_IDS as readonly string[]).includes(id);
}

/** Register (or, on hot reload, replace) the component of one page. */
export function registerPage(id: PageId | EmbedPageId, component: ComponentType<PageProps>): void {
  pages.set(id, component);
  version += 1;
  for (const l of listeners) l();
}

function subscribe(onChange: () => void): () => void {
  listeners.add(onChange);
  return () => listeners.delete(onChange);
}

/** The registered component of a page, or undefined when no feature provides it. */
export function usePage(id: PageId | EmbedPageId | string): ComponentType<PageProps> | undefined {
  useSyncExternalStore(
    subscribe,
    () => version,
    () => version,
  );
  return isPageId(id) || isEmbedPage(id) ? pages.get(id) : undefined;
}

/** The ids registered so far (for the nav and diagnostics). */
export function registeredPages(): PageId[] {
  return PAGE_IDS.filter((id) => pages.has(id));
}
