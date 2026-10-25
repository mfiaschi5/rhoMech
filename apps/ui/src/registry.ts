// The feature seam (design §11.1). main.tsx imports every src/features/*/index.{ts,tsx} eagerly;
// each feature calls registerPage() for the pages it provides, so the shell never names a
// feature file and builds before any feature exists. Features never import each other: they
// share only src/api, src/chem, src/components, src/state, router.ts and this file.
//
//   // src/features/predict/index.tsx
//   import { registerPage } from "../../registry";
//   registerPage("predict", PredictPage);   // PredictPage: (props: PageProps) => JSX.Element
import { useSyncExternalStore, type ComponentType } from "react";
import type { Route } from "./router";

export type PageId = "predict" | "correct" | "curate" | "browse";

export const PAGE_IDS: readonly PageId[] = ["predict", "correct", "curate", "browse"];

export interface PageProps {
  route: Route;
}

const pages = new Map<PageId, ComponentType<PageProps>>();
const listeners = new Set<() => void>();
let version = 0;

export function isPageId(id: string): id is PageId {
  return (PAGE_IDS as readonly string[]).includes(id);
}

/** Register (or, on hot reload, replace) the component of one page. */
export function registerPage(id: PageId, component: ComponentType<PageProps>): void {
  pages.set(id, component);
  version += 1;
  for (const l of listeners) l();
}

function subscribe(onChange: () => void): () => void {
  listeners.add(onChange);
  return () => listeners.delete(onChange);
}

/** The registered component of a page, or undefined when no feature provides it. */
export function usePage(id: PageId | string): ComponentType<PageProps> | undefined {
  useSyncExternalStore(
    subscribe,
    () => version,
    () => version,
  );
  return isPageId(id) ? pages.get(id) : undefined;
}

/** The ids registered so far (for the nav and diagnostics). */
export function registeredPages(): PageId[] {
  return PAGE_IDS.filter((id) => pages.has(id));
}
