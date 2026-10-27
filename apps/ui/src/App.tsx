// The shell of ChemRho: the section nav (registry.NAV_SECTIONS; ClotoMech's pages in a second
// tab row, testids subnav-<page>), the app-wide banners (from /api/meta) and the current page, or a
// "page not installed" placeholder when no feature registered it (design §11.1).
import { useEffect } from "react";
import { ErrorBanner } from "./components/ErrorBanner";
import { SyntheticBanner } from "./components/SyntheticBanner";
import { UserMenu } from "./components/UserMenu";
import { isEmbedPage, isPageId, NAV_SECTIONS, PAGE_IDS, PRODUCT_NAME, sectionOf, usePage, type PageId } from "./registry";
import { useHashRoute, type Route } from "./router";
import { loadAuth } from "./state/auth";
import { appErrorStore, capabilityOf, clearError, loadMeta, metaStore, useStore } from "./state/store";

const NAV_LABELS: Record<PageId, string> = {
  workbench: "ChemPtah",
  predict: "Predict",
  correct: "Correct",
  curate: "Curate",
  browse: "ChemThot",
  wiki: "Wiki",
  login: "Sign in",
  users: "Users",
};

/** The capability a whole page needs (/api/meta.capabilities); off: greyed with the reason. */
const PAGE_CAPABILITY: Partial<Record<PageId, string>> = {
  workbench: "workbench",
  predict: "predict",
  correct: "correct",
};

/** A capability-aware nav link (a section, or a page of the ClotoMech tab row). */
function NavLink(p: { href: string; testId: string; label: string; hint: string; page?: PageId; active: boolean; className: string }) {
  const meta = useStore(metaStore, (s) => s.data);
  const needs = p.page ? PAGE_CAPABILITY[p.page] : undefined;
  const cap = needs ? capabilityOf(meta, needs) : null;
  const off = cap !== null && !cap.enabled;
  return (
    <a
      href={p.href}
      data-testid={p.testId}
      data-disabled={off ? "true" : undefined}
      aria-disabled={off ? true : undefined}
      title={off ? `${p.label} is disabled: ${cap?.reason ?? ""}` : p.hint}
      className={`${p.className}${p.active ? " active" : ""}${off ? " disabled" : ""}`}
      aria-current={p.active ? "page" : undefined}
    >
      {p.label}
    </a>
  );
}

function Nav({ current }: { current: string }) {
  const section = sectionOf(current);
  return (
    <nav className="nav" aria-label="sections">
      <span className="brand" data-testid="brand" title={`${PRODUCT_NAME}: mechanisms from orbital interactions`}>
        {PRODUCT_NAME}
      </span>
      {NAV_SECTIONS.map((s) => (
        <NavLink
          key={s.id}
          href={`#/${s.pages[0]}`}
          testId={`nav-${s.id}`}
          label={s.label}
          hint={s.hint}
          page={s.pages.length === 1 ? s.pages[0] : undefined}
          active={section?.id === s.id}
          className="nav-link"
        />
      ))}
    </nav>
  );
}

const SUB_HINTS: Partial<Record<PageId, string>> = {
  predict: "Predict a mechanism from the reactants",
  correct: "Correct a predicted step (reaction matrix editor)",
  curate: "Curate mechanisms into the curation database",
};

/** The second tab row of a section of several pages (ClotoMech), while one of them is open. */
function SubNav({ current }: { current: string }) {
  const section = sectionOf(current);
  if (!section || section.pages.length < 2) return null;
  return (
    <nav className="subnav" aria-label={`${section.label} pages`} data-testid={`subnav-${section.id}`}>
      <span className="subnav-title muted">{section.label}</span>
      {section.pages.map((id) => (
        <NavLink key={id} href={`#/${id}`} testId={`subnav-${id}`} label={NAV_LABELS[id]} hint={SUB_HINTS[id] ?? NAV_LABELS[id]} page={id} active={current === id} className="subnav-link" />
      ))}
    </nav>
  );
}

/** document.title: "ChemRho", or "ChemRho · <section>" (· <page> inside ClotoMech). */
export function pageTitle(page: string): string {
  const section = sectionOf(page);
  if (!section) return PRODUCT_NAME;
  const sub = section.pages.length > 1 && isPageId(page) ? ` · ${NAV_LABELS[page]}` : "";
  return `${PRODUCT_NAME} · ${section.label}${sub}`;
}

function Page({ route }: { route: Route }) {
  const Component = usePage(route.page);
  if (Component) return <Component route={route} />;
  const known = (PAGE_IDS as readonly string[]).includes(route.page);
  return (
    <section className="panel placeholder" data-testid="page-not-installed" data-page={route.page}>
      <h2>{known ? `${NAV_LABELS[route.page as PageId]}: page not installed` : `unknown page "${route.page}"`}</h2>
      <p className="muted">
        {known
          ? `No feature registered the "${route.page}" page (src/features/*/index.tsx calls registerPage).`
          : `Use one of: ${PAGE_IDS.map((p) => `#/${p}`).join(", ")}.`}
      </p>
    </section>
  );
}

function MetaLine() {
  const { data: meta, error } = useStore(metaStore);
  if (error) return <ErrorBanner error={error} title="Cannot read /api/meta" />;
  if (!meta) return null;
  return (
    <div className="meta-line muted" data-testid="meta-line" data-chemistry={meta.chemistry}>
      <span>
        spec <code>{meta.spec_version}</code>
      </span>
      <span>chemistry {meta.chemistry}</span>
      <span>
        worker {meta.worker.alive ? "alive" : "down"}
        {meta.worker.busy && meta.worker.current_label ? ` (busy: ${meta.worker.current_label})` : ""}
      </span>
      {meta.dev && <span className="badge">dev: Vite on :5173</span>}
    </div>
  );
}

export function App() {
  const route = useHashRoute();
  const meta = useStore(metaStore, (s) => s.data);
  const appError = useStore(appErrorStore);
  const embed = isEmbedPage(route.page);
  useEffect(() => {
    document.title = pageTitle(route.page);
  }, [route.page]);
  useEffect(() => {
    if (!embed) {
      void loadMeta();
      void loadAuth();
    }
  }, [embed]);
  if (embed) {
    return (
      <main className="app-main app-embed" data-page={route.page}>
        <Page route={route} />
      </main>
    );
  }
  return (
    <div className="app-shell" data-page={route.page}>
      <header className="app-header">
        <Nav current={route.page} />
        <UserMenu />
        <MetaLine />
        <SubNav current={route.page} />
      </header>
      <SyntheticBanner meta={meta} />
      <ErrorBanner error={appError} onDismiss={clearError} />
      <main className="app-main">
        <Page route={route} />
      </main>
    </div>
  );
}
