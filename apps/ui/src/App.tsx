// The shell: navigation, the app-wide banners (from /api/meta) and the current page, or a
// "page not installed" placeholder when no feature registered it (design §11.1).
import { useEffect } from "react";
import { ErrorBanner } from "./components/ErrorBanner";
import { SyntheticBanner } from "./components/SyntheticBanner";
import { PAGE_IDS, usePage, type PageId } from "./registry";
import { useHashRoute, type Route } from "./router";
import { appErrorStore, clearError, loadMeta, metaStore, useStore } from "./state/store";

const NAV_LABELS: Record<PageId, string> = {
  predict: "Predict",
  correct: "Correct",
  curate: "Curate",
  browse: "Browse",
};

function Nav({ current }: { current: string }) {
  return (
    <nav className="nav" aria-label="pages">
      <span className="brand">ksmech</span>
      {PAGE_IDS.map((id) => (
        <a
          key={id}
          href={`#/${id}`}
          data-testid={`nav-${id}`}
          className={`nav-link${current === id ? " active" : ""}`}
          aria-current={current === id ? "page" : undefined}
        >
          {NAV_LABELS[id]}
        </a>
      ))}
    </nav>
  );
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
  useEffect(() => {
    void loadMeta();
  }, []);
  return (
    <div className="app-shell" data-page={route.page}>
      <header className="app-header">
        <Nav current={route.page} />
        <MetaLine />
      </header>
      <SyntheticBanner meta={meta} />
      <ErrorBanner error={appError} onDismiss={clearError} />
      <main className="app-main">
        <Page route={route} />
      </main>
    </div>
  );
}
