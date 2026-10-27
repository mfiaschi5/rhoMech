// WikiPage, #/wiki and #/wiki/<slug>: ChemRho's wiki. The left column lists the pages (GET
// /api/wiki, `?q=` searches title, summary, tags and text; the query lives in the hash), the
// article is the server's HTML (raw HTML disabled there) with its table of contents, a link to
// its Markdown source (`?view=source`) and its scene figures hydrated in place: every
// <figure class="wiki-scene" data-scene-src="scenes/<name>.json"> gets the scene viewer
// (components/SceneEmbed.tsx, @ksmech/viz's SceneView, files of this site only) through a portal.
// Links #/wiki/<slug> in the HTML navigate in the app (they are hash routes); a link to a
// heading (#<id>) and the TOC scroll to it instead of replacing the route.
//
// DOM: <div data-testid="wiki-page" data-slug>: wiki-search (input), <ul data-testid="wiki-list"
// data-count> of a[data-testid=wiki-link-<slug>] (aria-current on the open page); without a slug
// <section data-testid="wiki-index">; with one <article data-testid="wiki-article" data-slug
// data-state="loading|ready|error" data-scenes> holding wiki-title, wiki-tags, wiki-toc
// (buttons wiki-toc-<id>), wiki-source-link, wiki-html (the article body; its figures carry
// <div data-testid="wiki-scene-view" data-state data-src>) or wiki-markdown (the source).
import { useLayoutEffect, useMemo, useState, type MouseEvent } from "react";
import { createPortal } from "react-dom";
import { useCached } from "../../api/cache";
import { useDebounced } from "../../api/hooks";
import { getWikiIndex, getWikiPage, type WikiPageOut, type WikiPageSummary } from "../../api/wiki";
import { ErrorBanner } from "../../components/ErrorBanner";
import { SceneEmbed } from "../../components/SceneEmbed";
import type { PageProps } from "../../registry";
import { formatRoute, navigate, wikiHash, type Route } from "../../router";

/** The hash of the wiki with `q` and `view` set (empty values dropped). */
function wikiRoute(slug: string | null, q: string, view?: string): string {
  return formatRoute({ page: "wiki", parts: slug ? [slug] : [], params: { q, view } });
}

function PageList({ route, slug }: { route: Route; slug: string | null }) {
  const [text, setText] = useState(route.params.get("q") ?? "");
  const q = useDebounced(text.trim(), 250);
  const index = useCached(`/wiki?q=${q}`, () => getWikiIndex(q));
  useLayoutEffect(() => {
    if (q !== (route.params.get("q") ?? "")) navigate(wikiRoute(slug, q, route.params.get("view") ?? undefined), { replace: true });
    // only a settled query rewrites the hash
  }, [q]);
  const pages = index.data?.pages ?? [];
  return (
    <aside className="panel wiki-side stack">
      <input
        type="search"
        className="wiki-search"
        data-testid="wiki-search"
        placeholder="Search the wiki"
        aria-label="Search the wiki"
        value={text}
        onChange={(e) => setText(e.target.value)}
      />
      <ErrorBanner error={index.error} title="Cannot read the wiki index" />
      {index.loading && <p className="muted">loading…</p>}
      <ul className="wiki-list" data-testid="wiki-list" data-count={pages.length}>
        {pages.map((p) => (
          <li key={p.slug}>
            <a
              href={wikiRoute(p.slug, q)}
              data-testid={`wiki-link-${p.slug}`}
              className={p.slug === slug ? "active" : undefined}
              aria-current={p.slug === slug ? "page" : undefined}
              title={p.summary}
            >
              {p.title}
            </a>
          </li>
        ))}
      </ul>
      {!index.loading && !index.error && pages.length === 0 && <p className="muted">{q ? `No page matches "${q}".` : "The wiki has no pages yet."}</p>}
    </aside>
  );
}

function Tags({ tags }: { tags: readonly string[] }) {
  if (!tags.length) return null;
  return (
    <span className="wiki-tags" data-testid="wiki-tags">
      {tags.map((t) => (
        <span key={t} className="badge">
          {t}
        </span>
      ))}
    </span>
  );
}

function WikiIndex({ pages }: { pages: readonly WikiPageSummary[] }) {
  return (
    <section className="panel stack wiki-main" data-testid="wiki-index">
      <h2>Wiki</h2>
      <p className="muted">Guides to ChemRho and worked mechanisms with live scenes. Pick a page:</p>
      <ul className="wiki-cards">
        {pages.map((p) => (
          <li key={p.slug}>
            <a href={wikiHash(p.slug)}>{p.title}</a> <Tags tags={p.tags} />
            <div className="muted">{p.summary}</div>
          </li>
        ))}
      </ul>
    </section>
  );
}

/** The scene figures of the article body, found after every new HTML. */
function useSceneFigures(body: HTMLElement | null, html: string): HTMLElement[] {
  const [figures, setFigures] = useState<HTMLElement[]>([]);
  useLayoutEffect(() => {
    setFigures(body ? [...body.querySelectorAll<HTMLElement>("figure.wiki-scene[data-scene-src]")] : []);
  }, [body, html]);
  return figures;
}

/** Scroll to the heading `id` of the article (never a route change). */
function scrollToHeading(body: HTMLElement | null, id: string) {
  const el = body?.querySelector(`[id="${CSS.escape(id)}"]`);
  el?.scrollIntoView({ behavior: "smooth", block: "start" });
}

function Toc({ page, onPick }: { page: WikiPageOut; onPick(id: string): void }) {
  if (!page.toc.length) return null;
  const top = Math.min(...page.toc.map((t) => t.level));
  return (
    <nav className="wiki-toc" data-testid="wiki-toc" aria-label="Contents">
      <strong>Contents</strong>
      <ul>
        {page.toc.map((t) => (
          <li key={t.id} style={{ paddingLeft: `${(t.level - top) * 12}px` }}>
            <button type="button" className="wiki-toc-link" data-testid={`wiki-toc-${t.id}`} onClick={() => onPick(t.id)}>
              {t.text}
            </button>
          </li>
        ))}
      </ul>
    </nav>
  );
}

function ArticleBody({ page }: { page: WikiPageOut }) {
  const [body, setBody] = useState<HTMLDivElement | null>(null);
  const figures = useSceneFigures(body, page.html);
  const html = useMemo(() => ({ __html: page.html }), [page.html]);
  const onClick = (e: MouseEvent<HTMLDivElement>) => {
    const a = (e.target as HTMLElement).closest("a");
    const href = a?.getAttribute("href") ?? "";
    if (href.startsWith("#") && !href.startsWith("#/")) {
      e.preventDefault();
      scrollToHeading(body, decodeURIComponent(href.slice(1)));
    }
  };
  return (
    <>
      <Toc page={page} onPick={(id) => scrollToHeading(body, id)} />
      <div className="wiki-html" data-testid="wiki-html" ref={setBody} onClick={onClick} dangerouslySetInnerHTML={html} />
      {figures.map((fig, k) => createPortal(<SceneEmbed src={fig.dataset.sceneSrc ?? ""} testId="wiki-scene-view" />, fig, `${k}:${fig.dataset.sceneSrc}`))}
    </>
  );
}

function Article({ slug, route }: { slug: string; route: Route }) {
  const page = useCached(`/wiki/${slug}`, () => getWikiPage(slug));
  const source = route.params.get("view") === "source";
  const q = route.params.get("q") ?? "";
  const state = page.error ? "error" : page.data ? "ready" : "loading";
  const p = page.data;
  return (
    <article className="panel stack wiki-main" data-testid="wiki-article" data-slug={slug} data-state={state} data-scenes={p?.scenes.length ?? 0}>
      <ErrorBanner error={page.error} title={`Cannot open the wiki page "${slug}"`} />
      {state === "loading" && <p className="muted">loading…</p>}
      {p && (
        <>
          <header className="wiki-head">
            <h2 data-testid="wiki-title">{p.title}</h2>
            <div className="wiki-meta muted">
              <Tags tags={p.tags} />
              {p.updated && <span>updated {p.updated}</span>}
              <a href={wikiRoute(slug, q, source ? undefined : "source")} data-testid="wiki-source-link">
                {source ? "View page" : "View Markdown source"}
              </a>
            </div>
            {p.summary && <p className="wiki-summary">{p.summary}</p>}
          </header>
          {source ? (
            <pre className="pre wiki-markdown" data-testid="wiki-markdown">
              {p.markdown}
            </pre>
          ) : (
            <ArticleBody page={p} />
          )}
        </>
      )}
    </article>
  );
}

export function WikiPage({ route }: PageProps) {
  const slug = route.parts[0] ?? null;
  const index = useCached(`/wiki?q=`, () => getWikiIndex());
  return (
    <div className="wiki-page" data-testid="wiki-page" data-slug={slug ?? ""}>
      <PageList route={route} slug={slug} />
      {slug ? <Article key={slug} slug={slug} route={route} /> : <WikiIndex pages={index.data?.pages ?? []} />}
    </div>
  );
}
