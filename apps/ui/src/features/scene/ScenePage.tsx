// ScenePage, #/scene?src=<url>: one scene file (packages/viz scene format) drawn by @ksmech/viz's
// SceneView with no API call at all: the proof that a static page (the wiki) can embed the app's
// views from precomputed JSON. `src` is resolved against the page and must stay on its origin
// (a scene is data, but only this site's files are drawn); the loading and drawing is the shared
// components/SceneEmbed.tsx, which the wiki's scene figures use too. Without `src`: the list of
// public/scenes/index.json (the committed examples, written by `npm run scene:export`).
//
// DOM: <div data-testid="scene-page" data-state="loading|ready|error" data-src> with the SceneView
// "scene-view", or scene-error (+ scene-errors > li per validation error), or the list
// scene-index > a[data-testid=scene-link-<file>].
import type { PageProps } from "../../registry";
import { SceneEmbed, useLoad } from "../../components/SceneEmbed";

interface IndexEntry {
  file: string;
  type: string;
  title: string;
}

async function loadIndex(): Promise<IndexEntry[]> {
  const res = await fetch(new URL("scenes/index.json", window.location.href).href);
  if (!res.ok) throw new Error(`scenes/index.json: HTTP ${res.status}`);
  return (await res.json()) as IndexEntry[];
}

function SceneIndex() {
  const index = useLoad("index", loadIndex);
  return (
    <section className="panel stack" data-testid="scene-index" data-state={index.state}>
      <h2>Scenes</h2>
      <p className="muted">Precomputed views (packages/viz scene format), drawn without the server. Open one:</p>
      {index.state === "error" && <div className="banner banner-error">{index.message}</div>}
      <ul>
        {index.state === "ready" &&
          index.value.map((e) => (
            <li key={e.file}>
              <a href={`#/scene?src=scenes/${encodeURIComponent(e.file)}`} data-testid={`scene-link-${e.file}`}>
                {e.title}
              </a>{" "}
              <span className="muted">({e.type})</span>
            </li>
          ))}
      </ul>
    </section>
  );
}

export function ScenePage({ route }: PageProps) {
  const src = route.params.get("src");
  return src ? <SceneEmbed key={src} src={src} /> : <SceneIndex />;
}
