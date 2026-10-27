// ScenePage, #/scene?src=<url>: one scene file (packages/viz scene format) drawn by @ksmech/viz's
// SceneView with no API call at all: the proof that a static page (the wiki) can embed the app's
// views from precomputed JSON. `src` is resolved against the page and must stay on its origin
// (a scene is data, but only this site's files are drawn). Without `src`: the list of
// public/scenes/index.json (the committed examples, written by `npm run scene:export`).
//
// DOM: <div data-testid="scene-page" data-state="loading|ready|error" data-src> with the
// SceneView "scene-view", or scene-error (+ scene-errors > li per validation error), or the list
// scene-index > a[data-testid=scene-link-<file>].
import { loadScene, SceneView, type LoadedScene } from "@ksmech/viz";
import { useEffect, useState } from "react";
import type { PageProps } from "../../registry";
import { sameOriginUrl } from "./sceneUrl";

interface IndexEntry {
  file: string;
  type: string;
  title: string;
}

type Load<T> = { state: "loading" } | { state: "ready"; value: T } | { state: "error"; message: string; errors: readonly string[] };

function useLoad<T>(key: string, load: () => Promise<T>): Load<T> {
  const [s, setS] = useState<Load<T>>({ state: "loading" });
  useEffect(() => {
    let live = true;
    setS({ state: "loading" });
    load().then(
      (value) => live && setS({ state: "ready", value }),
      (err: unknown) => live && setS({ state: "error", message: err instanceof Error ? err.message : String(err), errors: (err as { errors?: string[] }).errors ?? [] }),
    );
    return () => {
      live = false;
    };
    // `key` stands for `load`
  }, [key]);
  return s;
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

function SceneFile({ src }: { src: string }) {
  const url = sameOriginUrl(src, window.location.href);
  const scene = useLoad<LoadedScene>(url ?? "", () => (url ? loadScene(url) : Promise.reject(new Error(`${src}: only scenes of this site are drawn`))));
  return (
    <div className="stack" data-testid="scene-page" data-state={scene.state} data-src={src}>
      {scene.state === "loading" && <p className="muted">loading {src}…</p>}
      {scene.state === "error" && (
        <div className="banner banner-error" data-testid="scene-error">
          {scene.message}
          {scene.errors.length > 0 && (
            <ul data-testid="scene-errors">
              {scene.errors.slice(0, 20).map((e, k) => (
                <li key={k}>
                  <code className="code">{e}</code>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
      {scene.state === "ready" && <SceneView scene={scene.value.doc} assets={scene.value.assets} />}
    </div>
  );
}

export function ScenePage({ route }: PageProps) {
  const src = route.params.get("src");
  return src ? <SceneFile key={src} src={src} /> : <SceneIndex />;
}
