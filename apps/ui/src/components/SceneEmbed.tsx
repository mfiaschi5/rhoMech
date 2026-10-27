// SceneEmbed: one scene file (packages/viz scene format) loaded from this site and drawn by
// @ksmech/viz's SceneView, without any API call. Shared by the scene viewer (#/scene) and the
// wiki's scene figures (<figure class="wiki-scene" data-scene-src=...>).
//
// Props:
//   src     string   the scene URL, resolved against the page; another origin is refused
//   testId  string   default "scene-page"
//
// DOM: <div data-testid=<testId> data-state="loading|ready|error" data-src> with the SceneView
// "scene-view", or scene-error (+ scene-errors > li per validation error).
import { loadScene, sameOriginUrl, SceneView, type LoadedScene } from "@ksmech/viz";
import { useEffect, useState } from "react";

export type Load<T> = { state: "loading" } | { state: "ready"; value: T } | { state: "error"; message: string; errors: readonly string[] };

/** A promise as a loading / ready / error state, reloaded when `key` changes. */
export function useLoad<T>(key: string, load: () => Promise<T>): Load<T> {
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

export function SceneEmbed({ src, testId = "scene-page" }: { src: string; testId?: string }) {
  const url = sameOriginUrl(src, window.location.href);
  const scene = useLoad<LoadedScene>(url ?? "", () => (url ? loadScene(url) : Promise.reject(new Error(`${src}: only scenes of this site are drawn`))));
  return (
    <div className="stack" data-testid={testId} data-state={scene.state} data-src={src}>
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
