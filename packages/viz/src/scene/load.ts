// loadScene: a scene file and its assets as static files (plain GETs of the scene URL and of
// paths relative to it; never the ksmech-serve API). The document is validated before it is
// returned, and every asset is checked against its byte size and sha256 (when the page has
// crypto.subtle: https or localhost) before a view reads it. This module is the only one of the
// package that fetches (vizBoundary.test.ts).
import type { AssetRef, SceneDoc } from "./types";
// @ts-ignore TS5097: see validate.ts
import { isRelativeAssetPath, validateScene } from "./validate.ts";

export class SceneError extends Error {
  /** The validation errors (JSON paths) when the document did not pass. */
  readonly errors: readonly string[];
  constructor(message: string, errors: readonly string[] = []) {
    super(message);
    this.name = "SceneError";
    this.errors = errors;
  }
}

/** The text of one asset of the scene (cached per path; a failed load is retried). */
export type AssetLoader = (ref: AssetRef) => Promise<string>;

export interface LoadedScene {
  doc: SceneDoc;
  /** The absolute URL the scene was read from (its assets are relative to it). */
  url: string;
  assets: AssetLoader;
}

type Fetch = (url: string, init?: { signal?: AbortSignal }) => Promise<Response>;

export interface LoadOptions {
  /** A fetch to use instead of the global one (tests, a wiki build step). */
  fetch?: Fetch;
  signal?: AbortSignal;
}

const pageUrl = () => (typeof location === "undefined" ? "http://localhost/" : location.href);

export async function loadScene(url: string, o: LoadOptions = {}): Promise<LoadedScene> {
  const get: Fetch = o.fetch ?? ((u, init) => fetch(u, init));
  const at = new URL(url, pageUrl());
  const res = await get(at.href, { signal: o.signal });
  if (!res.ok) throw new SceneError(`${url}: HTTP ${res.status}`);
  let json: unknown;
  try {
    json = await res.json();
  } catch (err) {
    throw new SceneError(`${url} is not JSON: ${String(err)}`);
  }
  const v = validateScene(json);
  if (!v.ok) throw new SceneError(`${url} is not a valid ksmech scene (${v.errors.length} error${v.errors.length === 1 ? "" : "s"})`, v.errors);
  return { doc: v.doc, url: at.href, assets: assetLoader(at, get) };
}

/** The lowercase hex sha256 of `bytes`, or null without WebCrypto (an insecure origin). */
export async function sha256Hex(bytes: Uint8Array): Promise<string | null> {
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) return null;
  const digest = new Uint8Array(await subtle.digest("SHA-256", bytes as BufferSource));
  return [...digest].map((b) => b.toString(16).padStart(2, "0")).join("");
}

/** Throws when `bytes` are not the file `ref` describes. */
export async function checkAssetBytes(ref: AssetRef, bytes: Uint8Array): Promise<void> {
  if (bytes.length !== ref.bytes) throw new SceneError(`asset ${ref.path}: ${bytes.length} bytes, the scene says ${ref.bytes}`);
  const hash = await sha256Hex(bytes);
  if (hash !== null && hash !== ref.sha256) throw new SceneError(`asset ${ref.path}: sha256 ${hash.slice(0, 12)}…, the scene says ${ref.sha256.slice(0, 12)}…`);
}

async function fetchAsset(base: URL, ref: AssetRef, get: Fetch): Promise<string> {
  if (!isRelativeAssetPath(ref.path)) throw new SceneError(`asset ${ref.path}: not a relative path inside the scene's folder`);
  const res = await get(new URL(ref.path, base).href);
  if (!res.ok) throw new SceneError(`asset ${ref.path}: HTTP ${res.status}`);
  const bytes = new Uint8Array(await res.arrayBuffer());
  await checkAssetBytes(ref, bytes);
  return new TextDecoder().decode(bytes);
}

/** The asset loader of a scene read from `base`. */
export function assetLoader(base: URL, get: Fetch): AssetLoader {
  const cache = new Map<string, Promise<string>>();
  return (ref) => {
    let p = cache.get(ref.path);
    if (!p) {
      p = fetchAsset(base, ref, get);
      cache.set(ref.path, p);
      p.catch(() => cache.delete(ref.path));
    }
    return p;
  };
}

/** For scenes without assets: every asset read fails. */
export const NO_ASSETS: AssetLoader = (ref) => Promise.reject(new SceneError(`asset ${ref.path}: this scene was given no asset loader`));
