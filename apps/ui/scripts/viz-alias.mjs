// @ksmech/viz (packages/viz) is a source package: no install and no build step. Its TypeScript
// is compiled by whoever imports it, so three resolvers learn its name here:
//   - Vite (vite.config.ts, and the UI probe of tests/test_serve_e2e.py): vizResolve();
//   - node (`npm test`, `npm run scene:export`, through scripts/viz-node-hooks.mjs): the resolve() hook below;
//   - tsc: the "paths" of tsconfig.json (the same mapping, written out).
// `@ksmech/viz` is src/index.ts and `@ksmech/viz/<path>` is src/<path> (.ts / .tsx added when
// the specifier has no extension). The package has no node_modules of its own: its bare imports
// (react, react-dom) resolve from apps/ui (Vite: resolve.dedupe; node: the parent rewritten).
import { existsSync } from "node:fs";
import { dirname, extname, join, resolve as resolvePath } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

export const UI_DIR = resolvePath(dirname(fileURLToPath(import.meta.url)), "..");
export const VIZ_DIR = resolvePath(UI_DIR, "..", "..", "packages", "viz");
export const VIZ_SRC = join(VIZ_DIR, "src");

/** Vite's resolve options for the package: aliases first, then dedupe for its bare imports. */
export function vizResolve() {
  return {
    alias: [
      { find: /^@ksmech\/viz$/, replacement: join(VIZ_SRC, "index.ts") },
      { find: /^@ksmech\/viz\/scene\.schema\.json$/, replacement: join(VIZ_DIR, "scene.schema.json") },
      { find: /^@ksmech\/viz\/(.*)$/, replacement: `${VIZ_SRC}/$1` },
    ],
    dedupe: ["react", "react-dom"],
  };
}

/** The file of a `@ksmech/viz[/path]` specifier, or null for any other specifier. */
export function vizFile(specifier) {
  if (specifier === "@ksmech/viz") return join(VIZ_SRC, "index.ts");
  if (!specifier.startsWith("@ksmech/viz/")) return null;
  const path = join(VIZ_SRC, specifier.slice("@ksmech/viz/".length));
  if (extname(path)) return path;
  return [".ts", ".tsx"].map((ext) => path + ext).find((p) => existsSync(p)) ?? `${path}.ts`;
}

const VIZ_URL = pathToFileURL(VIZ_DIR).href + "/";
const UI_URL = pathToFileURL(join(UI_DIR, "package.json")).href;
const bare = (s) => !/^(\.|\/|node:|file:|data:)/.test(s);

/** node:module resolve hook (registered by scripts/viz-node-hooks.mjs). */
export async function resolve(specifier, context, next) {
  const file = vizFile(specifier);
  if (file) return next(pathToFileURL(file).href, context);
  if (bare(specifier) && context.parentURL?.startsWith(VIZ_URL)) return next(specifier, { ...context, parentURL: UI_URL });
  return next(specifier, context);
}
