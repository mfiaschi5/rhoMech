# @ksmech/viz

The ksmech visualizations as React components that take **props only**: a molecule in 2D and
3D, its orbitals, one elementary step (Textbook, Animated, Orbitals) and the mechanism network
(Scheme, Pathways). No component fetches, reads the app's store or router: data comes in as
props typed by the ksmech-serve API schemas. The same components run in the app (`apps/ui`,
fed by the live API) and in static pages such as the wiki (fed by precomputed JSON **scenes**,
so reading a page never triggers QM compute).

## Wiring

A source package: TypeScript compiled by whoever imports it, no install and no build step.

- **apps/ui** resolves `@ksmech/viz` (= `src/index.ts`) and `@ksmech/viz/<path>` (= `src/<path>`)
  through one module, `apps/ui/scripts/viz-alias.mjs`: Vite (`vite.config.ts`: alias, `dedupe`
  of react, `server.fs.allow`), node (`scripts/viz-node-hooks.mjs`, a resolve hook for `npm test`
  and `npm run scene:export`) and tsc (`tsconfig.json` `paths`, the same mapping written out).
  The package has no `node_modules`: its bare imports resolve from `apps/ui`.
- **Peer dependencies** (`package.json`): `react`, `react-dom`, `@rdkit/rdkit`, `3dmol`. RDKit.js
  and 3Dmol.js are not bundled: the host page loads their classic scripts, which set
  `window.initRDKitModule` and `window.$3Dmol` (`/vendor/RDKit_minimal.js` with
  `/vendor/RDKit_minimal.wasm`, `/vendor/3Dmol-min.js`; `apps/ui/scripts/vendor.mjs` copies
  them). Only their types are imported.
- **Styles**: `src/theme.css` holds the colour tokens (light and dark); each view imports its own
  stylesheet (`step/step.css`, `network/netview.css`, `orbitals/orbitals.css`,
  `scene/scene.css`). The shared component classes (`btn`, `badge`, `muted`, `code`, `stack`,
  `mol-svg`, `pv-*`) are still in `apps/ui/src/styles.css` and `features/predict/predict.css`;
  a host page loads those too until they move here.
- **API types**: `src/apiTypes.ts` re-exports `apps/ui/src/api/types.gen.ts` (generated from the
  pydantic schemas by `ksmech-serve schema-ts`) as types only. It is the package's one link to
  the app and is erased at runtime.
- **The boundary** (`src/vizBoundary.test.ts`, run by `npm --prefix apps/ui test`): nothing under
  `packages/viz` imports `apps/ui` (apart from that type-only bridge), a package outside the peer
  dependencies, or calls `fetch` / XHR / WebSocket; `scene/load.ts` is the one module that
  fetches, and it reads static scene files only.

## Layout and exports

| Directory | What it holds |
|---|---|
| `scene/` | the scene format (`types.ts`), `validateScene`, `loadScene`, `SceneView` and its three views |
| `step/` | `StepTextbook`, `StepAnimated`, `StepOrbital`, `StepScheme`, `ProductView` and their pure helpers (`stepSides`, `stepGeometry`, `stepMorph`, `stepScheme`, `stepScene`, `stepRoute`, `stepText`, `stepCube`, `stepExport`) |
| `network/` | `SchemeView`, `PathwayExplorer`, `StateCard`, `netLayout`, card thumbnails, `network.ts` (labels and captions) |
| `pool/` | a pool state read as a Lewis structure (`poolStructure`), its 2D layout (`poolView`), the reaction matrix (`rmatrix`), arrow paths and the R overlay |
| `sketch/` | the 2D renderer of the workbench sketcher: model, geometry, styles (ACS 1996 and the screen styles), drawing, labels, molfile reader, SVG/PNG export |
| `chem/` | RDKit.js (`rdkit.ts`, `depict.ts` with exact atom positions, `svgFit.ts`, `arrowGeom.ts`) and 3Dmol.js helpers (`threed.ts`) |
| `viewer/` | the 3Dmol viewer hook, the molecule model and its overlays (`scene.ts`), `Molecule3D` |
| `orbitals/` | `LevelDiagram` and its level arithmetic |
| `components/` | `MolDepiction`, `SmilesDepiction`, `ArrowOverlay` |

`src/index.ts` is the public surface: the scene API (`SCENE_KIND`, `SCENE_VERSION`,
`validateScene`, `loadScene`, `SceneError`, `assetLoader`, `NO_ASSETS`, `SceneView`,
`MoleculeSceneView`, `StepSceneView`, `NetworkSceneView`), the molecule components
(`MolDepiction`, `SmilesDepiction`, `ArrowOverlay`, `Molecule3D`, `LevelDiagram`, `useGLViewer`,
`loadRDKit`, `useRDKit`), the step views (`StepTextbook`, `StepAnimated`, `StepOrbital`,
`StepScheme`, `ProductView`, `StepEnergy`, `StepCaption`, with `predictedStep`, `stepSidesOf`,
`sidesOf`, `stepCaptions`, `hoverOf`, `NO_HOVER`), the network views (`SchemeView`,
`PathwayExplorer`, `StateCard`, `NETWORK_VARIANTS`, the layouts of `netLayout`), and the API types
(`Api.*`, plus `EdgeOut`, `NetworkOut`, `NodeOut`, `MatrixIO`, ...). The sub-modules are
importable as `@ksmech/viz/<dir>/<module>`.

### The main props

- `SceneView { scene: SceneDoc; assets?: AssetLoader; testId? }`: one scene, dispatched on
  `scene.scene.type`.
- The step views share `StepVariantProps { sides: StepSides; step: ShownStep; prefix;
  productSmiles; sigmaKeys; label(i); above; below; hover; onHover }`. `step` is
  `predictedStep(edge)` (or an alternative the app checked), `sides` is
  `stepSidesOf(sourceDepiction, productDepiction, edge, step)`; the app builds the same values
  from the API with its hooks. `StepOrbital` also takes `orbital: OrbitalData { key; geometry:
  GeometryOut; depiction: DepictionOut; loadCube(token) => Promise<string>; pending?; error? }`.
- `SchemeView` / `PathwayExplorer`: `NetViewProps { net: NetworkOut; selectedNode;
  selectedEdge; pathway; pathwayFocus; probabilities; onSelectNode; onSelectEdge;
  onSelectPathway; onExpand?; expandStatus? }`. Without `onExpand` a state cannot be expanded
  (a scene is a snapshot); the app passes one that posts `/jobs/{id}/expand`.
- `MolDepiction { depiction: DepictionOut; fetchServer?; ... }`: the server tier of the exact
  atom positions is a prop (`apps/ui/src/components/MolDepiction.tsx` passes the app's).
- `Molecule3D { geometry: MoleculeGeometryOut; cube?; occupied?; iso?; rep?; height? }`,
  `LevelDiagram { canonical: OrbitalRowOut[]; lmos; selected; window; onPickCanonical; ... }`.

## The scene format (version 1)

One JSON document, its shape in `src/scene/types.ts` and in the JSON Schema
`scene.schema.json`:

```json
{ "kind": "ksmech-scene", "version": 1,
  "scene": { "type": "molecule" | "step" | "network", "title": "...", ... },
  "provenance": { "exporter", "exported_at", "server": {spec_version, chemistry, qm_method, ...},
                  "request": {...}, "predict": Provenance | null, "notes": [...] } }
```

Each type holds exactly what its views read, taken from the API responses (the types of
`types.gen.ts`, narrowed with `Pick` where a view reads less):

| Type | Holds | From |
|---|---|---|
| `molecule` | `smiles`, `molecule` (id, key, charge, multiplicity, chemistry, method), `geometry: MoleculeGeometryOut`, `orbitals` (the canonical `OrbitalSetOut` rows, HOMO index), `cubes` by orbital index, `selected` | `POST /molecule`, `GET /molecule/{id}/geometry`, `/orbitals`, `/cube` |
| `step` | `edge` (the `EdgeOut` fields the views read), `probabilities`, `source` (`depiction`, `geometry: GeometryOut \| null`, `cubes` by donor / acceptor token), `product` (`depiction`) or null, `variants` | `GET /jobs/{id}`, `/pool/{state}/depiction`, `/geometry`, `/orbital/{token}` |
| `network` | `network: NetworkOut`, `variant` | `GET /jobs/{id}` |

Large payloads are **assets**: files next to the scene, `{path, bytes, sha256, media_type}` with
`path` relative to the scene file (no `..`, no scheme). `loadScene(url)` fetches and validates
the document and returns `{doc, url, assets}`; `assets(ref)` fetches an asset once and checks its
size and sha256 (with WebCrypto: https or localhost) before a view reads it. `validateScene(x)`
is a hand-written structural check (no schema library): every field a view relies on, integer
indices, asset references and ids that must point into the document (a cube's token among the
readings, an edge's ends among the nodes), with errors as JSON paths. A reader refuses any other
`kind` or `version`; a change of shape is a new version.

Not in a scene (yet): a step's alternatives (they need `POST /step/check`), the workbench's
density / ESP / Fukui surfaces, the interactive pool canvas of the Correct page.

## Making scenes

`apps/ui/scripts/scene-export.ts` reads a running server and writes `DIR/NAME.json`, its assets
under `DIR/NAME/` and the list `DIR/index.json`:

```bash
npm --prefix apps/ui run scene:export -- --type step --job J --edge E --out DIR \
    [--cubes all|dominant|none] [--cube-spacing 0.35] [--cube-padding 2.5] [--no-3d]
npm --prefix apps/ui run scene:export -- --type molecule --smiles 'C=CC=O' --out DIR \
    [--orbitals homo,lumo] [--cube-stride 2]
npm --prefix apps/ui run scene:export -- --type network --job J --out DIR
# common: --url http://127.0.0.1:8000 --name NAME --title TEXT --token BEARER
```

Pool-orbital cubes are asked at a coarser grid (`spacing`, `padding` of
`GET /pool/{state}/orbital/{token}`); workbench cubes come at a fixed draft grid and are thinned
(`--cube-stride`: every n-th point per axis). The examples in `apps/ui/public/scenes/` (a
methoxide + chloromethane SN2 step and its network from the floor predictor, acrolein's frontier
orbitals) were made this way from user-style input, never from dataset samples; `#/scene` in the
app lists them and `#/scene?src=scenes/<file>` draws one with no API call
(`apps/ui/e2e/scene.spec.ts` aborts every `/api/` request and checks the views).

## Embedding in the wiki

The wiki (static HTML built from MDX, e.g. Astro) ships scene files with its pages and mounts the
views as client-side islands:

```tsx
// wiki/src/components/Scene.tsx
import { loadScene, SceneView, type LoadedScene } from "@ksmech/viz";
import "@ksmech/viz/theme.css";
import { useEffect, useState } from "react";

export default function Scene({ src }: { src: string }) {
  const [scene, setScene] = useState<LoadedScene | null>(null);
  useEffect(() => void loadScene(src).then(setScene), [src]);
  return scene ? <SceneView scene={scene.doc} assets={scene.assets} /> : <p>loading…</p>;
}
```

```mdx
import Scene from "../components/Scene.tsx";

The lone pair of the methoxide oxygen attacks the σ* orbital of the C–Cl bond:

<Scene client:visible src="/scenes/sn2-step.json" />
```

The wiki's build resolves `@ksmech/viz` with the same alias (or a workspace link), loads the
RDKit.js and 3Dmol.js scripts in its layout, and can run `validateScene` over its scene files so
a broken scene fails the build rather than a page.
