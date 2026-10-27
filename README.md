# ksmech

Predict organic reaction mechanisms as sequences of elementary steps, where each step is the
change it makes to the Lewis structure (a reaction matrix), predicted by a transformer over atom
and localized-orbital tokens whose pair track carries geometry-free Klopman-Salem quantities (the
spectral gap factor and the cross-centre coupling).

See [CLAUDE.md](CLAUDE.md) for the design and engineering rules, and
[docs/decisions.md](docs/decisions.md) for the running decision log.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.11 is managed by uv.

```bash
uv sync
uv run pre-commit install
uv run pytest
uv run ruff check .
```

torch comes from one of two dependency groups. `rocm` (the default) is the ROCm 7.2 build
for AMD cards. Run with `HSA_OVERRIDE_GFX_VERSION=10.3.0` on an RX 6600 / 6600 XT (gfx1032),
which runs the gfx1030 kernels. On an NVIDIA host, install the CUDA build instead:
`uv sync --no-group rocm --group cuda`, then `export UV_NO_SYNC=1`. Without it every `uv run`
syncs back to the default groups and reinstalls the ROCm build.

### tblite / xtb

`tblite` (Python bindings for GFN2-xTB) installs from PyPI on most Linux/macOS setups. Check with:

```bash
uv run python -c "import tblite"
```

If that fails (no wheel for your platform), install the bindings and the `xtb` binary with conda
instead, and make them visible to your environment:

```bash
conda install -c conda-forge tblite-python xtb
```

`ksmech.qm` is planned to keep a shell fallback that calls the `xtb` binary when the Python
bindings are unavailable (not yet implemented; arrives with the QM phase).

## Phases

1. **Data**: dataset download, parsing, canonical schema, splits
2. **QM featurization**: 3D embedding, xTB/tblite calculations, caching
3. **Orbital tokens**: localization, atom and orbital token construction
4. **Arrow alignment**: arrow parsing, arrow-to-orbital alignment, step application
5. **Physics baseline**: geometry-free pair features, the untrained gap-factor ranking and a
   LightGBM reference
6. **Model**: transformer with a pair track, edit decoder, losses
7. **Evaluation**: metrics, benchmarks, reports
8. **Scale-up**: FlowER-size training, QMugs pretraining

## Web app (ksmech-serve)

A local web application to predict mechanisms as reaction networks of elementary steps, read
each step as curved arrows and the orbital pairs that drive it, correct steps, curate and
export validated mechanisms, and browse PMechDB and curated steps. Backend in
`src/ksmech/serve/` (FastAPI, SQLite), frontend in `apps/ui/` (Vite, React, RDKit.js, 3Dmol.js).

The app is an add-on to the engine: it calls the engine's modules, the engine never imports
`ksmech.serve` (a test enforces it), and its web packages are the `serve` dependency group, not
dependencies of `ksmech` (`uv sync` installs the group by default; `uv sync --no-group serve`
gives the engine alone).

**Spec v9 (Predict and Correct).** Every step is a reaction matrix R = Delta(BE) on the pool
atoms (every H an atom), anchored on a forming bond. A step is applied as B + R with RDKit's
valence checks, the new pool is keyed by the sorted multiset of its sigma keys, equal pool keys
merge, and the network grows by beam search (beam 5, 8 steps by default) with cycles closed.
Each edge carries R, its anchor, its arrow reading (the decomposition of R into electron-pair
moves, chosen by the directed pooled gap factor where several exist), the orbital reading of
each forming bond (the channel (r, s) with the largest term of G_atom and the dominant
canonical MO pair (k, m) with its energies) and the product SMILES and sigma keys. Predictors
(`GET /api/predictors`):

- `floor`: the untrained physics floor. Anchors are atom pairs ranked by |G_atom|, the pooled gap
  factor over the valid donor/acceptor orbital pairs; the best are completed from their dominant
  channels with upstream `apply_predicted_arrows` (one candidate per allowed leaving bond). Its
  score is |G_atom| (1/eV), not a probability.
- `gbdt`: the geometry-free LightGBM ranker of `ksmech-physics gbdt`
  (`gbdt_lgbm_molecule_geomfree.npz` under the shared processed directory), ranking orbital
  pairs that are completed the same way; a ranking score.
- `network`: the spec-v9 model through a local checkpoint (`data/models/<source>/<run>/<file>.pt`,
  `GET /api/models`). `PredictIn.checkpoint {run, file, source?}` chooses one; the default is the
  best usable non-smoke run by validation top-1 (a smoke run is a `--limit` run or one named
  `smoke*`). The decoder is serve's (`serve/network_decode.py`, until the model gets one): the
  pool is encoded once on the CPU, the B (= beam) most probable anchors are completed by an exact
  best-first search over their most probable non-zero entries (at most 8 bond and 2 atom
  changes), completions that break charge conservation, a valence (B + R) or leave a non-metal at
  |formal charge| >= 2 are dropped with reason codes, and the rest are ranked by log P(R) and
  merged by product. Scores are probabilities: an edge's `logp` is log P(R) and a pathway's
  probability the product along it. END_MECH is untrained on PMechDB and ignored. The job's
  provenance records the checkpoint (run, file, sha256, epoch, val top-1); expansions keep it.
  The predictor is unavailable, with the reason, when torch or the model code is missing, no
  checkpoint exists, every checkpoint is refused (a dirty run) or every usable one failed to load.

**Embedding trace** (capability `embedding`: torch and the model code importable and one usable
local checkpoint). `GET /api/models` lists the checkpoints `data/models/<source>/<run>/<file>.pt`
of the shared root and the data root, with the encoder and the model size from `run.json`, the
epoch and validation metrics from `metrics.jsonl`, and whether the run is clean (a dirty run is
listed with its reason and refused, as `load_model` refuses it). `POST /api/embedding
{state_id | smiles, run, file?, source?}` is a job (kind `embedding`): it featurizes missing
fragments, then the compute worker loads the model on the CPU (kept per file version), traces
the pool (`ksmech-eval embedding`: h(0) by parts, every layer's attention and updates, the
anchor logits by term) and stores the page, the payload and a meta under `data/serve/embedding/`
(gitignored), keyed by the checkpoint's sha256, the pool state and its token digest, so a second
request answers at once. `GET /api/embedding/{id}` is the payload, `GET
/api/embedding/{id}/page[?theme=light|dark]` the self-contained page. In the UI: the step
panel's **Embedding** view (the step's parent state, a checkpoint picker, the page in an iframe
whose view hash is kept, "open in new tab") and the Workbench's **Model trace** button for the
computed molecule; both greyed with the reason while the capability is off.

Correct: `POST /api/step/check {state_id, matrix}` validates an edited R live (issue codes with
the atoms involved, the product, the arrows and readings); `POST /api/jobs/{id}/override
{parent_state_id, matrix}` adds a user step (predictor "user") and its child node, and `POST
/api/jobs/{id}/expand {node_id}` expands a node. `POST /api/predict` with `predictor: "user"`
starts a network from the reactants alone. Curated mechanisms store R and the reading of every
step; the re-check ranks the curated product among a predictor's products (sigma keys);
`GET /api/curate/export/pmechdb` downloads the curated steps as PMechDB raw records (mapped
reaction SMILES and the arrow code of the reading), a file, never training data. The spec-v3
panels (`candidates`, `physics`) stay off; the contact view was removed with the contact
model. The contract the UI is built against is `docs/serve_v9_contract.md`.

**Workbench** (`#/workbench`, the default page; capability `workbench`). Draw a molecule in the
ChemDraw-like sketcher (bond, chain, ring, charge and eraser tools, ChemDraw keys on the hovered
atom, SMILES / molfile / InChI import and export) or open `#/workbench?smiles=...`, press
Compute (`POST /api/molecule`: the QM record and tokens, display only, never the pair analysis),
then look at it:

- a 3D view (3Dmol.js) of the record geometry: ball and stick, sticks, spacefill, wireframe or
  lines, hydrogens on or off, labels (element, index, Mulliken charge or any atom property),
  atoms coloured by any property with a colour bar, Wiberg bond labels, orbital centroids,
  click-to-measure distances, angles and dihedrals, and PNG / xyz / molfile export;
- surface layers, several at once, each with its own isovalue, opacity, solid or mesh and
  colours (client-side; only kind, set, index or grid quality refetch), with the grid integral
  shown and a cube download: an orbital psi or |psi|^2, the total valence density, the
  frozen-orbital Fukui f+ / f- and the dual descriptor, and the ESP painted on the density;
- the orbital sets: canonical MOs, IBO and Pipek-Mezey (Mulliken), each LMO set from a Cholesky
  or a canonical start (ibo/cholesky is the model's own token set); an energy-level diagram
  (degenerate levels side by side, occupations, HOMO / LUMO; for an LMO, its spectral weights
  on the canonical levels) and the orbital list;
- sortable tables: global (energy, HOMO, LUMO, gap, mu, eta, omega, provenance), atoms (Mulliken,
  finite-difference and frozen-orbital Fukui, dual, frontier weights, %Vbur, Wiberg valence,
  chirality; hybridization, aromaticity and functional group greyed as diagnostics, never model
  input), bonds (Wiberg index, length) and the RDKit.js descriptors of the sketch. Clicking an
  atom row selects it in 3D and highlights the drawn atom in the sketch, while the sketch is
  unchanged since the compute.
- 2D layouts from the server (`POST /api/depict2d`, RDKit): "Clean" with a style next to it,
  textbook (RDKit's ring templates or CoordGen, whichever has fewer defects) or perspective (a
  library of chemist-style drawings: chair cyclohexanes and pyranoses, norbornanes such as
  camphor, bicyclo[2.2.2]octane, tropane, adamantane, cubane, cis- and trans-decalin, the steroid
  core; used only where it fits the molecule's 3D shape); without the server Clean falls back
  to RDKit.js's CoordGen with a notice. Once computed, "3D→2D" (Shift+D) redraws the sketch as
  the 3D view shows the record now (RDKit's depiction matched to the view, the view itself as a
  ring template, or a perspective template, whichever reads best), wedges keeping the record's
  stereo and the bond behind each crossing drawn broken; "Break crossings" (Shift+X) toggles
  the breaks. The info bar reports the method, the defects and the stereo check. The sketch
  stays linked to the record, since only coordinates and wedges that keep its stereo change.
  `uv run --no-sync ksmech-serve gallery [--out DIR]` draws a fixed gallery of 22 hard
  molecules with every method, to judge the layouts by eye.
- rotate and scale in the sketcher: with two or more atoms selected (rectangle or Ctrl+A) a
  frame shows a rotation handle (drag; Shift snaps to 15°) and corner handles (drag to scale;
  Shift for 5 % steps); the action row rotates by 15° (R / Shift+R), flips (Alt+H / Alt+V; a
  flip swaps wedges and hashes so the drawing stays the same stereoisomer), and its "⋯" menu
  turns by 90°, scales by 10 % (Ctrl+= / Ctrl+-) or resets the bond length. Without a selection
  they apply to the whole drawing; each is one undo step and keeps the sketch linked.
- radicals in the sketcher: the radical tool (one dot) and the diradical / carbene tool (two
  dots) on an atom, again to clear; the "." key cycles none, one, two on the hovered atom or the
  selection. The dots sit in the largest free angle by the atom, and the molfile carries them as
  `M  RAD` (2 = doublet, 3 = triplet), so RDKit drops the H count (`C[CH2]`, `[CH2]`). Radicals
  can be drawn but not computed yet: Compute is off and its tooltip and a note say why (the
  server answers RADICAL_NOT_SUPPORTED). Clean and 3D→2D keep them by atom id.
- an R/S toggle in the sketcher: RDKit.js's CIP labels (new labeler) beside the stereocentres
  (R, S, pseudo-asymmetric r, s, and "?" for a possible stereocentre the drawing leaves
  undefined) and E/Z beside the stereo double bonds, plus a list in the info bar ("C2 S · C3=C4
  E"). They follow every edit; a flip keeps the stereoisomer, so the labels stay.
- the orbital panel shows the energy-level diagram with the orbital list and set stats beside
  it; the thin bar between them folds the list away (a narrow panel stacks them). The diagram
  fills its panel; the mouse wheel zooms about the cursor, dragging (or Shift+wheel) pans, a
  double-click resets, the + / − / all / frontier buttons zoom, fit every level or show the
  frontier window (+, −, 0 and the arrow keys when it has focus). Degenerate MOs sit side by
  side with their occupation arrows, the HOMO−1 / HOMO / LUMO / LUMO+1 blocks are labelled
  with their energies, and the "degenerate ≤" field (0.001–0.5 eV, default 0.05, the server's
  value; kept with the preferences) regroups the levels and the frontier blocks.
- atom labels are placed collision-aware in every drawing style, on the canvas and in the
  SVG / PDF / PNG export and the chemfig hints: the H count and the charge move to a free side,
  then the subscript and charge shrink, and as a last resort a terminal atom's label moves off
  with a short leader (what still overlaps is counted in the canvas's data-label-issues).
  Separate fragments (a salt's ions) that come too close are spread apart after every Clean,
  as its own undo step, and by the Tidy fragments button (whole fragments move; bonds and
  stereo never change).
- a customizable layout: every panel (sketch, 3D view, orbitals, surfaces, properties) has a
  header that collapses it, hides it and moves it left, right, up or down (into a third column
  or the full-width row); drag the splitters between columns and under the sketch, the 3D view,
  the level diagram and the tables. The "Layout" menu has the presets (default, sketch-, 3D- and
  orbitals-focused), reset layout, a show / hide box per panel, and the preferences: sketch
  font size, bond line width, atom colours, label style (skeletal, all atoms, no implicit H),
  the default 3D representation, the default orbital isovalue and the theme (auto, light,
  dark). Layout and preferences are kept per browser (localStorage).
- drawing styles and publication export in the sketcher: a style select in the toolbar (also in
  Layout → Preferences) offers Screen (the app's look), ACS Document 1996 (ChemDraw's style
  sheet: 14.4 pt = 0.508 cm bonds, 0.6 pt lines, 2.0 pt wedges, 2.5 pt hash spacing, 18 %
  double-bond spacing, 1.6 pt margins, Arial 10 pt, black and white), Total synthesis (the
  app's own bold preset, not an official standard: 15 pt bonds, 1.1 pt lines, 3.2 pt tapered
  wedges, 20 % spacing, Arial 11 pt) and Custom; the ✎ editor has every field (bond length,
  line / bold / hash widths, hash and double-bond spacing, inner-line shortening, terminal C=X
  rule, margin, label font, size, weight, subscript and charge size, superscript or circled
  charges, colour, crossing gap, wedge shape, chain angle, labelled carbons or methyls) and
  "Actual size". Styles are render-only (the drawing, its stereo and its undo history never
  change). The Export panel's image section writes the drawing in the style at its true size:
  SVG (pt), PNG at 150 to 1200 dpi with the resolution in the file, a vector PDF (Helvetica, no
  embedded fonts), optional transparent background and R/S labels; "Copy image" (PNG + SVG),
  "Copy SVG", "Copy LaTeX" (a figure including scheme.pdf at its physical width) and "Copy
  chemfig" (`\usepackage{chemfig}` code that redraws the molecule natively). The thumbnail in
  the canvas corner is a drag handle carrying scheme.png, scheme.pdf and scheme.svg; what each
  target (Word, Overleaf, a file manager) accepts is listed in docs/decisions.md.
- the point group (`GET /api/molecule/{id}/symmetry?tolerance=`, numpy in the API process): the
  Global tab shows the Schoenflies symbol and order of the record geometry (C1 ... Ih, C∞v,
  D∞h), a tolerance in Å (0.01–0.5, default 0.1: every atom must land within it of an atom of
  the same element), the RMS of the symmetrized structure, a hint when a looser tolerance gives
  a higher group ("would be Td at 0.30 Å", a click applies it), the symmetry elements (C_n with
  C2′ / C2″, S_n, σh / σv / σd, i; a click highlights one in 3D) and the equivalent-atom
  classes (a click selects them). The 3D toolbar's "symmetry" draws every axis (labelled rods),
  mirror plane (translucent disc) and the inversion centre; "classes" colours the atoms by
  class. The response also carries the symmetrized coordinates.

Defaults and approximations: the density is valence-only on GFN2's minimal basis (an STO-nG
fit, so its integral is shown next to N_valence), the ESP is an approximate point-charge
potential from the Mulliken charges, and there is no Foster-Boys localization (tblite exposes no
dipole integrals). Selection state is in the hash (`m`, `set`, `start`, `orb`, `rep`, `layers`,
`q`); the drawing persists per tab in sessionStorage.

```bash
uv sync
PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1 npm --prefix apps/ui ci && npm --prefix apps/ui run build
uv run ksmech-serve --port 8000                       # threads are pinned to 1 by the CLI
# from your PC:
ssh -L 8000:localhost:8000 <server>                    # then open http://localhost:8000
# dev mode (hot reload of the UI):
uv run ksmech-serve --port 8000 --dev                  # on the server
npm --prefix apps/ui run dev                           # Vite on 127.0.0.1:5173, proxies /api
ssh -L 5173:localhost:5173 <server>                    # open http://localhost:5173
# UI development without xTB:
uv run ksmech-serve --port 8000 --chemistry synthetic  # banner: synthetic chemistry
```

Other commands: `uv run ksmech-serve recheck [--stale-only] [--predictor P] [--mechanism ID
...]` (offline, refused with exit 2 while a server runs on the same data root: use the web UI
or `POST /api/curate/recheck` then), `uv run ksmech-serve export --out FILE [--status ...]`
(the PMechDB-format file of the curated steps), `uv run ksmech-serve schema-ts`
(regenerates `apps/ui/src/api/types.gen.ts` from the pydantic models), `uv run ksmech-serve
probe-cache --smiles S ...` (exit 0 when every species loads from the shared token cache). The
API is under `/api` (OpenAPI page at `/api/docs`); `/api/health` and `/api/meta` report the
worker, the chemistry, the spec version and the physics fingerprint.

**Chemistry modes** (`--chemistry` or `KSMECH_SERVE_CHEMISTRY`):

- `real` (default): QM records and tokens come from the caches; a missing species runs
  GFN2-xTB in the compute worker (seconds to minutes per species).
- `cached`: the caches only; a missing species is `FEATURIZATION_DISABLED`, never xTB.
- `synthetic`: charge-iterated extended Hueckel on the GFN2 basis (`SYNTHETIC-EHT/v1`), for
  tests and UI development; every page shows a banner and every response says so.

**Environment variables.**

| Variable | Default | Meaning |
|---|---|---|
| `KSMECH_DATA_ROOT` | the current directory | root of `data/` (serve writes only `data/serve/` and `data/curation/`) |
| `KSMECH_SHARED_CACHE_ROOT` | `KSMECH_DATA_ROOT` | read-only shared `data/cache/qm`, `data/cache/tokens`, `data/cache/bond_index`, the GBDT model and `data/models/` |
| `KSMECH_SERVE_CHEMISTRY` | `real` | `real`, `cached` or `synthetic` |
| `KSMECH_SERVE_WORKERS` | 1 | compute worker processes, clamped to [1, 2] |
| `KSMECH_SERVE_GBDT` | `<shared>/data/processed/pmechdb/gbdt_lgbm_molecule_geomfree.npz` | the LightGBM ranker of the `gbdt` predictor |
| `KSMECH_SERVE_AUTO_INDEX` | 0 | 1 (or `--index-on-start`) builds the PMechDB browse index at startup |
| `KSMECH_SERVE_DIST` | `apps/ui/dist` | the built frontend |

**Data locations.**

- `data/serve/`: `qm/`, `tokens/` and `bond_index/` (species computed by the server and their
  Wiberg sidecars), `pools/<state_id>/` (pool states and their pooled-gap analyses), `cubes/`,
  `jobs/<job_id>/` (job, network, events), `index/` (PMechDB browse index),
  `logs/failures.jsonl` (every reason-coded skip), `run/` (server lock, task pid files).
- `data/curation/curation.sqlite`: curated mechanisms, steps and re-checks. The export is a
  download (`GET /api/curate/export/pmechdb`, `ksmech-serve export --out FILE`); nothing is
  written to `data/processed` and curated steps are never training data.
- The shared `data/cache/` and `data/processed/pmechdb/` are only read, never written.

**Host safety.** All heavy compute (xTB, localization, pair analyses, cube grids) runs in one
spawned compute worker with one BLAS / OpenMP / torch thread. Do not raise
`KSMECH_SERVE_WORKERS` while a QM batch runs, and keep heavy processes at most 4 host-wide. The
PMechDB browse index (about 10 minutes on the worker) is built only on request: the "Build
PMechDB index" button (`POST /api/browse/index`) or `--index-on-start`.

**Security.** There is no authentication. By default the server binds only to 127.0.0.1 and is
reached through the SSH tunnel above. `uv run ksmech-serve --port 8000 --host 0.0.0.0` (or one
of the machine's addresses) opens it to the local network at `http://<server-ip>:8000`: anyone
on that network can then run predictions, edit the curation database and download exports. The
Host allow-list and the same-origin guard then also accept the machine's addresses and host
names; any other name is refused.

**Tests.** The serve tests (`tests/test_serve_*.py`) use synthetic or toy chemistry and never
run xTB; on a loaded host run them one file at a time with `OMP_NUM_THREADS=1`. The Playwright smoke test uses the system Chrome or Chromium (Playwright browsers are
never downloaded): `KSMECH_E2E=1 uv run pytest tests/test_serve_e2e.py`, or `npm --prefix
apps/ui run e2e`.
