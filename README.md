# ksmech

Predict organic reaction mechanisms as sequences of elementary steps, where each step is a set of
arrows from a donor localized orbital to an acceptor localized orbital, scored by a transformer
whose attention is biased by the Klopman-Salem equation.

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
5. **Physics baseline**: Klopman-Salem terms and pair features, scored without learning
6. **Model**: transformer with KS attention bias, decoder, losses
7. **Evaluation**: metrics, benchmarks, reports
8. **Scale-up**: FlowER-size training, QMugs pretraining

## Web app (ksmech-serve)

A local web application to predict mechanisms as reaction networks, inspect the orbital
physics of every arrow (KS1, KS2, KS3, G, beta^2, the dominant canonical MO pair, the oriented
contact in 3D), correct steps, curate and export validated mechanisms, and browse PMechDB and
curated steps. Backend in `src/ksmech/serve/` (FastAPI, SQLite), frontend in `apps/ui/` (Vite,
React, RDKit.js, 3Dmol.js, cytoscape).

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
...]` and `uv run ksmech-serve export` (offline, refused with exit 2 while a server runs on the
same data root: use the web UI or `POST /api/curate/...` then), `uv run ksmech-serve schema-ts`
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
| `KSMECH_SHARED_CACHE_ROOT` | `KSMECH_DATA_ROOT` | read-only shared `data/cache/qm` and `data/cache/tokens` |
| `KSMECH_SERVE_CHEMISTRY` | `real` | `real`, `cached` or `synthetic` |
| `KSMECH_SERVE_WORKERS` | 1 | compute worker processes, clamped to [1, 2] |
| `KSMECH_SERVE_GBDT` | `data/processed/pmechdb/baseline_gbdt_random.npz` | the GBDT ranker of the `gbdt` predictor |
| `KSMECH_SERVE_STUB_PREDICTOR` | 0 | 1 adds the diagnostic `stub` predictor (floor plus rule bonuses) |
| `KSMECH_SERVE_AUTO_INDEX` | 0 | 1 (or `--index-on-start`) builds the PMechDB browse index at startup |
| `KSMECH_SERVE_DIST` | `apps/ui/dist` | the built frontend |

**Data locations.**

- `data/serve/`: `qm/` and `tokens/` (species computed by the server), `oriented/<fingerprint>/`
  (oriented-block cache), `pools/<state_id>/` (pool states and their analyses), `cubes/`,
  `contact/`, `jobs/<job_id>/` (job, network, events), `index/` (PMechDB browse index),
  `logs/failures.jsonl` (every reason-coded skip), `run/` (server lock, task pid files).
- `data/curation/curation.sqlite`: curated mechanisms, steps, re-checks and the export ledger.
- `data/curation/export_raw/steps.parquet` and `data/processed/user_curated/` (`steps.parquet`,
  `rejections.csv`, `ingest_report.json`, `pmechdb_arrow_codes.csv`, `export_report.json`):
  the export. Curated steps are pinned to the training split.
- The shared `data/cache/` and `data/processed/pmechdb/` are only read, never written.

**Host safety.** All heavy compute (xTB, localization, oriented blocks, cube grids) runs in one
spawned compute worker with one BLAS / OpenMP / torch thread. Do not raise
`KSMECH_SERVE_WORKERS` while a QM batch runs, and keep heavy processes at most 4 host-wide. The
PMechDB browse index (about 10 minutes on the worker) is built only on request: the "Build
PMechDB index" button (`POST /api/browse/index`), `--index-on-start`, or an export.

**Security.** There is no authentication. By default the server binds only to 127.0.0.1 and is
reached through the SSH tunnel above. `uv run ksmech-serve --port 8000 --host 0.0.0.0` (or one
of the machine's addresses) opens it to the local network at `http://<server-ip>:8000`: anyone
on that network can then run predictions, edit the curation database and start exports. The
Host allow-list and the same-origin guard then also accept the machine's addresses and host
names; any other name is refused.

**Tests.** The serve tests (`tests/test_serve_*.py`) use synthetic or stub chemistry and never
run xTB; on a loaded host run them one file at a time with `OMP_NUM_THREADS=1`. The Playwright smoke test uses the system Chrome or Chromium (Playwright browsers are
never downloaded): `KSMECH_E2E=1 uv run pytest tests/test_serve_e2e.py`, or `npm --prefix
apps/ui run e2e`.
