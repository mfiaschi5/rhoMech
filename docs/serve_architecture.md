# ksmech-serve: the service map

`ksmech.serve` is the web add-on of the engine (`uv sync` installs it with the `serve` group;
`uv sync --no-group serve` gives the engine alone). The engine never imports it
(`tests/test_engine_boundary.py`). Inside, one process holds thirteen sub-packages; the compute
runs in one spawned worker process. The HTTP contract is `docs/serve_v9_contract.md`; this page
is the code map.

## Layers

```
app ─────────────► pool  workbench  embedding  browse  curate      (features: one router each,
                     │       │          │         │       │         never one another)
                     ▼       ▼          ▼         ▼       ▼
runtime ◄── predict ◄── science ◄── chem ◄── compute ◄── core       (foundation: each layer may
                                                                     import every layer to its right)
depiction ──► core                                                  (pool, workbench, browse use it)
```

A module-level import between sub-packages must follow these arrows
(`tests/test_serve_layers.py`). Imports inside a function (to keep torch and the heavy
chemistry out of the API process, or to break a cycle) and under `TYPE_CHECKING` are not
checked.

## The packages

| Package | What it holds | Routes |
|---|---|---|
| `core` | settings, error codes, service contracts, the curation DB, cache keys, versions and the token / physics fingerprints, capabilities, the HTTP schemas (`schemas.py`, mirrored to `apps/ui/src/api/types.gen.ts`), the shared HTTP plumbing (`http.py`: `Services`, `Svc`, the path-id checks), test fakes | |
| `compute` | the worker process (`worker.py`) and its scheduler (`procsched.py`) | |
| `chem` | a pool's chemistry: normalization, featurization (QM, tokens, the sample), pool states, geometry, metal geometries, the Gaussian basis (`gto.py`, `data/sto_ng.json`), orbital cubes, token rows | |
| `science` | the spec-v9 readings: G^atom, the arrow and orbital readings of R, B + R (`stepapply.py`), the step check, the edge step | |
| `predict` | the predictors (floor, gbdt, network over `ksmech.model.decode`), checkpoints, the mechanism network, prediction jobs and expansion | `/predictors`, `/predict`, `/jobs/*`, `/step/check`, `/models` |
| `runtime` | the shared context of the services (`context.py`) and the task table the worker runs (`tasks.py`) | |
| `pool` | the view of one pool state | `/normalize`, `/pools`, `/pool/*` |
| `workbench` | the molecule workbench: molecules, orbital sets, densities, properties, point groups | `/molecule/*`, `/depict2d` |
| `depiction` | 2D drawing: RDKit depictions, ring templates (`data/`), the gallery and its metrics | |
| `embedding` | the embedding trace of a sample through a checkpoint, as a page | `/embedding/*` |
| `browse` | the dataset browser: the PMechDB index, search, the step view | `/browse/*` |
| `curate` | curated mechanisms, recheck, export | `/curate/*` |
| `app` | the FastAPI app (`api.py`: error envelope, middleware, wiring, `/health`, `/meta`), the `ksmech-serve` CLI, the static frontend | |

## Rules

- A route is thin: it checks its ids (`core.http.check_id`), calls one service function and
  returns its value. A new service is a new sub-package with a `router.py`, added to
  `app.api.ROUTERS` and to `FEATURES` in `tests/test_serve_layers.py`.
- Science that the engine also needs moves into the engine and serve imports it: the step
  decoder already lives in `ksmech.model.decode` (`ksmech-eval decode` uses it on stored
  samples). `science` is the next candidate (G^atom, the readings, B + R).
- The token and physics fingerprints hash their files by path (`core/versions.py`); moving one
  of those files changes the fingerprint once, and the caches it keys are recomputed on demand.
