# ksmech-serve: the service map

`ksmech.serve` is the web add-on of the engine (`uv sync` installs it with the `serve` group;
`uv sync --no-group serve` gives the engine alone). The engine never imports it
(`tests/test_engine_boundary.py`). Inside, one process holds fourteen sub-packages; the compute
runs in one spawned worker process. The HTTP contract is `docs/serve_v9_contract.md`; this page
is the code map.

## Layers

```
app ─────────────► pool  workbench  embedding  browse  curate  ml  (features: one router each,
                     │       │          │         │       │         never one another)
                     ▼       ▼          ▼         ▼       ▼
runtime ◄── predict ◄── science ◄── chem ◄── compute ◄── core       (foundation: each layer may
                                                                     import every layer to its right)
depiction ──► core                                                  (pool, workbench, browse use it)
ml ─────────► core                    (a feature on core alone: predict, runtime and the features
                                       read its checkpoint listing, ml.checkpoints)
```

A module-level import between sub-packages must follow these arrows
(`tests/test_serve_layers.py`). Imports inside a function (to keep torch and the heavy
chemistry out of the API process, or to break a cycle) and under `TYPE_CHECKING` are not
checked.

## The packages

| Package | What it holds | Routes |
|---|---|---|
| `core` | settings (the profile, the tokens file), error codes, service contracts, the curation DB, cache keys, versions and the token / physics fingerprints, capabilities and profiles, the auth seam (`auth.py`: principals, roles, bearer tokens, `AuthGate`), the audit log (`audit.py`), the HTTP schemas (`schemas.py`, mirrored to `apps/ui/src/api/types.gen.ts`), the shared HTTP plumbing (`http.py`: `Services`, `Svc`, the path-id checks, the error envelope), test fakes | |
| `compute` | the worker process (`worker.py`) and its scheduler (`procsched.py`) | |
| `chem` | a pool's chemistry: normalization, featurization (QM, tokens, the sample), pool states, geometry, metal geometries, the Gaussian basis (`gto.py`, `data/sto_ng.json`), orbital cubes, token rows | |
| `science` | the spec-v9 readings: G^atom, the arrow and orbital readings of R, B + R (`stepapply.py`), the step check, the edge step | |
| `predict` | the predictors (floor, gbdt, network over `ksmech.model.decode`), the mechanism network, prediction jobs and expansion | `/predictors`, `/predict`, `/jobs/*`, `/step/check` |
| `runtime` | the shared context of the services (`context.py`) and the task table the worker runs (`tasks.py`) | |
| `pool` | the view of one pool state | `/normalize`, `/pools`, `/pool/*` |
| `workbench` | the molecule workbench: molecules, orbital sets, densities, properties, point groups | `/molecule/*`, `/depict2d` |
| `depiction` | 2D drawing: RDKit depictions, ring templates (`data/`), the gallery and its metrics | |
| `embedding` | the embedding trace of a sample through a checkpoint, as a page | `/embedding/*` |
| `browse` | the dataset browser: the PMechDB index, search, the step view | `/browse/*` |
| `curate` | curated mechanisms, recheck, export | `/curate/*` |
| `ml` | the local checkpoints (`checkpoints.py`, read without torch), the training runs and the physics / eval reports on disk (`runs.py`, passed through) | `/models`, `/ml/*` |
| `app` | the FastAPI app (`api.py`: error handlers, middleware, wiring, `/health`, `/meta`, `/whoami`), the `ksmech-serve` CLI, the static frontend | |

## Profiles, roles and the audit log

One codebase, two deployments (`KSMECH_SERVE_PROFILE`, `ksmech-serve run --profile`):

- `lab` (the default): every capability. Without `KSMECH_SERVE_TOKENS` every request is the
  local admin (the private lab behind an SSH tunnel), so the UI and the tests need no token.
- `public` (the online platform): the workbench, predict and correct on user input. The
  `browse`, `curate`, `curate_edit`, `curate_recheck`, `export`, `embedding` and `ml`
  capabilities are not in the profile (`core.capabilities.PROFILES`): their routers answer 503
  FEATURE_DISABLED "not in the public profile" before any service runs, `/api/meta` reports them
  off, and the jobs of other kinds than predict / override / expand are JOB_NOT_FOUND. A request
  without a token is the anonymous reader; a bearer token raises the role. A public deployment
  takes its own data root: the profile hides the lab's routes, not the lab's files.

The routes of a dataset, curation or ML package are gated as a whole router
(`APIRouter(dependencies=[require(name)])`, or `admit(name)` where the routes of stored results
stay readable while the feature is off). `tests/test_serve_access.py` walks every route of the
public app: a route off its explicit allow-list must answer FEATURE_DISABLED, so a new route
cannot leak by accident.

Roles are reader < user < curator < admin (`core.auth`):

| Capability / route | Public profile | Least role |
|---|---|---|
| health, meta, whoami, `/models`, pools (`/normalize`, `/pools`, `/pool/*`), job reads | yes | reader |
| `workbench` (`/molecule/*`, `/depict2d`) | yes | reader |
| `predict`: `GET /predictors` | yes | reader |
| `predict`: `POST /predict`, `/jobs/{id}/expand`; `POST /jobs/{id}/cancel` | yes | user |
| `correct`: `/jobs/{id}/override`, `/step/check` | yes | user |
| `embedding`: stored traces / `POST /embedding` | no | reader / user |
| `browse` (`POST /browse/index`: curator) | no | reader |
| `curate` (reads) | no | reader |
| `curate_edit`, `curate_recheck`, `export` | no | curator |
| `ml` (`/ml/*`) | no | curator |

`core.audit.AuditLog` appends one JSON line per non-GET request (time, request id, principal,
method, route template, path, status) to `data/serve/logs/audit.jsonl`. The middlewares run, from
the outside in: Host allow-list, gzip, request context (id, 500 envelope), audit log,
same-origin guard, auth gate, the routes.

## Rules

- A route is thin: it checks its ids (`core.http.check_id`), calls one service function and
  returns its value. A new service is a new sub-package with a `router.py`, added to
  `app.api.ROUTERS` and to `FEATURES` in `tests/test_serve_layers.py`; a route meant for the
  public profile also goes on `PUBLIC_ROUTES` in `tests/test_serve_access.py`.
- Science that the engine also needs moves into the engine and serve imports it: the step
  decoder already lives in `ksmech.model.decode` (`ksmech-eval decode` uses it on stored
  samples). `science` is the next candidate (G^atom, the readings, B + R).
- The token and physics fingerprints hash their files by path (`core/versions.py`); moving one
  of those files changes the fingerprint once, and the caches it keys are recomputed on demand.
