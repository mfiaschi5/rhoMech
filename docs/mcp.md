# ksmech-mcp: the MCP server of the ksmech API

`ksmech-mcp` lets an LLM client (Claude Code, Claude Desktop, any MCP client) use ksmech:
normalize reactants, predict a mechanism, read the orbital picture of a step, check a step of
its own, inspect molecules and, on a lab server, read the training runs, the reports, the
dataset and the curated mechanisms.

It is a thin FastMCP server (the official `mcp` SDK, 1.x line) over
`ksmech.client.KsmechClient`. It never imports the server: every tool is an HTTP call to
ksmech-serve with `X-Ksmech-Via: mcp` and the bearer token of `KSMECH_TOKEN`, so an LLM gets
the same permissions, quotas and audit log as the web UI and the `ksmech` CLI.

## Running it

```bash
uv run ksmech-serve --port 8000                    # the API (any profile)
uv run ksmech-mcp                                  # stdio (what Claude Code starts)
uv run ksmech-mcp --transport http --port 8765     # streamable HTTP at http://127.0.0.1:8765/mcp
```

| Option | Default | Meaning |
|---|---|---|
| `--url` / `KSMECH_URL` | `http://127.0.0.1:8000` | the ksmech-serve API root |
| `--token` / `KSMECH_TOKEN` | none | bearer token sent as `Authorization: Bearer <token>` |
| `--transport` | `stdio` | `stdio` or `http` (streamable HTTP, endpoint `/mcp`) |
| `--host`, `--port` | `127.0.0.1`, `8765` | the HTTP transport's bind address |
| `--timeout` | 60 | seconds per API request (a prediction waits separately, `wait_s`) |

At startup the server reads `GET /api/meta` and exits with code 1 when the API does not answer
(start ksmech-serve first). Logs go to stderr; stdout is the stdio transport.

## Adding it to Claude Code

```bash
# stdio, the server on this machine (the venv's script: no uv sync on every start)
claude mcp add ksmech -e KSMECH_URL=http://127.0.0.1:8000 -- /path/to/ksmech/.venv/bin/ksmech-mcp
# with a token (KSMECH_SERVE_TOKENS on the server names its role)
claude mcp add ksmech -e KSMECH_TOKEN=<token> -- /path/to/ksmech/.venv/bin/ksmech-mcp
# a running HTTP transport
claude mcp add --transport http ksmech http://127.0.0.1:8765/mcp
```

`--scope user` makes it available in every project, `--scope project` writes `.mcp.json` for the
repository. `claude mcp list` shows whether it connected; `/mcp` inside Claude Code lists its
tools. For a remote lab server, tunnel the API (`ssh -L 8000:localhost:8000 <server>`) and run
`ksmech-mcp` locally on a checkout (`uv sync`): it imports only the client, never the
engine or the server.

Claude Desktop: add to `claude_desktop_config.json`
`{"mcpServers": {"ksmech": {"command": "/path/to/ksmech/.venv/bin/ksmech-mcp", "env":
{"KSMECH_URL": "http://127.0.0.1:8000"}}}}`.

## Authentication and who acts

Every API call carries `X-Ksmech-Via: mcp` and, with `KSMECH_TOKEN`, the bearer token; the
server's audit log records both, and the token's role decides what the calls may do (a tool
whose route the role may not use answers the API's error). `GET /api/whoami` (`ksmech whoami`)
shows the principal a token gives. Over the HTTP transport every MCP client acts with the
server's one token: bind it to 127.0.0.1 (the default, with FastMCP's DNS-rebinding check) and
do not expose it.

## Tools

Registered at startup from the capabilities of `/api/meta`: a tool whose capability is off is
not offered at all (a `public` server has `browse`, `curate` and `ml` off, so it never offers
the dataset, curation or ML tools). A capability the server does not list counts as on only for
a server that predates profiles (no `profile` in its meta).

Every tool answers compact JSON text (rounded numbers, summaries; `detail=true` where offered
returns more). Annotations: every tool has `openWorldHint=false` and `destructiveHint=false`;
all are `readOnlyHint=true`, `idempotentHint=true` except `predict_mechanism`, which submits a
prediction job (compute: not read-only, not idempotent, never destructive).

| Area | Tool | Capability | What it answers |
|---|---|---|---|
| meta | `whoami` | always | the principal the API sees for the token: name, role, via |
| chem | `normalize_reactants` | always | ok, pool key, state id, mapped SMILES, one row per species (errors with codes) |
| chem | `depict_svg` | always | a pool's 2D drawing as SVG, its state id and mapped SMILES (builds the pool state) |
| chem | `molecule_properties` | `workbench` | HOMO, LUMO, gap, mu, eta, omega; per atom charges, Fukui f+ / f-, dual descriptor, %Vbur (`detail`: bonds, Wiberg) |
| chem | `molecule_symmetry` | `workbench` | point group, order, top, RMS, equivalent atoms, a looser group (`detail`: elements) |
| chem | `orbital_summary` | `workbench` | orbitals around the frontier: energy, occupation, label, main centres (`set`: canonical / ibo / pm_mulliken) |
| predict | `list_predictors` | `predict` | floor, gbdt, network: available, reason, whether scores are probabilities |
| predict | `predict_mechanism` | `predict` | submits a job, waits `wait_s`, returns the top pathways with per-step products, rank, p or score, anchor, arrows, PMechDB code (`detail`: matrices, orbital readings) |
| predict | `get_job` | always | a job's status, progress or error, and its pathway summary once done |
| predict | `step_readings` | `predict` | one predicted step: the matrix, arrows, orbital readings (g_atom, donor / acceptor, HOMO->LUMO pair, gap), alternatives |
| predict | `check_step` | `correct` | validates a reaction matrix on a pool state: ok or issues, products, arrows, readings |
| ml | `list_checkpoints` | `ml` | the local checkpoints: epoch, encoder, validation metrics, usable, smoke |
| ml | `list_training_runs` | `ml` | the runs: encoder, epochs, best epoch's metrics, checkpoints, clean |
| ml | `training_run_metrics` | `ml` | one run: config, metrics per epoch (first and last 5; `detail`: up to 60), its reports |
| ml | `list_reports` | `ml` | the physics floor and LightGBM reports with their headline numbers (`detail`: the JSON) |
| datasets | `dataset_index_status` | `browse` | the PMechDB search index state and counts |
| datasets | `search_dataset_steps` | `browse` | steps by species (exact) or SMARTS (substructure): ref, reaction, arrows, PMechDB code |
| datasets | `view_dataset_step` | `browse` | one step: reaction, dataset arrows on map numbers, its curated mechanism |
| curate | `list_curated_mechanisms` | `curate` | curated mechanisms: id, status, reactants, steps, predictor, stale |
| curate | `get_curated_mechanism` | `curate` | one mechanism and its steps (`detail`: the step documents) |

Resources (with `ml`): `ksmech://checkpoints` (the checkpoint list) and the template
`ksmech://runs/{source}/{run}` (a run with its per-epoch metrics), both JSON.

There are no write tools for curated mechanisms or datasets (saving, validating, editing,
exporting, rebuilding the index): such changes will go through a review queue. The server's
instructions say so to the LLM.

## Adding a tool

A tool is an async method of the area's class in `src/ksmech/mcp/tools/<area>.py` that calls
`KsmechClient` through `blocking(...)` (a worker thread: the client is synchronous) and returns
`dumps(summary)`; its docstring is the description the LLM reads, its parameters are typed with
`Annotated[..., Field(description=...)]`. Register it in the area's `register` under its
capability with `READ` or `COMPUTE` annotations, add the summary to `ksmech.client.summary`,
and add it to the table above and to `tests/test_mcp_server.py`.
