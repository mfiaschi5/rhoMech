# ksmech

## Project goal

ksmech is a deep-learning tool that predicts organic reaction mechanisms as sequences of
elementary steps. Each step is predicted as the change it makes to the Lewis structure: the
reaction matrix R = Delta(bond-electron matrix), which bonds form and break and where electron
pairs end up. Molecules are encoded as atom and localized-orbital tokens, and the transformer's
pair track carries the geometry-free orbital-interaction quantities of the Klopman-Salem picture
(the spectral gap factor and the cross-centre coupling), so the model starts from a physical
picture of orbital interactions instead of learning it from scratch. Curved arrows are not predicted:
they are a reading of R, reconstructed afterwards with the orbital pairs that drive it.

The prediction target is an autoregressive sequence of elementary steps. A step is one reaction
matrix (bond-order changes of +-1 on atom pairs, lone-pair changes of +-2 on atoms), decoded
around an anchor bond that forms. The product is B + R; the intermediate is re-featurized (3D
embedding, xTB, localization, new tokens) and fed back in to predict the next step, until the
model emits a stop token. Labels come from the dataset arrows applied as one edit, taken
modulo resonance as the parsimonious class (spec, "Edit labels"). Resonance identity is the
sigma key (`data.resonance.sigma_key`, spec "Resonance"): the sigma skeleton with every H count
frozen and the charges zeroed, plus the total charge. Steps whose two sides share it with the
atom maps kept (no atom changes its sigma bonds or H) leave the training target and are kept as
resonance pairs; degenerate exchanges (the same molecules, atoms that change partners) are real
steps and stay. Training draws random valid resonance forms of the reactant pool as inputs, with
the labels recomputed.

## Representation

`docs/ksmech_architecture.tex` (spec v9) is the source of truth for the representation and
the model; this section summarizes it.

Two token families share one sequence: atom tokens and localized-orbital tokens. Orbitals are
Pipek-Mezey localized in the occupied and the virtual block separately (the singly occupied
MOs of a radical apart from the doubly occupied ones), started deterministically from the
canonical MOs, so one QM record always gives the same tokens.

An orbital token carries only continuous physical descriptors; they are its whole model
input:

- occupation: 2, 0, or 1 for a SOMO;
- the number of centres (atoms carrying at least `tau_atom = 0.15` of the population) and
  the four largest centre weights;
- the s/p/d shares on the four leading centres;
- for the two leading centres: the |cos| of each p vector with the axis (sigma vs pi as a
  number), the signed overlap population (positive bonding, negative antibonding) and the
  distance;
- the radius of gyration;
- the spectral set: (energy, weight) over the canonical MOs of the block, weight >= 0.01, at
  most 8 entries, renormalized. It is embedded by a set encoder.

The LMO energy E_r, the full spectrum (which the gap kernel G uses), the centroid, the centre
atoms and the coefficient vector are bookkeeping. An orbital inherits its atoms' embeddings
through its centre weights.

An atom token's model input:

- atomic number and formal charge (embedded);
- Mulliken charge;
- frozen-orbital f+ and f-;
- the finite-difference dual descriptor;
- the frontier weights for HOMO-1, HOMO, LUMO and LUMO+1, with degenerate blocks within
  0.05 eV summed;
- %Vbur;
- the chirality sign, set only where the SMILES specifies the stereocentre.

No absolute coordinates are embedded: geometry enters only through distances.

**No chemist's taxonomy as model input.** Sigma / pi / lone-pair labels are diagnostics,
never model inputs. These are computed and stored with `model_input=False`:

- orbital types (lone pair, sigma, pi, sigma*, pi*, empty p, ...);
- hybridization;
- aromaticity;
- functional-group ids.

They serve tests (`pytest -m diagnostic`), alignment diagnostics, the interface and the
"added back as input" ablations. They never decide which arrows are legal: the arrow
candidate masks use descriptors. A donor is a doubly occupied token. An acceptor is an empty
token that is not a d-type polarization function (unless configured) and not the
antibonding partner of the donor's own bond. Until the radical extension, a token with
0.2 < n < 1.8 (a SOMO, or a Fermi-smeared level) is neither; its fractional occupation stays
the continuous input. A molecule with an MO more than 0.05 from its aufbau occupation is
flagged `fermi_smeared`: its GFN2 gap vanishes and it lies outside the closed-shell domain.

The reason is that orbital character must generalize across elements: sigma*(C-Cl),
sigma*(C-Br) and sigma*(C-I) should look to the model like the same kind of acceptor at
different energies.

## Physics

The physics enters as geometry-free pair features between an occupied and a virtual orbital
token (spec, "Geometry-free pair features"):

1. the spectral gap factor G and log|G|: the energy denominator of the Klopman-Salem
   filled-empty term over the canonical spectra, for every pair;
2. within a molecule, in the reactant pose: the Wolfsberg-Helmholz cross-centre coupling, the
   centroid distance and the stereoelectronic alignment (lobe-axis and dihedral cosines);
3. the fragment flag and, on the Lewis structure, the bonded flag, bond order, Wiberg index and
   graph distance.

They initialize the pair track, which every head reads as a learned bias; all heads are
learned. There is no contact geometry, no Klopman-Salem attention bias and no frozen head. That
design (configurational contact, the three KS terms at contact, a frozen head 0) is archived
with its code on the git branch `archive/stage-b` (spec v8.1): untrained, its contact terms
ranked a step's first arrow worse than G alone.

## Data sources

| Dataset | Role |
|---|---|
| PMechDB | Curated polar mechanistic steps. Development set. |
| ReactMech | 67 reaction classes. Out-of-distribution splits. |
| FlowER dataset | About 1.4M steps. Scale-up. |
| mech-USPTO-31K | Held-out benchmark. Never trained on. |
| QMugs | QM pretraining. |

## Engineering rules

- Never fabricate a dataset column. Open and inspect the real file first, then document the
  schema in `docs/data_schema.md`.
- Every featurization function has a unit test on the fixed molecule set in
  `tests/fixtures/molecules.py`: ethane, ethanol, acetone, acetaldehyde, methyl acetate,
  chloromethane, methylamine, acetonitrile, benzene, acrolein.
- No silent failures. Every skipped molecule is logged with a reason code and counted, and
  the counts are reported at the end of the run.
- All QM results are cached on disk, keyed by canonical SMILES plus conformer id.
- Run `pytest` before saying any task is done, and paste the summary line.
- Record every non-obvious design choice in `docs/decisions.md`, with a date.
- Prefer plain numpy over adding dependencies.
- Keep functions under 60 lines.

## Commands

```bash
uv sync                       # install everything into .venv (torch: the ROCm build)
uv sync --no-group rocm --group cuda   # an NVIDIA host: the CUDA build of torch
export UV_NO_SYNC=1           # then, on that host: else uv run re-syncs to the default rocm group
uv run pytest -n 14 -m "not realdata"  # unit tests in 14 processes (pytest-xdist; -x stops at the first failure)
uv run pytest -m realdata     # real-data tests one at a time, each pass in 14 spawned processes
uv run ruff check .           # lint
uv run ruff format .          # format
uv run pre-commit install     # enable the commit hook: ruff, then the two pytest runs above with -x
```

Main CLI entry points are added phase by phase and listed here as they land:

- Data: `uv run ksmech-data ingest --source pmechdb` (steps parquet, `resonance_pairs.parquet`,
  rejections and report under `data/processed/pmechdb/`), `uv run ksmech-data stats --source pmechdb` (writes `stats.json`)
- Provenance: `uv run ksmech-data provenance <artifact>` (prints what an artifact was built from and
  whether it is stale), `uv run ksmech-data provenance --backfill qm` (one-off: headers on QM records
  written before 2026-10-21; never while a QM batch runs). Loaders refuse stale artifacts
  (STALE_ARTIFACT); only `textbook` takes `--allow-stale`.
- QM featurization: `uv run ksmech-qm run --source pmechdb --workers 8` (GFN2-xTB records under `data/cache/qm/`, manifest + reason-code table; see `docs/decisions.md`;
  `--timeout`, `--maxiter`, `--scf-maxiter` relax convergence for a retry of failures and do not
  change the method string; `--retry-reason REASON ...` retries only the failures with those
  codes), `uv run ksmech-qm bond-index`
  (Wiberg bond indices of the Lewis bonds of every cached record, `data/cache/bond_index/`; the
  token cache needs them, so re-run after the QM batch)
- Orbital tokens: `uv run ksmech-orbitals build --source pmechdb --workers 4` (PM-localized orbital
  and atom tokens, one `.npz` per molecule under `data/cache/tokens/` plus `summary.json`; molecules
  without a QM record are skipped and counted, so re-run after the QM batch), `uv run
  ksmech-orbitals show --smiles 'CC(=O)C'` (token table of one cached molecule),
  `uv run ksmech-orbitals fg-vocab --source pmechdb` (regenerate the functional-group vocabulary)
- Arrows: `uv run ksmech-arrows roundtrip --source pmechdb` (dataset arrows applied as one
  BE-matrix edit per step vs. the annotated products, `arrows_roundtrip.json`), `uv run
  ksmech-arrows align --source pmechdb` (arrow-to-orbital labels by atom, unaligned groups,
  predicted-arrow round trip, `alignment_report.json`; `--workers N` spawned processes), `uv run
  ksmech-arrows samples --source
  pmechdb` (the sample store `samples/shard_*.npz` + `index.json`: every training step with
  tokens, its reaction-matrix label class and anchors, the reactant's Lewis structure on the token
  atoms (bonds, lone pairs, Wiberg indices), the geometry-free pair features, the other
  resonance drawings of the pool with their labels (`arrows.drawings`), and the orbital-pair labels
  where every arrow aligns; one `samples_<strategy>_<part>.json` view per
  split of `molecule`, `scaffold` and
  `loco`; the per-step `random_leaky` split only with `--strategy random_leaky`; `--workers N`
  spawned processes, up to 15 heavy processes on the host in total; re-run both after a token rebuild), `uv run ksmech-arrows edit-labels --source pmechdb` (the spec-v8 reaction-matrix
  labels: parsimonious class, anchors, coverage gate >= 95%, `edit_labels_report.json`)
- Physics: `uv run ksmech-physics gbdt --source pmechdb [--split molecule|scaffold] [--workers 8]
  [--n-jobs 14]` (the fast LightGBM lambdarank reference on the geometry-free pair features over
  all training, ranking each step's first arrow: all positives + 20 hard (best by the gap kernel
  G) + 30 random negatives per step, early stopping on val top-5, the untrained gap kernel as the
  floor; `gbdt_lgbm_report_<split>_geomfree.json` with times and peak RSS, the models
  `gbdt_lgbm_<split>_geomfree[_textbook-held-out].npz`; `--workers` load the samples, `--n-jobs`
  LightGBM threads fit and score, one after the other), `uv run ksmech-physics textbook [--gbdt
  PATH] [--allow-stale]` (five textbook selectivity cases, expected vs top-3, with the untrained
  gap kernel and the two geometry-free LightGBM models of the molecule split, cases seen in a
  model's training marked; builds missing QM records and tokens). After a token rebuild:
  `ksmech-arrows samples` (geometry-free pair features and `pair_feature_stats_<strategy>.json`
  from each training part), then `ksmech-physics gbdt`, then `textbook`.
- Serve: `uv run ksmech-serve --port 8000` (web app on 127.0.0.1 only, `--host 0.0.0.0` for the
  LAN; data under `data/serve/`, curation DB `data/curation/curation.sqlite`; one compute worker;
  `--chemistry real|cached|synthetic`, `--dev`, `--no-ui`, `--index-on-start`), `uv run
  ksmech-serve schema-ts` (regenerate `apps/ui/src/api/types.gen.ts`), `uv run ksmech-serve
  probe-cache --smiles 'C[O-]'` (exit 0 when the shared token cache holds every species),
  `uv run ksmech-serve recheck`, `uv run ksmech-serve export --out FILE` (curated steps as
  PMechDB raw records, never training data); Predict and Correct work on spec-v9 reaction
  matrices (predictors floor, gbdt, network; contract in `docs/serve_v9_contract.md`; the
  v3 panels candidates and physics are off, `serve/capabilities.py`, also in `/api/meta`);
  embedding traces of a pool through a local checkpoint: `GET /api/models`, `POST /api/embedding`
  (a job in the worker; pages under `data/serve/embedding/`), the step panel's Embedding view
  and the Workbench's Model trace
- UI: `npm --prefix apps/ui run build | dev | e2e` (e2e: Playwright with the system Chrome,
  `KSMECH_E2E=1 uv run pytest tests/test_serve_e2e.py`)
- Training: `HSA_OVERRIDE_GFX_VERSION=10.3.0 uv run ksmech-train fit --run NAME [--split molecule]
  [--encoder dense|flex|sparse] [--epochs 30] [--cell-budget 1100000] [--max-tokens 1024]
  [--resume] [--no-resonance-draws]` (the spec-v9 model on one split; each epoch every training
  sample in a random one of its resonance drawings unless `--no-resonance-draws`; `--encoder`: `dense` the pair track of spec v9,
  `flex` FlexAttention with a relative bias and no pair state (GPU only), `sparse` top-k keys from a
  learned indexer, dense during the LR warm-up; `--stop-weight` is lambda_stop of the END_MECH head,
  by default 0.5 and 0 on PMechDB; `data/models/<source>/<run>/`: `run.json` with the provenance of
  the views and pair statistics, `train.log` in the `step n / total` format `ksmech-eta` reads,
  `metrics.jsonl` per epoch, `last.pt` and `best.pt` by validation anchor top-1; batches by a
  padded-cell budget, samples over `--max-tokens` skipped as TOO_LARGE and counted; the override is
  for the RX 6600 XT (gfx1032 runs the gfx1030 kernels); every `ModelConfig` field is an option,
  e.g. `--drop-pair-features g log_abs_g` for the G ablation; `--limit 64 --epochs 1` for a smoke
  run)
- Evaluation: `uv run ksmech-eval floor --source pmechdb [--split molecule]` (the physics floor:
  anchors ranked by |G^atom| over the valid orbital pairs, no learned parameter, val and test
  top-1/3/5, `anchor_floor_report_<split>.json`), `uv run ksmech-eval embedding --run NAME
  [--step-id ID | --index K [--part val] | --smiles 'CC(=O)C.[OH-]'] [--drawing k]
  [--checkpoint best|last]` (one sample through a trained model as a self-contained HTML page
  under `data/models/<source>/<run>/embedding/`: the tokens on the Lewis structure, the parts
  of h(0), every layer's attention and token updates (rebuilt from the recorded layer inputs and
  checked against the model's own forward), the decoder's anchor logits split into site, pair
  and physics terms; CPU by default; `--smiles` builds missing tokens with GFN2-xTB)
