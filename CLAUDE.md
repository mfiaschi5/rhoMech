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
matrix (bond-order changes of +-1 on atom pairs, changes of +-2 in an atom's non-bonding
electrons N, and +-1 for a source of one-electron steps: `DomainPolicy.open_shell_labels`, a
label rule only), decoded around an anchor bond that forms. The product is B + R; the intermediate is re-featurized (3D
embedding, xTB, localization, new tokens) and fed back in to predict the next step, until the
model emits a stop token. Labels come from the dataset arrows applied as one edit, taken
modulo resonance as the parsimonious class (spec, "Edit labels"), whose members must have a
balanced electron flow when any candidate does (`ReactionMatrix.balanced`). Resonance identity
is the identity key v2 (`data.resonance.identity_key`, spec v10 "Resonance"): for a closed
shell the sigma key (`sigma_key`: the sigma skeleton with every H count frozen and the charges
zeroed, plus the total charge), for an open shell that key plus each fragment's (sigma SMILES,
charge, unpaired count), u from the drawing's odd-N atoms (`data.spin.spin_of_drawing`, raised
by the species table's high-spin rows, never RDKit's radical perception). Steps whose two sides
share it with the atom maps kept and whose electron flow balances (electrons move only between
bonded atoms, `resonance.balanced_flow`) leave the training target and are kept as resonance
pairs; identical sides are NULL_STEP and a shared key without a balanced flow
NONCONJUGATED_SHIFT, both counted exclusions; degenerate exchanges (the same molecules, atoms
that change partners) are real steps and stay. Training draws random valid resonance forms of the reactant pool as inputs, with
the labels recomputed.

## Representation

`docs/ksmech_architecture.tex` (spec v9.2; on the Release B branches the v10 draft of the
open-shell plan P6, to be approved before its build) is the source of truth for the
representation and the model; this section summarizes it.

Two token families share one sequence: atom tokens and localized-orbital tokens. Orbitals are
localized in the occupied and the virtual block separately (the singly occupied MOs of a
radical apart from the doubly occupied ones) by IBO, the Pipek-Mezey functional with Lowdin
populations and exponent 4, started deterministically from a pivoted Cholesky factorization of
each block's density matrix (`orbitals.localize.PMConfig`; Pipek-Mezey with Mulliken
populations is only the comparison), so one QM record always gives the same tokens.

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

The full spectrum (which the gap kernel G uses), the centroid, the centre atoms and the
coefficient vector are bookkeeping. The LMO energy E_r and the spectral weight the set keeps
before renormalizing are stored as inputs a model selects (`ModelConfig.orbital_extra`, off by
default; spec v9.1). An orbital inherits its atoms' embeddings through its centre weights.

An atom token's model input:

- atomic number and formal charge (embedded);
- Mulliken charge;
- frozen-orbital f+ and f-;
- the finite-difference dual descriptor;
- the frontier weights for HOMO-1, HOMO, LUMO and LUMO+1, with degenerate blocks within
  0.05 eV summed (a second set with blocks within 1e-4 eV is stored beside it,
  `ModelConfig.frontier_blocks = 'exact'`; the finite-difference f+ / f- are stored too,
  unselected);
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
antibonding partner of the donor's own bond. Since v18 these exclusions are stored as 0/1 pair
flags on every cell (`physics.pairs.pair_flags`) rather than deleting the cells:
`ModelConfig.pair_cells = 'valid'` (default) reads only the `pair_valid` cells, as before;
`'all'` reads every cell with the flags as channels. A token with 0.2 < n < 1.8 is neither,
except an open-shell token of a record run as an open shell (P2b's plateau group, not a
Fermi-smeared level; `tokens.open_shell_tokens`): under `CandidateConfig.somo_roles` it is a
donor and its hole, a VIRTUAL twin built at sample assembly (`tokens.with_holes`, never in a
token file), an acceptor; within a fragment an open-shell row and a hole column are never valid;
metal-d open shells take no role unless `metal_d_open_shell` (open-shell plan P7). Its
fractional occupation stays the continuous input; the sample adds the orbital column
`spin_role` (+1 donor role, -1 hole) and the atom column `somo_population` (rho_A, restricted
SOMO population), which a model selects (`ModelConfig.spin_role`, `atom_spin_input`). A
molecule with an MO more than 0.05 from its aufbau occupation is flagged `fermi_smeared`: its
GFN2 gap vanishes and it lies outside the closed-shell domain.

The reason is that orbital character must generalize across elements: sigma*(C-Cl),
sigma*(C-Br) and sigma*(C-I) should look to the model like the same kind of acceptor at
different energies.

## Physics

The physics enters as geometry-free pair features between an occupied and a virtual orbital
token (spec, "Geometry-free pair features"):

1. the spectral gap factor G and log|G|: the energy denominator of the Klopman-Salem
   filled-empty term over the canonical spectra, for every pair, weighted by the occupation
   difference (2 on every closed-shell pair, bit for bit as before; 1 for a SOMO or a hole; 0,
   undefined, for an open-shell row with a hole); since v19 beside it the two-level column g2
   (`ks_terms.gap_kernel_two_level`, beta the cross-centre coupling within a molecule and
   `params.BETA_INTER` = 0.40 eV between molecules), the own-pair channel (own_g, own_g2 of a
   bond with its own antibond, pooled into B^own outside G^atom) and the recombination channel
   (somo_pair, somo_level_diff), each selected by name (`stage_a.SAMPLE_PAIR_FEATURES`);
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
| RMechDB (source `rmechdb`) | Radical (one-electron) steps: FlowER's RS/RC lines now, the official CSV later. 4,535 steps after dedup; split `rc_molecule` (RC by molecule, RS in training). CC-BY-NC-ND: artifacts local, private rhoMech-data and pods only, never public. |
| FlowER dataset | About 2.14M numeric-suffix lines (train+val+test). Scale-up. Its RS/RC lines are ingested as rmechdb; PM and PC never. |
| mech-USPTO-31K | Held-out benchmark. Never trained on. |
| QMugs | QM pretraining. |

Ingest screens every step with one law per domain phenomenon, for every element and parity
(`data.domain`, spec v9.1): a -1 on a four-bonded group-14 atom is removed; an onium bonded to
a metal or B is a dative donor and the acceptor is re-charged where that leaves the fragment
even, the side's net charge >= 0 and both sides' totals equal (rule 'D dative onium->M'); a
non-s-block metal with exactly one neighbour and charge <= 0 is a ligand-dropped placeholder
(METAL_PLACEHOLDER) unless it compensates an onium donor (ML1); an even fragment with a 3d
metal left d1-d9 by the bookkeeping is flagged METAL_SUSPECT (`domain_class`), never excluded.
Since P8 one spin law holds for every source (`data.domain`, spec v10): the species table
decides a drawn species' multiplicity; a singlet drawn with odd-N atoms is redrawn closed shell
where bonded electron moves reach one (PAIRED_DRAWING: N#N, ozone, Criegee) and a Hund-case
species drawn paired is redrawn unpaired (HIGH_SPIN_DRAWING: O=O); a species left with u >= 2
that is not a Hund case, or a singlet no closed-shell drawing reaches, is MULTI_RADICAL_SPECIES
except in a counted exemption (`curation/multi_radical_exemptions.json`); the spin census keeps
such species out of QM (SPIN_CENSUS_REFUSED). ReactMech re-reads odd metals by the spin law; a
source of one-electron steps (rmechdb) conserves u only where R moves no single electron.

## Engineering rules

- Never fabricate a dataset column. Open and inspect the real file first, then document the
  schema in `docs/data_schema.md`.
- Every featurization function has a unit test on the fixed molecule set in
  `tests/fixtures/molecules.py`: ethane, ethanol, acetone, acetaldehyde, methyl acetate,
  chloromethane, methylamine, acetonitrile, benzene, acrolein. A function an open-shell phase
  changes is also tested on the hand-written `OPEN_SHELL` list there (radicals, superoxide, the
  benzene radical anion, triplet O2, Br., .OH, NO, Cl[C]Cl, [C]=O, [N]=[N], ozone, CuCl2,
  FeBr4-).
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

- Data: `uv run ksmech-data ingest --source pmechdb|reactmech|rmechdb [--sample N --seed S]` (steps parquet,
  `resonance_pairs.parquet`, rejections and report under `data/processed/<source>/`; `--sample`
  keeps N random accepted steps for a pilot; ReactMech reads its ID and OOD split files once each,
  classes from the OOD mechanism files; rmechdb streams FlowER's RS/RC lines, strips bath gases,
  keys each line by its map-free reaction after the spin redrawings and keeps one step per id,
  RC over RS, `flower_suffix` / `flower_part`), `uv run ksmech-data stats --source pmechdb` (writes
  `stats.json`), `uv run ksmech-data leakage-audit [--write-cohorts]` (rmechdb's leakage audit
  in both directions, species by key v2, `leakage_audit.json`; the cohorts `leak_guard_joint`,
  `rc_heldout_overlap`, `rc_heldout_shared`, `closed_base_rmechdb_overlap`, each pinned once)
- Provenance: `uv run ksmech-data provenance <artifact>` (prints what an artifact was built from and
  whether it is stale), `uv run ksmech-data provenance --backfill qm` (one-off: headers on QM records
  written before 2026-10-21; never while a QM batch runs). Loaders refuse stale artifacts
  (STALE_ARTIFACT); only `textbook` takes `--allow-stale`. `uv run ksmech-data
  open-shell-survey --source S` (read-only: a subclass for every excluded step and every kept
  step with a suspicious metal, the SPIN_PARITY failures by kind, `open_shell_survey.json`; the
  open-shell plan's oracle), `uv run ksmech-data freeze-sample --source reactmech` (the pilot's
  step ids as `sample_frozen.json`; `ingest --sample` then keeps exactly them),
  `uv run ksmech-arrows freeze-splits --source S --strategy molecule scaffold ...` (the split
  files as `split_frozen_<strategy>.json`, FROZEN_SPLIT_SCHEMA, no file inputs: `samples` then
  keeps every frozen step in its part, places new groups by hash and counts SPLIT_CONFLICT;
  a snapshot is replaced only by hand). The open-shell plan is `docs/open_shell_plan.md`.
- QM featurization: `uv run ksmech-qm run --source pmechdb --workers 8 [--steps-from FILE]` (GFN2-xTB records under `data/cache/qm/`, manifest + reason-code table; see `docs/decisions.md`;
  the spawned workers run tblite on the system's LP64 OpenBLAS instead of the wheel's netlib copy
  (`qm.blas`, `KSMECH_BLAS=auto|bundled|PATH`; `--workers 1` and the web app keep the netlib copy;
  a pod needs `libopenblas0-serial`); `--optimizer tblite-anc` optimizes in approximate normal
  coordinates with RFO steps and restarted SCFs (`qm.anc`, method `.../tblite-anc`, 2-4x fewer
  energies than the default L-BFGS-B `.../tblite-lbfgs`; its `--maxiter` counts energies); `--steps-from` only the fragments of the steps a JSON file lists, `qm_steps` or `ids`;
  `--timeout`, `--maxiter`, `--scf-maxiter` relax convergence for a retry of failures and do not
  change the method string; `--retry-reason REASON ...` retries only the failures with those
  codes; a failure that would recur leaves a tombstone `<key>.failed.json` next to the records,
  and a later run of any source, or the web app, with the same settings and no larger budget
  answers from it instead of rerunning xTB), `uv run ksmech-qm bond-index`
  (Wiberg bond indices of the Lewis bonds of every cached record, `data/cache/bond_index/`; the
  token cache needs them, so re-run after the QM batch), `uv run ksmech-qm index stats [--root
  data/cache --root data/serve]` (the SQLite index `<root>/index.sqlite` of records, tombstones,
  tokens and Wiberg sidecars, read from the file headers and refreshed incrementally first;
  the records of species `data.spin` reassigns, under the tagged method `+spin-v1-u<u>`
  (`qm.spin`), listed by spin source;
  `index refresh [--rebuild]` only refreshes; the files stay the source of truth)
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
  `loco` (rmechdb: `rc_molecule`, RC split by molecule with RS in training except LEAK_GUARD,
  and `flower_given`, FlowER's own parts); the per-step `random_leaky` split and `loco_all` (the
  class held out as test, the unlabelled steps in train and val) only when named with
  `--strategy`; `--views-only` writes
  the named strategies' views over the store already built; `--workers N`
  spawned processes, up to 15 heavy processes on the host in total; re-run both after a token rebuild), `uv run ksmech-arrows edit-labels --source pmechdb` (the spec-v8 reaction-matrix
  labels: parsimonious class, anchors, coverage gate >= 95%, `edit_labels_report.json`)
- Physics: `uv run ksmech-physics gbdt --source pmechdb [--split molecule|scaffold] [--workers 8]
  [--n-jobs 14] [--cells valid|all]` (the fast LightGBM lambdarank reference on the geometry-free pair features over
  all training, ranking each step's first arrow: all positives + 20 hard (best by the gap kernel
  G) + 30 random negatives per step, early stopping on val top-5, the untrained gap kernel as the
  floor; `gbdt_lgbm_report_<split>_geomfree.json` with times and peak RSS, the models
  `gbdt_lgbm_<split>_geomfree[_textbook-held-out].npz`; `--cells all`: every pair a candidate,
  the exclusion flags as features, files tagged `_allcells`; `--workers` load the samples, `--n-jobs`
  LightGBM threads fit and score, one after the other), `uv run ksmech-physics textbook [--gbdt
  PATH] [--allow-stale]` (five textbook selectivity cases, expected vs top-3, with the untrained
  gap kernel and the two geometry-free LightGBM models of the molecule split, cases seen in a
  model's training marked; builds missing QM records and tokens; `--radical`: the radical
  selectivity suite of open-shell plan P7, untrained kernel, g and g2, diagnostics, or with
  `--checkpoint PATH` a trained model's anchors). After a
  token rebuild:
  `ksmech-arrows samples` (geometry-free pair features and `pair_feature_stats_<strategy>.json`
  from each training part), then `ksmech-physics gbdt`, then `textbook`.
- Serve (an optional add-on: the engine never imports `ksmech.serve`; its web packages are the
  `serve` dependency group; seventeen sub-packages in layers, map in `docs/serve_architecture.md`,
  checked by `tests/test_serve_layers.py`): `uv run ksmech-serve --port 8000` (web app on 127.0.0.1 only, `--host 0.0.0.0` for the
  LAN; data under `data/serve/`, curation DB `data/curation/curation.sqlite`; one compute worker;
  `--chemistry real|cached|synthetic`, `--dev`, `--no-ui`, `--index-on-start`), `uv run
  ksmech-serve schema-ts` (regenerate `apps/ui/src/api/types.gen.ts`), `uv run ksmech-serve
  probe-cache --smiles 'C[O-]'` (exit 0 when the shared token cache holds every species),
  `uv run ksmech-serve users list | add LOGIN --name N --role admin --generate | set LOGIN
  --role R` (accounts of `serve.accounts`: local passwords, OIDC sign-in through
  `KSMECH_SERVE_AUTH`, sessions, job quotas, admins unlimited; `docs/accounts.md`),
  `uv run ksmech-serve recheck`, `uv run ksmech-serve rekey-pools [--dry-run] [--purge]` (the
  stored pool states under the identity key v2, `PoolState.key_version`; once, at the Release B
  merge; refuses while a server runs), `uv run ksmech-serve export --out FILE` (curated steps as
  PMechDB raw records, never training data), `uv run ksmech-serve ocr-setup [--relock]
  [--skip-env] [--skip-weights]` (the OCR engine MolScribe 1.1.1: its isolated environment
  `tools/ocr` synced into `data/ocr/venv` and the sha256-checked checkpoint under
  `data/ocr/weights/`, resumable; `ocr` is off until it has run; `POST /api/ocr`,
  `docs/ocr.md`); the wiki is `content/wiki/*.md` (TOML front matter, rendered on the server:
  `GET /api/wiki`, `/api/wiki/{slug}`, `/api/wiki/llms.txt`; format in `docs/wiki.md`); Predict and Correct work on spec-v9 reaction
  matrices (predictors floor, gbdt, network; contract in `docs/serve_v9_contract.md`; the
  v3 panels candidates and physics are off, `serve/core/capabilities.py`, also in `/api/meta`);
  embedding traces of a pool through a local checkpoint: `GET /api/models`, `POST /api/embedding`
  (a job in the worker; pages under `data/serve/embedding/`), the step panel's Embedding view
  and the Workbench's Model trace
- UI: `npm --prefix apps/ui run build | dev | e2e` (e2e: Playwright with the system Chrome,
  `KSMECH_E2E=1 uv run pytest tests/test_serve_e2e.py`; `KSMECH_E2E_PORT=<p>` beside another
  server). The app is branded ChemRho, in four nav sections: ChemPtah (`#/workbench`), ClotoMech
  (`#/predict`, `#/correct`, `#/curate`), ChemThot (`#/browse`) and the Wiki (`#/wiki`); the
  section names are hash aliases (`docs/serve_architecture.md`)
- Training: `HSA_OVERRIDE_GFX_VERSION=10.3.0 uv run ksmech-train fit --run NAME [--split molecule]
  [--encoder dense|flex|sparse] [--epochs 30] [--cell-budget 1100000] [--max-tokens 1024]
  [--resume] [--no-resonance-draws] [--eval-test] [--preset general|legacy|open_shell]
  [--drop-domain-class C...] [--val-cohort pmechdb=closed_base]` (the spec-v9 model on one split;
  `--preset` sets the flags of `model.config.PRESETS` as option defaults: general = every pair cell and the
  orbital extras, legacy = the v17 inputs, open_shell = general with the atom head over
  {-2,-1,0,+1,+2}, the pair features of `OPEN_SHELL_BLOCK` (statistics fitted on the run's training
  samples) and the P7 inputs (P9); an explicit option wins; `--source pmechdb+reactmech`
  pools several sources' training parts, validates each and the pool, pair statistics fitted on the pool,
  `--split pmechdb=molecule,rmechdb=rc_molecule` names each source's split, a run with pmechdb and
  rmechdb leaves `cohort_leak_guard_joint` out of training (LEAK_GUARD, counted), `--val-cohort`
  restricts a source's validation part to a cohort, and run.json states the selection and the
  statistics; each epoch every training
  sample in a random one of its resonance drawings unless `--no-resonance-draws`; `--encoder`: `dense` the pair track of spec v9,
  `flex` FlexAttention with a relative bias and no pair state (GPU only), `sparse` top-k keys from a
  learned indexer, dense during the LR warm-up; `--stop-weight` is lambda_stop of the END_MECH head,
  by default 0.5 and 0 on PMechDB; `data/models/<source>/<run>/`: `run.json` with the provenance of
  the views and pair statistics, `train.log` in the `step n / total` format `ksmech-eta` reads,
  `metrics.jsonl` per epoch, `last.pt` and `best.pt` by validation anchor top-1; batches by a
  padded-cell budget, samples over `--max-tokens` skipped as TOO_LARGE and counted; the override is
  for the RX 6600 XT (gfx1032 runs the gfx1030 kernels); `--eval-test` also scores the test part each epoch (`test_*`, never for `best.pt`);
  `--drop-domain-class OSM_CU2 OSM_FE3` leaves those classes' training samples out (P4's metal
  ablation; evaluation keeps them; counted as DROPPED_DOMAIN_CLASS);
  every `ModelConfig` field is an option,
  e.g. `--drop-pair-features g log_abs_g` for the G ablation, `--no-somo-channels`,
  `--no-own-pair-channel`; metrics.jsonl also holds `open_top1`, `open_r_exact`, `open_n` over the
  pools with an unpaired electron; `--limit 64 --epochs 1` for a smoke run)
- Evaluation: `uv run ksmech-eval floor --source pmechdb [--split molecule] [--cells valid|all]
  [--floor legacy|own_break] [--kernel g|g2] [--drop-intra-open]` (the physics floor:
  anchors ranked by |G^atom| over the valid orbital pairs, or every pair, no learned parameter, val and test
  top-1/3/5, `anchor_floor_report_<split>.json` and per-sample `anchor_floor_ranks_<split>.json`;
  `own_break` adds a break anchor on every bonded pair scored by |B^own|, so a homolysis is
  reachable; files tagged by floor and kernel when not the default),
  `uv run ksmech-eval decode --run NAME [--run-source pmechdb+reactmech] [--part val|test]
  [--limit N] [--beam 5] [--scope all|inter] [--caps legacy|labels] [--max-bonds B --max-atoms A
  --max-candidates C] [--closed-minus1-filter]` (the beam decoder of
  `model.decode` on the stored samples, B + R checked on the sample's Lewis structure: full-R
  top-1/3/5, a hit when a completion is a variant of the label class, the drop counts, the
  checkpoint's and the view's sample versions (a mismatch warns); caps legacy = 8 bonds, 2
  atoms, 48 candidates, labels = the label maxima of --source's training part, caps other than
  legacy tag the files `_caps<b>-<a>-<c>`; on a pool with no unpaired electron the closed-shell
  screen drops a +1 atom entry off every broken bond (ODD_OFF_BROKEN_BOND), `--closed-minus1-filter`
  also any -1 atom entry (CLOSED_MINUS1, files tagged `_minus1filter`, off until P9's ablation sets
  it); `odd_top1` and per-sample `screened` in the report and the hit file; `rank_any` and
  `top{k}_any` against every product the source records for the same reactant pool (identity key
  v2, `eval.siblings`; siblings from the whole source, those of another part counted), with the
  multi-outcome subset (`multi_outcome`);
  `decode_report_<split>_<part>[_<source>][_<cohort>][_n<limit>].json` and per-sample
  `decode_hits_...json` beside the checkpoint; `--run-source` is the run's models directory when
  it differs from `--source`, the samples decoded); both take `--cohort NAME` (only the steps of
  `cohort_<NAME>.json`, `eval.cohorts`) and `--slice fermi_smeared|bond_stretched` (with and
  without), and report the covered fraction; `uv run ksmech-eval cohort --source pmechdb --name
  open_shell --domain-class RADICAL_SPECTATOR RADICAL_REACTIVE` (the training steps of some
  domain_class values pinned once as a cohort); `uv run ksmech-eval pin --source pmechdb --name closed_base [--workers N]`
  (open-shell plan P0b: the molecule split's samples as a cohort, its GBDT subsets `<name>_gbdt_valid|all`
  and `baseline_<name>.json`, per-sample digests of every named column, `eval.pin`; each written once),
  `uv run ksmech-eval compare --a HITS... --b HITS... [--k 1] [--metric own|any] [--out FILE]` (two arms of decode runs
  paired by sample: A fails when its mean top-k is below B's by more than max(2 x pooled SD, 1 point),
  the exact one-sided McNemar test on majority-of-seeds hits and every seed pair; arms decoded on other
  samples, cohorts, caps or decode versions refused, `eval.paired`); `uv run ksmech-physics gbdt --score-only [--model PATH
  --allow-stale]` scores the stored ranker of `--split` (or a reference copy) without a refit
  (`gbdt_lgbm_ranks_<split>_geomfree[_<model>].json`); P0a: `uv run python scripts/p0a_changes.py
  --source S` (every step the P0a build changed, removed, returned or flagged, by rule,
  `p0a_changes.json`, against the v17 copy in `data/processed_v17/`), `uv run python
  scripts/probes/dative_audit.py` (the 20-case oxidation-state audit of the dative law,
  read-only); P5: `uv run python scripts/p5_rescore.py [--pin]` (the two ReactMech checkpoints'
  decode hits by slice: unchanged since P0, P0a's Kekule and dative changes, P5's returns via
  `cohort_p5_returns.json`; `p5_rescore.json`); P6: `uv run python scripts/p6_checks.py dump
  --source S --out FILE` under two code versions, then `compare --old A --new B` (edit labels,
  resonance flags and keys of the v18 steps by content, closed and open shells apart), `uv run
  python scripts/probes/flower_label_dryrun.py --source flower|pmechdb|reactmech --out DIR`
  (FlowER's RS/RC lines and the v18 labels through the P6 rules: codes, R shapes, P9's masks and
  screens; read-only); P7: `uv run python scripts/probes/open_shell_g_table.py --out DIR`
  (|g| and |g2| by cell kind and the intramolecular |beta| by graph distance, from which
  BETA_INTER is stated; read-only); `uv run ksmech-eval importance --run NAME [--part val]
  [--scope columns|blocks|both] [--repeats 3] [--limit N] [--cohort C]` (permutation importance
  of the token inputs: each column, and each block of related columns, permuted across the
  part's tokens of its kind and the part scored again; the rise of the anchor NLL and the fall of
  top-1/3 and greedy R against the baseline; pair features untouched, `spin_role` left out;
  `token_importance_<split>_<part>[...].json` beside the checkpoint, `eval.importance`);
  `uv run ksmech-eval embedding --run NAME
  [--step-id ID | --index K [--part val] | --smiles 'CC(=O)C.[OH-]'] [--drawing k]
  [--checkpoint best|last]` (one sample through a trained model as a self-contained HTML page
  under `data/models/<source>/<run>/embedding/`: the tokens on the Lewis structure, the parts
  of h(0), every layer's attention and token updates (rebuilt from the recorded layer inputs and
  checked against the model's own forward), the decoder's anchor logits split into site, pair
  and physics terms; CPU by default; `--smiles` builds missing tokens with GFN2-xTB)
- Network: `uv run ksmech-network search --run NAME [--run-source pmechdb] --start SMILES --target
  SMILES [--k 5] [--max-expansions 50] [--max-depth 6] [--step-cost 0] [--levels EXACT SKELETON]
  [--beam 5] [--build-missing] [--out FILE]` (the engine's network, `ksmech.network`: pool states
  expanded by a checkpoint on the CPU, the k best pathways to the target by best-first search over
  simple paths (A* with `--step-cost` x the remaining-change bound when the target is mapped on the
  start pool, else uniform-cost; MAPPING_FAILED counted), arrival EXACT (identity keys) or SKELETON
  (heavy-atom connectivity, up to protons), mechanism classes up to the order of independent steps
  with p(class | reaches); the web app's network imports `network.graph` and `network.keys`)
- API client: `uv run ksmech [--url URL] [--token T] [--json] COMMAND` (the HTTP client
  `ksmech.client.KsmechClient`, `X-Ksmech-Via: cli`; `KSMECH_URL`, default http://127.0.0.1:8000,
  `KSMECH_TOKEN`): `health`, `meta`, `whoami`, `predictors`, `predict SMILES... [--predictor P]
  [--beam 5] [--checkpoint RUN[/FILE]] [--no-wait]`, `jobs list|get|wait|cancel|pathways`,
  `normalize`, `check STATE --bond I,J,D --lone I,D`, `pool create|get|svg`, `molecule
  create|get|props|symmetry|orbitals`, `depict2d`, `ocr IMAGE [--molfile OUT]`, `models`, `ml runs|run|reports`, `embedding
  create|get`, `curate list|get`, `browse status|search|step`; exit 0 ok, 1 API error, 2 usage)
- MCP server: `uv run ksmech-mcp [--transport stdio|http] [--port 8765]` (FastMCP over the
  client, `X-Ksmech-Via: mcp`; tools per `/api/meta` capability, read-only except
  predict_mechanism; `docs/mcp.md`; Claude Code: `claude mcp add ksmech --
  <repo>/.venv/bin/ksmech-mcp`)
