# ksmech

## Project goal

ksmech is a deep-learning tool that predicts organic reaction mechanisms as sequences of
elementary steps. Each step is a set of arrows, and each arrow runs from a donor localized
orbital to an acceptor localized orbital. A transformer scores candidate arrows, and its
attention is biased by the Klopman-Salem equation, so the model starts from a physical
picture of orbital interactions instead of learning it from scratch.

The prediction target is an autoregressive sequence of elementary steps. A step is a list of
(donor orbital token, acceptor orbital token) arrows. After each step the arrows are applied
as an edit to the bond-electron matrix, giving an intermediate. The intermediate is
re-featurized (3D embedding, xTB, localization, new tokens) and fed back in to predict the
next step, until the model emits a stop token.

## Representation

`docs/ksmech_architecture.tex` (spec v7.2) is the source of truth for the representation and
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

The three Klopman-Salem terms are computed as pair features between orbital tokens:

1. filled-filled (closed-shell) repulsion,
2. Coulomb interaction between the charges on the two sites,
3. filled-empty orbital interaction (overlap squared over the energy gap).

These features enter the transformer as per-head additive attention biases. One head is
hard-wired to the pure KS kernel (no learned parameters) so the ablation "physics only vs.
learned" is a flag away.

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
uv sync                       # install everything into .venv
uv run pytest                 # run tests (use -x to stop at first failure)
uv run ruff check .           # lint
uv run ruff format .          # format
uv run pre-commit install     # enable the ruff + pytest -x commit hook
```

Main CLI entry points are added phase by phase and listed here as they land:

- Data: `uv run ksmech-data ingest --source pmechdb` (parquet + rejections + report under
  `data/processed/pmechdb/`), `uv run ksmech-data stats --source pmechdb` (writes `stats.json`)
- Provenance: `uv run ksmech-data provenance <artifact>` (prints what an artifact was built from and
  whether it is stale), `uv run ksmech-data provenance --backfill qm` (one-off: headers on QM records
  written before 2026-10-21; never while a QM batch runs). Loaders refuse stale artifacts
  (STALE_ARTIFACT); only `ks-rank` and `textbook` take `--allow-stale`.
- QM featurization: `uv run ksmech-qm run --source pmechdb --workers 8` (GFN2-xTB records under `data/cache/qm/`, manifest + reason-code table; see `docs/decisions.md`), `uv run ksmech-qm bond-index`
  (Wiberg bond indices of the Lewis bonds of every cached record, `data/cache/bond_index/`; the
  token cache needs them, so re-run after the QM batch)
- Orbital tokens: `uv run ksmech-orbitals build --source pmechdb --workers 4` (PM-localized orbital
  and atom tokens, one `.npz` per molecule under `data/cache/tokens/` plus `summary.json`; molecules
  without a QM record are skipped and counted, so re-run after the QM batch), `uv run
  ksmech-orbitals show --smiles 'CC(=O)C'` (token table of one cached molecule),
  `uv run ksmech-orbitals fg-vocab --source pmechdb` (regenerate the functional-group vocabulary),
  `uv run ksmech-orbitals contact-table --source pmechdb` (the contact overlap table of the KS
  terms, `data/cache/contact_table.json`, from the token cache; re-run after a token rebuild)
- Arrows: `uv run ksmech-arrows roundtrip --source pmechdb` (dataset arrows applied as one
  BE-matrix edit per step vs. the annotated products, `arrows_roundtrip.json`), `uv run
  ksmech-arrows align --source pmechdb` (arrow-to-orbital labels by atom, unaligned groups,
  predicted-arrow round trip, `alignment_report.json`), `uv run ksmech-arrows samples --source
  pmechdb` (the sample store `samples/shard_*.npz` + `index.json` with the geometry-free stage-A
  pair features, one `samples_<strategy>_<part>.json` view per split of `molecule`, `scaffold` and
  `loco`; the per-step `random_leaky` split only with `--strategy random_leaky`; `--workers N`
  spawned processes, keep the host total <= 4; re-run both after a token rebuild), `uv run ksmech-arrows contacts --source pmechdb [--split molecule]` (the legacy
  contact sidecar `samples/contacts_<split>/`: the 11 KS pair features with the oriented kernel at
  2.0, 2.3 and 3.0 Angstrom for val, test and a 3,000-step training subsample; needs `samples`)
- Physics: `uv run ksmech-physics ks-rank --source pmechdb [--split molecule|scaffold] [--kernel
  oriented|table|analytic] [--distance 2.0 2.3 ...]` (untrained KS ranking of each step's
  first arrow: the floor -(KS3 + KS2) and the frozen head on train + val and test, scorer
  table, failure classes, per-class table, `ks_rank_report_<split>_<kernel>[_d<d>].json`;
  oriented blocks are cached under `data/cache/oriented/`), `uv run ksmech-physics baseline
  --source pmechdb [--split molecule|scaffold]` (logistic regression + GBDT ranker,
  `baseline_report_<split>.json`, the models and a GBDT with the textbook pools held out), `uv
  run ksmech-physics textbook` (five textbook selectivity cases, expected vs top-3, cases seen in
  a model's training marked; builds missing QM records and tokens). The oriented kernel is the
  frozen baseline `physics/contact/legacy_fixed23.py`. `ks-rank` and `baseline` read the samples
  joined with the contact sidecar (train = its subsample). After a token rebuild: `ksmech-arrows
  samples` (stage-A features and `pair_feature_stats_<strategy>.json` from each training part),
  then `ksmech-arrows contacts`, then `ksmech-orbitals contact-table` for the table ablation, then
  `ks-rank`, `baseline`, `textbook`.
- Training: _not yet_
- Evaluation: _not yet_
