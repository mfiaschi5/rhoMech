# Data schemas

One section per dataset, written only after inspecting the real file. Never document a column
that was not seen in the file.

## PMechDB (via the PMechRP data release)

Inspected 2026-10-19.

### Where the files are

The download is in `data/raw/pmechrp/`, not `data/raw/pmechdb/`. It is the data release of the
PMechRP paper (Baldi lab), which repackages the manually curated PMechDB polar steps into model
splits. Its `README` says each fold has an `mc_only` ("manually curated only") set and a `mixed`
set. The `mixed` set adds combinatorially generated proton-transfer steps.

The layout, with `foldN` for N = 0..4:

| Path | Content |
|---|---|
| `two_stage_models/mc_only_splits/foldN/raw/{train,val,test}.txt` | **Ingested.** Curated steps, one per line. |
| `two_stage_models/mc_only_splits/foldN/raw/train_val_combined.txt` | train + val concatenated |
| `two_stage_models/mc_only_splits/foldN/reformatted/{train,val,test}.txt` | same steps as 4 CSV fields (see below) |
| `two_stage_models/mc_only_splits/foldN/allids/train_val_combined.json` | `{graph-topology-feature-string: int}` lookup, 6273 keys; not step data |
| `two_stage_models/mixed_splits/...` | same layout; train has +10,000 combinatorial proton transfers |
| `scaffold_split/{train,val,test}.txt` | same 13,104 curated steps, Murcko-scaffold split (11076/1014/1014) |
| `chemformer/`, `graph2smiles/`, `molecular_transformer/`, `t5chem/` | model-specific tokenized copies (no arrows) |
| `pathway_datasets/human_benchmark/`, `pathway_datasets/ORD_USPTO/` | multi-step pathway benchmarks: unmapped SMILES, no arrows |

Checks that were run (`uv run python -I <script> data/raw/pmechrp`):

- The union of fold0 `raw/{train,val,test}.txt` has 13,104 lines, all unique.
- Every fold has the same union and the same test set (1,337 lines). Only train and val
  rotate (10,430 / 1,337).
- `scaffold_split/` holds exactly the same 13,104 records.
- The `mixed` train set contains 10,000 records that are not in the curated set.

The ingester reads the 13,104 curated records once from
`two_stage_models/mc_only_splits/fold0/raw/{train,val,test}.txt`. The PMechRP fold and split
membership is **not** stored in our schema.

### Raw step file format (`raw/*.txt`)

- A headerless, one-column CSV that happens to have a `.txt` extension. Each line is a single
  field wrapped in double quotes (all 13,104 lines start and end with `"`).
- Inside the quotes are two space-separated tokens (all 13,104 lines have exactly two):
  1. `reaction`: reaction SMILES `reactants>>products`. Fragments are joined by `.`. Kekulé form
     only: no aromatic lowercase atoms were found in the curated set.
  2. `arrow_code`: arrows separated by `;`. Each arrow is `source=sink`.

There is **no header** and **no other column**. In particular, the file has no step id, no
mechanism class, no pathway id or step index, and no conditions or reagent text. These fields are
missing from the release.

Header and three rows exactly as they appear in `fold0/raw/test.txt` (there is no header line):

```text
"[B:20](C1=C(N=CC=C1)Cl)(OC(C)C)OC(C)C.[OH-:10].[Li+].CC(C)[O-].[Na+]>>[B-:20](C1=C(N=CC=C1)Cl)([OH:10])(OC(C)C)OC(C)C.[Li+].CC(C)[O-].[Na+] 10=20,"
"C[CH+:20]C.C1=CN[CH:10]=[CH:11]1>>C[CH:20](C)[CH:10]1[CH+:11]C=CN1 10,11=10,20"
"[H:20][O+:21](CC)C(C)([O-])OC(=O)C.[OH2:10]>>[H:20][OH2+:10].CC[O:21]C(C)([O-])OC(=O)C 10=20;20,21=21"
```

### Arrow code grammar (measured over all 13,104 records)

| Shape | Count | Meaning |
|---|---|---|
| `a=b` (1 atom → 1 atom) | 10,682 | lone pair on `a` attacks atom `b`; a bond `a–b` forms or its order increases |
| `a,b=c` with `c ∈ {a,b}` (bond → atom) | 10,800 | the pair of bond `a–b` moves onto `c` as a lone pair |
| `a,b=c,d` sharing one atom (bond → bond) | 4,593 | the pair of bond `a–b` moves to bond `c–d` |

Measured facts about the grammar:

- Arrows per step: 1 → 2,414; 2 → 8,571; 3 → 1,960; 4 → 156; 5 → 3.
- Never seen: a lone pair pointing at a bond (`a=b,c`), a bond pointing at an atom outside it, a
  bond pointing at a disjoint bond, or `a=a`.
- No radical (single-electron) notation. PMechDB is polar only.
- 1,155 records end the arrow code with a stray `,` (e.g. `10=20,`). See `docs/decisions.md`.
- Atom indices in the arrow code are **atom-map numbers** in the reaction SMILES. Usually only
  the atoms the arrows touch are mapped, numbered from 10 (nucleophile side) and 20
  (electrophile side). 18 records map every atom (map numbers from 1).

### Data quality (RDKit 2026.09.1, `removeHs=False`)

| Check | Failures |
|---|---|
| map number used twice on one side | 0 |
| reactant map set ≠ product map set | 1 (`C[CH2:11]O[Na-:10][OH+:20]C(C)C>>CC(C)[OH:20].CCO[Na:10]`: map 11 vanishes) |
| arrow uses a map number absent from the reactants | 0 |
| heavy-atom element counts differ | 0 |
| total H count differs | 0 |
| total formal charge differs | 0 |
| RDKit sanitization fails (valence) | 12 (hypervalent ate complexes such as `Br[Br-:20][Br:10]`) |
| the arrows applied to the reactants do not give the products (`ARROWS_MISMATCH`) | 1 (pmechdb-9946a45b6a5d6f2e: the product drops the dative bond `[Na+]<-[C-:20]` without an arrow) |

Other notes:

- 55 records write a metal as a bare bracket atom with no H and no charge (`[K]`, `[CuH2]`).
  RDKit gives these atoms radical electrons. They are spectators and no arrow touches them.
- Explicit `[H]` atoms appear both mapped (`[H:20]`) and unmapped (`[H][C@]...`). Mapped H
  atoms must be kept, because arrows reference them.

### Reformatted file (`reformatted/*.txt`)

The same records as a 4-field CSV (no header), with fields quoted when they contain commas. The
PMechRP README names the fields "rxn, arrow, source, sink". Fields 3 and 4 are single-molecule
SMILES with one atom mapped `:1`: the source and sink molecules. Example:

```text
C[CH+:20]C.C1=CN[CH:10]=[CH:11]1>>C[CH:20](C)[CH:10]1[CH+:11]C=CN1,"10,11=10,20",c1c[cH:1][nH]c1,C[CH+:1]C
```

We do not ingest this file because it adds nothing beyond `raw/`.

### Pathway benchmark files (not ingested)

- `pathway_datasets/human_benchmark/pathways/350_HumanBenchmark.csv` (350 rows). Header:
  `Temp (≠ RT),Reactants,Final Product,Prod Formula,Min Steps,"Clayden, 2nd Ed Textbook Citation",Step 1 Target,...,Step 7 Target`.
  Unmapped SMILES, no arrows.
- `pathway_datasets/human_benchmark/pathways/raw/depthK.csv` (K = 1..7). Header:
  `Reactants, Final Target, Depth, Step 1 Target, ..., Step K Target`.
- `pathway_datasets/ORD_USPTO/raw/ORD_USPTO_transformations.csv` (133 lines). No header; one
  unmapped reaction SMILES per line.

## Processed: `data/processed/<source>/steps.parquet`

There is one row per `ksmech.data.schema.ElementaryStep`:

| Column | Arrow type | Notes |
|---|---|---|
| `step_id` | string | `pmechdb-` + first 16 hex chars of SHA-1 of the raw record (the text inside the quotes) |
| `source` | string | `pmechdb`, `rmechdb`, `reactmech`, `mech_uspto`, `flower` |
| `pathway_id` | string, nullable | null for PMechDB (not in the release) |
| `step_index` | int32, nullable | null when `pathway_id` is null |
| `reactants_smiles` | string | canonical atom-mapped SMILES (see `docs/decisions.md`) |
| `products_smiles` | string | same canonicalization |
| `arrows` | list<struct<source_kind, source_atoms: list<int32>, sink_kind, sink_atoms: list<int32>>> | in written order |
| `mechanism_class` | string, nullable | null for PMechDB (not in the release) |
| `conditions` | string, nullable | null for PMechDB (not in the release) |

Rejected records go to `data/processed/<source>/rejections.csv`, with columns `step_id`,
`reason`, `detail` and `raw`.

## Processed: `data/processed/<source>/samples_<split>.pt`

Written by `ksmech-arrows samples` (`ksmech.arrows.steps`). The pair features use the contact
kernel given by `--kernel`: `oriented` by default (d = 2.3 Angstrom, blocks cached under
`data/cache/oriented/`, site-based lobes and clash guard since version 3); `table` needs `data/cache/contact_table.json` and `mlp`
`data/cache/contact_mlp.npz`. `<split>` is `<strategy>_<part>`:
strategy `random` (by pathway), `scaffold` or `loco-<class>`; part `train`, `val`, `test`.
A `torch.save` dict, loadable with `weights_only=True` (tensors, lists, strings, numbers only).
Fields listed from a real file (`samples_random_val.pt`, PMechDB, version 3, 2026-10-20);
`load_samples` refuses any other version:

- top level: `version` (3), `split`, `n_steps` (steps in the split), `n_samples`, `skipped`
  (reason code -> count: `MISSING_TOKENS`, `CORRESPONDENCE`, `UNALIGNED_ARROW`), `n_labels`,
  `n_fermi_smeared`, `samples` (list). The same counts are in `samples_<split>.json`.
- one sample (`MechanismSample.to_dict`), `A` atoms and `O` orbitals over all reactant
  fragments; global token ids are atoms `0..A-1`, then orbitals `A..A+O-1`:

| Field | dtype, shape | Notes |
|---|---|---|
| `step_id`, `source` | str | |
| `fragment_keys` | list[str] | token-cache key (map-free canonical SMILES) of each fragment, in fragment order |
| `atom_categorical` | int64, (A, 2) | atomic number, formal charge (record values) |
| `atom_features`, `atom_missing` | float32 / bool, (A, 10) | `atom_feature_names()`; missing values are 0 and flagged |
| `atom_fragment`, `orbital_fragment` | int64, (A,) / (O,) | fragment index |
| `atom_map` | int64, (A,) | atom-map number of the drawn atom, 0 when unmapped or an implicit H |
| `atom_positions`, `orbital_centroids` | float32, (A, 3) / (O, 3) | Angstrom, record geometries; for distances only |
| `orbital_features` | float32, (O, 23) | `feature_names()` |
| `orbital_spectra` | float32, (O, 8, 2) | (E eV, p) spectral set |
| `orbital_occupied`, `donor_mask`, `acceptor_mask` | bool, (O,) | candidate masks (`CandidateConfig`); the donor-partner exclusion is in `pair_valid` |
| `incidence` | float32, (E, 3) | rows (orbital index, atom index, weight) for every centre of every orbital |
| `pair_z` | float32, (O_occ, O_virt, 7) | Klopman-Salem pair vector of every (occupied, virtual) pair, rows and columns in orbital order within each block; entries `PAIR_FEATURES` = ks3, coupling_sq, gap_kernel, ks2, ks1, inter, distance, computed at K = eps = 1 (multiply by `physics.params.scale`; `physics.ranking.physical_pair_z`). With the oriented kernel ks3 = (K S_rs (E_r + E_s) / 2)^2 G |
| `pair_defined` | bool, (O_occ, O_virt, 7) | where the entry exists (coupling_sq only within a fragment, ks2 not on one site, ks1 not when %Vbur is missing); undefined entries are 0 |
| `pair_valid` | bool, (O_occ, O_virt) | arrow candidates: donor and acceptor masks, minus every pair sharing a centre atom within a fragment |
| `pair_contact_sites` | int64, (O_occ, O_virt, 2) | oriented kernel: the (donor site, acceptor site) atoms (global indices) whose placement gave the pair's contact amplitude, the maximum over the two tokens' centres; -1 with another kernel |
| `pair_clash` | bool, (O_occ, O_virt) | oriented kernel: every placement of that site pair clashed (atoms closer than 1.5 Angstrom); the amplitude is then the smallest found |
| `atom_coulomb` | float32, (A, A) | symmetric KS2 between atoms at eps = 1: geometric distance within a fragment, van der Waals contact across; 0 diagonal |
| `labels` | list[dict] | `ArrowLabel.to_dict`: `donor_ids`, `acceptor_ids` (global token ids), `donor_site`, `acceptor_site` (atom indices), `flags`, `arrow_index`, `source_kind`, `donor_partner`, `sink_partners`, `absorbed` |
| `product_key`, `product_key_flag` | str | products' canonical resonance key and its `ResonanceFlag` |
| `fermi_smeared` | bool | a reactant fragment is flagged `fermi_smeared` |

`ksmech.arrows.steps.collate_samples` pads a batch to `A_max` atoms and `O_max` orbitals,
lays tokens out as atoms `[0, A_max)` then orbitals `[A_max, A_max + O_max)`, and returns
`token_mask`, `is_atom`, `is_orbital`, `occupied`, `virtual`, `donor_mask`, `acceptor_mask`,
`fragment` (-1 padding), `same_fragment` (B, N, N), `incidence` (B, N, A_max; atom rows
one-hot), `donor_sets` / `acceptor_sets` (B, K, N multi-hot), `donor_site` / `acceptor_site`
(B, K; -1 padding) and `label_mask`, plus the padded input arrays.

## QM cache: `data/cache/qm/<sha256>.npz`

Written by `ksmech.qm` (one file per molecule and conformer). Fields were listed from a real
cached file (acetone, 10 atoms, 22 AOs). `n` = atoms, `nao` = AOs, `nmo` = MOs (= `nao`). The file
is a plain `.npz` (no pickle). `ksmech.qm.records.load_record` returns a `QMRecord`.

The key is `sha256(f"{smiles}|{conformer_id}|{method}")`, where `smiles` is the map-free canonical
SMILES (`ksmech.data.chem.unmapped_canonical`) and `method` looks like
`GFN2-xTB/ALPB-water/tblite-lbfgs` (so the solvent model and optimizer are part of the key).

| Field | dtype, shape | Notes |
|---|---|---|
| `smiles`, `atom_order_smiles` | str, () | the second one defines the atom order: `Chem.AddHs(Chem.MolFromSmiles(atom_order_smiles))` |
| `conformer_id`, `charge`, `unpaired` | int, () | charge = sum of formal charges, unpaired = sum of radical electrons |
| `method`, `schema_version` | str / int, () | |
| `atomic_numbers` | int32, (n,) | |
| `coords_ang` | float64, (n, 3) | optimized geometry, Angstrom |
| `energy_ha` | float64, () | total energy, Hartree (includes the ALPB term) |
| `orbital_energies_ev` | float64, (nmo,) | eV |
| `occupations` | float64, (nmo,) | restricted; radicals have a fractional SOMO (1.0) |
| `C` | float64, (nao, nmo) | AO x MO, S-orthonormal: `C.T @ S @ C = 1` |
| `S`, `H_core`, `fock` | float64, (nao, nao) | Hartree for the last two. `H_core` is tblite's core Hamiltonian; `fock = S C diag(eps) C.T S` is rebuilt from the converged MOs |
| `charges` | float64, (n,) | Mulliken, positive = electron-poor |
| `ao_atom`, `ao_l` | int32, (nao,) | atom index and angular momentum (0 s, 1 p, 2 d) of every AO; order per atom is s, p x3, d x5 |
| `fukui_plus`, `fukui_minus`, `fukui_zero`, `fukui_dual` | float64, (n,) | finite difference N+-1 on Mulliken charges, NaN if unavailable |
| `fukui_fmo_plus`, `fukui_fmo_minus` | float64, (n,) | frozen-orbital: Mulliken population of LUMO / HOMO |
| `homo_ev`, `lumo_ev`, `chemical_potential_ev`, `hardness_ev`, `electrophilicity_ev` | float64, () | NaN without a frontier pair |
| `buried_volume` | float64, (n,) | %Vbur, radius 3.5 A, hydrogens included, each atom as centre |
| `warnings` | str, () | comma-separated codes: `FUKUI_SCF_FAIL`, `VBUR_FAIL`, `NO_FRONTIER`, `RESTRAINED_GEOM`, `EMBED_UFF_FALLBACK`, `EMBED_RANDOM_COORDS` |

### Manifest: `data/processed/<source>/qm_manifest.parquet`

One row per (`smiles`, `conformer_id`, `method`): `smiles`, `conformer_id`, `cache_path`,
`status` (`ok` / `failed`), `reason` (a `ReasonCode`, empty when ok), `detail`, `warnings`,
`seconds`, `n_embedded`, `method`. Fragments that cannot be canonicalized get `conformer_id = -1`.

## ksmech-serve outputs (inspected 2026-10-21)

Everything below was read back from real files: a synthetic run (MeO- + MeCl with the stub
predictor, beam 2, max steps 1; validate the SN2 pathway; override; export; re-check) through
the FastAPI TestClient on a scratch data root, and an index build over the mini PMechDB fixture.
The shared `data/cache/` and `data/processed/pmechdb/` are never written.

### Export: `data/curation/export_raw/steps.parquet` and `data/processed/user_curated/steps.parquet`

Both use the `STEP_SCHEMA` of `data/processed/<source>/steps.parquet` above (plus the
`resonance_step` bool column `build_step` computes). The raw file is what the export writes;
the processed one is its re-ingest through `ksmech.data.user_curated.load` and
`write_outputs`, with identical rows. One row per exported curated step:

| Column | Value |
|---|---|
| `step_id` | `user_curated-` + SHA-1[:16] of `"<reactants>>><products> <pmechdb code>"` |
| `source` | `user_curated` |
| `pathway_id` | `user_curated-path-<mechanism id>` |
| `step_index` | int32: the step's `order_group` (steps that share it are independent / unordered) |
| `reactants_smiles`, `products_smiles` | Kekule, every atom mapped and every H explicit, e.g. `[C:1]([O-:2])([H:3])([H:4])[H:5].[C:6]([Cl:7])([H:8])([H:9])[H:10]` |
| `arrows` | the dataset arrows on map numbers, e.g. `[{lone_pair [2] -> atom [6]}, {bond [6, 7] -> atom [7]}]` |
| `mechanism_class` | null |
| `conditions` | the mechanism's conditions, or null |
| `resonance_step` | recomputed by `build_step` (false for the SN2) |

The same directory holds `rejections.csv` (header `step_id,reason,detail,raw`, empty for a
clean export), `ingest_report.json` (`source`, `inputs`, `n_records`, `n_accepted`,
`n_resonance_steps`, `n_rejected`, `rejected_fraction`, `rejections_by_reason`, `notes`),
`export_report.json` (the `ExportReportOut` fields `export_id` (`x` + 16 hex), `created_at`,
`spec_version`, `paths`, `n_mechanisms`, `n_steps_considered`, `n_steps_written`, `skipped`,
`ingest_rejections`, `duplicates_pmechdb`, `duplicates_in_export`, `withdrawn`, `changed_ids`,
`steps` (per step: `mechanism_id`, `step_index`, `step_id`, `code`, `detail`, `provenance`
{edge_id, edge_kind, job_id, physics_fingerprint, predictor, predictor_version,
spec_version}), `sha256` (of the processed `steps.parquet`), plus `warnings`, `statuses` and
`withdrawn_steps`) and `pmechdb_arrow_codes.csv`: no header, every field quoted, one record per
line in the PMechDB raw format with the H atoms the arrows do not name folded into their heavy
atoms, e.g.

```
"[CH3:1][O-:2].[CH3:6][Cl:7]>>[CH3:1][O:2][CH3:6].[Cl-:7] 2=6;6,7=7"
```

### Curation database: `data/curation/curation.sqlite`

SQLite in WAL mode; datetimes are stored as naive UTC and read back as aware UTC. Foreign keys
are declared but not enforced.

| Table | Columns (primary key first) |
|---|---|
| `mechanism` | `id` int; `created_at`; `reactants` (JSON list of SMILES); `stoichiometry` (JSON list of int); `conditions` (nullable); `predictor`; `checkpoint_hash` (the predictor version string, e.g. `stub/fce42b87c0cd/bf7faeb705a9` or `floor/ks3+ks2/d2.3/<fp>`); `spec_version`; `status` (predicted / validated / corrected / rejected); `notes`. Indexes on created_at, predictor, spec_version, status |
| `curatedstep` | (`mechanism_id`, `step_index`); `order_group`; `reactant_pool_key`, `product_pool_key` (e.g. `CCl.C[O-]`, `COC.[Cl-]`; indexed); `arrows_json` (the StepDoc below); `was_corrected` bool; `original_arrows_json` (the original prediction's StepDoc of a corrected step, else null) |
| `recheck` | `id`; `mechanism_id`, `step_index`, `created_at`, `predictor`, `predictor_version`, `spec_version`, `physics_fingerprint`, `aligned` bool, `first_arrow_rank` (1-based, nullable), `n_valid`, `top1`, `top5`, `product_match` (`exact`, ..., `UNCHECKED` when the replay could not run), `top1_reproduces_product`, `reason` (null when the step was evaluated) |
| `exportrecord` | `export_id`; `created_at`, `path` (the processed steps.parquet), `sha256`, `n_steps`, `report_json` (the export report) |
| `exportedstep` | `step_key` (JSON `[reactant_pool_key, product_pool_key]`); `step_id`, `mechanism_id`, `step_index`, `first_export_id`, `last_export_id` (the ledger; withdrawn keys stay) |

The StepDoc (`arrows_json`, format `ksmech-serve/step-v1`) has the keys `format`, `job_id`,
`edge_id`, `edge_kind` (predicted / corrected), `reactant_state_id`, `product_state_id`,
`reactants_mapped`, `products_mapped`, `arrows` (per token arrow: `donor`, `donor_site`,
`acceptor`, `acceptor_site` (global token and atom ids of the reactant state), `donor_site_map`,
`acceptor_site_map`, `donor_element`, `acceptor_element`, `source_kind`, `source_partner_map`,
`sink_partner_map`, `donor_centres_map`, `acceptor_centres_map` (map number -> centre
weight)), `dataset_arrows`, `pmechdb_code` (e.g. `2=6;6,7=7`), `encode_reason`, `score`,
`probability`, `rank`, `predictor`, `predictor_version`, `spec_version`,
`physics_fingerprint` and `touched_maps` (e.g. `[2, 6, 7]`). An original StepDoc also has
`target_pool_key`.

### Serve-owned files: `data/serve/`

| Path | Content |
|---|---|
| `qm/<sha256>.npz`, `tokens/<sha256>.npz` | QM records and token files in the formats of `data/cache/qm` and `data/cache/tokens`, for species the server computed (synthetic ones under the method `SYNTHETIC-EHT/v1`) |
| `oriented/<oriented fp>/<block key>.npz` | oriented-block cache (upstream format) namespaced by the oriented fingerprint; `token_digests.json` maps each token key to the sha256[:16] of its token file |
| `pools/<state_id>/state.json` | the `PoolState`: `state_id`, `mapped_smiles`, `pool_key`, `pool_id`, `species` (species_key, token_keys, count, charge, resonance_flag), `fragment_token_keys`, `n_atoms` |
| `pools/<state_id>/analysis-<physics fp>-<dig12>.npz` and `.json` | the pool analysis (members below); the JSON holds `analysis_digest`, `candidate_config`, `chemistry`, `created_at`, `ks_params`, `n_non_finite`, `n_valid`, `physics_fingerprint`, `predictor_errors`, `predictor_versions`, `seconds`, `state_id`, `token_digests`, `versions` |
| `pools/<state_id>/analysis-failure-<fp>.json` | a recorded analysis or featurization failure (`code`, `upstream`, `detail`, `error_detail`, `created_at`, `digest`, `stage`, `key`) |
| `pools/<state_id>/scores-<sha12>.npz` | extra predictor scores (member `scores`) |
| `cubes/<sha32>.cube`, `cubes/<sha32>-donor.cube`, `-acceptor.cube` | Gaussian cubes in bohr (an orbital, and the two cubes of a contact view on one grid) |
| `contact/<sha32>.json` | the `ContactOut` fields of a placement plus `cube_paths`, `placement_version`, `created_at`, `acceptor_context_note` |
| `jobs/<job_id>/job.json` | the `Job` (`id`, `kind`, `status`, `created_at`, `started_at`, `finished_at`, `request`, `progress`, `error`, `parent_job_id`, `network_id`, `result_path`, `label`, `skipped`, `provenance`, `cancel_requested`) |
| `jobs/<job_id>/network.json` | predict jobs: `network_to_dict`, format `ksmech-serve/network-v1` (`network_id`, `revision`, `complete`, `root`, `nodes`, `edges`, `cycles`, `pathways`, `failed_proposals`, `skipped`, `params`, `predictor`, `chemistry`) |
| `jobs/<job_id>/result.json`, `events.jsonl` | other kinds' results (override, export, recheck, index); one reason-coded skip per line (`time`, `code`, `detail`, `count`, where) |
| `index/pmechdb-<sha12>.parquet` and `.json` | the PMechDB browse index (below) and its meta (`built_at`, `index_version`, `job_id`, `n_indexed`, `n_steps`, `pool_key_version`, `skipped`, `source`, `source_sha12`, `split_badges`) |
| `logs/failures.jsonl` | every reason-coded skip of every job (`time`, `code`, `detail`, where) |
| `run/server.lock`, `run/task-<seq>.pid` | the server lock (holder pid) and the pid of each running worker task |

Members of an analysis npz (MeO- + MeCl, synthetic: 14 occupied x 13 virtual orbital tokens);
every pair matrix is (n_occ, n_virt), row i is the occupied token `rows[i]`, column j the
virtual token `cols[j]` (global token ids):

| Member | dtype, shape | Meaning |
|---|---|---|
| `rows`, `cols` | int64, (n_occ,), (n_virt,) | global token id of each row / column |
| `valid` | bool | `pair_valid` of the sample (the candidate masks) |
| `reasons` | int8 | `candidates.PairReason` code of each cell (0 = valid) |
| `contact_sites` | int64, (n_occ, n_virt, 2) | the oriented kernel's contact site pair (global atoms), -1 when undefined |
| `gap_sign` | int8 | sign of G (`MechanismSample.pair_gap_sign`, bookkeeping) |
| `ks1`, `ks2`, `ks3` | float64 | the three KS terms (eV), with `ks1_ok`, `ks2_ok` (bool) where defined |
| `gap_kernel` | float64 | G (1/eV) |
| `amp` | float64 | the contact overlap amplitude S (`kernel.amplitude_matrix`) |
| `clash` | bool | the oriented placement's clash flag |
| `inter` | bool | intermolecular pair (`scope_mask(rt, "inter")`) |
| `comp_log_abs_ks1`, `comp_fbar` | float64 | the pair components the analysis rebuilds `pair_components` from |
| `beta` | float64 | signed beta (eV) of the main 2.3 Angstrom kernel; KS3 = beta^2 G |
| `beta2_d2.0`, `beta2_d2.3`, `beta2_d3.0` | float64 | beta^2 per contact distance (eV^2), NaN where undefined |
| `floor` | float64 | -(KS3 + KS2), bit-equal to the floor scorer |
| `frozen` | float64 | `SCORERS["frozen_head"]` of the components (the untrained frozen-head logit) |
| `score_<predictor>` | float64 | every predictor available at analysis time (`score_floor`, `score_stub`, ...), -inf where not valid |
| `fermi_smeared` | bool, () | any fragment flagged `fermi_smeared` |

PMechDB browse index rows (`index/pmechdb-<sha12>.parquet`, one per accepted step of
`data/processed/pmechdb/steps.parquet`): `step_id`, `reactants_smiles`, `products_smiles`,
`pmechdb_code`, `n_arrows` (int32), `resonance_step`, `reactant_pool_key`, `product_pool_key`,
`arrows_json` (the dataset arrows as JSON), `reactant_species`, `product_species`,
`reactant_token_keys`, `product_token_keys`, `flags` (lists of string), `split_random`,
`split_scaffold` (`train` / `val` / `test` / `resonance`, from `arrows.steps.split_ids`).
