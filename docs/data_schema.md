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
| `resonance_step` | bool | reactants and products are resonance structures; dropped from the training target |
| `domain_exclusion` | string, nullable | `METAL_PLACEHOLDER`, `OPEN_SHELL_METAL` or `CHARGE_UNREPAIRED` (`ksmech.data.domain`): a fragment on either side is outside the closed-shell domain; dropped from the training target and the metric denominators (`read_training_steps`, without renumbering `step_index`). Null otherwise |
| `charge_repair` | string, nullable | JSON `{"repairs": [{"side", "rule", "before", "after"}], "reactants_before", "products_before"}` when an explicit ingest rule changed a formal charge (`CHARGE_REPAIRED`; the step stays); the SMILES columns hold the repaired sides |

Rejected records go to `data/processed/<source>/rejections.csv`, with columns `step_id`,
`reason`, `detail` and `raw`. Charge repairs are also listed one per row in
`data/processed/<source>/charge_repairs.csv` (`step_id`, `side`, `rule`, `before`, `after`,
`domain_exclusion`; fragments as map-free canonical SMILES). `ingest_report.json` counts
`domain_exclusions` by code and `n_charge_repaired`; `stats.json` has the same counts under
`domain`, with `RESONANCE_STEP`, and `n_training_steps` excludes both.

## Processed: `data/processed/<source>/samples/` and `samples_<split>.json`

Written by `ksmech-arrows samples` (`ksmech.arrows.steps`, `ksmech.arrows.store`; schema
`SAMPLE_VERSION` 15). The store holds every training step's sample once:

- `samples/shard_00000.npz` ...: 256 samples each, members `<i>/<field>` and a JSON `meta`
  member, with a provenance header whose inputs are the steps parquet and every token file its
  samples used;
- `samples/index.json`: step id -> shard, the shards as provenance inputs, and the skipped steps
  with their reason codes (such as `MISSING_TOKENS`).

A split is a view, `samples_<strategy>_<part>.json` next to the store (its ids and counts, with
the index and the split file as inputs); strategy `molecule` (default), `scaffold`,
`loco-<class>` or, only when named, `random_leaky`; part `train`, `val`, `test`. Loaders refuse
another schema version and stale or dirty artifacts (STALE_ARTIFACT). On disk the pair arrays
keep the valid cells only (`pair_valid`): `pair_z` and `atom_coulomb` as float16, the 0/1
features (`inter`, `bonded`) and `pair_defined` bit-packed, `atom_coulomb` as its upper
triangle; in memory every array is dense float32 again.

One sample (`MechanismSample`), `A` atoms and `O` orbitals over all reactant fragments; global
token ids are atoms `0..A-1`, then orbitals `A..A+O-1`:

| Field | dtype, shape | Notes |
|---|---|---|
| `step_id`, `source` | str | |
| `fragment_keys` | list[str] | token-cache key (map-free canonical SMILES) of each fragment, in fragment order |
| `atom_categorical` | int64, (A, 2) | atomic number, formal charge (record values) |
| `atom_features`, `atom_missing` | float32 / bool, (A, F_a) | `atom_feature_names()`; missing values are 0 and flagged |
| `atom_fragment`, `orbital_fragment` | int64, (A,) / (O,) | fragment index |
| `atom_map` | int64, (A,) | atom-map number of the drawn atom, 0 when unmapped or an implicit H |
| `atom_positions`, `orbital_centroids` | float32, (A, 3) / (O, 3) | Angstrom, record geometries; for distances only |
| `orbital_features` | float32, (O, F_o) | `feature_names()` |
| `orbital_spectra` | float32, (O, 8, 2) | (E eV, p) spectral set |
| `orbital_occupied`, `donor_mask`, `acceptor_mask` | bool, (O,) | candidate masks (`CandidateConfig`); the donor-partner exclusion is in `pair_valid` |
| `incidence` | float32, (E, 3) | rows (orbital index, atom index, weight) for every centre of every orbital |
| `pair_z` | float32, (O_occ, O_virt, 11) | geometry-free pair features of every (occupied, virtual) pair, rows and columns in orbital order within each block, computed at K = 1 (`physics.params.REDUCED`; `physics.params.to_params` scales `cross_coupling` to another K). Columns, `pair_features` (`physics.stage_a.STAGE_A_FEATURES`) in order: `g` (the spectral gap factor G, eV^-1), `log_abs_g` (log \|G\|, floored at 1e-12), `inter` (1 between fragments), `bonded`, `bond_order`, `graph_distance` (between the two centre sets on the record's Lewis structure: bonded when some centre of r is bonded to some centre of s, the largest such order, the shortest graph distance, 0 when they share an atom, capped at 6; across fragments 0, 0, 6), `wbi` (the largest Wiberg index over the Lewis bonds joining the centre sets, 0 when not bonded), `centroid_distance` (Angstrom), `cross_coupling` (\|l_r^T B l_s\| over the AO blocks between the centres), `align_cos` (\|cos\| between the p vectors of r and s on their main centres), `dihedral_cos` (site-partner-partner-site dihedral when the two sit on adjacent bonds) |
| `pair_defined` | bool, (O_occ, O_virt, 11) | where the entry exists: `centroid_distance`, `cross_coupling`, `align_cos` and `dihedral_cos` only within a fragment, `align_cos` only with p character on both main centres, `dihedral_cos` only on adjacent bonds; undefined entries are stored as 0 |
| `pair_gap_sign` | int8, (O_occ, O_virt) | sign of G (-1 for the normal order E_r < E_s); bookkeeping, not a feature |
| `pair_valid` | bool, (O_occ, O_virt) | arrow candidates: donor and acceptor masks, minus, within a fragment, the donor's own antibond and every pair whose site pairs (donor centre a, acceptor centre b) all have a and b identical or bonded with the acceptor's weight on a or the donor's on b at least 0.20 (`orbitals.tokens.site_excluded_matrix`, `CandidateConfig.site_weight`) |
| `pair_contact_sites` | int64, (O_occ, O_virt, 2) | always -1. A field of the removed contact model, kept only so that the schema-15 store stays readable; not a model input, and it leaves at the next sample schema bump |
| `atom_coulomb` | float32, (A, A) | symmetric q q / R between atoms at eps = 1 (geometric distance within a fragment, van der Waals contact across; 0 diagonal). Kept only so that the schema-15 store stays readable; not a model input, and it leaves at the next sample schema bump |
| `labels` | list[ArrowLabel] | the orbital reading of the arrows (`donor_ids`, `acceptor_ids`, sites, flags, ...); empty unless `orbital_reading` is `aligned` |
| `product_key` | str | sigma key of the products (`data.resonance.sigma_key`) |
| `edit_label` | EditLabel | the target: the parsimonious reaction-matrix class and its anchors on the global atoms (`arrows.delta_be`) |
| `bonds`, `lone_pairs`, `wiberg` | int16 (E, 3) / int8 (A,) / float16 (E,) | the reactant's Lewis structure as drawn (Kekulé, implicit H included): (i, j, order) with i < j, lone pairs per atom, the Wiberg index of each bond (`arrows.lewis`) |
| `orbital_reading` | str | `aligned`, `LABEL_FILTERED` or the first unaligned reason (e.g. `HETEROLYSIS_ONLY`) |
| `fermi_smeared` | bool | a reactant fragment is flagged `fermi_smeared` |
| `label_filter` | dict[str, int] | labeled (donor, acceptor) cells that `pair_valid` rejects, by group |
| `inputs` | dict[str, str] | provenance: every token file the sample was built from (`file:<path>` -> sha256) |
| `pair_features` | tuple[str] | the column names of `pair_z` / `pair_defined` |

**`pair_feature_stats_<strategy>.json`** (same directory, one per split strategy, written by
`ksmech-arrows samples`): a provenance header (kind `pair_stats`, schema `SAMPLE_VERSION`;
inputs the training view `samples_<strategy>_train.json` and the store index; config the
feature names, `cells: valid` and the physical constants) and `features`:
`{feature: {mean, std, count}}` over the defined entries of the valid cells (`pair_valid`) of
the strategy's training part, with the values rounded to float16 as the store keeps them, at
the physical K (`arrows.steps.pair_stats`, `physics.pairs.PairStats`). A refit on the loaded
samples gives the same numbers. Read it with `arrows.steps.load_pair_stats`, which refuses a
stale or dirty file (STALE_ARTIFACT) like the samples. Standardization of the model's pair
features (`physics.pairs.standardize`) uses these, never val or test.

`ksmech.arrows.batches.collate_samples` pads a batch to `A_max` atoms and `O_max` orbitals,
lays tokens out as atoms `[0, A_max)` then orbitals `[A_max, A_max + O_max)`, and returns
`token_mask`, `is_atom`, `is_orbital`, `occupied`, `virtual`, `donor_mask`, `acceptor_mask`,
`fragment` (-1 padding), `same_fragment` (B, N, N), `incidence` (B, N, A_max; atom rows
one-hot), `donor_sets` / `acceptor_sets` (B, K, N multi-hot), `donor_site` / `acceptor_site`
(B, K; -1 padding) and `label_mask`, plus the padded input arrays. The pair inputs keep each
sample's (occupied, virtual) layout, padded to the batch's largest counts (R, C): `pair_z`
(B, R, C, F; at K = eps = 1, unstandardized), `pair_defined` (B, R, C, F), `pair_valid` and
`pair_gap_sign` (B, R, C), `pair_row_pos` (B, R) and `pair_col_pos` (B, C), the batch token
position of each row's occupied and each column's virtual token (-1 padding), for scattering
into the N x N pair track, and `pair_features` (the column names); `atom_coulomb` and
`pair_contact_sites` are not collated. The reactant's Lewis structure and the reaction-matrix
targets are the `_reactant_tensors` and `_edit_tensors` keys.

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

## Serve workbench: `data/serve/workbench/`

Display-only artifacts of the molecule workbench (`ksmech-serve`, `src/ksmech/serve`), written
by the serve worker; never read by training or evaluation. `<sha>` is sha256[:32] of the JSON
of the cache key named below; every key includes the QM record's content sha (sha256[:16] of
the file bytes), the QM method, the workbench version constants and the oriented fingerprint.

| Path | Content | Key |
|---|---|---|
| `molecules/<molecule_id>.json` | registry entry: `molecule_id`, `key` (unmapped canonical SMILES), `method`, `failure` (the last featurize error body, or null), `first_posted`, `last_posted`, `version` | `molecule_id = "m" + sha256("molecule-v1|" + key)[:20]` |
| `orbitals/<sha>.npz` | the tokens of a non-model orbital set, the upstream token format with its provenance header (QM record input, `token_config(PMConfig(method, start))`) | key, set, start |
| `orbitals/<sha>.json` | the rows of one orbital set (`OrbitalSetOut` without `molecule_id`) | key, set, start, the model token file's digest (model set only) |
| `cubes/<sha>.cube` | Gaussian cube (bohr): two comment lines with kind, key, set, start, index, units, integral, expected value and notes | key, kind, set, start, index, quality or spacing, padding, the model token digest (model-set orbitals only) |
| `cubes/<sha>.json` | the cube's metadata: request, units, integral, expected, notes, spacing, grid shape, origin, min, max | as the cube |
| `properties/<sha>.json` | `MoleculePropertiesOut` without `molecule_id` and `geometry`; NaN written as null | key, the model token digest |
