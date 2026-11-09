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
  RDKit gives these atoms radical electrons. They are not all spectators: since 2026-10-21
  (odd fragments) and 2026-10-24 (RADICAL_DRAWN) the domain screen excludes them, and the
  open-shell survey (`ksmech-data open-shell-survey`, `open_shell_survey.json`) gives each one a
  subclass (BARE_ALKALI, CUH2_LIGAND, SN_PLACEHOLDER, PLACEHOLDER, ...).
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

## ReactMech (via the DeepMech data release)

Inspected 2026-10-26.

### Where the files are

Zenodo record 10.5281/zenodo.20305780 (CC BY 4.0), `DeepMech_Dataset_Model.zip` (45 MB, 139 MB
unpacked), from DeepMech (Das, Hoque, Baranwal, Sunoj, Chem. Sci. 2026, arXiv:2509.15872). Only
its `data/` folder is unpacked, under `data/raw/reactmech/DeepMech_Dataset_Model/`; the model
checkpoint `models/DeepMech_Split42.pth` is not.

| Path under `data/` | Content |
|---|---|
| `ID_Split42/{train,val,test}.txt` | **Ingested.** 73,474 / 10,497 / 20,993 steps, no line in two parts |
| `OOD_Split42/{train,val,test}.txt` | **Ingested.** 71,129 / 10,162 / 20,323 steps: the same steps partitioned by class, plus 9,235 not in the ID split |
| `OOD_Split42/OOD_CRM_dataset/OOD_Set1.csv` | **Read for classes.** 3,340 mechanisms, 5 classes |
| `OOD_Split42/OOD_CRM_dataset/OOD_Set2.csv` | **Read for classes.** 438 mechanisms, 4 classes; none matches a step |
| `ID_Split42/ID_CRM_dataset/ID_CRM.csv` | 2,751 mechanisms, no class column. Not read |
| `PrebioMech/{train,val,test}.txt` | 21,588 / 3,066 / 6,133 prebiotic steps, same line format. Not read |

### Step file format (`*_Split42/*.txt`)

One elementary step per line: an atom-mapped reaction SMILES `reactants>>products`, nothing else
(no `|`, no space, no header, no arrows). Implicit H on mapped atoms (`[CH3:1]`), aromatic
lowercase atoms, charges as drawn. Every line of the six files has `>>`.

- The two splits give 114,199 distinct lines (ID 104,964 + 9,235 OOD-only); 92,379 lines repeat
  one already read. The ingester reads each line once and counts the repeats
  (`repeat_across_files`); neither split's membership is stored.
- Ingest of all six files (RDKit 2026.09.1): 114,046 accepted, 153 rejected (0.13%):
  `CHARGE_UNBALANCED` 70, `MAP_NOT_BIJECTIVE` 48, `UNBALANCED` 35. 6,528 resonance steps.
  Domain exclusions: `OPEN_SHELL_METAL` 1,251, `METAL_PLACEHOLDER` 1,122, `RADICAL_DRAWN` 792.
- The reaction matrix comes from the two mapped sides; there are no arrows to apply. A hydrogen
  that moves is a count on both sides (`[OH+:5]` -> `[O:5]`, `[Cl-:1]` -> `[ClH:1]`): 4,454 of
  the pilot's 9,183 training steps move one or two. `arrows.hydrogens` writes each as a mapped
  atom before R is taken (docs/decisions.md, 2026-10-27).
- d-metal species are drawn with charge-separated dative bonds to a neutral metal, e.g. a Pd(0)
  phosphine as `CC(C)(C)[P+]([Pd])(C(C)(C)C)C(C)(C)C`: an odd electron count, which the domain
  screen excludes as `OPEN_SHELL_METAL` (Pd is not among the d0 / d10 acceptors of rule R2).
  Chromic acid is drawn `[O]=[Cr](=[O])([OH])[OH]` and passes. The most common metals are Pd
  (1,711 distinct reactant molecules), Cr (223), Ni (138), Ag (`[Ag+]`).

### Mechanism files (`*_CRM_dataset/*.csv`)

CSV with a header: `mechanisms` (ID_CRM) or `mechanistic_class,mechanisms` (OOD sets). A
`mechanisms` cell is a Python list literal of unmapped SMILES, one species (the whole pool) per
step boundary: 2-11 species in ID_CRM, 4 in Set1, 5 in Set2.

- **Classes.** A step takes a class when its map-free canonical reactants and products are two
  consecutive species of a classified mechanism; no pair is claimed by two classes. 9,948 steps
  get one of Set1's 5 classes: acid_halide 4,365, sul_halide 2,730, anhydride 1,584,
  haloformate 1,071, carbamic_halide 198. Set2's 4 classes (CO_coupling, bidentate_Ni_bha,
  bidentate_bha, intramolecular_bha: Buchwald-Hartwig aminations with octylamine, Pd and Ni)
  match no step: the split files' Ni steps are other couplings (aryl boronic acids). The other
  58 of the paper's 67 classes are not released.
- Not used yet: the pathways themselves (step order), ID_CRM's mechanisms.

## rmechdb (RMechDB via FlowER's data release)

Inspected 2026-11-05 (`scripts/probes/flower_inspect.py`, read-only; open-shell plan P8 step 1).

### Where the files are

FlowER (Joung et al., github.com/schwallergroup/FlowER; data figshare 32513667),
`flower_new_dataset/{train,val,test}.txt` under `data/raw/flower/`. Its RS and RC lines are
RMechDB's radical steps (Tavakoli, Chiu, Baldi et al., J. Chem. Inf. Model. 2023, PMC9976277).
License chain: RMechDB is CC-BY-NC-ND per its article; FlowER's repository and figshare record
carry an MIT label, which is not relied on for the RMechDB lines. Artifacts built from them are
kept as PMechDB's are (U18): local, the private rhoMech-data and pods, never the public
repository (not legal advice; the NC clause matters if ChemRho is offered commercially). No
real RS/RC line is tracked (`tests/test_p8_no_flower_lines.py`); `tests/fixtures/mini_flower.txt`
is written by hand.

| File | Lines | Numeric suffix | RC | RS | PM | PC |
|---|---|---|---|---|---|---|
| `train.txt` (3.1 GB, streamed) | 1,933,750 | 1,902,076 | 1,383 | 3,223 | 12,322 | 14,746 |
| `val.txt` | 21,723 | 21,723 | 0 | 0 | 0 | 0 |
| `test.txt` | 218,997 | 216,568 | 136 | 346 | 294 | 1,653 |

2,174,470 lines in all, 2,140,367 with a numeric suffix (about 2.14M).

### Line format

`mapped reactants>>mapped products|SUFFIX`, one elementary step per line, no header. Every atom
is mapped (RXNMapper style) with explicit H atoms: 0 unmapped atoms and 0 heavy atoms with
implicit H on the RS/RC lines. The suffix meanings are **inferred**, not documented: RS and RC
RMechDB's specific and core sets, PM PMechDB, PC unknown; numeric suffixes FlowER's own
generated steps. Only RS and RC are ingested; the stored column is the raw suffix
(`flower_suffix`), and the inferred core / specific reading lives only here, to be replaced
when the official RMechDB CSV joins (D1).

Measured on the RS/RC lines:

- bath gases: `[Ar]` and `[He]` third bodies on 165 lines (RC 137, RS 28), in every case one
  unbonded atom with one map number on both sides;
- O2 drawn only `[O][O]` (RC 89, RS 1,210 lines); CO only `[C]=O` (RC 5, RS 28); ozone `[O]O[O]`
  (RC 10, RS 10) or `[O][O+][O-]` (RC 1, RS 2); N2 `[N]=[N]` on 5 RC lines and `N#N` on 8;
- metals on RC lines only: Sn 56, Na 12, Si 10, K 3, Mg 3, B 1; Na and K are often drawn
  covalently (`[O][Na]` ketyls, the Na...benzene Birch adduct, K persulfate).

### Ingest (`ksmech-data ingest --source rmechdb`, `ksmech.data.rmechdb`)

Before the id, in this order: the bath gases are stripped (an unbonded noble-gas atom with one
map number on both sides; counted per line as the note `NOBLE_GAS_STRIPPED`), then the spin
redrawings (`domain.redrawn_side`: PAIRED_DRAWING, HIGH_SPIN_DRAWING) give the key. `step_id` is
`rmechdb-` + 16 hex of SHA-1 of the map-free canonical reaction after both, so a line drawn with
other maps or with a bath gas shares its twin's id. One step per id: the RC copy wins over RS
in any file order (note `rc_wins_over_rs`), otherwise the first line in file order (train, val,
test); every other line is rejected as `DUPLICATE_STEP`. A reaction in both FlowER parts has
`flower_part` `train+test` (note `in_train_and_test`). The steps carry no arrows: R is the
difference of the mapped sides. `ingest_report.json` records `via: flower`.

Measured (2026-11-05): 5,088 lines -> 4,535 steps; 553 DUPLICATE_STEP, 165 bath-gas lines, 83
RC copies kept over RS, 120 reactions in both parts. Domain codes: MULTIPLICITY_AMBIGUOUS 27,
MULTI_RADICAL_SPECIES 23, NULL_STEP 4 (3 of them identities only after the pairing redraw: an
ozone, a DMSO2 and an enediyne drawn as biradicals on one side), SPIN_NOT_CONSERVED 1 (O(3P)
inserting into BrO., R even); 149 resonance steps (147 resonance pairs). PAIRED_DRAWING on 39
steps. Kept: RC 1,188 and RS 3,145 training steps. Labels (`edit-labels`): 4,293 of 4,333
(99.08%; MISSING_TOKENS 16, NOT_ELEMENTARY 12, DISSOCIATIVE_ET_STEP 12, 11 of them radical
reactions of the paired ozone). A step whose drawn reactant gives no class is labeled on the
reactant drawing that gives one (spec v10; `drawings.source_label`), and its sample holds that
drawing.

## Processed: `data/processed/<source>/steps.parquet`

There is one row per `ksmech.data.schema.ElementaryStep`:

| Column | Arrow type | Notes |
|---|---|---|
| `step_id` | string | `pmechdb-` + first 16 hex chars of SHA-1 of the raw record (the text inside the quotes); `reactmech-` of the line; `rmechdb-` of the map-free canonical reaction after the bath-gas strip and the spin redrawings |
| `source` | string | `pmechdb`, `rmechdb`, `reactmech`, `mech_uspto`, `flower` |
| `pathway_id` | string, nullable | null for PMechDB (not in the release) |
| `step_index` | int32, nullable | null when `pathway_id` is null |
| `reactants_smiles` | string | canonical atom-mapped SMILES (see `docs/decisions.md`) |
| `products_smiles` | string | same canonicalization |
| `arrows` | list<struct<source_kind, source_atoms: list<int32>, sink_kind, sink_atoms: list<int32>>> | in written order |
| `mechanism_class` | string, nullable | null for PMechDB (not in the release) |
| `conditions` | string, nullable | null for PMechDB (not in the release) |
| `resonance_step` | bool | reactants and products are resonance structures: one identity key v2 with the atom maps kept and a balanced electron flow (`data.resonance.classify_step`, open-shell plan P6); dropped from the training target |
| `domain_exclusion` | string, nullable | `METAL_PLACEHOLDER`, `OPEN_SHELL_METAL`, `CHARGE_UNREPAIRED` or `RADICAL_DRAWN` (`ksmech.data.domain`): a fragment on either side is outside the closed-shell domain; dropped from the training target and the metric denominators (`read_training_steps`, without renumbering `step_index`). On a source that admits open shells by rule (PMechDB, open-shell plan P3) a RADICAL_DRAWN step is re-read by the open-shell laws, and a failed law puts its own code in place: `MULTIPLICITY_UNTABULATED`, `MULTIPLICITY_AMBIGUOUS` (a u >= 2 species the species table, `data.spin_table` v2, lacks or does not assign), `SPIN_NOT_CONSERVED` or `NOT_ELEMENTARY` (an odd atom entry of the arrows' R); a curated step keeps `RADICAL_DRAWN`. `SPIN_RULE_PENDING` (the table re-assigned a species) is P3's code, which old artifacts carry; since P4 such a count stands. Since P4 the same laws re-read an `OPEN_SHELL_METAL` step of a source that admits open-shell metals by rule (PMechDB): a metal fragment the metal table (`spin_tables/metal_v1.json`) assigns no count is `MULTIPLICITY_AMBIGUOUS` (or `MULTIPLICITY_UNTABULATED` when the species table lacks it), and a fragment code the laws do not re-read (`CHARGE_UNREPAIRED`, `METAL_PLACEHOLDER`) keeps the step out under that code. Since P6 the electron-flow law adds two step codes, set where no other code stands (`data.resonance.classify_step`): `NULL_STEP` (identical sides, R = 0; formerly a resonance pair) and `NONCONJUGATED_SHIFT` (one identity key with the atom maps kept, but no balanced flow between any reactant drawing and product form). Since P8 one spin and domain law holds for every source: a species with u >= 2 under `data.spin` that is not a Hund case (one degenerate open group: triplet O2, O(3P)), or one the species table decides singlet that no closed-shell drawing reaches (an open-shell singlet biradical), is `MULTI_RADICAL_SPECIES`, except in a counted exemption (`curation/multi_radical_exemptions.json`); a source that keeps drawn radicals out (ReactMech) relabels such a `RADICAL_DRAWN` step with the code too. ReactMech re-reads odd metals under the spin law since P8 (its 2 odd-Pd pilot steps are `MULTIPLICITY_AMBIGUOUS`). A source of one-electron steps (rmechdb) conserves u only in a step whose R moves no single electron, and law (R) does not apply to it. Null otherwise |
| `charge_repair` | string, nullable | JSON `{"repairs": [{"side", "rule", "before", "after"}], "reactants_before", "products_before"}` when an ingest repair of `ksmech.data.domain` changed the drawing (`CHARGE_REPAIRED`; the step stays): a formal charge (R1, the dative law `D dative onium->M`, A1 `A1 bare alkali +1`) or missing H (V1 `V1 methyl [C] +3H` and `V1 imidazol-2-yl C +H`, the named repairs of `src/ksmech/data/repairs_v1.json`, rule `BL_NAMED`); one row per side, fragment and rule; the SMILES columns hold the repaired sides |
| `domain_class` | string, nullable | the repairs and flags of the domain laws a step carries (`DomainClass`), several joined by `+` in enum order: `REPAIRED_DATIVE`, `REPAIRED_V1`, `REPAIRED_A1`, `REPAIRED_BL`, `METAL_SUSPECT` (R1 gives none); an admitted open shell `RADICAL_SPECTATOR` or `RADICAL_REACTIVE` (an R entry on an atom drawn with radical electrons); a curated open-shell exclusion `RADICAL_AMBIGUOUS` or `DRAWING_ERROR_UNREPAIRED` (with `RADICAL_DRAWN`); since P4 `CARBENE_SINGLET` (the species table pairs a species drawn with radical electrons: a carbene, CO drawn `[C]=O`) and `OSM_CU2` / `OSM_FE3` (an admitted Cu(II) or high-spin Fe(III) fragment, the open_shell_metal cohort); since P8 `PAIRED_DRAWING` (a species the species table decides singlet, drawn with odd-N atoms, redrawn as the closed-shell Lewis structure its bonded electrons reach: N#N, ozone O=[O+][O-], the Criegee C=[O+][O-]; every source admits it), `HIGH_SPIN_DRAWING` (a Hund-case species drawn paired, redrawn unpaired: O=O -> [O][O]; none on disk) and `RADICAL_STEP` (every admitted rmechdb step). A training candidate is kept only when its source's admit set (`domain.ADMIT`: the five P0a/P1 classes for every source, plus `RADICAL_SPECTATOR`, `RADICAL_REACTIVE` and `CARBENE_SINGLET` where the source admits open shells by rule and `OSM_CU2`, `OSM_FE3` where it admits open-shell metals by rule) holds all of its classes (rmechdb's also `RADICAL_STEP` and `HIGH_SPIN_DRAWING`). Null otherwise |
| `flower_suffix` | string, nullable | rmechdb only (P8): FlowER's raw suffix of the line kept, `RS` or `RC`; null for every other source and in files written before it |
| `flower_part` | string, nullable | rmechdb only (P8): `train`, `test`, or `train+test` for a reaction FlowER holds in both parts |

Rejected records go to `data/processed/<source>/rejections.csv`, with columns `step_id`,
`reason`, `detail` and `raw`. Charge repairs are also listed one per row in
`data/processed/<source>/charge_repairs.csv` (`step_id`, `side`, `rule`, `before`, `after`,
`domain_exclusion`; fragments as map-free canonical SMILES). `ingest_report.json` counts
`domain_exclusions` by code, `n_charge_repaired`, `charge_repairs_by_rule` (steps per rule) and
`domain_class` (steps per value); `stats.json` has the same counts under `domain` (with
`RESONANCE_STEP`), `charge_repairs_by_rule` and `domain_class`, and `n_training_steps`
excludes the resonance steps, the excluded ones and those the admit set leaves out.
`ingest_report.json` also holds `open_shell` (admitted steps by class, the admitted ones that
hold a non-metal species with u >= 2 that is not a Hund case (each must be an exemption), the
count of those holding a Hund-case species, the curated exclusions by reason) and `spin` (`data.spin`'s
rule version, the species table and the metal table with their content hashes).

Two tracked spin tables (`src/ksmech/data/spin_tables/`), read by `data.spin.assign`:

- `species_v2.json` (with `theory_v2.json`; v1 kept as P3's record): map-free canonical SMILES ->
  unpaired count or `MULTIPLICITY_AMBIGUOUS`, each with its spGFN2 states, both verdicts (A, B)
  and the case of rule C (`data.spin_table`).
- `metal_v1.json` (open-shell plan P4): `rows` of `id`, `element`, `oxidation_state`,
  `unpaired` (or `redox_incompatible`), optional `donors_any` / `donors_none` (donor element
  symbols, `H` for a hydride), `basis` and `citations`; `citations`, `notes`, `version`,
  `schema`, `content_hash` (over every other field). A row holds for a species only where the
  species table's rule C has A and B agree on the row's count.

Two tracked curation files (open-shell plan P3), step ids and reasons, no SMILES:

- `src/ksmech/data/curation/pmechdb_open_shell.json`: `entries` of `step_id`, `reason`
  (`RADICAL_AMBIGUOUS` or `DRAWING_ERROR_UNREPAIRED`), `decided` (date), `why`: steps the
  open-shell laws admit and the hand reading rejects. Each must pass the laws.
- `src/ksmech/data/curation/multi_radical_exemptions.json`: `entries` of `source`, `step_id`,
  `species` (map-free SMILES of its u >= 2 species), `reason`: the counted exemptions of P8's
  MULTI_RADICAL_SPECIES policy (an exempt step is neither redrawn nor named multi-radical).
  Every admitted step with a non-Hund u >= 2 species must be listed. One entry: PMechDB
  5d007b38 (Et3B + O2), which the spin law still holds back as SPIN_NOT_CONSERVED (the table
  gives the adduct u = 0).

`data/processed/rmechdb/leakage_audit.json` (`ksmech-data leakage-audit`, `data.leakage`; P8
step 6): species matched by the identity key v2 of every non-trivial (> 3 heavy atoms)
fragment; every overlapping step id of RC val/test with PMechDB's training part and any
PMechDB, ReactMech or mech-USPTO species, of RC val/test reactions with PMechDB and ReactMech
steps, of rmechdb training with closed_base val/test (LEAK_GUARD for joint training) and the
reverse. `--write-cohorts` pins `cohort_leak_guard_joint` and `cohort_rc_heldout_overlap`
(rmechdb), `cohort_rc_heldout_shared` and `cohort_closed_base_rmechdb_overlap` (pmechdb).

## Processed: `data/processed/<source>/samples/` and `samples_<split>.json`

`samples_report.json` lists, under `SKIPPED_BOND_STRETCHED`, the samples whose label changes a
reactant bond to Li, Na, Mg or K that the QM record holds further apart than 1.3 x the
covalent-radius sum (`steps.stretched_bonds`; a counted flag and the eval slice
`bond_stretched`, never a model input).

Written by `ksmech-arrows samples` (`ksmech.arrows.steps`, `ksmech.arrows.store`; schema
`SAMPLE_VERSION` 19 since open-shell plan P6: N and u per atom and drawing, the atom
categorical's u column and the identity key v2; 18 on main). The store holds every training
step's sample once:

- `samples/shard_00000.npz` ...: 256 samples each, members `<i>/<field>` and a JSON `meta`
  member, with a provenance header whose inputs are the steps parquet and every token file its
  samples used;
- `samples/index.json`: step id -> shard, the shards as provenance inputs, and the skipped steps
  with their reason codes (such as `MISSING_TOKENS`; since P6 also the label's flow codes
  `SET_STEP`, R with atom entries only, and `DISSOCIATIVE_ET_STEP`, bond entries whose
  electron flow is unbalanced in every member of the class, `arrows.delta_be.flow_skip`).

A split is a view, `samples_<strategy>_<part>.json` next to the store (its ids and counts, with
the index and the split file as inputs); strategy `molecule` (default), `scaffold`,
`loco-<class>` or, only when named, `random_leaky`; for rmechdb (P8) `rc_molecule` (default:
the molecule split over RC, every RS step in training except those sharing a non-trivial
species with an RC val/test step, LEAK_GUARD, counted in `samples_report.json` under
`excluded`) and `flower_given` (FlowER's own parts, val drawn as 10% of the FlowER-train RC
molecule groups; leaky by construction, for the RMechRP comparison only); part `train`, `val`,
`test`. Loaders refuse
another schema version and stale or dirty artifacts (STALE_ARTIFACT). On disk the pair arrays
keep every (occupied, virtual) cell, flattened cell by cell (since v18; v17 kept the
`pair_valid` cells only, about 89%): `pair_z` as float16 byte planes, the 0/1 features
(`inter`, `bonded` and the three exclusion flags) and `pair_defined` bit-packed, `pair_valid`
as the legacy mask; in memory every array is dense float32 again.

One sample (`MechanismSample`), `A` atoms and `O` orbitals over all reactant fragments; global
token ids are atoms `0..A-1`, then orbitals `A..A+O-1`:

| Field | dtype, shape | Notes |
|---|---|---|
| `step_id`, `source` | str | |
| `fragment_keys` | list[str] | token-cache key (map-free canonical SMILES) of each fragment, in fragment order |
| `atom_categorical` | int64, (A, 3) | atomic number, formal charge (record values; model inputs), and (v19) the unpaired count u of the atom in the drawing (`data.spin.spin_of_drawing`; bookkeeping, `model_input=False`: keys, diagnostics and P9's categorical-u ablation read it); `drawn(k)` sets the charge and u of drawing k |
| `atom_features`, `atom_missing` | float32 / bool, (A, F_a) | columns `atom_columns`; missing values are 0 and flagged |
| `atom_columns` | tuple[str] | `orbitals.tokens.STORED_ATOM_FEATURES` (v18): the v17 inputs `atom_feature_names()`, then the extras `ATOM_EXTRA_FEATURES`: `f_plus`, `f_minus` (finite-difference, NaN -> missing when FUKUI_SCF_FAIL), `f_plus_fmo_exact`, `f_minus_fmo_exact`, `fmo_exact_<level>` (frontier blocks within 1e-4 eV); since v19 (`steps.SAMPLE_ATOM_COLUMNS`, open-shell plan P7) then `somo_population`, the restricted SOMO population rho_A = sum over the open-shell tokens t of min(n_t, 2 - n_t) w_tA (centre weights renormalized; 0 on a closed shell; the record's, not the drawing's). A model selects its columns by name (`model.inputs.token_columns`; `somo_population` with `ModelConfig.atom_spin_input = 'somo_population'`, at a fixed scale) |
| `atom_fragment`, `orbital_fragment` | int64, (A,) / (O,) | fragment index |
| `atom_map` | int64, (A,) | atom-map number of the drawn atom, 0 when unmapped or an implicit H |
| `atom_positions`, `orbital_centroids` | float32, (A, 3) / (O, 3) | Angstrom, record geometries; for distances only |
| `orbital_features` | float32, (O, F_o) | columns `orbital_columns` |
| `orbital_columns` | tuple[str] | `orbitals.tokens.STORED_ORBITAL_FEATURES` (v18): `feature_names()`, then `retained_fraction` (the spectral weight kept before renormalization) and `energy_ev` (E_r); since v19 (`steps.SAMPLE_ORBITAL_COLUMNS`) then `spin_role`: +1 on an open-shell token in its donor role, -1 on its hole, 0 elsewhere (selected with `ModelConfig.spin_role`) |
| `orbital_spectra` | float32, (O, 8, 2) | (E eV, p) spectral set |
| `orbital_occupied`, `donor_mask`, `acceptor_mask` | bool, (O,) | candidate masks (`CandidateConfig`); the donor-partner exclusion is in `pair_valid`. Since v19 the orbitals of an open-shell fragment end with one hole per open-shell token (`tokens.with_holes`, P7): a virtual twin (same coefficients, centres, descriptors, occupation, spectrum) that is an acceptor, while the open-shell token is a donor (`CandidateConfig.somo_roles`; metal-d open shells only with `metal_d_open_shell`). Holes exist in samples only, never in a token file |
| `incidence` | float32, (E, 3) | rows (orbital index, atom index, weight) for every centre of every orbital |
| `pair_z` | float32, (O_occ, O_virt, 14) | geometry-free pair features of every (occupied, virtual) pair, rows and columns in orbital order within each block, computed at K = 1 (`physics.params.REDUCED`; `physics.params.to_params` scales `cross_coupling` to another K). Columns, `pair_features` (`physics.stage_a.STAGE_A_FEATURES`) in order: `g` (the spectral gap factor G, eV^-1), `log_abs_g` (log \|G\|, floored at 1e-12), `inter` (1 between fragments), `bonded`, `bond_order`, `graph_distance` (between the two centre sets on the record's Lewis structure: bonded when some centre of r is bonded to some centre of s, the largest such order, the shortest graph distance, 0 when they share an atom, capped at 6; across fragments 0, 0, 6), `wbi` (the largest Wiberg index over the Lewis bonds joining the centre sets, 0 when not bonded), `centroid_distance` (Angstrom), `cross_coupling` (\|l_r^T B l_s\| over the AO blocks between the centres), `align_cos` (\|cos\| between the p vectors of r and s on their main centres), `dihedral_cos` (site-partner-partner-site dihedral when the two sit on adjacent bonds); then (v18, `physics.stage_a.STORED_PAIR_FEATURES`) the 0/1 exclusion flags of `physics.pairs.pair_flags`: `d_acceptor` (the virtual token is a d polarization function), `own_antibond` (the donor's own antibond), `site_self` (every site pair is a bond interacting with itself), defined on every cell |
| `pair_z` columns of v19 | float32 | `physics.stage_a.SAMPLE_PAIR_FEATURES`: the 14 v18 columns, then (open-shell plan P7) `g2`, `log_abs_g2` (`G2_FEATURES`: the two-level gap factor on every cell of nonzero weight, beta the cross-centre coupling at K = 1.75 within a fragment and `params.BETA_INTER` = 0.40 eV across; stored at those values, not scaled), `own_g`, `own_g2` (`OWN_PAIR_FEATURES`: g and g2 of a bonding token with its own antibond, defined on the `own_antibond` cells only), `somo_pair` (bit-packed), `somo_level_diff` (`OPEN_SHELL_PAIR_FEATURES`: 1 and E_r - E_s, defined on the open-shell rows x hole columns of two fragments only). Since v19 `g` is weighted by the occupation difference (2 on every closed-shell cell, bit for bit; 1 for a SOMO or a hole) and, with `log_abs_g`, undefined on an open-shell row with a hole column (weight 0). `pair_valid` is False on every open-shell row x hole column of one fragment |
| `pair_defined` | bool, (O_occ, O_virt, 14) | where the entry exists: `centroid_distance`, `cross_coupling`, `align_cos` and `dihedral_cos` only within a fragment, `align_cos` only with p character on both main centres, `dihedral_cos` only on adjacent bonds; undefined entries are stored as 0 |
| `pair_gap_sign` | int8, (O_occ, O_virt) | sign of G (-1 for the normal order E_r < E_s); bookkeeping, not a feature |
| `pair_valid` | bool, (O_occ, O_virt) | the legacy mask (`ModelConfig.pair_cells = 'valid'`), = donor x empty non-d acceptor x no exclusion flag: donor and acceptor masks, minus, within a fragment, the donor's own antibond and every pair whose site pairs (donor centre a, acceptor centre b) all have a and b identical or bonded with the acceptor's weight on a or the donor's on b at least 0.20 (`orbitals.tokens.site_excluded_matrix`, `CandidateConfig.site_weight`) |
| `labels` | list[ArrowLabel] | the orbital reading of the arrows (`donor_ids`, `acceptor_ids`, sites, flags, ...) restricted to `pair_valid`; empty unless `orbital_reading` is `aligned` |
| `labels_all` | list[ArrowLabel] | (v18) the same labels on every cell, the teacher pair chosen among all of them; empty when an arrow does not align. Read by `--cells all` |
| `product_key` | str | identity key v2 of the products (`data.resonance.identity_key`: the sigma key for a closed shell, plus `\|<sigma SMILES>,<charge>,<u>;...` per fragment for an open shell) |
| `edit_label` | EditLabel | the target: the parsimonious reaction-matrix class and its anchors on the global atoms (`arrows.delta_be`) |
| `bonds`, `lone_pairs`, `wiberg` | int16 (E, 3) / int8 (A,) / float16 (E,) | the reactant's Lewis structure as drawn (Kekulé, implicit H included): (i, j, order) with i < j, lone pairs per atom (N // 2), the Wiberg index of each bond (`arrows.lewis`) |
| `nonbonding`, `unpaired` | int8, (A,) / (A,) | (v19) the non-bonding electrons N of every atom (B_ii: odd on a radical centre; `lewis.lewis_nonbonding`) and its unpaired count u (`lewis.lewis_unpaired`); implicit H atoms 0. Collate refuses a sample without N for every atom |
| `drawings` | list[Drawing] | the pool's other resonance drawings, stacked on disk as `drawing_charges`, `drawing_orders`, `drawing_lone_pairs` and (v19) `drawing_nonbonding`, `drawing_unpaired` (int8, one row per drawing), their labels in the JSON; a radical's drawings come through the anion proxy (`resonance.radical_forms`, one per radical site) |
| `orbital_reading` | str | `aligned`, `LABEL_FILTERED`, the first unaligned reason (e.g. `HETEROLYSIS_ONLY`), or `NO_ARROWS` for a source without arrows (ReactMech) |
| `fermi_smeared` | bool | a reactant fragment is flagged `fermi_smeared` (a closed shell, or an odd Cu(II) / Fe(III) fragment RDKit counts 0: kept and sliced; a fragment drawn u > 0 that stays an open shell under its record's count skips the step as `SOMO_NOT_INTEGRAL`, law (S) of open-shell plan P3) |
| `label_filter` | dict[str, int] | labeled (donor, acceptor) cells that `pair_valid` rejects, by group |
| `inputs` | dict[str, str] | provenance: every token file the sample was built from (`file:<path>` -> sha256) |
| `pair_features` | tuple[str] | the column names of `pair_z` / `pair_defined` |

**`pair_feature_stats_<strategy>.json`** and **`pair_feature_stats_<strategy>_allcells.json`**
(same directory, one pair per split strategy, written by `ksmech-arrows samples`): a provenance
header (kind `pair_stats`, schema `SAMPLE_VERSION`; inputs the training view
`samples_<strategy>_train.json` and the store index; config the feature names, `cells` and the
physical constants) and `features`: `{feature: {mean, std, count}}`. `valid`: the v17 features
over the defined entries of the `pair_valid` cells; `all`: every stored feature and
`pair_valid` as a 0/1 feature over every cell (`physics.stage_a.cell_features`). Both over
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
(B, R, C, F; at K = eps = 1, unstandardized; the model selects its features by name), `pair_defined` (B, R, C, F), `pair_valid` and
`pair_gap_sign` (B, R, C), `pair_row_pos` (B, R) and `pair_col_pos` (B, C), the batch token
position of each row's occupied and each column's virtual token (-1 padding), for scattering
into the N x N pair track, and `pair_features` (the column names), with `atom_columns` and
`orbital_columns` for the token inputs. The reactant's Lewis structure and the reaction-matrix
targets are the `_reactant_tensors` and `_edit_tensors` keys.

**`anchor_floor_report_<strategy>.json`** (same directory, `ksmech-eval floor`): a provenance
header (kind `anchor_floor`; inputs the val and test views; config the split) and, per part,
`top1`, `top3`, `top5`, `median_rank`, `n` and `unreachable`: the physics floor, anchors
ranked by |G^atom| over the valid pairs (`eval.floor`); with `--cells all`,
`anchor_floor_report_<strategy>_allcells.json`, pooled over every pair.

**`cohort_<name>.json`** (same directory, `eval.cohorts.write_cohort`; written once, replaced
only by hand): a provenance header (kind `cohort`, schema `COHORT_SCHEMA` 1, no file inputs, the
content hash of the sorted ids, the name in its config and a free-form note) and `ids`, sorted
step ids. `ksmech-eval floor|decode --cohort <name>` scores only those steps. **`baseline_<name>.json`**
(`eval.pin.write_baseline`): kind `cohort_baseline`, schema 1, `columns` (the column names of
`eval.pin.column_digests`, once) and `digests`, step id -> one 16-hex-digit digest per column
("" where the sample lacks it); `eval.pin.load_baseline` checks it by its content hash. The
open-shell plan's cohorts: PMechDB `closed_base` (+ `_gbdt_valid`, `_gbdt_all`, its baseline;
P0b), `added`, `open_shell`, `carbene_singlet`, `open_shell_metal`; ReactMech (P5)
`p5_returns` (the 225 samples P5 added to the pilot, every sample outside `baseline_p5_pre`),
`p5_b_kekule` (68) and `p5_c_dative` (29) (the molecule val's samples P0a re-labelled
Kekule-closest or re-charged by the dative law, from `p0a_changes.json`), and
`baseline_p5_pre.json` (the column digests of the 9,031 pre-P5 samples).
`data/processed/reactmech/p5_rescore.json` (`scripts/p5_rescore.py`, unstamped) holds the
two ReactMech checkpoints' top-k per slice and the returns' parts per view.

## Training runs: `data/models/<source>/<run>/` (`ksmech-train fit`)

- `run.json`: a provenance header (kind `train_run`, schema `train.loop.TRAIN_VERSION`). Its
  inputs are the train and val views and `pair_feature_stats_<split>.json`; its config is
  `{train, model, sample_version}`. `extra.parts` holds, per part, `n`, `used` and `skipped`
  (`{TOO_LARGE: count}`). Beside the header: the device and the autocast dtype.
- `train.log`: timestamped `step n / total ...` lines (the format `ksmech-eta` reads) and one
  `epoch e of E done: ...` line per epoch.
- `metrics.jsonl`: one JSON object per epoch:
  - `train_loss` and `seconds`;
  - `val_loss`, `val_anchor_nll`, `val_top1/3/5` (anchor, set marginalization), `val_r_exact`
    (greedy R in the label class), `val_p_stop` (mean P(END_MECH)) and `val_n`;
  - `epoch`, `step` and `skipped`.
- `last.pt`, and `best.pt` (best `val_top1`, ties broken by `val_loss`): a torch dict with
  - `provenance` (the run header), `model_config`, `feature_stats`, `pair_stats`,
    `state_dict`;
  - `optimizer`, `scheduler`, `scaler`, `epoch`, `step`, `best`, `metrics`.

  Read it with `train.loop.load_model`, which refuses a run trained from a dirty tree.

## Token files: `data/cache/tokens/<sha256>.npz` (`ksmech-orbitals build`)

One file per molecule and conformer (`orbitals.tokens.save_tokens`, no pickle), with a
provenance header (kind `tokens`, schema `orbitals.tokens.SCHEMA_VERSION` 10, config
`token_config()`: the descriptor thresholds and the localization settings; input the QM
record). Members: the metadata (`smiles`, `method`, `conformer_id`, PM sweeps, `fermi_smeared`,
...), one `atom_<field>` array per atom field (`AtomToken`: model inputs and diagnostics), the
orbital diagnostics `orb_*`, one `desc_<field>` array per orbital descriptor (ragged ones padded
with their lengths), `C_loc`, `S`, `F`, `ao_atom`, `ao_l`. Schema 10 (2026-10-29) adds
`atom_f_plus_fmo_exact`, `atom_f_minus_fmo_exact` and `atom_fmo_weights_exact` (4 levels): the
frozen-orbital f+ / f- and frontier weights with blocks within 1e-4 eV
(`orbitals.frontier.EXACT_EV`) beside the 0.05 eV ones, whose content is unchanged; every other
member is schema 9's. A build that replaces a file compares it field by field
(`tokens.legacy_differences`); the build summary (`summary.json` beside the files, and per
source `data/processed/<source>/tokens_summary.json`) counts under `replaced` the files whose
stored fields changed, and under `frontier_blocks` the closed-shell molecules whose blocks at
0.01, 0.05 and 0.1 eV differ from the exact ones.

Open shells (open-shell plan P2b, 2026-10-31). The file of a molecule with u > 0 (RDKit's radical
count of its `smiles`, `tokens.drawn_unpaired`, the u its QM record was run at) carries one more
config key, `open_shell_rule` = `occ-plateau-v1`: its blocks come from the plateau reference
occupations (`localize.reference_occupations`) and its frontier columns from the radical
convention (`frontier.record_frontier_blocks`: HOMO = LUMO = the open group, frozen f+ = f-).
Closed-shell headers do not carry the key and their payloads are unchanged; the schema stays 10.
`load_tokens` adds the key to whatever config its caller passes when the file's own molecule is
an open shell, so a u > 0 file built before the rule is STALE_ARTIFACT under every reader, and
`save_tokens` stamps it. The build summary's `open_shell` block counts the u > 0 molecules (by u),
their open MOs (0 < reference < 2) and open plateaus, the molecules whose energy-window reference
(0.01 / 0.05 / 0.1 eV, `build.window_reference`) differs from the plateau one, the smeared ones,
and the distributions (median, p90, max) of max|n - aufbau| and max|n - reference|; the
`frontier_blocks` comparison covers closed shells only.

## QM cache: `data/cache/qm/<sha256>.npz`

Written by `ksmech.qm` (one file per molecule and conformer). Fields were listed from a real
cached file (acetone, 10 atoms, 22 AOs). `n` = atoms, `nao` = AOs, `nmo` = MOs (= `nao`). The file
is a plain `.npz` (no pickle). `ksmech.qm.records.load_record` returns a `QMRecord`.

The key is `sha256(f"{smiles}|{conformer_id}|{method}")`, where `smiles` is the map-free canonical
SMILES (`ksmech.data.chem.unmapped_canonical`) and `method` looks like
`GFN2-xTB/ALPB-water/tblite-lbfgs` (so the solvent model and optimizer are part of the key;
`--optimizer tblite-anc` writes `.../tblite-anc`, `xtb` writes `.../xtb-opt`). A
species whose spin `data.spin` reassigns (a singlet carbene or CO drawn with two radical
electrons, an odd Cu(II) or Fe(III) fragment RDKit counts 0; open-shell plan P4) has the tag
`+spin-v1-u<u>` right after the optimizer part, before any later suffix:
`GFN2-xTB/ALPB-water/tblite-lbfgs+spin-v1-u1`, serve's metal path
`...tblite-lbfgs+spin-v1-u1+metal-geom-v1` (`qm.spin.method_for`). Every other key is unchanged.
A tagged species is optimized from the embedding and from it kicked (0.02 Angstrom, seed 0), an
optimum with a bond the drawing lacks is dropped (`BOND_FORMED` when every start is), and the
lower energy is kept. The untagged records of those species (the old triplet `Cl[C]Cl`) stay in
the cache, unused by the engine.

| Field | dtype, shape | Notes |
|---|---|---|
| `smiles`, `atom_order_smiles` | str, () | the second one defines the atom order: `Chem.AddHs(Chem.MolFromSmiles(atom_order_smiles))` |
| `conformer_id`, `charge`, `unpaired` | int, () | charge = sum of formal charges, unpaired = sum of radical electrons, or the count of the method's spin tag |
| `method`, `schema_version` | str / int, () | |
| `atomic_numbers` | int32, (n,) | |
| `coords_ang` | float64, (n, 3) | optimized geometry, Angstrom |
| `energy_ha` | float64, () | total energy, Hartree (includes the ALPB term) |
| `orbital_energies_ev` | float64, (nmo,) | eV |
| `occupations` | float64, (nmo,) | restricted (one set of spatial orbitals, no spin polarization); a SOMO has occupation 1.0, a high-spin Fe(III) five of them |
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
`seconds`, `n_embedded`, `method`. Fragments that cannot be canonicalized get `conformer_id = -1`,
and so do the species the spin census refuses before QM (P8: `SPIN_CENSUS_REFUSED`, a
non-metal with u >= 2 that is not a Hund case or has no count, unless a counted exemption's).
`method` is per row: a tagged species' rows carry its tagged method, and a run's view keeps each
row whose method is its own species' method under the run's configuration
(`qm.batch.row_method`).

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
## QMugs (encoder pretraining): `data/raw/qmugs/`

Isert, Atz, Jiménez-Luna and Schneider, *Sci. Data* **9**, 273 (2022); CC BY-SA 4.0. Downloaded
on 2026-10-26 from ETH's libdrive share `X5vOBNSITAG5vzM` (the links OpenQDC's loader uses):
`summary.csv` (2,026,848,085 bytes) and `structures.tar.gz` (7,180,016,346 bytes), with their
sha256 in `SHA256SUMS`. The wavefunctions (over 7 TB) and the spectra were not downloaded.
Inspected from the real files:

- `structures.tar.gz` holds `structures/<CHEMBL id>/conf_00.sdf`, `conf_01.sdf` and
  `conf_02.sdf`, in ChEMBL-id order sorted as strings. Each is one V2000 molfile: the title
  line `QMugs: Quantum Mechanical Properties of Drug-like Molecules`, every H explicit, Kekulé
  bonds, the GFN2-xTB-optimized geometry in Angstrom.
- Its SD tags, read from `CHEMBL1/conf_00.sdf`. Vectors are `|`-separated, per atom or per bond
  in molfile order.
  - `CHEMBL_ID` and `CONF_ID`.
  - GFN2-xTB:
    - energies: `GFN2:TOTAL_ENERGY`, `ATOMIC_ENERGY`, `FORMATION_ENERGY`, `TOTAL_ENTHALPY`,
      `TOTAL_FREE_ENERGY`;
    - `DIPOLE` (x, y, z, total), `QUADRUPOLE` (6) and `ROT_CONSTANTS` (3);
    - `ENTHALPY`, `HEAT_CAPACITY` and `ENTROPY` (vib, rot, transl, total);
    - `HOMO_ENERGY`, `LUMO_ENERGY`, `HOMO_LUMO_GAP` and `FERMI_LEVEL` (Hartree);
    - per atom: `MULLIKEN_CHARGES`, `COVALENT_COORDINATION_NUMBER`,
      `DISPERSION_COEFFICIENT_ATOMIC` and `POLARIZABILITY_ATOMIC`;
    - `DISPERSION_COEFFICIENT_MOLECULAR` and `POLARIZABILITY_MOLECULAR`;
    - `WIBERG_BOND_ORDER` (per bond) and `TOTAL_WIBERG_BOND_ORDER` (per atom).
  - DFT (wB97X-D/def2-SVP):
    - energies: `DFT:TOTAL_ENERGY`, `ATOMIC_ENERGY`, `FORMATION_ENERGY`, `XC_ENERGY`,
      `NUCLEAR_REPULSION_ENERGY`, `ONE_ELECTRON_ENERGY` and `TWO_ELECTRON_ENERGY`;
    - per atom: `ESP_AT_NUCLEI`, `LOWDIN_CHARGES` and `MULLIKEN_CHARGES`;
    - `ROT_CONSTANTS` and `DIPOLE`;
    - `HOMO_ENERGY`, `LUMO_ENERGY` and `HOMO_LUMO_GAP`;
    - more below; the full list is in the files.
- `summary.csv`: one row per conformer.
  - Identity: `chembl_id`, `conf_id` (`conf_00` ...) and `smiles` (every H explicit, stereo).
  - Descriptors: `mw`, `atoms`, `heavy_atoms`, `heteroatoms`, `rotatable_bonds`,
    `stereocenters`, `rings`, `hbond_acceptors` and `hbond_donors`.
  - Flags: `significant_negative_wavenumbers` and `nonunique_smiles`.
  - The scalar GFN2 and DFT properties above, as `GFN2_*` and `DFT_*` columns.
  - Measured on the `conf_00` rows: 664,303 molecules. 20,350 have a charge in their SMILES.
    Median 30 heavy atoms, at most 100.

### Single points: `data/cache/qmugs/qm/` and `data/processed/qmugs/qm_manifest.parquet`

`ksmech-qm qmugs` (`qm.given`) runs GFN2-xTB at the QMugs geometry of each conformer, without
re-optimizing (method `GFN2-xTB/ALPB-water/given`). Each record has the fields of the QM cache
above, with the QMugs geometry as `coords_ang`. Its atoms are in the record order of the
map-free canonical SMILES, not the molfile's: `data.qmugs.record_mol` maps them with
`arrows.reactants.record_correspondence`.

The manifest has one row per (`chembl_id`, `conformer_id`, `method`):
- `chembl_id`, `conformer_id`, `smiles` (the canonical key) and `n_atoms`;
- `cache_path`, `status` (`ok` / `failed`), `reason` and `detail`;
- `warnings`, `seconds` and `method`.

Two QMugs-specific reason codes: `SDF_UNREADABLE` (RDKit cannot read the molfile) and
`RECORD_ORDER` (its atoms do not all map onto the record's). A geometry in which a bond of the
Lewis structure is longer than 1.3 times the covalent radii is `CONNECTIVITY_CHANGED`.
