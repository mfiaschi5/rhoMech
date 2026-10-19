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
| `pathway_id` | string, nullable | PMechDB: not in the release, rebuilt by chaining steps (`ksmech.data.pathways`, see `docs/decisions.md`); null for the 3,820 unlinked steps |
| `step_index` | int32, nullable | BFS depth in the pathway (branches can share one); null when `pathway_id` is null |
| `reactants_smiles` | string | canonical atom-mapped SMILES (see `docs/decisions.md`) |
| `products_smiles` | string | same canonicalization |
| `arrows` | list<struct<source_kind, source_atoms: list<int32>, sink_kind, sink_atoms: list<int32>>> | in written order |
| `mechanism_class` | string, nullable | null for PMechDB (not in the release) |
| `conditions` | string, nullable | null for PMechDB (not in the release) |

Rejected records go to `data/processed/<source>/rejections.csv`, with columns `step_id`,
`reason`, `detail` and `raw`.

`ingest_report.json` has a `pathways` block for PMechDB, the `PathwayReport` counts from
`ksmech.data.pathways`: links, reverse pairs, cuts, pathway sizes and coverage.
