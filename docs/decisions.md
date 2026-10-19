# Design decisions

Running log. One entry per non-obvious choice, newest last, each with a date.

## 2026-10-19 — Project scaffold

- **uv + uv_build, src layout, Python pinned to 3.11** (`requires-python = ">=3.11,<3.12"`).
  Keeps the torch/rdkit/tblite wheel set predictable.
- **tblite installs from PyPI** (0.7.0 wheel) on this machine, so the conda route is only a
  documented fallback. A `xtb`-binary shell fallback is planned for `ksmech.qm` (not yet written).
- **pre-commit uses `language: system` local hooks** that call `uv run`, so hook tool versions
  always match the locked environment.

## 2026-10-19 — Data layer (`ksmech.data`)

### Raw PMechDB location and scope

- The download lives in `data/raw/pmechrp/`: the PMechRP data release, not a raw PMechDB
  export. `pmechdb.resolve_inputs` tries `data/raw/pmechdb/` first, then `data/raw/pmechrp/`.
  `--raw` can point at another directory or a single file.
- We ingest only the 13,104 manually curated (`mc_only`) records, read once from
  `fold0/raw/{train,val,test}.txt`. Every fold, and `scaffold_split/`, is a permutation of that
  same set. The `mixed` splits add 10,000 combinatorially generated proton transfers, which are
  not PMechDB curation and are left out.
- `step_id = "pmechdb-" + sha1(raw record text)[:16]`. The id stays the same whichever fold file
  a record is read from, so steps can be traced back to the raw line.

### Canonicalizing atom-mapped SMILES (`ksmech.data.chem`)

- Parse with `removeHs=False`. Mapped explicit H atoms (`[H:20]`) are arrow endpoints, and
  RDKit's default `RemoveHs` would delete them, since `removeMapped` defaults to true. Unmapped
  explicit `[H]` are kept too, so the graph is exactly what the curator wrote.
- Sanitize with `SANITIZE_ALL ^ SANITIZE_SETAROMATICITY`, so **no aromaticity perception**.
  PMechDB is written in Kekulé form, and arrows name specific single and double bonds
  (`10,11=10,20` starts from the double bond 10=11). Resonance steps only move pi bonds.
  Aromatizing would erase that, and re-kekulizing could put the double bond elsewhere. Aromatic
  input, which other sources may contain, is kekulized with RDKit's own choice and its flags are
  cleared. For such input, an arrow on a ring bond is only as reliable as that choice. PMechDB has
  none (verified: 0 of 26,182 canonical strings carry aromatic atoms).
- Canonical string: `Chem.MolToSmiles(mol, canonical=True, kekuleSmiles=True)` on the whole
  multi-fragment side. RDKit includes atom-map numbers in its canonical ranking. Two Kekulé forms
  of the same molecule therefore give different strings, and that is intended. Verified
  idempotent on all 13,091 accepted steps.
- Molecule *identity* (stats, scaffolds) uses a separate key: maps removed, H suppressed, standard
  aromatic canonical SMILES (`chem.unmapped_canonical`).

### Lone-pair source vs bond source (`ksmech.data.schema.Arrow`)

- `Arrow` stores the arrow **as drawn**. The tail is `lone_pair` (1 atom), `bond` (2 atoms) or
  `radical` (1 atom; defined, never seen in PMechDB). The head is `atom` (1) or `bond` (2). Bond
  tuples are stored sorted, because a bond is an unordered pair and direction comes from the
  sink. Arrow order within a step is preserved.
- Where the pair ends up is derived by `Arrow.destination()` rather than stored:
  - lone pair on `a` → atom `b`: bond `a–b` forms or gains order (PMechDB `10=20`).
  - bond `a–b` → atom `b`: lone pair on `b` (heterolysis).
  - bond `a–b` → bond `b–c`: pair shifts to `b–c`. Exactly one shared atom is required.
  - lone pair on `a` → bond `a–b`: bond `a–b`. Never written in PMechDB, but mech-USPTO uses it
    (`(9, [9, 1])`).
  - Any other shape is a `ValueError` and becomes `ARROW_INVALID`.
- Why not store the destination instead? The raw notation and the curly-arrow drawing both name
  the attacked atom. Keeping that is lossless, and the mapping to destinations is 1:1 for every
  shape seen in the data.

### Ambiguities found in the raw format

- **Stray trailing comma**: 1,155 records end their arrow code with `,` (e.g. `10=20,`). It is
  treated as an export artifact and dropped. It is counted as the `stray_trailing_comma` note,
  not as a rejection. A comma anywhere else is still an `ARROW_PARSE` error.
- **No mechanism class, pathway id, step order or conditions** exist anywhere in the release,
  despite the brief. The fields are `None`, and `step_index` is `None` when `pathway_id` is
  `None`. Consequences:
  - "Random by pathway" degenerates to a per-step random split. Consecutive steps of one
    reaction may land in different sets. (Superseded the same day: pathways are rebuilt by
    chaining. See "Rebuilding PMechDB pathways" below.)
  - Leave-one-class-out cannot run on PMechDB, and raises a clear error instead.
- **Bare metals**: 55 records write a metal as a bare bracket atom (`[K]`, `[CuH2]`). RDKit
  gives it radical electrons. These are left untouched: they are spectators that no arrow
  touches.
- **Hypervalent ate complexes** (12 records: `Br[Br-]Br`-type, 5-valent Sn, neutral 4-valent B)
  fail RDKit valence checks. They are rejected as `SANITIZE_FAILED`, not sanitized leniently,
  because downstream 3D embedding and xTB need valid valences.
- **One broken mapping**: map 11 vanishes from the products. Rejected as `MAP_NOT_BIJECTIVE`.
- **Partial mapping**: maps usually cover only the atoms the arrows touch (10.. nucleophile side,
  20.. electrophile side). 18 records are fully mapped. "Bijection" therefore means: unique on
  each side, and the same map set on both sides.

### Validation order

Parse → heavy-atom balance (`UNBALANCED`) → H balance → charge balance → map uniqueness and
bijection → arrow atoms mapped → bond-source atoms bonded in the reactants (`BOND_NOT_FOUND`) →
`apply_arrows_stub` (no-op until `ksmech.arrows` lands).

Balance runs first, so a step broken in several ways reports the most basic fault. H and charge
balance were not requested. They cost nothing, and 0 real records fail them.

### Splits (`ksmech.data.splits`)

- Every strategy packs whole groups greedily, after a seeded shuffle of the sorted group keys.
  Groups larger than half the val or test target go first, so they land in train, as in
  chemprop's balanced scaffold split. A target is never overshot, and is missed by less than one
  group.
- Scaffold split: the scaffold of the largest reactant fragment, by heavy atoms with ties broken
  by unmapped SMILES. The acyclic scaffold `''` is one group (4,912 PMechDB steps, 37.5%) and
  ends up in train. A step with a known pathway takes the scaffold of its pathway's first step, so
  pathways never straddle sets.
- Leave-one-class-out: test is the held-out class, and train/val are split by pathway. Excluded
  (and reported in `Split.excluded`): unlabelled steps, and other-class steps that share a
  pathway with a held-out step.

### Sources without inspected raw files

- **ReactMech** (DeepMech, Chem. Sci. 2026): the published files hold mapped steps with **no
  arrows** and **no class labels**. They are ingested with `arrows=[]` via `require_arrows=False`,
  and arrow alignment fills them in later. The 67 classes exist only in the paper's SI, and the
  OOD set may not be released.
- **mech-USPTO-31K**: one row per reaction, holding one flat arrow list with no step boundaries
  and implicit-H references (`5.1`). `read_mechanisms` parses it (H refs become `HydrogenRef`).
  `load` raises `NotImplementedError` until arrow replay exists. A partial conversion would have
  produced fake "elementary" steps.
- Both modules carry `untested_on_real_data=True` and must be checked against the real file (and
  `docs/data_schema.md` filled in) before use.

## 2026-10-19 — Proceeding without PMechDB pathway grouping

- PMechDB steps keep `pathway_id = None` and `step_index = None`. Both stay nullable in
  `STEP_SCHEMA`. Rebuilding pathways by chaining steps is deferred to the
  `claude/adoring-edison-6a4f37` branch.
- **Pathway-level metrics skip missing ids instead of reporting zero.**
  - When no step has an id, `stats.pathway_stats` returns `n_pathways = None` and
    `steps_per_pathway = None`, and counts the id-less steps in `n_steps_without_pathway`.
  - `ksmech-data stats` then prints `n/a` and leaves out the steps-per-pathway table.
  - When only some steps have ids, the metrics cover those steps only.
- **Splits already fall back.** `split_by_pathway` makes every id-less step its own group. A
  test now checks that an all-`None` set gives an exact per-step 48/6/6 split. The scaffold split
  anchors an id-less step on itself.
- **Cost, if `split_by_pathway` is used.** Without ids it is a per-step random split, so
  consecutive steps of one reaction can land in different sets. Measured with the chaining links
  from the grouping branch: 3,505 of 10,522 links cross sets at seed 0. The PMechDB default is
  therefore `split_by_molecule` (next entry), which leaks none of them, with or without pathway
  ids.
- **Why.** Merging the grouping later only fills the two columns, and no metric or split code has
  to change for it.

## 2026-10-19 — `split_by_molecule` is the PMechDB default split

- **Rule.** Steps that share any non-trivial molecule form one group, and so do steps that share a
  pathway id (when there are ids). Groups are packed like every other strategy, so no molecule
  appears in more than one of train/val/test. `splits.default_split("pmechdb")` returns this
  split. Other sources still default to `split_by_pathway`.
  - A molecule is a map-free canonical SMILES with stereo kept (`chem.nontrivial_species`).
  - It must have more than 3 heavy atoms. Water, H3O+, OH-, halides, metal ions, MeOH and NH3
    are ignored.
  - Molecules are taken from **both reactants and products**.
- **Why it is the default.** It prevents the cross-set leak whether or not pathways are known.
  PMechDB has no pathway ids (entry above).
- **Why products count.** The brief asked for reactant molecules only, which was measured first.
  It leaves 3,383 of 10,522 chain links (32%) crossing sets, about the same as a random split.
  Step X → Y and step Y → Z share no reactant, so it cannot keep them together. With products
  included, 0 links cross.
- **Cost.** Common reagents put 10,255 steps (78.3%) into one group. Above 200 steps each, the
  most shared molecules are HSO4-, H2SO4, AcO-, AcOH, TsO-, Et3NH+, TFA- and pyridine. That
  group lands in train. Val and test are drawn from the 21.7% of steps that share no molecule
  with it: an out-of-distribution test, harder than a random one. Sizes at seed 0 with 80/10/10
  are 10,472 / 1,309 / 1,310.
- **Leakage at seed 0, 80/10/10.** The chain links come from the grouping branch's analysis.

  | Split | chain links crossing sets | molecules in >1 set |
  |---|---|---|
  | `split_by_pathway`, no ids (per step) | 3,505 / 10,522 | 3,458 / 14,072 |
  | `split_by_molecule` | 0 | 0 |

- **Changed packing.** `_assign_groups` now puts a group larger than the whole train target into
  train, with a warning, instead of letting it fall through to test. The 78% group would have
  fallen through with fractions such as (0.7, 0.15, 0.15).

## 2026-10-19 — Rebuilding PMechDB pathways

This section ends "Proceeding without PMechDB pathway grouping" above. It fills `pathway_id` and
`step_index` and changes no metric or split code. The pathway metrics start reporting numbers, and
`split_by_molecule`, the PMechDB default, also joins steps that share a pathway id.

### No grouping column exists

Every file variant in `data/raw/pmechrp/` was checked:

- `raw/` and `scaffold_split/` hold the reaction and the arrow code.
- `reformatted/` adds the source and sink molecules.
- The Chemformer, Graph2SMILES, Molecular Transformer and T5Chem copies hold tokenized source and
  target only.

None of them has a reaction id, reference, DOI, textbook page or upload batch. Line order carries
no grouping either, because the files are shuffled: only 45 of 10,522 chain links (below) join
adjacent lines.

### Chaining rule (`ksmech.data.pathways`, called from `pmechdb.load`)

- **Link** i → j when the *non-trivial* products of i equal the non-trivial reactants of j.
  - Both sides are compared as multisets of map-free canonical SMILES, with stereo kept.
  - A species is trivial when it has at most 3 heavy atoms (`chem.nontrivial_species`): water,
    H3O+, OH-, halides, metal ions, MeOH, NH3 and the like. Curators add or drop these between
    consecutive steps, so they would break true links.
  - Empty keys never link. 378 steps have only trivial reactants and 342 only trivial products.
    Linking empty keys would join all of them into one pathway.
- **Pathway**: a weakly connected component of these links with at least 2 steps. Unlinked steps
  keep `pathway_id = None`.
- **Size cap of 15 steps.** An oversize component is cut at its shared intermediates.
  - A shared intermediate is a product key that more than one step makes or more than one step
    consumes.
  - Each cut removes every link through one such key: the one with the fewest heavy atoms, ties
    going to the most producer × consumer pairs. Cuts repeat until every piece is within the cap.
  - The cut is needed because a common substrate (made at the end of one reaction, used at the
    start of others) glues unrelated reactions together. The raw components reach 219 steps.
  - A piece with no shared intermediate is a plain chain, with in- and out-degree at most 1. It
    is **kept whole, however long** (user decision, 2026-10-19), because the cap exists to catch
    hub merges and these have none. Four such chains exist: 16, 18, 20 and 36 steps. The 36-step
    one is three Paal–Knorr pyrrole formations on a single tris(1,4-diketone).
- **`pathway_id`** is `<source>-path-` + sha1(id of the first step)[:16]. The first step has
  the smallest `(step_index, step_id)`.
- **`step_index`** is the BFS depth from the pathway's start steps. Start steps are found in this
  order:
  1. the steps that no other step of the pathway leads to;
  2. if every step has a predecessor (the release curates reverse steps, so 2-cycles are
     common), the steps that start a forward chain once reverse pairs are ignored;
  3. if there are none of those either, the smallest step_id.

  Branches share an index: 205 pathways have more than one step at index 0.
- The gate set before filling ids was "median component size ≥ 2 and no component over 15". The
  median is taken over pathways only, which gives 3. Counting the 3,820 unlinked steps as size-1
  components would give 1. The only steps over 15 are the four plain chains above.

Measured on the 13,091 accepted steps:

| Quantity | Value |
|---|---|
| links / reverse pairs | 10,522 / 829 |
| steps with >1 successor / >1 predecessor (reverse links ignored, before cuts) | 847 / 802 |
| raw components with ≥2 steps / largest | 2,168 / 219 |
| components that needed cutting / intermediates cut | 25 / 78 |
| **pathways / steps in them** | **2,379 / 9,271 (70.8%)** |
| median / max pathway size | 3 / 36 (15 without the plain chains) |

Size distribution: 2:1039, 3:388, 4:290, 5:143, 6:192, 7:124, 8:51, 9:31, 10:35, 11:15, 12:41,
13:17, 14:6, 15:3, 16:1, 18:1, 20:1, 36:1.

Alternatives that were measured:

- **Exact multiset of all species, spectators included.** 6,459 links, 2,589 pathways holding
  60.6% of steps, largest 23, no cutting needed. It is cleaner, but it misses steps where the
  curator added or dropped a small species. The rule above covers 10 points more.
- **Next step's arrow-touched fragments contained in the previous step's products.** This
  produces a 1,484-step component held together by common small molecules. Rejected.

Ingest now takes about 30 s instead of 14 s, because both sides of every step are re-parsed to
build the keys.

### Effect on splits (seed 0, 80/10/10)

| Split | chain links crossing sets | molecules in >1 set |
|---|---|---|
| `split_by_pathway`, no ids (before) | 3,505 / 10,522 | 3,458 / 14,072 |
| `split_by_pathway` with rebuilt ids | 689 (the cut links) | 1,241 |
| `split_by_molecule` (default, unchanged) | 0 | 0 |
