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
    reaction may land in different sets.
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
the arrows, applied to the reactants, give the products (`ARROWS_MISMATCH`; a no-op stub until
the arrow layer landed, see "Arrow replay in ingest validation", 2026-10-20).

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

## 2026-10-19 — QM layer (`ksmech.qm`)

### Optimizer

- **scipy L-BFGS-B on tblite energies and gradients** (Cartesian, Bohr) is the default, because
  tblite has no optimizer. Converged when max |gradient| <= 3e-4 Hartree/Bohr (`QMConfig.fmax`),
  at most 500 iterations; otherwise `OPT_NOT_CONVERGED`. Start geometry: lowest-energy MMFF (UFF
  if MMFF lacks parameters) conformer from ETKDGv3. Typical cost: 10-40 atoms in 1-4 s (gas
  phase benchmark); a 103-atom steroid takes ~300 s with ALPB on one core.
- `QMConfig.optimizer="xtb"` shells out to `xtb --opt --gfn 2 [--alpb water]`, then runs the same
  tblite single point. **The `xtb` binary is not installed on this machine**, so that backend is
  untested here: its test is `skipif(no xtb)`, and without the binary it returns `XTB_NOT_FOUND`.
  The restraint fallback below exists only for the tblite backend.
- Per-molecule timeout: 900 s (`SIGALRM` in the worker, so it cannot interrupt a long C/Fortran
  call mid-way). The first pilot used 300 s and timed out three 90-100-heavy-atom steroids.
- The sterics step uses morfeus with `density=0.004` instead of the default 0.001: 4.5x faster
  for a 33-atom molecule and %Vbur moves by < 0.15 percentage points (benchmarked on one
  molecule).

### Solvation (user decision, 2026-10-19)

- **ALPB water is the default for every optimization and single point**, including the Fukui
  N+-1 calculations (`QMConfig.solvent="water"`, `None` = gas phase). Reason: PMechDB species are
  mostly charged intermediates. On a 300-molecule pilot, gas-phase relaxation broke a bond in
  19.7% of them (protonated tetrahedral intermediates lose water, C(O-)Cl loses Cl-) and ALPB
  water cut that to 12.3%.
- The solvent model is part of the cache key (`method` string `GFN2-xTB/ALPB-water/tblite-lbfgs`),
  and the manifest has a `method` column, so gas-phase and solvated results never mix. Orbital
  energies, charges and Fukui values are therefore solvated values.

### Connectivity test and restraint fallback (user decision, 2026-10-19)

- `CONNECTIVITY_CHANGED` is raised only when an **input bond stretched beyond 1.3x the sum of
  covalent radii** (RDKit radii) in the final geometry. New close contacts are ignored. Bonds
  to Li, Na, Mg and K are skipped (their contacts are electrostatic). The first version compared
  the full RDKit-perceived bond graph and flagged 14 of 58 failures that only had a new
  near-covalent contact (carbocation bridges, Li-C/O).
- Pilot results on the same 300 molecules, in the order the changes were made: gas phase strict
  graph 78.7% -> ALPB water 86.0% -> stretch test 89.0% -> restraint fallback 98.7%.
- **Fallback**: if the free optimization breaks a bond, optimize again from the force-field
  geometry with a flat-bottom harmonic wall on every tested input bond:
  `E = k * max(0, d - 1.2 * (r_i + r_j))^2`, k = 1.0 Hartree/Bohr^2. The final single point is
  unrestrained. The record gets warning `RESTRAINED_GEOM`. It stays a failure if a bond is still
  stretched beyond 1.3x. Such geometries are **not minima**: they are the species "as drawn" in
  the mechanism. Treat `RESTRAINED_GEOM` records as lower-trust for energies and charges.
- An earlier implementation edit dropped the line that attaches `RESTRAINED_GEOM`. It was caught
  because the pilot's warning table had no such entry. The affected cache records were deleted
  and recomputed.

### Fukui convention and sign

- Charges q are Mulliken (positive = electron-poor), from the same method as the geometry, at
  the **same geometry** (no re-optimization):
  `f+ = q(N) - q(N+1)`, `f- = q(N-1) - q(N)`, `f0 = (f+ + f-)/2`, `dual = f+ - f-`
  (dual > 0: electrophilic site). Each of f+, f-, f0 sums to 1.
- N+1 uses charge-1, N-1 uses charge+1. Unpaired electrons for N+-1: 1 if N is closed shell, else
  `unpaired - 1`.
- **Both finite-difference and frozen-orbital Fukui are stored.** The finite-difference f+ of
  acetaldehyde peaks on oxygen (0.34), not on the carbonyl carbon (0.17), because the relaxed
  Mulliken response moves charge to the electronegative atom. The frozen-orbital value (Mulliken
  population of the LUMO/HOMO) gives C 0.60 and N 0.83 (methylamine f-). The validation asserts
  use the frozen-orbital values; a slow test records that the finite-difference acetaldehyde f+
  still peaks on oxygen.
- A failed N+-1 SCF gives NaN finite-difference arrays and warning `FUKUI_SCF_FAIL`; it does not
  fail the molecule. Zero-electron species (`[H+]`, `[Na+]`) get a record with NaN frontier
  data and warning `NO_FRONTIER`.
- Global descriptors (eV, Koopmans-style): HOMO = highest MO with occupation >= 0.5,
  `mu = (E_HOMO + E_LUMO)/2`, `eta = (E_LUMO - E_HOMO)/2` (half gap), `omega = mu^2/(2 eta)`.

### Charge and multiplicity

- Total charge = sum of RDKit formal charges. Unpaired electrons = sum of RDKit radical electrons.
  If `n_electrons - unpaired` is odd (or negative) the molecule fails with `SPIN_PARITY` and is
  not guessed at.
- tblite runs open-shell systems **restricted with fractional occupations**, with no spin
  polarization (a radical's SOMO has occupation 1.0). Bare metals that RDKit gives radical
  electrons follow the same rule. Spin states of transition metals are not scanned.

### Matrices, AO map and atom order

- tblite's `hamiltonian-matrix` is the core Hamiltonian, not the converged Fock. Both are stored;
  `fock = S C diag(eps) C^T S` (valid because `C^T S C = 1`, measured to 1e-15).
- tblite exposes no AO-to-atom map. The AO count per element is probed from single-atom
  calculations (1, 4 or 9 AOs), and the layout is s, p x3, d x5 per atom. The check "AO gross
  population + Mulliken charge is a whole number for every atom" validates the **atom
  assignment**. It cannot detect a permutation *within* a shell, which only matters to the
  orbital-localization phase. Re-verify there.
- Overlap diagonal is 1 only to ~2e-10 (chlorine), so the unit-diagonal test tolerance is 1e-8.
- Record atom order = `Chem.AddHs(Chem.MolFromSmiles(canonical_smiles))` (`prepare_mol`). The
  batch deduplicates on the map-free canonical SMILES, so a later step recovers the
  correspondence to its atom-mapped molecule by substructure match. `GFN2Calculator.run`
  rejects a mol whose atom order differs from `prepare_mol`.
- `embed` mutates its input in place (RDKit idiom), requires explicit hydrogens, and renumbers
  kept conformers 0..k-1 by MMFF energy, so conformer 0 is the lowest.

### Batch

- The PMechDB run uses **1 conformer** per molecule (user decision); `embed` defaults to 3.
- Workers are `spawn`ed with `OMP_NUM_THREADS=1`. A hard crash of the pool re-runs the in-flight
  tasks one at a time to pin it on a molecule (`WORKER_CRASH`).
- **Operational note (host instability, 2026-10-19):** the machine (Ryzen 7 2700X, 16 threads,
  23 GB RAM, `/tmp` is a 12 GB tmpfs) hard-rebooted four times under the full PMechDB run: three
  times within 1-2 minutes of starting 14 workers, and once after 34 minutes with 8 workers inside
  a 10 GB memory cgroup. Each time the journal just stops: no OOM-killer entry, no thermal or
  machine-check message, memory flat at ~5.5 GB used, CPU 48-58 C when read afterwards. That
  points to a power or hardware stability problem under sustained all-core floating-point load
  (the board reports a 1.42 V core voltage), not to memory. Not proven.
  Consequences handled in code: cache files are now `fsync`ed before the atomic rename, and a
  zero-length `.npz` (50+ were left behind by the crashes) is not a cache hit. All empty files
  were deleted. Runs are resumable, so a crash loses only the molecules in flight.

## 2026-10-19 — Orbital tokens (`ksmech.orbitals`)

### Pipek-Mezey, not Boys

- **Pipek-Mezey with Mulliken populations on the GFN2 AO basis**, in plain numpy. The
  functional `P = sum_i sum_A (Q_A^i)^2` is maximized by 2x2 Jacobi rotations; a sweep is
  converged when `P` changes by less than `1e-8` (absolute), at most 200 sweeps
  (`PMConfig`). Occupied (`occupation >= 0.5`, the same rule as the Fukui HOMO) and virtual
  orbitals are localized as separate blocks. A non-converged block is kept as is and flagged
  (`MoleculeTokens.pm_converged`, counted in the build summary).
- **No Foster-Boys fallback.** tblite's Python interface (`save-integrals`) exposes only the
  overlap and Hamiltonian matrices, no dipole integrals, so Boys cannot be built without a new
  dependency. Documented, not implemented.
- **One sweep = n-1 round-robin rounds of n/2 disjoint pairs**, each round vectorized. The
  angle of a pair depends only on `Q_ii`, `Q_jj` and `Q_ij`, which a rotation of a disjoint pair
  does not change, so a round is exactly a sequence of ordinary Jacobi rotations and `P` never
  decreases. The first version looped over pairs in Python and maintained the full
  `(n_atoms, n, n)` population tensor: 47 s for a 96-atom PMechDB molecule (134 occupied
  orbitals, 33 + 57 sweeps). The round version takes 8 s for that molecule and 0.07 s for a
  typical 31-atom one. Large molecules need many sweeps because the tolerance is absolute while
  `P` grows with the orbital count (`P ~ 76` for 134 orbitals).
- **Valence-only assertion**: every atom must carry 1, 4 or 9 AOs and the electron count must
  equal the sum of GFN2 valence electrons minus the charge (`valence_electrons`: group number;
  Zn, Cd, Hg = 2; lanthanides = 3; checked against tblite single-atom occupations for Z 1-86).
- **AO order within a p shell is (py, pz, px)** (real spherical harmonics m = -1, 0, +1), the
  "permutation within a shell" the QM layer could not check. Verified on HF placed along each
  Cartesian axis (the sigma MO's p coefficient sits on that axis); `P_AXIS_ORDER` and a test
  pin it.
- **Random starts do not change the result**: on all ten fixtures, three Haar-random orthogonal
  starting rotations of each block reach the same `P` (to 1e-4) as the canonical start, so no
  restarts are run. Benzene's three 3-centre pi orbitals (below) are the PM maximum in this
  basis, not a symmetry trap.

### Classification thresholds (`ClassifyConfig`)

| Threshold | Value | Why |
|---|---|---|
| `one_atom_frac` | 0.85 | Lone pairs on the fixtures carry 0.96-1.01 on one atom; polarized bond orbitals top out at 0.71 (acetone C=O pi: O 0.71 / C 0.29). 0.85 sits in the empty band between them. |
| `two_atom_frac` | 0.85 | Two-centre bonds on the fixtures carry 0.97-1.02 on two atoms; benzene's PM pi orbitals carry 0.74 (0.50 + 0.24). |
| `sigma_cos` | 0.7 | Sigma p vectors on the fixtures have \|cos\| > 0.95 to the bond axis, pi ones < 0.1 (planar C=O, C=C) and < 0.3 for the slightly bent acetonitrile pi set. 0.7 (45 degrees) splits them with margin. |
| `min_p_norm` | 0.05 | An atom whose p coefficients are below this has no direction to test (H, or an s-only contribution). |
| `max_s_share_pi` | 0.25 | A 3+ centre orbital is pi-type only if nearly all its population is on p AOs; benzene pi: s share 0.00. |
| `member_min_pop` | 0.05 | Atoms listed as parents of a delocalized orbital (benzene pi: 0.50 / 0.24 / 0.21 / 0.05 carries four). |

- **Bonding sign check**: the Mulliken overlap population between the two parents must be
  positive for an occupied two-centre orbital and negative for a virtual one. That is the "sign
  change across the bond" test, done on the interatomic overlap rather than on raw coefficient
  phases. A two-centre orbital failing it, or with one sigma-like and one pi-like atom, is
  `unknown`.
- **Types beyond the brief's enum**: `empty_s` (one-atom virtual dominated by s, e.g. the 1s of
  `[H+]`, the acceptor of every proton transfer), `empty_d` (one-atom virtual dominated by d:
  the five GFN2 polarization functions on Si, P, S, Cl, Br, I, which are basis artefacts, not
  acceptors), `delocalized_pi_star` (virtual counterpart of `delocalized_pi`), and `unknown`
  (the classifier failed). Without `empty_s`/`empty_d`, proton acceptors and Cl d functions
  would both have been `empty_p`. `delocalized_*` and `unknown` are what the summary reports as
  "unresolved".
- **Benzene gives 3 `delocalized_pi` + 3 `delocalized_pi_star`**, not Kekulé pi bonds: each PM
  pi orbital is 0.50 on one carbon, 0.24 and 0.21 on its neighbours, 0.05 para. Every other
  fixture is fully resolved (no `unknown`, no delocalized): acrolein gives separate C=C and C=O
  pi / pi* pairs.
- Orbital parents are listed by descending population (`parent_atoms[0]` is the dominant atom,
  e.g. Cl in the C-Cl sigma: 0.63 / 0.37). Centroids are population-weighted positions with
  negative Mulliken populations clipped to 0.

### Tokens

- Atom and orbital tokens share one id sequence: atoms `0..n-1` (id = record atom index), then
  occupied orbitals by ascending energy, then virtual orbitals by ascending energy. Orbital
  energies are `c^T F c` in the localized basis (the diagonal of `U^T F U`); the occupied sum
  equals the canonical occupied sum (trace invariance, tested).
- **Parents are record atom indices, not map numbers.** A token file belongs to a map-free
  molecule that appears in many steps with different maps. `AtomToken.map_number` is 0 and is
  filled per step by the arrow-alignment phase (`MoleculeTokens.assign_map_numbers`), after the
  substructure match the QM layer already requires.
- Atom features copied from the QM record: Mulliken charge, finite-difference f+, f-, dual (NaN
  when `FUKUI_SCF_FAIL`), frozen-orbital f+ and f-, %Vbur, position; from RDKit (aromaticity on,
  `prepare_mol` order): formal charge, hybridization, aromatic flag.
- **Functional groups: RDKit Contrib IFG (Ertl)** on the heavy-atom mol, label = the group's own
  canonical SMILES (`C=O`, `O=CO`, `[O-]`, `n`), not the variant with neighbouring carbons
  (4,777 distinct vs 2,717). Vocabulary `src/ksmech/orbitals/fg_vocab.json`: labels seen at
  least 5 times on the 14,383 unique PMechDB molecules, 413 labels covering 90.4% of group
  occurrences (min_count 10 would give 233 / 86.9%; 2 gives 1,000 / 94.9%); index 0 = no group,
  1 = UNK. Regenerate with `ksmech-orbitals fg-vocab`. H atoms are never in a group.
- Token files: one `.npz` (no pickle) per (molecule, conformer) under `data/cache/tokens/`,
  same key as the QM cache, columnar: `atom_*` arrays, `orb_*` arrays (parents padded with
  -1, weights with NaN), `C_loc` (nao x n_orbitals), `S`, `F` (Hartree), `ao_atom`, `ao_l`,
  PM stats and the QM warnings. `ksmech-orbitals build` loads an existing file instead of
  recomputing unless `--force`, skips and counts molecules without a QM record (current
  method only, so gas-phase pilot records are ignored), and exits 1 when more than
  `--max-unresolved` (10%) of occupied orbitals are delocalized or unknown, printing the worst
  20 molecules.

### PMechDB statistics (991 molecules, the records cached while the QM batch was running)

Rebuilt 2026-10-20 with the follow-up rules below, from `data/cache/tokens/summary.json`;
13,392 molecules had no current-method record yet and were skipped and counted. Re-run
`ksmech-orbitals build` after the batch and refresh these numbers.

- **Occupied orbitals (41,144): 3.93% unresolved** = 3.92% `delocalized_pi` + 0.01% `unknown`
  (5 orbitals), under the 10% acceptance limit. Types: sigma 78.0%, lone pair 13.1% (135 of
  them metal-bound), pi 4.9%. Virtual (38,518): sigma* 83.4%, `empty_d` 7.0% (the heavy-atom d
  functions), `delocalized_pi_star` 5.5%, pi* 3.5%, `empty_p` 0.5%, `empty_s` 0.1%, unknown
  0.04% (15, metal-centred acceptors spread over Mg, Al or Ti and their ligands).
- **Every `delocalized_pi` is an aromatic ring**: the 20 worst molecules (15-20% of their
  occupied orbitals) are benzene, carbazole, phenyl-phosphonium and -phosphorane salts,
  stilbenes, pyridine, bromobenzene: three 3-centre pi orbitals per six-membered ring, as for
  the benzene fixture. Thresholds were not lowered: these orbitals *are* delocalized and the
  atom set is stored on the token.
- **The 5 `unknown` occupied orbitals**: three azide terminal-N lone pairs at 0.83 + 0.08 on the
  next N (they fall into the two-centre test and fail the sign check), one 3-centre-2-electron
  C-H-C bridge in a carbocation (0.35 / 0.33 / 0.32), and one C-Li orbital spread as C 0.70 /
  Li 0.15 / Si 0.06 / C 0.06, just under the two-centre threshold. Before the follow-up rules
  (first build, 878 molecules) there were 17: the rest were C-Mg / C-Li / Br-Mg bonds (now
  metal-bound lone pairs), two C-C sigma bonds at the cosine threshold (now ranked) and three
  borderline lone pairs on Cl+/O-I that the growing cache no longer shows in this set.
- **Tokens per molecule**: min 11, median 98, max 391 (one token per atom plus one per AO).
- **Functional groups**: 7.7% of heavy atoms carry the UNK label, 62.3% are in no Ertl group
  (saturated carbons, hydrocarbon rings).
- **PM convergence**: median 6 sweeps for both blocks; most blocks converge in 5-9 sweeps, but
  the tail is long and 20 molecules hit the 200-sweep cap in at least one block (flagged
  `pm_converged = False` in the token file, counted in the summary). The capped molecules are
  mostly small aromatic systems (bromobenzene, 2-methylpyridine, stilbene imines), not the
  largest molecules: the PM functional is nearly flat along rotations that mix the ring pi
  orbitals, so the absolute `1e-8` criterion is reached only slowly.
  Measured on all 24 capped blocks of the first build's 18 molecules: at sweep 200 the
  functional still changes by 1e-8 to 4e-6 per sweep, every block converges within 1-569
  further sweeps, and the total gain after sweep 200 is at most 5e-4 on functionals of 12-49.
  The capped results are therefore localized to all practical purposes; the cap and the
  absolute tolerance are kept as specified, and the flag stays so the tail can be revisited
  (see the diagnostic in the follow-ups below: only pi orbitals rotate).

## 2026-10-19 — Orbital tokens, follow-ups (user review of the first build)

### `empty_d` tokens stay, but are not acceptors

- `CandidateConfig(allow_empty_d_acceptors=False, allow_unknown=False)` in `ksmech.orbitals.tokens`
  decides which tokens may be arrow ends: `MoleculeTokens.donor_candidates()` (lone pair, sigma,
  pi, delocalized pi) and `acceptor_candidates()` (sigma*, pi*, empty p, empty s, delocalized
  pi*; `empty_d` only with the flag). The tokens themselves are kept, so the flag is a one-line
  ablation. `atoms_with_only_empty_d_virtuals()` lists the atoms whose every virtual token is
  `empty_d` (a bare halide anion: `[Cl-]` has 4 lone pairs and 5 `empty_d`); **the Prompt 4
  alignment must report how many dataset sinks land on such an atom**.

### Bonds to ionic metals are metal-bound lone pairs

- A two-centre occupied LMO with exactly one parent in `ClassifyConfig.ionic_metals` (Li, Na,
  Mg, Al, K, Zn) is `lone_pair` on the other atom with `metal_bound = True` and `metal_atom`
  set; the parent weight is the non-metal's population (CH3MgBr: C 0.54, Mg 0.46; C-Mg overlap
  population +0.23 is kept on the token). A one-atom lone pair whose second atom is such a metal
  with at least `metal_min_pop = 0.10` of the population is flagged too (Br-Mg in CH3MgBr: Br
  0.85 / Mg 0.13), while the free Br lone pairs (0.95 / 0.05) are not. The rule is not limited
  to carbon: an O-Li or Br-Mg "bond" is the same ionic situation. The flag survives the `.npz`
  round trip (schema 2; schema-1 files load with the flag unset) and `token_table` prints
  `[Mg1-bound]`. Metal-bound lone pairs are donor candidates: Grignard and organolithium
  arrows start there. The build summary counts them (`n_metal_bound_lone_pairs`).
- Why ionic metals only: a C-Zn or C-Mg LMO is 50-60% on carbon with an overlap population of
  0.2, i.e. a polarized bond whose electrons the carbon keeps in the reaction; transition
  metals (Cu, Ti, Pd) are left to the ordinary rules.

### Multi-LMO bonds are ranked, not thresholded

- `rank_bond_orbitals` (on by default, `ClassifyConfig.rank_bond_orbitals`): within one block,
  the two-centre LMOs that share the same parent pair and have the right bonding sign are
  sorted by mean |cos|; with 2 or 3 of them the largest is sigma(*), the others pi(*),
  regardless of `sigma_cos`. This fixes the terminal-alkyne sigma that PM tilts to |cos| 0.68 /
  0.70 (its pi partners sit at 0.15-0.23) and labels two banana orbitals of a bent double bond
  sigma + pi. A pair with a single LMO keeps the threshold test, which now calls the orbital
  sigma when at least one atom's p vector lies along the axis (the strained-polycycle case at
  0.63 / 0.89); a single LMO whose p vectors are both off-axis is still pi.

### Leaving-group physics (slow tests, optimized geometries)

- **Protonation lowers sigma*(C-O)**: methanol +2.71 eV -> CH3OH2+ -6.96 eV in the gas phase, a
  9.67 eV drop (test asserts >= 3 eV). **In ALPB water the drop is 2.92 eV** (+2.52 -> -0.39),
  just under the 3 eV asked for: the implicit solvent stabilizes the cation and lifts its
  orbitals by ~6 eV. The solvated test asserts only the sign and prints the value; the 3 eV
  criterion is enforced on the gas-phase pair.
- **sigma*(C-X) does not order Cl > Br > I.** PM LMO energies, ALPB water: C-Cl -2.07, C-Br
  -0.65, C-I -1.63 eV (gas phase -1.94, -0.29, -1.45): C-Cl is the lowest, C-Br the highest.
  The canonical LUMOs do follow the expected order (-4.21, -5.92, -6.28 eV; tested). A localized
  virtual's energy `c^T F c` is an average over the canonical virtuals it mixes, and the
  sigma*(C-X) LMO mixes with the C-H sigma* and the halogen d functions, whose energies differ
  between Cl, Br and I. The ordering test is kept as `xfail(strict=True)` with the numbers in
  its reason, so it fails loudly if a later change (energy definition, basis, localization of
  the virtual block) makes it pass. **Open question for the KS term:** the filled-empty gap may
  need the canonical LUMO projection rather than the LMO diagonal for virtuals.

### The sweep cap is a pi-system effect

- `scripts/pm_cap_diagnostic.py` re-runs every capped block to the cap, runs one more sweep,
  and classifies the orbitals that sweep still rotates (sin of the rotation angle from the
  diagonal of U). On the 18 capped PMechDB molecules (24 blocks): **every rotating orbital is
  `delocalized_pi` or `delocalized_pi_star`** (residual rotations 1e-4 to 2e-3), and no sigma or
  lone-pair orbital moves in any block. Exit code 1 would name the molecule otherwise.

## 2026-10-20 — Continuous orbital descriptors replace types as model input (Prompt 3b)

### Why types were demoted

- sigma / pi / lone-pair labels are a per-element chemistry heuristic with thresholds; a model
  that reads them learns the labeller, and the labels do not transfer to elements and bonding
  situations the thresholds were not tuned on (metals, hypervalent centres, 3-centre bonds,
  aromatic rings, which the first build left as `delocalized` or `unknown`). Orbital character
  must generalize across elements, so the token is the localized orbital itself, described by
  continuous physical quantities. The classifier stays (diagnostics, tables, the candidate
  masks `allow_empty_d_acceptors` / `allow_unknown`), its `type` field carries the dataclass
  metadata `model_input=False`, and `test_model_input_has_no_type_index` checks that changing
  every type leaves `MoleculeTokens.orbital_features()` unchanged.

### Descriptor list (`ksmech.orbitals.descriptors.OrbitalDescriptors`, 41 floats)

| group | entries | definition |
|---|---|---|
| occupation | 1 | spectral average of the canonical occupations, `sum_k p_rk n_k`: 2 occupied, 0 virtual, between 1 and 2 for LMOs that mix a SOMO (`[CH3]`: the seven occupied tokens sum to 7) |
| centres | 1 + 4 | number of atoms with population >= `tau_atom = 0.05`; sorted populations of the top 4 centres, zero-padded |
| composition | 4 x 3 | s, p, d shares of the orbital's gross population on each top centre (sum 1 per real centre, 0 on padding) |
| axis | 2 + 1 | `cos_a`, `cos_b`: \|cos\| between the p vector on centre a (b) and the a-b axis; `n_axis_centers` (0, 1, 2) says how many of the two carry p character, since a hydrogen has no p and its cos is a padding zero, not "perpendicular". `cos_character()` averages over those centres. |
| overlap | 2 + 1 | sign and magnitude of the a-b Mulliken overlap population; a-b distance (Angstrom); all 0 for one-centre orbitals |
| extent | 1 | population-weighted radius of gyration of the centres about the centroid (lone pair < sigma bond < ring pi, tested) |
| spectral | 8 + 8 | `(E_k, p_rk)` with `p_rk = U_kr^2` over the canonical MOs of the block: entries with `p >= 0.01`, at most 8 (largest first), renormalized to 1, zero-padded; the canonical indices are kept as bookkeeping |

- Bookkeeping that is **not** model input: centre atom indices, spectral MO indices, the
  centroid, and `E_r = sum_k p_rk E_k`, which equals the diagonal of `U^T F U` (tested to 1e-8)
  and orders the tokens within a block.
- `n_axis_centers` is the one entry the brief did not list; without it the C-H sigma* of
  acetone had cos-character 0.5 and failed the "> 0.7" criterion for a reason unrelated to
  the orbital.
- Spectral cutoff: on the fixtures the truncated set reproduces `E_r` to better than 0.5 eV and
  the three benzene pi MOs keep their full weight 3.000 over the occupied tokens.

### Deterministic localization

- Pipek-Mezey starts from the canonical MOs with the fixed round-robin pair schedule and no
  random rotation; the 200-sweep cap and `pm_converged` per block are kept. Two
  tokenizations of the same benzene record give bit-identical feature matrices, coefficient
  matrices and diagnostics (tested).
- Delocalized orbitals need no special path: a 4-centre ring pi orbital is a token with 4
  centres. The delocalized / unknown counts stay in the build summary as information; the
  build no longer fails on them (`--max-unresolved` removed), and prints the descriptor
  distributions (centres per token, cos-character, spectral entries per token).

### Term 3 in spectral form (`ksmech.physics.ks_terms`)

- `KS3(r, s) = (l_r^T B l_s)^2 * G(r, s)`, `G(r, s) = sum_k sum_m 2 p_rk p_sm / (E_k - E_m)`,
  canonical energies in the denominator, localized coefficients in the numerator; `G` is a
  pair feature of its own and needs only the two spectral sets, so it is defined between
  tokens of different molecules (`gap_kernel_matrix`). Gaps below 1e-3 eV are clamped.
- Checked on the fixtures: methanol O lone pair against chloromethane sigma*: G(C-Cl) =
  -0.219 1/eV < G(C-H) = -0.115 1/eV < 0.
- **Open point for the physics phase:** occupied and virtual LMOs of one converged molecule
  are S-orthogonal and F-decoupled, so with the molecule's own `S` or `F` as `B` the
  intramolecular coupling, and term 3 with it, is identically zero (tested). The coupling
  between tokens of different fragments needs an intermolecular `B` (overlap at a contact
  geometry, or a distance-free proxy); within one molecule a different coupling is needed
  altogether. Terms 1 and 2 are untouched (not yet written).

### Anisole: where the regiochemistry lives in the tokens

- The brief asked that the ring tokens' HOMO weight "resolved per centre" be larger at ortho
  and para than at meta. The per-centre sum `sum_r p_r,HOMO w_r(c)` does **not** reproduce
  the canonical HOMO population (GFN2 anisole, exact vs sum: ortho 0.110 / 0.112 and 0.141 /
  0.069, meta 0.048 / 0.157 and 0.020 / 0.110, para 0.239 / 0.203): it drops the cross terms
  between LMOs, which are large for 3-centre ring orbitals. What the tokens do carry: PM
  centres the three ring pi LMOs on ortho, ortho and para (never meta or ipso), and the
  para-centred token has the largest HOMO weight (0.404; ortho 0.212 and 0.128). The slow
  test asserts that, and the canonical HOMO itself has the ortho/para pattern.

### Resonance steps (`ksmech.data.resonance`)

- `canonical_resonance_key(mol)`: the lexicographically smallest canonical SMILES over RDKit's
  `ResonanceMolSupplier` (atom maps kept, cations and anions unconstrained, charge separation
  allowed, at most 1000 structures). At the cap the key is the plain canonical SMILES and the
  flag is `RESONANCE_TRUNCATED` (also `RESONANCE_FAILED` when sanitization fails); a
  truncated side therefore never matches, so truncation under-flags rather than over-flags.
- Atom maps stay in the key on purpose: a degenerate exchange (H3O+ + H2O -> H2O + H3O+) has
  the same map-free molecules on both sides but a different mapped skeleton and is a real
  step.
- Resonance structures share the sigma skeleton, so `skeleton_key` (all bonds single, charges
  zero, maps and explicit H kept) is checked first and the enumeration (48 ms per step, 7% of
  sides truncated on a 300-step probe) runs only when the skeletons agree.
- `ElementaryStep.resonance_step` is set in `build_step`; `steps.parquet` keeps every accepted
  step with the flag, `read_training_steps` drops the flagged ones and renumbers `step_index`
  0.. within each pathway (PMechDB has no pathways, so nothing is renumbered there), and both
  `ingest_report.json` and `stats.json` count them. The mini fixture's enolate shift is the
  one flagged step of its five.
- **PMechDB, re-ingested 2026-10-20: 147 of the 13,091 accepted steps (1.1%) are resonance
  steps**, so the training target has 12,944 steps (`stats.json`: `n_resonance_steps`,
  `n_training_steps`). The 13 rejections are unchanged. The ingest with the skeleton pre-filter
  took about a minute more than before.

### PMechDB rebuild with the new tokens (1,137 molecules cached by 2026-10-20)

- 47,139 occupied and 44,041 virtual tokens, 0 failures, 20 molecules at the sweep cap.
  Diagnostic types: occupied unresolved 3.67% (delocalized 3.66%, unknown 0.01%).
- **Centres per token**: 1: 8.7%, 2: 86.0%, 3: 3.6%, 4: 1.7%, 5-6: 37 tokens.
- **cos-character** (tokens with >= 2 centres): 0.9-1.0: 88.6%, 0.0-0.1: 10.4%, everything in
  between 1.0%. The sigma / pi split is bimodal and needs no threshold to be read.
- **Spectral entries per token**: 88.1% of tokens hit the cap of 8; only 0.4% are a single
  canonical MO. Before renormalization the kept entries carry a mean weight of 0.785 (median
  0.795, 10th percentile 0.58, minimum 0.34; 3.3% of LMOs keep less than half), measured on
  6,212 LMOs of 80 molecules. The cutoff `p >= 0.01`, 8 entries is therefore a real
  truncation on PMechDB-sized molecules, where an LMO spreads over 20-40 canonical MOs; the
  renormalized set keeps the dominant energies and `E_r` is still exact (full `p`). Raise
  `spectral_max` if the gap kernel turns out to need the tail.

## 2026-10-20 — Follow-ups to 3b: full spectra for G, contact kernel, frontier weights

### G is computed from the full spectral vectors

- Each token keeps its full `p_rk = U_kr^2` over the canonical MOs of its block and the
  block's energies as bookkeeping (padded on disk, trimmed on load; token schema 4).
  `token_gap_kernel` uses them; the 8 renormalized entries are model input only, and
  `token_gap_kernel_truncated` exists solely for the truncation test. Three model-input
  scalars summarize the full spectrum: `spectral_mean` (= `E_r`), `spectral_variance` and
  `retained_fraction` (weight of the kept entries before renormalization). Feature vector:
  44 floats.

### The contact kernel is used for every pair

- **Why the converged intramolecular KS3 is identically zero.** The occupied and virtual LMOs
  of one molecule are orthogonal rotations of the occupied and virtual canonical MOs of the
  same converged Fock operator: `l_r^T S l_s = 0` and `l_r^T F l_s = 0` for r occupied, s
  virtual. The second-order filled-empty energy of Klopman-Salem describes two reactants
  *approaching*: it is the perturbation that the other molecule's orbitals exert, and inside a
  molecule at its own SCF solution that perturbation has already been absorbed into the
  canonical MOs. A term that is zero by construction carries no information for the model.
- **Why the contact picture is right.** An arrow is drawn between a donor and an acceptor that
  come into contact (inter- or intramolecularly: 4-chlorobutanol cyclizes by exactly the O
  lone pair -> sigma*(C-Cl) interaction, through a folded conformer, not through the
  extended one's vanishing coupling). The coupling that matters is the one the two orbitals
  would have with one centre of each at bonding distance, lobes aligned. `ksmech.physics.contact`
  computes it from GFN2 itself: the s-s, s-p_sigma, p_sigma-s, p_sigma-p_sigma overlaps of the
  element-pair diatomic at the sum of the covalent radii are probed with tblite (cached per
  element pair, hydrogen has no p), combined with the orbital's |s| coefficient and p-vector
  norm on each centre in the maximal-overlap orientation, and the token pair takes its best
  centre pair. `ks3_contact = kappa^2 * G` applies to every pair, in one molecule or across
  two, and needs no common AO basis and no geometry of the complex.
- **Intramolecular extras** (`intramolecular_pair_features`): the centroid distance and the
  cross-centre coupling `l_r^T B l_s` restricted to the AO blocks of the donor's centres and
  the acceptor's centres. The latter is not zero (the orthogonality holds only for the full
  vectors), but it is geometric: in the extended MMFF conformer of 4-chlorobutanol, with the
  O three bonds from C-Cl, it is 1.9e-5 with `B = S`; the test asserts non-zero and prints it.
- Checked: methanol O lone pair / chloromethane C-Cl sigma*: kappa 0.05-2, `ks3_contact < 0`;
  the whole occupied x virtual matrix of methanol is negative.

### Per-atom frontier-orbital weights (`ksmech.orbitals.frontier`)

- `AtomToken.fmo_weights`: Mulliken population on the atom of the HOMO-1, HOMO, LUMO and
  LUMO+1 *blocks*, a block being every canonical MO within 0.05 eV of the level, summed. For
  a non-degenerate HOMO / LUMO this is exactly the frozen-orbital f- / f+ already stored;
  benzene's e1g / e2u pairs are summed so the weights are ring-symmetric. NaN when a level
  does not exist (`[H+]` has no occupied levels, `[H-]` no virtuals).
- Anisole, HOMO block per ring carbon: ipso 0.165, ortho 0.110 / 0.141, meta 0.048 / 0.020,
  para 0.239. Nitrobenzene, LUMO block: ipso 0.010, ortho 0.084 / 0.084, meta 0.001 / 0.001,
  para 0.094. Both ortho/para > meta, as the frontier picture of electrophilic and
  nucleophilic aromatic substitution requires; the orbital-level anisole test is kept too.
- **Truncation proof** (`test_truncation_matters_on_pmechdb_pairs`): over 49,605
  occupied x virtual pairs of the first 20 rebuilt PMechDB molecules, G from the 8-entry
  model-input spectra differs from G from the full spectra by up to 21.8%; the test requires
  more than 5%. Rebuilt 2026-10-20 on 1,267 cached molecules, 0 failures.

## 2026-10-20 — Reconciliation with spec v3

`docs/ksmech_architecture.tex` (spec v3, copied into the repo) is now the source of truth.
Two principles drive every change below: orbital character must generalize across elements,
so a token is described by physical quantities; and no chemist's taxonomy (orbital type,
hybridization, aromaticity, functional group) is a model input or decides which arrows are
legal. Token schema 5; schema-4 files are rebuilt on load.

### Model inputs
- **Orbital tokens: exactly the spec's v_r.** 23 continuous inputs (`orbital_features`) plus
  the 8 x (E, p) spectral set (`orbital_spectra`, for a set encoder) plus centre weights
  (`orbital_centres`, for the sum of w_a h_a). Moved to bookkeeping: `spectral_mean` (= E_r,
  which the spec keeps as a diagnostic; removed as a duplicate of `energy_ev`),
  `spectral_variance`, `retained_fraction`, `n_axis_centers` (the s/p/d shares already say
  whether a centre has p). `overlap_sign` + `overlap_magnitude` are now one signed
  `overlap_population`. This supersedes the 41/44-float lists above.
- **tau_atom 0.05 -> 0.15** (the spec value). Benzene's PM pi orbitals go from 4 centres to 3.
  The full centre set A_r is stored (`all_center_atoms` / `all_center_weights`), because the
  site softmax and the centre embedding run over all of A_r; the top 4 stay the input.
- **Atom tokens.** Every field is tagged. `atom_categorical` holds Z and formal charge;
  `atom_features` holds q, frozen-orbital f+ and f-, the finite-difference dual, the 4
  frontier weights, %Vbur and chirality. Tagged `model_input=False`: hybridization,
  aromaticity, fg_id, finite-difference f+/-, position. A missing value (dual after
  FUKUI_SCF_FAIL, the weight of an absent level) is 0 in `atom_features`, flagged in
  `atom_missing` and counted in the build summary.
- **Chirality** = -sign det of the vectors to the three highest legacy-CIP-ranked
  neighbours (+1 R-like). Only for stereocentres the SMILES specifies, because ETKDG picks an
  arbitrary enantiomer otherwise; a geometry that inverts the tag is logged. RDKit's legacy R/S
  labels are wrong for three-coordinate centres (sulfoxide, phosphine), so the test checks
  against `rdCIPLabeler`. PMechDB has stereo in only 2 of 13,091 steps.

### Candidate masks
- **Masks use descriptors, not classifier types.** A donor has occupation > 1.5. An
  acceptor has occupation < 0.5, is not d-type (its main centre's d share is the largest), and
  is not the donor's antibonding partner (same two leading centres, both with >= 2 centres).
  Renames and removals: `allow_empty_d_acceptors` -> `allow_d_acceptors`; `allow_unknown` is
  removed; `atoms_with_only_empty_d_virtuals` -> `atoms_with_only_d_virtuals`, and the
  alignment must still report the sinks that land there. The d-type rule picks exactly the
  classifier's `empty_d` tokens on 7 test molecules; the build summary counts mismatches
  (`d_type_label_mismatch`). Partially occupied radical tokens are neither donor nor acceptor,
  and are counted.

### Physics (`ksmech.physics`)
- **K and eps are parameters.** `KSParams(K=1.75, eps=4.0)`. Each feature is a fixed power of
  the parameters (`SCALING`), so the model applies learned values to features computed at
  K = eps = 1. Standardization statistics must be frozen at the initial values, otherwise a
  learned scale divides out.
- **Wolfsberg-Helmholz B**, zero within an atom. The intramolecular cross-centre coupling is
  now l_r^T B l_s over the full vectors. Before, it used S restricted to the centre blocks.
  Measured with K = 1.75: an occupied/virtual pair on the same bond has a median of 0.45 eV
  (up to 3 eV); 4-chlorobutanol's O lone pair -> sigma*(C-Cl) in the extended conformer
  gives 0.016 eV.
- **D1 (user decision): the contact kernel gives KS3 for every pair.** Spec section 5.4
  says to use the xTB matrices within a fragment. The coupling above shows how close two
  orbitals are through bonds in the computed conformer, not whether they can meet. The
  coupling squared and the centroid distance are therefore the intramolecular extras in z_ij.
  The first paragraph of section 5.4 needs a matching edit in the spec.
- **D2 (user decision): the kappa table is the default kernel.** Key: element plus
  dominant shell (by shell share) of each main centre. Samples come from non-bonded pairs
  (beyond 1.3x the covalent-radii sum) within +-0.3 Angstrom of the vdW sum. There are two
  tables (occupied-virtual, occupied-occupied); backoff is (Z,l,Z,l) -> (Z,Z) -> (l,l) ->
  global, needing at least 5 samples per level. The table is built from every cached token
  file of the source: it is a physical statistic, not derived from labels, so it cannot leak
  across splits. The tblite probe sits behind `ContactConfig(kernel="analytic")`; `mlp`
  raises NotImplementedError until the ablation phase.
- **Interpretation of F_rr in the contact beta.** beta = K a (F_r + F_s)/2 uses the Fock
  diagonal of the dominant shell on the main centre (WH at the AO level). An LMO's own Fock
  diagonal is negative for occupied tokens and often positive for virtual ones (methanol
  sigma*(C-O) +2.7 eV), so their mean can cross zero. The spec's S x w x w is kept as written,
  although the median S already contains typical coefficients.
- **KS2** = 14.3996 q q / (eps R) eV between sites (an atom, or an orbital's main centre);
  undefined when both tokens sit on the same atom.
- **KS1 uses electron populations** N_a = Z_val - q_a, not partial charges. Spec Eq. (ks)
  writes term 1 with q and term 2 with Q, as Klopman and Salem do. Within a fragment: the
  WH form, summed over occupied r' != r whose main centre is within 3.5 Angstrom of m(s).
  Between fragments the spec leaves the form open; pinned here as
  -(N + N) beta_c a %Vbur %Vbur / 100^2.
- **z_ij** (7 entries) for every ordered token pair. Undefined entries are 0 and carry a
  `defined` mask. Standardization uses defined entries only. A non-finite value (for
  example NaN %Vbur) is counted and left undefined.

### Tests
- Physics tests pick tokens by descriptors (`tests/token_select.py`). Label tests carry the
  `diagnostic` marker. `test_ks3_matrix_matches_the_scalar_form` was vacuous: it compared the
  full-spectrum matrix with a truncated-spectrum scalar and passed only because the S
  coupling is 0. It now uses a random symmetric B and full spectra.

### Operational
- The host hard-rebooted at 01:42, about 1.5 minutes after the token rebuild (3 workers)
  started next to the QM batch (4 workers). This is the same instability as in the
  2026-10-19 operational note.
- **The cause is unpinned BLAS threads, not the worker count.** `ksmech-qm` sets
  `OMP_NUM_THREADS=1` for its pool; `ksmech-orbitals build` did not. Each token worker's numpy
  therefore fanned out over all cores: one orphaned worker ran at 528% CPU after the restart,
  and the load average reached 21 with "3 + 4 workers". `run_build` now spawns its workers
  with `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` set to 1
  (`single_threaded_children`, tested), and each worker measures 100% CPU. This
  probably explains part of the 2026-10-19 crashes too.
- Keep the total number of busy cores at 4 or fewer anyway. The QM batch is resumable; it was
  stopped at 02:03 and is not restarted by this work.

### d-type rule refined after the PMechDB rebuild
- The first d-type rule (main centre mostly d) flagged 104 more tokens than the classifier's
  `empty_d`, and 53 + 4 of them sat on Ti and Cu. In groups 3-11 the d shell is valence, not
  polarization, and d0 Ti(IV) is a Lewis acid that accepts into it. `is_d_type` now also
  requires the main centre's element to have no valence d shell (`VALENCE_D`, Z 21-29, 39-47,
  57-79; Zn, Cd and Hg keep d10 in the core). A TiCl4 test pins this.
- 99 tokens remain where the rule and the label disagree, both on purpose:
  - 47 are d-dominated sigma* / empty-p tokens on Al, I and Mg, 14-19 eV up. They are
    masked; the hypervalent-iodine acceptors that react are the p-dominated I-O sigma*.
  - 52 are one-centre Ti / Cu d orbitals that the classifier calls `empty_d` and that are
    now acceptors.

### PMechDB rebuild with the spec-v3 tokens (2,001 molecules cached by 2026-10-20 02:10)
- **Build:** 2,001 of 14,383 molecules (12,382 still without a QM record, skipped and
  counted), 0 failures, 35 molecules at the PM sweep cap. Tokens per molecule: min 11,
  median 93, max 441. 83,155 occupied and 76,425 virtual tokens. Diagnostic types: occupied
  unresolved 3.93% (delocalized 3.91%, unknown 0.02%).
- **Centres per token at tau_atom = 0.15:** 1: 10.7%, 2: 86.6%, 3: 2.7%, 4: 2 tokens, none
  above 4. At 0.05 they were 8.7 / 86.0 / 3.6 / 1.7%.
- **cos-character** (tokens with >= 2 centres): 0.9-1.0: 89.6%, 0.0-0.1: 9.8%, in between
  0.6%.
- **Spectral set:** 87.7% of tokens hit the 8-entry cap, 0.4% are one canonical MO. The top
  entry's weight has median 0.30 (p5 0.18, p95 0.73). The kept entries carry median 0.79 of
  the weight before renormalization (p5 0.52).
- **Orbital inputs** (median, p5-p95):
  - main-centre weight 0.52 (0.50-1.00); second-centre weight 0.47 (0-0.50);
  - signed overlap population 0.0 (-1.77 to +0.39): one-centre tokens are 0, virtuals
    negative;
  - distance_ab 1.10 Angstrom (p95 1.54);
  - radius of gyration 0.55 Angstrom (0.01-0.78).
- **Atom inputs** (63,601 atoms; median, p5-p95):
  - Mulliken charge 0.04 (-0.36 to 0.24);
  - frozen-orbital f+ 0.0003 (p95 0.24) and f- 0.0013 (p95 0.17);
  - dual -0.002 (-0.064 to 0.076);
  - %Vbur 50.2 (33.0-77.2);
  - no missing values in this set;
  - chirality is 0 on every atom: no cached molecule has a stereocentre specified in its
    SMILES.
- **Masks:** 83,148 donors (7 partially occupied radical tokens are neither donor nor
  acceptor), 71,657 acceptors, 4,768 d-type tokens.
- **Contact table** (`data/cache/contact_table.json`): 352,985 occupied-virtual and 352,802
  occupied-occupied samples.
  - Full (Z, l, Z, l) keys: 251 and 234, of which 200 and 191 have >= 5 samples.
  - Global medians: 0.0042 and 0.0007.
  - The samples are dominated by contacts with H: (C p, H s) has 112,370 samples, median
    0.011; (C p, C p) has 90,650, median 0.0013.
- **G > 0 across fragments:** over 245,698 occupied x virtual pairs of different reactant
  fragments, on the 269 multi-fragment PMechDB reactant sides with all tokens cached, 90
  pairs (0.037%) have G > 0. They fall on 15 sides, all strong donor / acceptor pairs
  (enolate + Br2, propargylsilane + protonated formaldehyde, CBr3- + phosphonium), where a
  canonical donor level lies above an acceptor level and second-order perturbation theory
  no longer holds. Left as computed for now; the physics phase has to decide whether to
  clamp G to a negative value there.
- The truncation test now runs again on schema-5 files.

## 2026-10-20 — Prompt 3b re-check: SOMOs localized apart, tests tightened

The 3b brief (continuous descriptors as the only orbital input, `type` demoted to a
diagnostic, deterministic PM, spectral store, spectral KS3 with G as its own feature,
resonance keys) was re-run against the spec-v3 code. Everything was in place, except a
SOMO never got occupation 1. Three choices from the spec-v3 reconciliation still hold where
the brief words them differently:

- the overlap is one signed population, not a sign and a magnitude;
- the mask is `CandidateConfig.allow_d_acceptors`, not `allow_empty_d_acceptors`;
- KS3 uses the contact kernel for every pair (D1). The converged `(l_r^T B l_s)^2 G` form is
  `ks_terms.ks3` / `ks3_matrix`, and G is the pair feature `gap_kernel`.

The descriptor list, spectral cutoff, deterministic start, the reason types were demoted and
the 147 removed resonance steps are recorded in the 3b and spec-v3 entries above.

### SOMOs are localized apart from the doubly occupied MOs

- **Rule.** Within the occupied block (occupation >= 0.5), the MOs at or above
  `DOUBLE_THRESHOLD = 1.5` and the ones below it (SOMOs, and the rare smeared fractional
  levels) are localized separately. `U_occ` is block-diagonal, so every LMO's spectral
  weights still run over the whole occupied block and G is unchanged in form. The occupied
  `PMStats` sum both sub-blocks (sweeps: the max).
- **Why.** The density sum_i n_i |phi_i|^2 is invariant only under rotations among MOs of
  equal occupation. Mixing a SOMO into the doubly occupied block gave LMOs that no longer
  represent the computed density. It also spread the radical over many tokens. Br[C]Br had
  four tokens at 1.861 and two at 1.277. CC(=CC=O)C(O)O[O] had thirteen between 1.67
  and 1.99. Spec section 4: n_r in {2, 0}, 1 reserved for singly occupied orbitals. Spec
  §4.1 ("separately to the occupied valence block and to the virtual block") needs one
  sentence for the singly occupied sub-block.
- **The old `partially_occupied = 7` hid most of it.** That count only sees tokens that
  fall below the 1.5 donor threshold. Most radical-contaminated tokens stayed above 1.5 and
  were accepted as donors. 11 of the 2,001 schema-5 token files had a fractional occupied
  token. Now: one token at 1.000 per SOMO (Br[C]Br: two), the rest at 2 (or a smeared
  1.9x level, as before). A SOMO is neither donor nor acceptor until the radical extension.
- **Closed-shell records are unchanged.** 36 of the first 40 cached files rebuild
  bit-identically. 4 differ at floating-point noise level: the same LMOs (|<old|new>| = 1.0),
  the same sweep counts, and two fresh builds of each are bit-identical. The old files were
  most likely written with a different BLAS thread count. Token schema 5 -> 6, so every file
  is rebuilt on load.
- Tests: `[CH3]` gives occupations 1, 2, 2, 2, and its SOMO token's spectral set is one
  canonical MO. `[CH2]` (triplet) has a `U_occ` with zero SOMO / doubly occupied blocks.

### Tests tightened

- **Determinism** compares every model-input matrix (`orbital_features`, `orbital_spectra`,
  `orbital_centres`, `atom_features`), `C_loc` and the full spectral vectors of two
  tokenizations of the benzene record.
- **Anisole** now asserts the brief's form, with the HOMO weight resolved by main centre:
  each ring pi token gives its spectral weight on the HOMO to its main centre. Values:
  ortho 0.212 / 0.128, para 0.404, meta 0 / 0, ipso 0, so ortho and para > meta. The
  population-spread form `sum_r p_r,HOMO w_r(c)` gives ortho 0.109 / 0.065, meta 0.152 /
  0.100, para 0.202. It is printed, not asserted, for the cross-term reason above.
- The build summary adds an occupation histogram (0.1 bins) to the centres,
  cos-character and spectral-entry histograms.

### PMechDB rebuild, schema 6 (2,001 molecules, 2026-10-20 03:31)

- `ksmech-orbitals build --source pmechdb --workers 4`: 2,001 built, 0 failed, 12,382
  without a current-method QM record (counted). 35 molecules at the PM sweep cap. 8 min 17 s
  wall clock, each worker at ~100% of one core. Token counts, types, centres,
  cos-character, spectral entries and the input quantiles match the spec-v3 rebuild above;
  only the open-shell molecules changed.
- **Occupation** (0.1 bins): 2.0: 83,142; 0.0: 76,418; 1.0: 10 (the SOMOs of 6
  molecules); 10 tokens in between (0.1: 5, 0.2: 1, 0.4: 1, 1.3: 1, 1.9: 2).
- **The 10 tokens in between are Fermi smearing, not radicals.** Two small-gap carbocations
  (C=CCN1...[CH+]...: 1.91 occupied, 0.075 and 0.057 virtual) and `C=[O+][Ti-]` (1.27
  occupied; 0.42, 0.17 and 0.10 virtual). The virtual block is localized as one block, so a
  smeared level there still mixes with the empty ones, as before this change. They are too
  few to act on now. The 1.27 token is neither donor nor acceptor. The virtual tokens
  under 0.5 count as acceptors.
- Masks: 83,144 donors, 71,657 acceptors, 11 occupied tokens that are neither (10 SOMOs +
  the Ti 1.27 token), 4,768 d-type tokens.
- Contact table regenerated from the schema-6 cache: 352,990 occupied-virtual and 352,820
  occupied-occupied samples, global medians 0.0042 and 0.0007 (schema 5: 352,985 /
  352,802, same medians).
- The PMechDB truncation test runs again on the schema-6 files: G from the 8-entry spectra
  differs from the full-spectrum G by up to 21.8% over 50,323 pairs.

## 2026-10-20 — Fermi-smeared molecules: flag and fractional-occupation mask (user decision)

Supersedes "too few to act on now" in the schema-6 rebuild note above.

- **Fractional occupations stay the continuous `occupation` input** (`sum_k p_rk n_k`). They
  are not rounded or clipped.
- **Per-molecule flag `MoleculeTokens.fermi_smeared`** (token schema 6 -> 7). It is True
  when some canonical MO's occupation lies more than 0.05 from its aufbau value for the
  record's spin state (`localize.aufbau_occupations` / `is_fermi_smeared`).
  - The aufbau value is 2 on the (N - u) / 2 lowest MOs, 1 on the next u, 0 above.
  - For a closed shell this is exactly "some MO has 0.05 < n < 1.95". A radical's SOMO at
    n = 1 is its spin state, so it is not counted as smearing.
  - Taken literally, the 0.05 < n < 1.95 rule would also flag the 6 tokenized radicals
    (C=[O+][Sn], C=[O+][Sn-], Br[C]Br, two peroxyl radicals, one radical cation). Their gaps
    are 0.39-6.1 eV, not vanishing. The rule uses canonical occupations, not token
    occupations: the flagged silicate enedione has no token below 1.95 (canonical
    1.934 / 0.066, spread over several LMOs).
- **These molecules have a vanishing GFN2 gap and are outside the closed-shell domain.**
  - At the electronic temperature tblite uses, a gap of a few kT spreads electrons over the
    frontier levels, so the record is no longer a single closed-shell determinant.
  - The flagged molecules have exactly the four smallest HOMO-LUMO gaps of the 2,001
    tokenized molecules: C=[O+][Ti-] 0.046 eV (canonical n 1.271 / 0.457 / 0.272), two
    resonance forms of one allyl cation, C=CCN1...[CH+]..., 0.135 eV (1.863 / 0.137), and a
    pentacoordinate-silicate enedione 0.175 eV (1.934 / 0.066). The 1st percentile of the
    gap is 0.74 eV and the median 3.17 eV. The next gaps (0.21, 0.26, 0.32 eV) are not
    smeared beyond 0.05.
  - Their tokens, and steps through them, are not closed-shell evidence. Read the flag
    before training or evaluating on them.
- **Candidate mask `CandidateConfig.exclude_fractional` (on by default).** A token with
  0.2 < n < 1.8 (`fractional_band`) is neither donor nor acceptor until the radical
  extension. On top of the 1.5 / 0.5 thresholds, this means donors need n >= 1.8 and
  acceptors n <= 0.2. SOMOs (n = 1) were already excluded by the old thresholds.
- **PMechDB (2,001 of 14,383 molecules tokenized, schema-7 rebuild 2026-10-20):**
  - 4 molecules flagged.
  - 8 of the 13,091 accepted steps have a flagged reactant or product; 7 of the 12,944
    training steps.
  - Only 103 accepted steps (102 training) have every molecule tokenized, and none of
    those is affected. The counts are lower bounds until the QM batch finishes.
  - Mask: 12 tokens in the band (10 SOMOs, the Ti 1.27 and 0.42 tokens). Only 1 candidate
    is lost: the Ti 0.42 virtual, which had been an acceptor.
  - Totals: 83,144 donors and 71,656 acceptors.
  - The build summary reports all of this (`fermi_smeared`, `masks.fractional_band`,
    `masks.fractional_band_removed`).
  - Contact table regenerated, unchanged (352,990 / 352,820 samples).

## 2026-10-20 — Arrows: application, alignment, samples (`ksmech.arrows`, Prompt 4)

`ksmech-arrows roundtrip | align | samples`; reports in `data/processed/<source>/`
(`arrows_roundtrip.json`, `alignment_report.json`, `samples_report.json`).

### Charge bookkeeping (`apply.apply_arrows`)

- **One bond-electron-matrix edit per step.** Each arrow is one electron-pair move, from a
  lone pair (N -= 2) or a bond (order -= 1) to a lone pair (N += 2) or a bond (order += 1).
  All moves of a step are summed before the molecule is touched, and every touched atom's
  charge changes by `dq = -(dN + dB)` (from `q = V - N - B`). Results:
  - lone pair a -> atom b: a +1, b -1;
  - heterolysis (a, c) -> c: a +1, c -1;
  - bond (a, c) -> bond (a, b): c +1, b -1, a unchanged.

  There is no per-arrow charge rule. A sink that also loses its leaving group in the same step
  (SN2 carbon, E2 hydrogen) nets to 0 by the sum. The arrow order does not matter (tested on
  the E2 fixture).
- **Hydrogens and stereo.**
  - Touched atoms have their H count frozen before the edit, so RDKit never refills a valence.
  - A changed bond loses its E/Z flag.
  - An atom whose neighbour set changes loses its chiral tag. Stereo is not tracked through a
    step; PMechDB has stereo in 2 steps.
- **Reason codes.**
  - VALENCE: a lone-pair source on an atom with N < 2, a bond order outside 0..3, or an RDKit
    valence error.
  - SANITIZE: any other RDKit sanitization error.
  - UNKNOWN_ARROW: an unknown shape, an atom missing from the molecule, or a bond from an atom
    to itself.
  - Radical arrows raise `NotImplementedError`.
- **Lone pair -> bond (b, d) with the donor outside the bond** is the separate
  `Displacement(donor, attacked, leaving)`. `schema.Arrow` stores bond tuples sorted, so the
  attacked atom would be lost. PMechDB never writes this shape (0 ARROW_INVALID rejections).
- **Round trip (all 12,944 PMechDB training steps, 11 s): 12,943 = 99.99%.**
  - 12,942 steps reproduce the exact Kekulé SMILES. 1 matches only modulo resonance: the product
    writes a phenyl ring in the other Kekulé form.
  - With the 147 resonance steps included: 13,090 / 13,091.
  - The single failure (MISMATCH, bond>atom, pmechdb-9946a45b6a5d6f2e): the product drops the
    dative bond `[Na+]<-[C-:20]` without an arrow for it. It is left failing, not special-cased.
  - This measurement predates the ingest check below. Since "Arrow replay in ingest
    validation" the ingest rejects that step, and the round trip is 12,943 / 12,943.

### Site-based alignment policy (`align`)

- **Arrow roles.** An arrow that forms a bond (lone pair -> atom, lone pair -> own bond,
  bond -> bond) has:
  - a donor site a: the lone-pair atom, or the shared atom of bond -> bond;
  - a partner c for a bond source;
  - a sink b.

  A heterolysis (bond (p, q) -> lone pair on q) forms no bond. When another arrow of the step
  forms a bond at p, the heterolysis is that arrow's sink bond (b, d) = (p, q) and is absorbed
  into its acceptor set. A bond -> bond arrow whose new bond forms at q is a sink bond at p in
  the same way (the E2 relay). An unabsorbed heterolysis is unaligned (HETEROLYSIS_ONLY).
- **Sets**, with every type-sounding test read from descriptors:
  - "on an atom": the atom is a centre, i.e. weight >= tau_atom;
  - "one-centre": `n_centers == 1`;
  - "antibonding": overlap population < 0;
  - "d-type": `is_d_type`, the main centre mostly d on an element without a valence d shell.

  The rules themselves:
  - Donors: every donor-candidate token on a. For a bond source, only those also on c;
    SOURCE_RELAXED when none is.
  - Acceptors: every acceptor-candidate token on b. With sink bonds, only those also on some
    d; SINK_RELAXED when none is. SINK_RELAXED is not in the brief; it is the symmetric case.
  - A test changing every token's `type` leaves all labels unchanged.
- **Antibonding exclusion only for the arrow's own source bond (a, c).** A lone pair has no
  bond, so nothing is excluded for it. The first version excluded the partner of every token
  in the donor set, and a lone-pair set contains the sigma bonds of a, e.g. sigma(O-C) of a
  tetrahedral C(O-)Cl. That intermediate's acceptor, at its restrained, stretched C-Cl, is a
  three-centre token (C 0.60, O 0.20, Cl 0.15) whose leading centres are C and O. 5 arrows
  lost their only acceptor that way.
- **RESONANCE_FORM** (lone pair drawn on an atom with no one-centre occupied token).
  - "The multi-centre tokens on that atom" cannot be read literally: C-H sigma tokens have two
    centres too.
  - The donors are therefore the tokens the drawing has no bond for: tokens on a shared with a
    partner x by more tokens than the drawn order a-x (an implicit H counts as order 1), plus
    tokens with three or more centres.
  - Acetaldehyde enolate, GFN2: the carbanion pair is a two-centre C=C pi token (0.68 / 0.31),
    not a lone pair. The carbanion drawing (lone pair on CH2) and the enolate drawing (C=C
    bond source) both give donor site CH2 and donor set {sigma(C-C), pi(C=C)} on the same
    tokens (tested).
- **ALIGN_FALLBACK**: the nearest candidate tokens of the site's fragment by centroid
  distance, within 0.25 Angstrom of the nearest.
- **Sinks on only-d atoms** (the 2026-10-19 follow-up): flagged SINK_ONLY_D and counted
  (`n_sinks_on_only_d_atoms`) whether they align by fallback or not. A bare halide sink has no
  acceptor candidate in its fragment, so it is unaligned NO_ACCEPTOR.
- **Teacher representatives** (r*, s*): the member with the largest weight on its site. Ties go
  to the higher-energy donor and the lower-energy acceptor.
- **Measured.** Only 539 of the 12,944 training steps have every reactant fragment tokenized;
  12,405 were skipped as MISSING_TOKENS, because the QM batch is stopped at 2,001 of 14,383
  molecules.
  - Arrows: 1,092 in total. 580 labels and 440 absorbed heterolyses = **93.4% aligned**; 100%
    of the bond-forming arrows have a label.
  - **ALIGN_FALLBACK 0%.** SOURCE_RELAXED, SINK_RELAXED and RESONANCE_FORM 1.55% each (9 labels
    each: beta-silicate C-Si pairs held on C, stretched C-Cl in restrained tetrahedral
    intermediates, ylides and diazo/diazonium N).
  - Mean set size: donor 3.15, acceptor 1.67.
  - Sinks on only-d atoms: 0.
  - 467 of the 539 steps are complete.
- **Open spec point: a lone heterolysis has no 4-tuple.** All 72 unaligned arrows are
  unimolecular ionizations (Si-O, C-O, P-Br, I-O, B-O, Si-Cl, Si-F, C-N, ... bonds breaking with
  no bond forming). Over the whole training set, without tokens: 805 of 12,944 steps (6.2%),
  805 of 25,782 arrows (3.1%). The tokenized subset over-represents them (13% of its steps),
  because small molecules were featurized first. They need a decoder action the spec does not
  have, for example "ionize: donor r on (p, q), site q, no acceptor". Until then these steps
  are skipped as UNALIGNED_ARROW.

### Kekulization and sink rules in `apply_predicted_arrows`

The spec rule taken literally reproduced 421 / 467 = 90.1% of the completely aligned steps
(VALENCE 23, MISMATCH 23). The rule was fixed, not the steps. Arrows are applied in order;
each electron pair moves once per step (a bond decremented earlier in the step is not
decremented again).

- **Source.**
  - Lone pair when the site is negative in the current state, or carries a one-centre occupied
    token, **and** has at least 2 nonbonding electrons in the current Lewis structure.
  - Otherwise a bond a-c, chosen in this order:
    1. a multiple bond to a centre of the donor token;
    2. the same after a Kekulé switch (`ResonanceMolSupplier(KEKULE_ALL)`, same formal charges,
       every accumulated bond change still in 0..3, at most 64 structures);
    3. a single bond to a donor-token centre (sigma donors: C-H of an E2, hydride shifts);
    4. a drawn bond that no occupied token spans (8 beta-silicates whose C-Si pair is a
       one-centre token on C, while the drawing gives that C no lone pair);
    5. a multiple bond to any neighbour.

    c is never the acceptor site. With no such c, a lone pair of the Lewis structure is used
    (an NH drawn neutral but delocalized, so it has no one-centre token).
- **Sink.**
  - A sink with an open valence shell before the new bond (fewer than 8 electrons, 2 for H)
    keeps all its bonds.
  - A full-shell sink breaks one bond b-d: d is a centre of the antibonding acceptor bonded to
    b, multiple bonds first, then by weight, never the donor site.
  - When the acceptor names no such centre, d is the neighbour whose bond pair already sits
    mostly off b: the smallest weight on b over the occupied tokens spanning b-d.

  The spec's "two-centre antibonding token" test failed both ways:
  - It broke a C-C single bond when a carbocation's empty p is conjugated onto a neighbour, a
    two-centre token C 0.60 / C 0.25 with overlap -0.19 (10 steps).
  - It broke nothing for a protonated carbonyl pi* or a tetrahedral collapse whose acceptor
    has three centres (10 VALENCE).
- **Measured: 462 / 467 = 98.9%** (460 exact, 2 modulo resonance). The 5 failures:
  - three hypervalent additions (O -> Si twice, O- -> P+ once): the same antibonding acceptors
    describe substitution at Si / P, and the 4-tuple cannot say whether the sink keeps all
    its bonds;
  - a magnesium alkoxide whose O-Mg pair the site rule takes as a lone pair, since O carries
    lone pairs;
  - a restrained sulfur intermediate where no acceptor token reaches the leaving O.
- **Set members.** Swapping one non-representative member into a reproduced step still
  reproduces the product for 1,233 / 1,236 donor and 364 / 380 acceptor swaps.

### Samples (`steps`)

- A `MechanismSample` holds the model inputs of all reactant fragments in one index space, the
  masks, the sparse token-to-centre incidence, the labels and the product resonance key.
- The full `MoleculeTokens` (C_loc, S, F) are not copied: about 1 MB per molecule, repeated
  across steps. `fragment_keys` reloads them.
- The token cache keeps the 1,024 most recently used molecules (LRU).
- A step becomes a sample only when it is completely aligned. Skip reasons:
  MISSING_TOKENS, CORRESPONDENCE, UNALIGNED_ARROW.
- **PMechDB, 2026-10-20: 467 samples** from the 12,944 training steps (12,405 MISSING_TOKENS,
  72 UNALIGNED_ARROW), about 21 KB each.
  - random: 370 / 47 / 50 (369 / 49 / 49 after the re-ingest below: removing one step
    reshuffles the random split);
  - scaffold: 431 / 17 / 19 (unchanged after the re-ingest);
  - leave-one-class-out does not apply to PMechDB (no classes) and is reported as such; it is
    tested on the mini fixture with classes added.
- Re-run `ksmech-arrows align` and `ksmech-arrows samples` after the QM batch and the token
  rebuild. Every number in this entry covers only the tokenized 4%.

## 2026-10-20 — Arrow replay in ingest validation (user decision)

- **`validation.check_arrows_reproduce_products` replaces `apply_arrows_stub`** as the last
  validation step. The step's arrows are applied to the reactants with
  `ksmech.arrows.apply.apply_arrows`, and the result must have the products' canonical Kekulé
  SMILES or their resonance key (`ksmech.arrows.compare.match_products`, shared with the
  round-trip report). Otherwise the step is rejected as **`ARROWS_MISMATCH`**. The detail is the
  edit's reason code and message (`VALENCE: ...`), or `MISMATCH` plus the properties that differ
  (`MISMATCH bond:C-O;charge:O`).
- Not checked:
  - steps without arrows (ReactMech, `require_arrows=False`): nothing to replay;
  - steps with radical arrows: replay is not implemented yet, so the check is logged and
    skipped. No source with radical arrows is ingested yet.
- `ksmech.arrows.compare` imports only RDKit, `ksmech.arrows.apply` and `ksmech.data`, so the
  data layer does not pull in the token or QM stack.
- **PMechDB re-ingested.**
  - 13,104 records: 13,090 accepted, 14 rejected (SANITIZE_FAILED 12, MAP_NOT_BIJECTIVE 1,
    ARROWS_MISMATCH 1).
  - The new rejection is pmechdb-9946a45b6a5d6f2e. Its product drops the dative bond
    `[Na+]<-[C-:20]` without an arrow.
  - Every other step is unchanged, compared row by row with the previous `steps.parquet`.
  - 147 resonance steps, so 12,943 training steps.
  - `stats.json` regenerated. The ingest took 27 s.
- **Arrow outputs regenerated.**
  - Round trip: 12,943 / 12,943 (12,942 exact, 1 modulo resonance).
  - Alignment and predicted-arrow round trip unchanged: 539 tokenized steps, 93.4% of arrows
    aligned, 462 / 467 = 98.9%. The removed step was not tokenized; MISSING_TOKENS is now
    12,404.
  - Samples rebuilt because the random split depends on the full step set: 467 samples,
    random 369 / 49 / 49, scaffold 431 / 17 / 19.

## 2026-10-20 — Physics module and the untrained-KS gate (Prompt 5)

`ksmech-physics ks-rank | contact-mlp`; reports in `data/processed/<source>/`
(`ks_rank_report[_<kernel>].json`).

**The gate failed.** Untrained KS3 puts a labeled pair in the top 5 for 8.2% of the
PMechDB test steps (bar: 50%). As the brief asks for that case, this step stops at the
diagnosis. Not built: the logistic regression, the gradient-boosting ranker
(`physics/baseline.py` holds only the evaluation and the untrained scorers) and the textbook
suite. They wait for a decision on the contact amplitude (below).

### Constants and the resonance integral

- **K = 1.75, eps = 4.0** (`KSParams`): the physical initialization. They stay parameters
  for Prompt 6. Every pair feature is a fixed power of them (`SCALING`). The samples store z
  at K = eps = 1 (`REDUCED`), and `baseline.physical_pair_z` applies the fixed physical
  values.
- **`ks_terms.resonance_integral(S_ab, H_aa, H_bb, K)`** is Wolfsberg-Helmholz:
  `K S_ab (H_aa + H_bb) / 2`. `wolfsberg_helmholz` moved from `params` to `ks_terms` and is
  built on it.
- **H_aa is the diagonal of the converged GFN2 Fock matrix**, the self-consistent GFN2
  Hamiltonian, not tblite's core `H_core`. Reasons:
  - the token files store S and F only;
  - the diagonal of F carries the charge shift that separates an anion's lone pair from a
    neutral one, which matters for ions;
  - H_core would need a token schema bump and a rebuild.

### The three amendments

- **(a) KS3 from the contact kernel for every pair** (as D1). The two intramolecular extras:
  - the centroid distance, in the `distance` entry;
  - the cross-centre coupling `l_r^T B l_s`, now **restricted to the AO blocks between the
    donor's centres A_r and the acceptor's centres A_s**. `coupling_sq` is its square.

  This reverts the spec-v3 choice of full vectors. Measured values:
  - 4-chlorobutanol, extended conformer: 3.6e-4 eV restricted, against 0.016 eV with full
    vectors;
  - a sigma / sigma* pair of one bond: above 0.1 eV, and over 100x the through-space pair
    (tested).
- **(b) G from the full spectral vectors**: unchanged since the 3b follow-ups. **G > 0 is
  kept signed**: no labeled first-arrow pair of the 467 samples has G > 0.
- **(c) K and eps** stay fixed in the baseline (above).

### Validity filters (all read from descriptors)

- **Donor:** occupation > 1.5 and outside the fractional band, so n >= 1.8.
- **Acceptor:**
  - occupation < 0.5 and outside the band, so n <= 0.2;
  - not d-type (the main centre's dominant shell is d on an element without a valence d
    shell) unless `allow_d_acceptors`;
  - not the donor's antibonding partner.
- **Partner rule, now with signs:** the same two leading centres, **plus** a positive overlap
  population on the donor and a negative one on the acceptor.
- **The rule applies only within one fragment.** Centre indices are per molecule. The first
  `valid_pairs` compared them across fragments and dropped 8 of the 117 methanol +
  chloromethane candidates (a C-H sigma of one molecule against a C-H sigma* of the other).
  A test caught it.
- The alignment keeps its own source-bond exclusion, so no label changes.

### Contact kernels

- **table** (default): the construction of D2.
  - Samples: non-bonded pairs beyond 1.3x the covalent-radii sum, within +-0.3 Angstrom of
    the vdW sum, value |l_r,a^T S_ab l_s,b|.
  - Median per (Z, l, Z, l); backoff (Z, Z) -> (l, l) -> global, at >= 5 samples per level.
  - Two tables, saved as JSON.
  - Lookups are now vectorized: one per distinct key pair, counted once per token pair.
- **mlp** (`contact_mlp.py`):
  - Input: the two tokens' 23 continuous descriptors (no element, no type).
  - Target: log of the same samples, floored at 1e-7.
  - Network: 2 x 64 SiLU, Adam 2e-3 with cosine decay, 20 epochs, batch 4096, seed 0; one
    network per kind.
  - Its output replaces S[kappa] in a = S w_m(r) w_m(s). Weights are saved as npz and
    evaluated in numpy.
  - Held-out molecules (10%, split by molecule), PMechDB 2,001 molecules:

    | kind | R2 of log, MLP / table | median abs log10 error, MLP / table |
    |---|---|---|
    | occupied-virtual | 0.459 / 0.412 | 0.289 / 0.253 |
    | occupied-occupied | 0.163 / 0.081 | 0.362 / 0.347 |

    Both predict a contact overlap poorly: they summarize contacts at whatever orientation
    the conformers happen to have.
- **oriented** (Phase 5b, stub): `contact.oriented_contact_overlap` and
  `ContactConfig(kernel="oriented")` raise NotImplementedError. It is the overlap of the two
  LMOs themselves at vdW contact, main lobes turned toward each other, evaluated with GFN2
  overlap integrals of the joint geometry: the long-term replacement for table and mlp.
- **analytic**: the existing tblite two-atom probe, unchanged.

### Pair features in the samples (`MechanismSample` version 2)

- New fields:
  - `pair_z` (n_occ, n_virt, 7), float32, at K = eps = 1;
  - `pair_defined`;
  - `pair_valid`;
  - `atom_coulomb` (A, A), at eps = 1.

  `docs/data_schema.md` lists them, from a real file.
- How the brief's entries are read:
  - **ks2** is between the two main centres;
  - **ks1** ("proxy for the donor's neighbourhood") is read as the repulsion the donor meets:
    the donor against the occupied tokens whose main centre lies within 3.5 Angstrom of the
    acceptor's main centre. Within a fragment it has the WH form; across fragments the
    contact form `-(N + N) beta_c a %Vbur %Vbur / 100^2`;
  - **distance** is the centroid distance within a fragment and the vdW contact of the main
    centres across fragments.
- About 95 KB per sample (21 KB before). The 467 samples rebuild in 61 s, and the splits are
  unchanged (random 369 / 49 / 49, scaffold 431 / 17 / 19).
- `ksmech-arrows samples` refuses to run without a contact table (exit 2).
- `collate_samples` does not batch the pair block yet. Prompt 6 chooses that layout.

### The gate: untrained ranking of the first arrow (467 samples, 2,001 molecules tokenized)

Setup:
- The candidates are the `pair_valid` pairs: median 746 per test step, 831 per training
  step, up to 16,509.
- Multi-label marginalized top-k, ties counted against the label.
- Leave-one-class-out cannot run on PMechDB (no classes).
- The pure scores need no training, so all 467 samples are reported next to the 49 test
  steps.

| scorer (kernel) | test top-1 / 3 / 5 | all top-1 / 3 / 5 | median rank, all |
|---|---|---|---|
| KS3 (table) | 0.0 / 8.2 / **8.2%** | 0.2 / 2.4 / 3.6% | 288 |
| KS3 + KS2 (table) | 4.1 / 4.1 / 18.4% | 8.1 / 10.1 / 12.8% | 66 |
| KS3 (mlp) | 4.1 / 10.2 / 12.2% | 1.5 / 5.6 / 9.4% | 256 |
| KS3 (analytic) | 12.2 / 20.4 / 28.6% | 9.6 / 16.1 / 19.1% | 38 |
| KS3 + KS2 (analytic) | 12.2 / 20.4 / 30.6% | 9.9 / 16.3 / 19.5% | 37 |
| G alone (diagnostic) | 30.6 / 42.9 / 44.9% | 29.3 / 38.1 / 43.7% | 9 |
| KS2 alone (diagnostic) | 0.0 / 0.0 / 4.1% | 1.5 / 2.4 / 4.3% | 66 |

At eps = 4, KS2 is about two orders of magnitude above KS3 over the candidates: median
|KS2| 5.2e-3 eV (p95 0.11), median |KS3| 6.8e-5 eV (p95 1.5e-3). KS3 + KS2 therefore ranks
almost by KS2.

**The amplitude is the failing factor.** KS3 = (K a Fbar)^2 G, ablated on the table kernel
(top-5, test / all):

| score | top-5 |
|---|---|
| G | 44.9 / 43.7% |
| Fbar^2 G | 46.9 / 42.2% |
| a^2 G | 8.2 / 6.0% |
| KS3 | 8.2 / 3.6% |

The Wolfsberg-Helmholz energy factor is harmless; the contact amplitude a^2 is what destroys
the ranking.

### Failure classes (KS3, table; 450 of 467 miss the top 5)

Diagnostic labels describe the pairs; they are never input.

- **The pair ranked first instead:**
  - its acceptor is on H in 90.9% of the misses;
  - its donor's main centre is s-rich in 81.1%, and the donor lies below -15 eV in 68.0%;
  - the most frequent such pairs: lone_pair(O) -> sigma*(H-O) 69, sigma(H-C) -> sigma*(H-C)
    59 (two C-H bonds of one carbon), lone_pair(O) -> sigma*(H-C) 48,
    lone_pair(Br) -> sigma*(H-Br) 34, lone_pair(Br) -> sigma*(H-C) 29,
    pi(N-C) -> sigma*(H-C) 27.
- **Labeled pairs that never reach the top 5:**

  | labeled pair | samples |
  |---|---|
  | lone_pair(O) -> sigma*(C-O) (tetrahedral collapse) | 46 |
  | lone_pair(O) -> delocalized pi*(C-O) | 43 |
  | pi(C=C) -> pi*(C=O) | 42 |
  | pi(C=C) -> pi*(C=N) | 32 |
  | pi(C=C) -> sigma*(H-Br) | 22 |
  | lone_pair(O) -> pi*(C=O) | 20 |
  | pi(C=C) -> pi*(C=C) | 18 |
  | pi(C=C) -> sigma*(Br-Br) | 11 |
  | pi(C=C) -> sigma*(C-O) | 11 |
  | pi(C=C) -> sigma*(C-Br) | 10 |

  Even proton transfers do poorly: lone_pair(O) -> sigma*(H-O) hits 36% (n 25).
  48.4% of the missed labels are intramolecular.
- **Root cause: the contact overlap is biased toward H 1s and s shells.** Table medians:

  | key | median |
  |---|---|
  | (Cl p, H s) | 0.0165 |
  | (Cl s, H s) | 0.0145 |
  | (O p, H s) | 0.0116 |
  | (C p, H s) | 0.0111 |
  | (O s, H s) | 0.0101 |
  | (O p, C p) | 0.0017 |
  | (C p, C p) | 0.0013 |
  | (N p, C p) | 0.0013 |

  So beta^2 is 50-100x larger for any pair that touches H, which G (0.06 vs 0.3-0.8 1/eV for
  the labeled pi*) cannot compensate. There are two reasons:
  - the vdW contact distance with H is about 0.5 Angstrom shorter;
  - an s function has no angular factor, while a p lobe at a random conformer orientation
    mostly points elsewhere.

  The deep s-rich lone pairs on top come from the same s-shell keys, not from Fbar.
  **No code error was found**: the table measures what it was specified to measure.
  Contacts at van der Waals distance and random orientation are not the overlap of a reactive
  approach. The MLP (top acceptor on H in 87% of misses) and the analytic probe (82%) show
  the same pattern.
- **G alone (44.9%) fails the other way.** It ranks the lowest pi* first:
  - 40% of its misses have an H acceptor: proton transfers, whose sigma*(X-H) lie high;
  - the sigma*(C-O) leaving-group steps miss too;
  - intermolecular labels hit 48.0%, intramolecular 38.8%.

**Open (user decision):** the contact amplitude. Options include the Phase 5b oriented
overlap; overlaps at a reactive distance rather than vdW; a kernel conditioned on
orientation; and dropping the amplitude from the untrained score.

## 2026-10-20 — Oriented contact kernel (Phase 5b brought forward), baselines, textbook suite

User decision: the analytic oriented overlap becomes the default contact kernel. The
vdW-contact table and the descriptor MLP stay as ablation baselines only. Gate: the full KS3
product must beat F^2 G alone at some contact distance d.

**It does not, at any d.** As instructed, the kernel work stops here with the per-class table
below, and **F^2 G is the untrained score** (`ksmech-physics textbook --untrained fbar2_g` is
the default). The baselines and the textbook suite were built on that basis.

### Why the vdW-contact table fails (recorded at the user's request)

- The table medians come from intramolecular contacts at van der Waals distance, at whatever
  orientation the conformers happen to have.
- Two effects favour H and s shells systematically:
  - a contact with H is about 0.5 Angstrom shorter;
  - an s function has no angular factor, while a p lobe at random orientation mostly points
    elsewhere.
- Result: keys with H 1s or an s shell have median overlaps 0.010-0.017, heavy-atom p-p keys
  0.0013-0.0017 (entry above).
- Such contacts anticorrelate with a reactive approach. a^2 G ranked the first arrow worse
  than G alone: 8.2 vs 44.9% top-5 on the test split.

### The GFN2 basis in numpy (`ksmech.physics.gto`)

- **Shells:** Slater exponents, principal quantum numbers and primitive counts come from the
  GFN2 parametrization tblite itself exports (`tblite.library.export_gfn2_param` ->
  `ksmech.qm.ao.gfn2_shells`). Nothing is copied by hand.
- **STO-nG expansions:** Stewart's tables as tblite ships them.
  - Extracted, numbers only, from tblite v0.7.0 `src/tblite/basis/slater.f90`; sha256 8a3df2db...
    is recorded in `physics/data/sto_ng.json`.
  - Fetched from the tblite GitHub repository at the installed version's tag; the file is
    identical at v0.4.0.
  - tblite's normalization: (2a/pi)^(3/4) (4a)^(l/2) / sqrt((2l-1)!!) per primitive, no
    renormalization of the contraction.
- **Component order** as in tblite's `lx` table and `dtrafo`:
  - p is (y, z, x);
  - d is (xy, yz, z2, xz, x2-y2), built from (xx, yy, zz, xy, xz, yz).
- **Overlaps:** Gaussian product rule with the Obara-Saika 1-D recursion, vectorized over
  primitive pairs and centre pairs.
- **Checks:**
  - the analytic S equals tblite's stored S to <= 4e-10 for every element in the token cache
    (51 records, 24 elements, Ti / Cu / Fe / Sc included);
  - the brief's test: H 1s - C 2pz at 1.09 Angstrom gives 0.44731, against 0.44654 for the
    methane record's C-H at its own 1.0922 Angstrom (|diff| 7.7e-4 < 1e-3). At the record's
    exact bond length the two agree to 1e-8.

### AO layout bug found and fixed (QM layer)

- **The bug:** `ksmech.qm.ao` assumed s, p, d on every atom. tblite orders shells per element,
  and for Sc-Cu, Y-Ag and La-Au the d shell comes first (Ti: 3d, 4s, 4p). The matrices were
  always right; the `ao_l` labels were wrong on those atoms.
- **What it affected:** 22 cached molecules (Ti 16, Cu 4, Fe 1, Sc 1). Their s/p/d shares,
  hence the dominant-shell keys and the d-type rule on Ti and Cu, were mislabelled. The earlier
  "52 one-centre Ti / Cu d orbitals that are now acceptors" count was made with wrong labels.
- **The fix:**
  - `ao_layout` now reads tblite's shell list;
  - `records.load_record` recomputes the AO map from the atomic numbers and logs any stored
    map that differs, so no QM cache file was rewritten;
  - the 22 token files were rebuilt;
  - tests: `tests/test_qm_ao.py`.

### The oriented kernel (`ksmech.physics.oriented`, `ContactConfig(kernel="oriented")`)

- **Lobe direction:** sample 200 Fibonacci directions on a 1.5 Angstrom sphere around the main
  centre and take the argmax of |amplitude|.
  - An s-dominated token instead points away from its bonded neighbours, weighted by its
    Mulliken population on each.
  - An antibonding acceptor takes the largest lobe in the hemisphere away from its bond
    partner.
  - **Deviation from the brief:** the amplitude is that of the LMO's part on the main centre
    (its AO block there), not of the whole LMO. With the whole LMO, a neighbour on or inside
    the 1.5 Angstrom sphere wins: the LMO's small tail coefficient on it times its AO values
    near its nucleus. Acetone's pi* then pointed 9 deg off a methyl carbon, acetyl chloride's
    sigma*(C-Cl) 18 deg off the oxygen, acrolein's C=C pi along the C-C bond.
  - With the main-centre part, as the brief's "larger lobe on the main centre" describes:
    - sigma*(C-Cl): 173-175 deg from Cl;
    - carbonyl pi*: 93 deg from O, with a 98-100 deg tilt away from O in acrolein / acetyl
      chloride;
    - pi donors and p-type lone pairs: perpendicular;
    - H-centred sigma*: 180 deg from the partner.
- **Placement:** the smallest rotation makes the acceptor lobe antiparallel to the donor lobe,
  and the acceptor's main centre goes to distance d along the donor lobe. An intramolecular
  pair uses a copy of the molecule.
- **Overlap:** S_rs = l_r^T S l_s over every AO pair of the two placed fragments.
  - Each atom pair is evaluated in its local frame, where a shell-pair block is diagonal in
    sigma / pi / delta. Those radial functions are the analytic integrals on a 0.005 bohr grid
    to 30 bohr, splined; atom-pair blocks reproduce the direct integrals to <= 9e-12.
  - Coefficients are rotated with exact real-harmonic rotations (checked against AO values on
    rotated points to 3e-16).
  - Atom pairs whose coefficient-norm x envelope bound is below 1e-9 are skipped (|change|
    <= 3e-9).
  - A whole S_rs matches a brute-force contraction over directly computed blocks to 2e-13.
- **Coupling:** beta = K |S_rs| (F_rr + F_ss) / 2 with the LMO Fock diagonals E_r, E_s;
  KS3 = beta^2 G. The ablation kernels keep their AO-shell Fock mean.
- **Caching:** blocks per (molecule, molecule, occ_virt | occ_occ, d, tol) in memory and as npz
  under `data/cache/oriented/`. The CLIs enable the disk cache; library calls use only the
  memory cache unless given a `cache_dir`.
- **Cost:** about 0.5 ms per pair. One distance over the 869,163 occupied x virtual pairs of
  the 467 samples takes about 13 minutes on one core; cached re-runs take a few minutes.

### The d sweep and the gate (467 samples; test = random_test, 49 steps)

Top-5, test / all:

| d (Angstrom) | KS3 full product | F^2 G | KS3 + KS2 | S^2 | S^2 G |
|---|---|---|---|---|---|
| 2.0 | 10.2 / 7.5 | 44.9 / 39.0 | 12.2 / 8.8 | 2.0 / 0.9 | 6.1 / 7.3 |
| 2.3 | 4.1 / 5.1 | 44.9 / 39.0 | 6.1 / 6.2 | 6.1 / 3.4 | 4.1 / 5.8 |
| 2.6 | 2.0 / 1.7 | 44.9 / 39.0 | 4.1 / 3.6 | 0.0 / 0.2 | 4.1 / 2.6 |
| 3.0 | 4.1 / 5.1 | 44.9 / 39.0 | 12.2 / 13.1 | 2.0 / 3.6 | 2.0 / 4.9 |

- For reference (all on the Ti/Cu-corrected tokens), KS3 top-5 with the ablation kernels:
  table 8.2 / 3.6, MLP 12.2 / 9.9, analytic two-atom probe 28.6 / 19.1. G alone: 44.9 / 43.7.
  The analytic probe (|s| and p-norm per centre, covalent distance, best centre pair) is the
  least harmful amplitude, but it still trails G.
- The stored sample features use d = 2.0, the best KS3 on train + val (30 / 418 hits, against
  22, 7, 22 at the other distances).

**Root cause: the donor lobe rule places the acceptor inside the donor's own bond.**
- A two-centre bond orbital's largest lobe points at its bond partner.
- At d = 2.0 the placed acceptor's main centre lies a median 0.49 Angstrom from another donor
  atom for the top-ranked pairs (under 1 Angstrom in 96%), against 0.70 Angstrom for the
  labeled pairs. The overlap then measures the clash with the partner atom: median |S| 0.31,
  against 0.19 for the labeled pair.
- 97% of the top-ranked donors have two or more centres. The pairs ranked first in the 432
  misses: sigma(C-C) -> pi*(C-O) 101, sigma(C-C) -> pi*(C-C) 63, sigma(C-C) ->
  delocalized pi*(C-O) 45, sigma(C-C) -> pi*(C-N) 38, sigma(C-Si) -> pi*(C-O) 24.
- The H and s-shell bias of the table is gone (an H acceptor ranks first in 0.5% of the
  misses), but the amplitude is still anti-informative: S^2 alone is at or near 0% in every
  class.

**Per-class top-5 (d = 2.0, all 467 samples).** A class is named by the first arrow's teacher
pair (`ranking.teacher_cell`, the set members with the largest weight on their sites);
diagnostic labels only.

| labeled pair | n | KS3 | F^2 G | S^2 G | S^2 | G |
|---|---|---|---|---|---|---|
| lone_pair(O) -> sigma*(C-O) | 50 | 0% | 6% | 0% | 0% | 12% |
| pi(C-C) -> pi*(C-O) | 42 | 17% | 90% | 19% | 0% | 90% |
| lone_pair(O) -> delocalized pi*(C-O) | 40 | 8% | 18% | 22% | 0% | 18% |
| pi(C-C) -> pi*(C-N) | 31 | 0% | 100% | 3% | 0% | 100% |
| lone_pair(O) -> sigma*(H-O) | 25 | 0% | 0% | 0% | 0% | 0% |
| lone_pair(O) -> pi*(C-O) | 20 | 0% | 100% | 0% | 0% | 100% |
| pi(C-C) -> sigma*(C-O) | 19 | 0% | 21% | 0% | 0% | 26% |
| pi(C-C) -> pi*(C-C) | 14 | 14% | 93% | 21% | 0% | 93% |
| sigma(C-C) -> sigma*(H-Br) | 11 | 0% | 0% | 0% | 0% | 0% |
| pi(C-C) -> sigma*(H-Br) | 11 | 0% | 0% | 0% | 0% | 0% |
| lone_pair(N) -> pi*(C-O) | 10 | 30% | 100% | 20% | 0% | 100% |
| pi(C-C) -> sigma*(C-Br) | 10 | 20% | 30% | 10% | 0% | 90% |
| sigma(C-H) -> pi*(C-C) | 9 | 0% | 22% | 0% | 0% | 33% |
| lone_pair(O) -> sigma*(H-N) | 8 | 0% | 0% | 0% | 0% | 0% |
| lone_pair(N) -> sigma*(H-O) | 8 | 0% | 0% | 0% | 0% | 0% |
| lone_pair(O) -> pi*(C-C) | 7 | 0% | 86% | 0% | 0% | 100% |
| pi(C-C) -> sigma*(Br-Br) | 7 | 29% | 86% | 29% | 14% | 86% |

- **F^2 G (and G) works for pi -> pi* and n -> pi*** (86-100%).
- **It fails on proton transfers** (every sigma*(X-H) class 0%), on O(-) -> sigma*(C-O)
  leaving-group collapse (6%) and on alkene -> sigma*(C-O / C-Br) (21-30%). For these the
  lowest-gap pi* of the reactant set ranks first: the F^2 G misses put lone_pair(O) -> pi*(C-O)
  39, -> pi*(C-C) 22 on top.
- With LMO energies, F^2 G is not better than G (39.0 vs 43.7% on all samples).

**Open (user decision):** the donor lobe of a bond orbital. Options include:
- its back lobe, away from the partner, as for the acceptor;
- a steric exclusion of placements within ~1.5 Angstrom of donor atoms;
- the overlap restricted to the donor's main-centre AOs;
- the maximum over a set of approach directions.

### Requested records

- **K = 1.75** (Wolfsberg-Helmholz) and **eps = 4.0**: parameters with that initialization,
  fixed in every untrained score and baseline (`KSParams`, `SCALING`). The samples store z at
  K = eps = 1.
- **Contact-overlap construction and backoff** (now ablation only): non-bonded pairs beyond
  1.3x the covalent-radii sum and within +-0.3 Angstrom of the vdW sum; |l_r,a^T S_ab l_s,b|;
  median per (Z, l, Z, l), backoff (Z, Z) -> (l, l) -> global at >= 5 samples; separate
  occupied-virtual and occupied-occupied tables.
  - Rebuilt after the Ti/Cu token fix: 352,990 / 352,820 samples, global medians 0.0042 /
    0.0007.
  - The MLP refit is unchanged: held-out R2 of log 0.459 vs table 0.412.
- **Validity filters** (all from descriptors):
  - donor: occupation >= 1.8;
  - acceptor: occupation <= 0.2, and not d-type (dominant d share on the main centre of an
    element without a valence d shell) unless allowed;
  - not the donor's own antibond: same two leading centres, donor overlap population > 0,
    acceptor < 0, same fragment.

### Baselines (`ksmech-physics baseline`, `physics/baseline.py`)

- **Features (57 per candidate pair, no type one-hot):**
  - the 7 KS pair features at the physical K and eps (oriented kernel, d = 2.0);
  - the 23 continuous descriptors of donor and of acceptor;
  - frozen-orbital f- on the donor's main centre and f+ on the acceptor's;
  - the LMO energies E_r, E_s.
- **Targets:** the positives are the first arrow's labeled pairs. The other arrows' labeled
  pairs are left out of training, never out of the evaluation.
- **(a) Logistic regression:** pointwise, on asinh-scaled and standardized features, each step
  weighted equally, L2 picked on validation top-5 from 0.01-10 (0.01 on random, 0.1 on
  scaffold), L-BFGS.
- **(b) GBDT ranker:** numpy histogram trees.
  - 64 quantile bins, depth 4, Newton leaves with L2 1.0, shrinkage 0.1, half the rows per
    tree;
  - listwise loss -log sum_P softmax of each step, multi-label marginalized;
  - early stopping on validation top-5: 170 trees on random.
  - No new dependency.
- **Results** (top-1 / 3 / 5 on the test split, multi-label marginalized, ties counted against
  the label):

  | scorer | random_test (49) | scaffold_test (19) |
  |---|---|---|
  | untrained KS3 (oriented, d = 2.0) | 2.0 / 6.1 / 10.2 | 0.0 / 0.0 / 10.5 |
  | untrained F^2 G (the untrained score) | 26.5 / 38.8 / 44.9 | 21.1 / 31.6 / 36.8 |
  | untrained G | 30.6 / 42.9 / 44.9 | 15.8 / 31.6 / 36.8 |
  | logistic regression | 36.7 / 53.1 / 61.2 | 36.8 / 47.4 / 57.9 |
  | GBDT ranker | **71.4 / 89.8 / 91.8** | **57.9 / 68.4 / 78.9** |

- Leave-one-class-out does not apply to PMechDB (no mechanism classes); the CLI reports it.
- Training takes about 7 minutes on 2 threads (617,712 training candidate pairs).
- The test splits are small (49 and 19 steps), so one step is 2.0 or 5.3 points.

### Textbook suite (`ksmech-physics textbook`, `physics/textbook.py`)

- **Setup:**
  - five atom-mapped reactant sets;
  - every valid pair, within and across molecules, is scored;
  - a token's site is its main centre;
  - expected atoms are matched up to graph symmetry (`CanonicalRankAtoms`, no tie breaking).
- **Molecules:** the 4 that had no token file (C[O-], CCBr, [C-]#N, [OH3+]) got GFN2 records
  in the QM cache, with the dataset's method, and token files.
- **Results:**

  | case | untrained F^2 G | GBDT |
  |---|---|---|
  | MeO(-) + MeCl: O -> C | rank 1, top 3 | rank 0 |
  | MeO(-) + EtBr: SN2 O -> C(Br) | rank 1, top 3; SN2 13.8 vs E2 2.16 (rank 95) | rank 0; SN2 5.49 vs E2 1.22 (rank 3) |
  | CN(-) + acrolein: 1,4 C -> beta C | rank 23, miss; 1,4 23.8 vs 1,2 17.1 | rank 6, miss; 1,4 4.33 vs 1,2 4.29 |
  | MeNH2 + AcCl: N -> carbonyl C | rank 0 | rank 0 |
  | H3O(+) + acetone: O -> H | rank 14, miss | rank 0 |
  | **expected pair in the top 3** | **3 / 5** | **4 / 5** |

- **The untrained score misses** cyanide (its top pairs are the nitrile pi -> pi*(C=O)) and
  the protonation (acetone's own n -> pi*(C=O) ranks first), the two classes the per-class
  table already shows.
- **The GBDT misses** only cyanide. Its top pair is pi(C-N) -> pi*(C=C) at the beta carbon,
  i.e. the 1,4 acceptor, but with the pi(C-N) token as donor, whose main centre is N.
  - The alignment's convention would count it, since the donor set of an arrow from C holds
    every donor token on C.
  - The main-centre site definition does not count it.
- **In both scores** SN2 outranks E2 and 1,4 outranks 1,2.

## 2026-10-20 — Site-based lobe placement, clash guard, shared-centre filter (user decisions)

The user's five decisions after the first oriented sweep, and the re-run of the sweep.

**Outcome:** the full product still loses to F^2 G on train + val at every d. Per the
instruction, **F^2 G + KS2 is the untrained score**, and the amplitude moves to Prompt 6 as a
learnable pair feature rather than a fixed factor.

### Lobes at the site atom (`oriented.site_candidates`)

- **Site, not main centre.** The lobe of token r is evaluated at a site atom a, one of its
  centres. Only a's AO part of the LMO is used (the earlier deviation, kept).
- **Hemisphere rule.** A direction n on the 1.5 Angstrom sphere is admissible only if
  n . (x_c - x_a) <= 0 for every other centre c of r. A one-centre token uses the hemisphere
  away from a's bonded neighbours, weighted by the token's Mulliken population on each. When
  no sampled direction satisfies every centre (a site between other centres), the
  least-violating directions within 0.05 are used.
- **Ties.** Candidates are local maxima of |amplitude| over the admissible directions, using 6
  nearest neighbours on the 200-point sphere. Neighbouring points of one lobe also lie within
  5% (they are 14 deg apart), so a tie is a second *local maximum* within 5%: the two faces of
  a pi.
- **s functions.** A site whose AO part has no direction (max/min amplitude below 1.05: an H
  1s) points away from the other centres, weighted, or from the weighted neighbours. That is
  how sigma(C-H) at H points out beyond H.
- **Checked on chloromethane, acetone and acrolein:**
  - sigma bonds and sigma* at either centre: 172-180 deg from the partner;
  - sigma(C-H) at H: 180 deg;
  - pi / pi* and p-type lone pairs: 83-99 deg, two tied faces;
  - s-rich lone pairs: 173-176 deg from the neighbour.

### Clash guard

- A placement clashes when any acceptor-fragment atom lies within 1.5 Angstrom of a
  donor-fragment atom. An intramolecular pair is placed against a copy of the molecule.
- **Order of tries per site pair:**
  1. every combination of the two tokens' tied lobes; the largest |S| among the clash-free
     placements is kept;
  2. if all clash: the opposite face of the first lobe on either side;
  3. if those clash too: the smallest |S| found over both tiers, flagged CLASH.
- **The pair-track value** is the maximum over the donor's centres x the acceptor's centres.
  A clash-free site pair always wins over a CLASH one.
- **CLASH shares at d = 2.3** (576 samples, valid pairs):
  - 55.1% of the intramolecular pairs and 35.1% of the intermolecular ones;
  - 25.6% of the first arrows' teacher pairs;
  - 11.1% of the pairs KS3 ranks first.

  The shares fall with d: 59.9 / 40.2% at 2.0, 37.5 / 19.0% at 3.0.

### Site-resolved value and the stored diagnostics

- `oriented.site_contact(t_F, r, a, t_G, s, b)` returns (|S|, CLASH) for known sites: the form
  for labels, the decoder's acceptor step and applying an arrow.
- The samples store the argmax site pair of the pair-track value as `pair_contact_sites`
  (global atoms) and its flag as `pair_clash`. Sample version 3.
- Tests:
  - the pair value equals the best `site_contact`;
  - an explicit placement equals a brute-force contraction to 1e-9;
  - with every placement clashing, every pair is flagged and its value is at most the
    clash-free one.

### Shared-centre validity filter

- **Rule:** a donor-acceptor pair of one fragment whose centre sets intersect is not a
  candidate (`tokens.shares_centres`, also in `acceptor_candidates(donor=...)`). It replaces
  the own-antibond rule, which it contains.
- **Tests:** acetone n -> pi*(C=O) is excluded; a 1,2-hydride shift sigma(C1-H) -> p(C2) in the
  isopropyl cation is kept.
- **Effect:**
  - 10.7% of the donor x acceptor pairs are excluded (the own-antibond rule removed about 1%),
    leaving a median of 640 candidates per step;
  - 10 of the 576 steps lose every labeled pair and count as misses.

### Cost and caching

- **Exact rewrites:**
  - shell coefficients are rotated directly (p: index permutation and a 3 x 3 product; d:
    M C M^T), instead of building rotation matrices per atom pair;
  - `rotation_to` is vectorized.
  - Together these make a block about 2x faster.
- **Screening tolerance 1e-9 -> 1e-6.** On 1,370 pairs of 6 blocks: |dS| <= 1.3e-5, 99th
  percentile relative change 4e-4, no argmax site changed; 1.5x faster.
- **Time per distance on one core:**
  - all 946 occupied x virtual molecule-pair blocks of the samples: 59-73 minutes;
  - the 516-732 inter-fragment occupied x occupied blocks: 6-9 minutes.
- **Cache:** the blocks were pre-computed per d in parallel (four single-threaded processes),
  so the sample builds and reports read the cache. `data/cache/oriented/` holds 7,561 blocks,
  123 MB. Blocks of earlier versions were deleted.

### More samples

- The textbook suite tokenized C[O-], CCBr, [C-]#N and [OH3+], which completes 109 more
  PMechDB steps.
- **Samples: 576** (random 454 / 56 / 66, scaffold 533 / 23 / 20), against 467 before.
- **Alignment refreshed:** 648 tokenized steps, 94.6% of arrows aligned; predicted-arrow round
  trip 569 / 576 = 98.8%.
- Every number below is on the 576 samples.

### The re-run sweep and the gate (train + val = 510 samples; test = 66)

Top-5, train + val (test):

| d | KS3 | F^2 G | F^2 G + KS2 | KS3 + KS2 | S^2 | S^2 G | G |
|---|---|---|---|---|---|---|---|
| 2.0 | 29.0 (36.4) | 35.1 (39.4) | 35.3 (39.4) | 39.6 (43.9) | 12.4 | 35.1 | 40.0 (42.4) |
| 2.3 | 29.4 (34.8) | 35.1 (39.4) | 35.3 (39.4) | **44.9 (47.0)** | 10.6 | 34.9 | 40.0 (42.4) |
| 2.6 | 2.5 (1.5) | 35.1 (39.4) | 35.3 (39.4) | 6.7 (6.1) | 2.4 | 3.7 | 40.0 (42.4) |
| 3.0 | 2.9 (3.0) | 35.1 (39.4) | 35.3 (39.4) | 20.0 (19.7) | 3.3 | 4.5 | 40.0 (42.4) |

- **The gate fails at every d.** The full product (KS3) never beats F^2 G on train + val:
  29.4% at best, against 35.1%.
- **Untrained score: F^2 G + KS2** (`ksmech-physics textbook` default).
- **Stored features:** d = 2.3, the best KS3 on train + val.
- **Site placement fixed the bond-clash artefact at short d.** KS3 top-5 at d = 2.0 rose from
  7.5% (all 467 samples, main-centre rules) to 29.0% (train + val of the 576).
- **Collapse above 2.3 Angstrom:** the H-site placements of sigma(C-H) stop clashing and win.
  The pairs ranked first in the KS3 misses at 2.6 are sigma(C-H) -> pi*(C-O) 115,
  sigma(C-H) -> pi*(C-C) 79, lone_pair(Br) -> sigma*(H-C) 73.
- **Recorded for the Prompt 6 decision:** KS3 + KS2 at d = 2.3 reaches 44.9% on train + val,
  ahead of G (40.0%) and of the untrained score (35.3%). The amplitude carries signal once it
  is combined with the Coulomb term. It is not the gate's criterion, so it was not adopted.

**Per-class top-5 at d = 2.3** (all 576; classes by the teacher pair; diagnostic labels).
Proton transfers are the ones to watch.

| labeled pair | n | KS3 | F^2 G | F^2 G + KS2 | KS3 + KS2 | S^2 G | G |
|---|---|---|---|---|---|---|---|
| lone_pair(O) -> sigma*(H-O) | 57 | 7% | 0% | 0% | 39% | 23% | 4% |
| lone_pair(O) -> sigma*(C-O) | 51 | 8% | 8% | 10% | 63% | 8% | 12% |
| pi(C-C) -> pi*(C-O) | 42 | 79% | 90% | 90% | 71% | 69% | 90% |
| pi(C-C) -> pi*(C-N) | 31 | 100% | 100% | 100% | 100% | 100% | 100% |
| lone_pair(N) -> sigma*(H-O) | 27 | 4% | 0% | 0% | 37% | 41% | 4% |
| lone_pair(O) -> delocalized pi*(C-O) | 24 | 21% | 21% | 21% | 96% | 21% | 25% |
| lone_pair(O) -> pi*(C-O) | 23 | 96% | 100% | 100% | 87% | 87% | 100% |
| pi(C-C) -> sigma*(H-O) | 22 | 9% | 0% | 0% | 32% | 77% | 0% |
| pi(C-C) -> sigma*(C-O) | 19 | 16% | 26% | 26% | 16% | 16% | 32% |
| pi(C-C) -> pi*(C-C) | 14 | 93% | 93% | 93% | 86% | 93% | 93% |
| pi(C-C) -> sigma*(C-Br) | 13 | 0% | 31% | 31% | 0% | 15% | 92% |
| sigma(C-C) / pi(C-C) / lone_pair(O) -> sigma*(H-Br) | 29 | 0% | 0% | 0% | 0% | 0% | 0-14% |
| lone_pair(N) -> pi*(C-O) | 10 | 60% | 100% | 100% | 70% | 60% | 100% |
| lone_pair(O) -> sigma*(H-N) | 8 | 12% | 0% | 0% | 12% | 0% | 0% |
| lone_pair(N) -> sigma*(H-N) | 6 | 0% | 0% | 0% | 0% | 33% | 0% |

- **F^2 G + KS2 scores 0% on every proton-transfer class.** KS2 adds nothing measurable
  (35.3 vs 35.1%).
- **The oriented amplitude carries the proton-transfer signal** (S^2 G 23-77% on the
  sigma*(O-H) classes) and the leaving-group signal (KS3 + KS2 63% on O(-) -> sigma*(C-O)).
- **It hurts where G is already right:** pi -> pi*, and pi -> sigma*(C-Br) (G 92%, KS3 0%).
- **The HBr protonations** (sigma*(H-Br), 29 steps) are missed by every untrained score.

### Baselines and textbook suite on the 576 samples (features at d = 2.3)

Top-1 / 3 / 5 on the test split:

| scorer | random_test (66) | scaffold_test (20) |
|---|---|---|
| untrained F^2 G + KS2 (the untrained score) | 22.7 / 31.8 / 39.4 | 20.0 / 30.0 / 35.0 |
| untrained KS3 | 25.8 / 30.3 / 34.8 | 5.0 / 10.0 / 10.0 |
| untrained KS3 + KS2 | 27.3 / 40.9 / 47.0 | 0.0 / 20.0 / 20.0 |
| untrained G | 25.8 / 39.4 / 42.4 | 15.0 / 30.0 / 35.0 |
| logistic regression | 22.7 / 50.0 / 63.6 | 35.0 / 40.0 / 50.0 |
| GBDT ranker | **68.2 / 84.8 / 95.5** | **55.0 / 75.0 / 75.0** |

- Model choices: logistic regression L2 0.01 on random, 1.0 on scaffold; GBDT 200 and 160
  trees.
- Leave-one-class-out does not apply to PMechDB.
- **Textbook** (expected pair in the top 3):
  - **F^2 G + KS2: 3 / 5.** It misses cyanide + acrolein (rank 15) and H3O(+) + acetone
    (rank 10).
  - **GBDT: 4 / 5.** It misses cyanide (rank 4).
  - In both, SN2 outranks E2 (F^2 G + KS2: 13.8 vs 2.24; GBDT: 3.99 vs 1.43) and 1,4 outranks
    1,2 (23.7 vs 17.1; 3.26 vs 2.41).
- These supersede the 467-sample numbers of the previous entry.

## 2026-10-20 — KS3 + KS2 floor, local-cap placement, narrowed filter, Prompt 6 pair features (user decisions)

The user's decisions after the site-based sweep, and the three checks they asked for before
Prompt 6. This entry supersedes the placement ("an intramolecular pair is placed against a
copy of the molecule"), the shared-centre rule and the pair-feature list of the entries above.
**Superseded in part by the next entry:** the orbital-level beta, the main-centre filter, the
geometric intramolecular KS2, the standardized frozen head and the sample / cache versions.

### The floor and the frozen-head kernel

- **Adopted:** -(KS3 + KS2) at d = 2.3 Angstrom is the untrained physics floor
  (`physics.cli.UNTRAINED_SCORE = "ks3+ks2"`, also the `textbook` default). With KS1 it is
  the kernel of the frozen head.
- **The earlier gate compared unequal term sets.** It set KS3 alone (29.4% top-5, train + val)
  against F^2 G (35.1%), and then adopted F^2 G + KS2. With KS2 on both sides the comparison
  is KS3 + KS2 44.9% against F^2 G + KS2 35.3% (whole-copy placement, strict shared-centre
  filter). So the amplitude did carry signal; the gate missed it because only one side had the
  Coulomb term.
- `ks-rank` reports the floor (`report["floor"]`) instead of a gate and always exits 0. Its
  tags are `<split>_<kernel>_d<d>` (`ks_rank_report_random_oriented_d2.3.json`).

### Only the acceptor's local cap is placed (`oriented.cap_atoms`, `_Side.cap`)

- **Rule (user):** for every pair, intra- and intermolecular, only the acceptor's cap is
  placed: its centre atoms plus their first bonded neighbours, hydrogens included. The
  acceptor's LMO is truncated to the cap atoms' AOs. The donor fragment stays whole.
- **Why:** the whole-copy placement made CLASH an artefact. The copy of the molecule landed on
  the molecule itself, giving CLASH on 55% of the intramolecular pairs and on 26% of the
  labeled pairs.
- **Check on 200 random valid intermolecular pairs** (d = 2.3, seed 0):
  - median relative change of |S| 0.13% (target < 5%), 90th percentile 33.5%;
  - 47 pairs change by more than 5%, 21 of which clash under one of the two placements; over
    the 170 pairs clash-free under both, the median change is 0.06% and the 90th percentile
    10.2%;
  - CLASH 15.0% -> 6.0%.
- **The plain cap left 6.5% of the labeled pairs flagged CLASH** (target < 5%). The remaining
  contacts were donor-side hydrogens within 1.5 Angstrom of the placed cap, in placements that
  a rotation about the approach axis avoids.
- **Deviation: roll search** (`N_ROLLS = 6`, `_rolled`). A clashing placement is also tried
  rolled about the approach axis in 60 deg steps. The roll keeps the two lobes antiparallel
  and the acceptor site at x_a + d u, so it is the same approach with the cap turned about
  it. It runs in both tiers (tied lobes, then opposite faces). When every placement clashes,
  the CLASH value is the smallest |S| over the *unrolled* placements, so a CLASH pair keeps
  the value it had without the search.
- **CLASH at d = 2.3 after the cap and the roll search** (576 samples, valid pairs):

  | pairs | whole copy | cap + roll |
  |---|---|---|
  | intramolecular | 55.1% | 23.1% |
  | intermolecular | 35.1% | 15.3% |
  | labeled (teacher) pairs | 25.6% | 4.0% (intra 12 / 219 = 5.5%, inter 11 / 357 = 3.1%) |
  | ranked first by the floor | 11.1% (KS3) | 0.9% |

- **Tests:**
  - brute-force overlaps on two strict-subset caps: methanol n(O) with ethanol sigma*(O-H)
    (intermolecular) and ethanol n(O) with its own methyl sigma*(C-H) (intramolecular). A
    mutant taking the global AO offsets fails both;
  - the exact cap of ethanol sigma*(O-H) is {O, H, C1};
  - over ethanol's sigma* tokens, some placements clash against the whole molecule but not
    against the cap, and a cap clash always implies a whole-molecule clash;
  - `site_contact` (the production path) on both strict-subset caps equals the maximum of the
    cap-truncated brute force over the tied-lobe placements; a mutant that places the whole
    molecule in `block_contacts` fails it;
  - a roll keeps the lobes antiparallel and turns every cap atom about the approach axis by
    k x 60 deg.
- **Cost:** `ORIENTED_VERSION = 3`. Per distance, all 1,210 occupied x virtual blocks of the
  samples take 1,143 s at 2.0, 1,048 s at 2.3 and 786 s at 3.0 on one core; the 732
  occupied x occupied blocks at 2.3 take 231 s. That is about 0.9 s a block, against
  3.7-4.6 s with the whole copy.
- **Disk:** a block stores the same per-pair arrays (S, sites, CLASH), so the cache shrinks in
  compute, not in bytes. The 4,362 version-3 blocks take about 78 MB. The 7,561 version-2
  blocks (123 MB) are still in `data/cache/oriented/` for the whole-copy comparison above; the
  code never reads them.

### Shared-centre filter: the 10 steps that lost every labeled pair

All 10 have the same first arrow: a lone pair on atom X pushed to the neighbouring atom Y,
with the Y-Z bond breaking (diagnostics in `diagnostics/lost_label_steps.json`):

| step | part | reactants | arrows |
|---|---|---|---|
| pmechdb-02d1522c8ec3ce63 | train | `C#CCC1=C([N:10]=[N:20][OH2+:21])C=CC=C1` | lp(10) -> 20; (20-21) -> 21 |
| pmechdb-c203c91f0a9dfe29 | train | `CC(=O)NC1=CC=C([N:10]=[N:20][OH2+:21])C=C1` | same |
| pmechdb-3b8d5ef8aeea049c | train | `CC(=O)C(C)(C)[C:20](C)([O-:10])[Cl:21]` | same |
| pmechdb-4c5143fdfc205f89 | train | `C=C[C:20](CC(C)=[N+]1CCCC1)([O-:10])[Cl:21]` | same |
| pmechdb-9b1653479e80c4c4 | train | `CC(=O)CC[C:20](C#N)([O-:10])[Cl:21]` | same |
| pmechdb-b2acc68b38d8dd9f | train | `CC(=O)OC(=O)CC[C:20](C#N)([O-:10])[Cl:21]` | same |
| pmechdb-e2112b6254124167 | train | `C=C[C:20](C#N)([O-:10])[Cl:21]` | same |
| pmechdb-4a7b049712ab483b | val | `C=C[C:20](CC(C)=O)([O-:10])[O:21]C(C)=O` | same |
| pmechdb-e174509f7b8dc941 | val | `...[N+](C)(C)[C:20](OC1=CC=CC=C1)([O-:10])[Cl:21]` | same |
| pmechdb-69915f38690dce51 | test | `CC(=O)O[C:20](C)([O-:10])[Cl:21]` | same |

- **Eight are tetrahedral collapses** (R-C(O-)(Cl or OAc) -> C=O + leaving group). **Two are
  diazohydroxide dehydrations** (Ar-N=N-OH2+ -> ArN2+ + H2O).
- **The acceptor contains the donor atom as a minor centre.** In the collapses the acceptor
  localizes as one delocalized antibonding token over C 0.62, O 0.20, Cl 0.15 (sigma*(C-Cl)
  mixed with the C-O(-) pi). Every donor on O therefore shares a centre with it.
- **Verdict:** a three-lens panel (mechanism, orbital picture, dataset annotation) judged each
  step independently: 30 of 30 votes "genuine elementary step", confidence 0.80-0.96. So no
  `resonance_step` tag.
- **Filter narrowed, per the instruction** (`tokens.excluded_pair`). Within one fragment, a
  pair is excluded when:
  - the acceptor is the donor's own antibond (same two leading centres, the donor bonding,
    the acceptor antibonding: the spec's rule, kept), or
  - donor and acceptor have the same main centre.
- **Tests:**
  - acetone n(O) -> pi*(C=O) is kept (main centres O and C);
  - the own pi -> pi* and a same-main-centre pair are excluded;
  - the 1,2-hydride shift is kept.
- **Effect:** all 10 steps have a valid labeled pair again.
- **Side effect on the floor: the narrowed rule re-admits same-atom pairs, which the floor
  ranks first.** n(O) -> sigma*(O-H) of the same O-H (main centre H) shares O but not the
  main centre, so it is a candidate again. Floor top-1/3/5, cap placement, d = 2.3, random
  split:

  | filter | train + val | test |
  |---|---|---|
  | narrowed (adopted) | 21.0 / 28.2 / 33.7 | 18.2 / 22.7 / 33.3 |
  | strict shared-centre | 24.9 / 39.4 / 45.7 | 28.8 / 39.4 / 47.0 |

  In 253 of the 576 steps the floor's top pair is one that only the narrowed rule allows:

  | re-admitted pair | steps where it ranks first |
  |---|---|
  | n(O) -> sigma*(O-H) | 117 |
  | n(O) -> delocalized pi*(C-O) | 45 |
  | n(O) -> pi*(C-O) | 33 |
  | n(Br) -> sigma*(H-Br) | 18 |

  The proton transfers carry the loss. Floor top-5 under the narrowed and the strict filter,
  same cap placement: lone_pair(O) -> sigma*(H-O) 0% and 35%, lone_pair(N) -> sigma*(H-O) 4%
  and 37%, pi(C-C) -> sigma*(H-O) 0% and 32%. A rule in between (for example, excluding acceptors that carry the donor's main centre above
  a weight) is a user decision; nothing was changed beyond the instruction.

### HBr protonations (pi(C=C) / sigma(C-C) -> sigma*(H-Br))

- 34 steps have sigma*(H-Br) as the labeled acceptor (the earlier count of 29 took only the
  three largest classes). The teacher pair's donor is:
  - a C-C token of the C=C (sigma or pi) in 22;
  - an O lone pair in 7;
  - another donor in 5: pi(O-N) 2, lone_pair(P), pi(N-C), lone_pair(C-Mg).
- **The floor misses all 34**, as do -KS3, F^2 G + KS2 and S^2 G. Some reach the top 5 under
  other scores:
  - -KS2 alone: 2 of the 11 pi(C-C) steps and 2 of the 7 O-lone-pair steps;
  - the frozen head: 3 of the 7 O-lone-pair steps.
- **Five of them** (`diagnostics/hbr_protonations.json`). The rank of the labeled pair under
  each score (lower is better):

  | step | reactants | -G | beta^2 | -KS2 | -KS3 | floor |
  |---|---|---|---|---|---|---|
  | 02fbef20 | 1,3,5-triphenylcyclopentene + HBr | 555 | 1382 | 6 | 1308 | 492 |
  | 054b3fa9 | 1,7,7-triphenylcycloheptene + HBr | 614 | 1718 | 16 | 1644 | 502 |
  | 0cf491aa | 1,6,6-triphenylcyclohexene + HBr | 585 | 1733 | 13 | 1542 | 499 |
  | 19c0ec8c | 1,4-cyclohexadiene + HBr | 33 | 243 | 28 | 175 | 79 |
  | 1b02a817 | 1-methyl-1-vinylcyclopentane + HBr | 22 | 532 | 9 | 429 | 94 |

- **Labeled pair:** G -0.12 to -0.18, beta^2 0.013-0.030, KS2 -0.001 to -0.046 eV.
- **What outranks it** (the floor's top five, the same in every step):
  - HBr's own n(Br) -> sigma*(H-Br), intramolecular: G -0.079, beta^2 4.47, KS2 -0.32 eV
    (the bonded H(+) and Br(-)). It is first in 4 of 5 steps. It is a re-admitted same-atom
    pair (above).
  - Br lone pairs -> pi*(C=C) of the alkene, intermolecular: G -0.12 to -0.30, beta^2 2-6.
- **Why beta^2 is 100x smaller: the Wolfsberg-Helmholz mean cancels.** beta = K |S| (E_r +
  E_s) / 2 with the LMO Fock diagonals. The sigma*(H-Br) LMO lies at E_s = +8.78 eV, so
  (E_r + E_s) / 2 = -1.18 eV, against -8.56 eV for n(Br) -> pi*(C=C). The overlaps are
  similar (|S| 0.084 against 0.108). A high-lying acceptor gets a near-zero resonance
  integral, whatever its overlap.
- -KS2 alone places the labeled pair at rank 6-28: the protonation is charge-controlled, and
  the floor's KS2 is too small next to KS3 to show it.

### Pair features for Prompt 6 (`physics.pairs.PAIR_FEATURES`, replacing the list)

| feature | definition | defined | K, eps dependence |
|---|---|---|---|
| `log_beta2_d2.0`, `_d2.3`, `_d3.0` | log beta^2, oriented contact at that d | all | + 2 log K |
| `log_abs_g` | log \|G\| (sign in `pair_gap_sign`) | all | none |
| `log_abs_ks3` | log \|beta^2 G\| at 2.3 | all | + 2 log K |
| `ks2` | signed Coulomb, main centres | main centres differ | x 1 / eps |
| `log_abs_ks1` | log KS1 (repulsion with the occupied tokens near m(s)) | KS1 > 0 | + log K |
| `inter` | 1 between fragments | all | none |
| `centroid_distance` | Angstrom | within a fragment | none |
| `cross_coupling` | \|cross-centre coupling\| | within a fragment | x K |
| `clash` | 1 where the argmax site pair's placements all clash | all | none |

- **Three beta^2 distances:** the decay from 2.0 to 3.0 Angstrom encodes how diffuse the
  lobes are.
- **Stored at K = eps = 1** (`REDUCED`). `params.to_params` applies the dependence above, so
  K and eps stay learnable. Logs are floored at 1e-12, and undefined entries are stored as 0.
- **Standardized on the training part only:** `pair_feature_stats_<strategy>.json`, with
  mean, std and count per feature over the defined entries of every (occupied, virtual) pair
  of the strategy's train samples, at K = 1.75, eps = 4. Written by `ksmech-arrows samples`.
  On the random split's train part: log|KS3| std 2.25, KS2 std 0.116 eV, log|KS1| std 2.31.
- **`pair_clash` is gone** (clash is a feature). Samples are version 5.
- **Frozen head** (`pairs.frozen_head`, no parameters):
  z(log|KS3|) - z(KS2) - z(log|KS1|), with z the standardized feature. Higher means more
  favourable:
  - a larger stabilization magnitude raises it;
  - an attractive (negative) KS2 raises it;
  - a larger closed-shell repulsion lowers it.

  It reads the magnitude of KS3, so an inverted-gap pair (G > 0, which the floor penalizes)
  is not penalized: 161 of the 833,972 valid pairs (0.02%), between fragments.

### KS1 is phase-free (found in review)

- **The bug:** within a molecule, KS1(r, r') was the bilinear form -l_r^T Q l_r'. Its sign
  follows the arbitrary phase of each Pipek-Mezey LMO, so summing it over the occupied tokens
  near the acceptor's main centre let terms of random sign cancel. log|KS1| therefore
  depended on the phases.
- **The fix:** KS1(r, r') = |l_r^T Q l_r'|, positive like the intermolecular contact form.
  This departs from spec v3's Eq. for term 1, written with the signed coefficients; the spec
  is unchanged.
- **Test:** flipping the sign of half of methanol's occupied LMOs leaves the matrix
  unchanged.
- **Undefined entries:** log|KS1| where KS1 = 0 (inside a single-atom fragment: no resonance
  integral within an atom) was stored as log(1e-12) = -27.6. It is now 0, as documented,
  tested on Br(-).
- The floor does not use KS1. The frozen head's top-5 on train + val went from 20.4% to
  24.5%.

### Floor and frozen head (d = 2.3, 576 samples)

Top-1 / 3 / 5:

| scorer | random train + val (510) | random test (66) | scaffold train + val (556) | scaffold test (20) |
|---|---|---|---|---|
| **floor -(KS3 + KS2)** | **21.0 / 28.2 / 33.7** | **18.2 / 22.7 / 33.3** | **21.0 / 27.9 / 33.8** | **10.0 / 20.0 / 30.0** |
| frozen head | 9.0 / 15.7 / 24.5 | 4.5 / 10.6 / 22.7 | 9.2 / 15.6 / 24.6 | 0.0 / 0.0 / 10.0 |
| -KS3 | 18.2 / 26.5 / 29.6 | 24.2 / 31.8 / 33.3 | 19.2 / 27.3 / 30.4 | 10.0 / 20.0 / 20.0 |
| F^2 G + KS2 | 18.8 / 30.0 / 35.3 | 22.7 / 31.8 / 37.9 | 19.2 / 30.2 / 35.6 | 20.0 / 30.0 / 35.0 |
| S^2 G | 22.0 / 31.6 / 34.7 | 25.8 / 34.8 / 43.9 | 22.7 / 32.2 / 36.0 | 15.0 / 25.0 / 30.0 |
| G | 26.5 / 35.3 / 40.6 | 25.8 / 36.4 / 39.4 | 26.8 / 35.6 / 40.6 | 15.0 / 30.0 / 35.0 |

- **The narrowed filter set the floor's level** (33.7% top-5 against 45.7% with the strict
  filter, above), not the cap.
- **The frozen head is weak untrained, because equal weights over-weight KS2.**
  Standardization multiplies KS2 by 1 / 0.116 against 1 / 2.25 for log|KS3|, about 20 times
  more weight than in the eV sum. Its terms on random train + val, top-5:

  | term | top-5 |
  |---|---|
  | z(log\|KS3\|) | 30.4 |
  | z(log\|KS3\|) - z(KS2) | 24.3 |
  | z(log\|KS3\|) - z(log\|KS1\|) | 22.0 |
  | -z(KS2) | 3.7 |
  | -z(log\|KS1\|) | 12.5 |

  The head is a fixed kernel the trained heads sit beside, so it was left as specified.

**Per-class floor at d = 2.3** (random split, all 576; classes with n >= 10):

| labeled pair | n | floor | -KS3 | F^2 G + KS2 | S^2 G | G | frozen head |
|---|---|---|---|---|---|---|---|
| lone_pair(O) -> sigma*(H-O) | 57 | 0% | 0% | 0% | 9% | 2% | 23% |
| lone_pair(O) -> sigma*(C-O) | 50 | 40% | 8% | 8% | 8% | 12% | 42% |
| pi(C-C) -> pi*(C-O) | 42 | 76% | 81% | 90% | 71% | 90% | 36% |
| lone_pair(O) -> delocalized pi*(C-O) | 40 | 65% | 12% | 18% | 12% | 18% | 82% |
| pi(C-C) -> pi*(C-N) | 31 | 100% | 100% | 100% | 100% | 100% | 19% |
| lone_pair(N) -> sigma*(H-O) | 27 | 4% | 0% | 0% | 37% | 4% | 4% |
| lone_pair(O) -> pi*(C-O) | 23 | 65% | 96% | 100% | 87% | 100% | 43% |
| pi(C-C) -> sigma*(H-O) | 22 | 0% | 0% | 0% | 73% | 0% | 0% |
| pi(C-C) -> sigma*(C-O) | 19 | 21% | 21% | 21% | 21% | 26% | 5% |
| pi(C-C) -> pi*(C-C) | 14 | 57% | 93% | 93% | 93% | 93% | 0% |
| pi(C-C) -> sigma*(C-Br) | 13 | 0% | 0% | 31% | 8% | 92% | 0% |
| sigma(C-C) -> sigma*(H-Br) | 11 | 0% | 0% | 0% | 0% | 0% | 0% |
| pi(C-C) -> sigma*(H-Br) | 11 | 0% | 0% | 0% | 0% | 0% | 0% |
| lone_pair(N) -> pi*(C-O) | 10 | 100% | 100% | 100% | 100% | 100% | 30% |

### Why the floor misses: it prefers intramolecular pairs

- **Measured on the 576 samples:**
  - the floor's top pair is intramolecular in 442 steps (77%);
  - the labeled pair is intermolecular in 357 (62%; 366 steps have two or more fragments).
- **The floor's top pairs, by how the two tokens' centres relate** (median -KS3 and -KS2 of
  that pair, eV):

  | top pair | steps | -KS3 | -KS2 |
  |---|---|---|---|
  | intramolecular, a shared centre (re-admitted) | 253 | 0.08 | 0.62 |
  | intermolecular | 134 | 1.14 | 0.03 |
  | intramolecular, farther | 111 | 1.08 | 0.02 |
  | intramolecular, centres bonded (geminal / vicinal) | 78 | 0.10 | 0.59 |

  The labeled intermolecular pairs have a median -KS3 of 0.05 eV and -KS2 of 0.07 eV.
- **Within a fragment, KS2 is taken between the main centres at their geometric distance.**
  For bonded atoms that is the bond's own polarity (O(-) H(+) at 1 Angstrom), about 0.6 eV,
  against at most a few tenths at the van der Waals contact used between fragments.
- **Ablation, not adopted:** intramolecular KS2 at the van der Waals contact. Floor top-5,
  train + val / test:
  - narrowed filter: 33.7 / 33.3 -> 36.5 / 33.3;
  - strict filter: 45.7 / 47.0 -> 40.2 / 40.9.

  The labeled geminal / vicinal pairs (111) need that Coulomb term too (median -KS2 0.43 eV).
- **The whole-copy CLASH had hidden this.** It flagged 55% of the intramolecular pairs and gave
  them the smallest |S| found, an accidental penalty on intramolecular pairs. The cap removed
  it.
- **Option for the user: a weighted shared-centre rule.** Within a fragment, exclude a pair
  when one atom carries at least w of the population in both tokens (own antibond kept).
  Floor top-1/3/5, random split:

  | w | train + val | test | lost steps with a label |
  |---|---|---|---|
  | 0.15 (= tau_atom, the strict rule) | 24.9 / 39.4 / 45.7 | 28.8 / 39.4 / 47.0 | 0 / 10 |
  | **0.25** | **23.9 / 38.4 / 45.7** | **28.8 / 39.4 / 45.5** | **10 / 10** |
  | 0.30 | 22.5 / 30.4 / 36.5 | 22.7 / 27.3 / 36.4 | 10 / 10 |
  | 0.35-0.40 | 21.6 / 28.2 / 33.7 | 19.7 / 22.7 / 33.3 | 10 / 10 |

  At w = 0.25 it keeps all 10 steps and the strict rule's floor. The window is narrow: the
  collapses' acceptor has O at 0.20, and by 0.30 the re-admitted pairs are back. Not adopted,
  because the instruction named the main-centre rule.

### Baselines (`ksmech-physics baseline`, 61 features, samples version 5)

Top-1 / 3 / 5. The trained models were fitted on train, so train + val is not a fair number
for them; their val and test are.

| scorer | random val (56) | random test (66) | scaffold val (23) | scaffold test (20) |
|---|---|---|---|---|
| floor -(KS3 + KS2) | 10.7 / 17.9 / 25.0 | 18.2 / 22.7 / 33.3 | 8.7 / 8.7 / 13.0 | 10.0 / 20.0 / 30.0 |
| frozen head | 5.4 / 8.9 / 19.6 | 4.5 / 10.6 / 22.7 | 4.3 / 4.3 / 8.7 | 0.0 / 0.0 / 10.0 |
| G | 17.9 / 30.4 / 37.5 | 25.8 / 36.4 / 39.4 | 8.7 / 13.0 / 17.4 | 15.0 / 30.0 / 35.0 |
| logistic regression | 28.6 / 48.2 / 53.6 | 34.8 / 57.6 / 66.7 | 8.7 / 30.4 / 34.8 | 35.0 / 55.0 / 60.0 |
| GBDT ranker | 71.4 / 80.4 / 91.1 | **69.7 / 87.9 / 90.9** | 30.4 / 52.2 / 60.9 | **55.0 / 70.0 / 75.0** |

- **Model choices:**
  - logistic regression: L2 0.1 on both splits;
  - GBDT: 120 trees (random) and 150 (scaffold).
- **Against version 3** (whole copy, strict filter, 7 features): GBDT random test 68.2 / 84.8
  / 95.5, scaffold test 55.0 / 75.0 / 75.0. The trained ranker is unchanged within the noise
  of 66 and 20 test steps, and it learns past the floor's intramolecular bias. Logistic
  regression improved on random test top-5 from 63.6 to 66.7.
- **Textbook suite** (expected pair in the top 3):
  - **floor 1 / 5.** F^2 G + KS2 scored 3 / 5. The floor misses:
    - MeO(-) + MeCl (rank 58: methoxide's n(O) -> its own sigma*(C-H));
    - MeO(-) + EtBr (rank 34: EtBr's n(Br) -> its own sigma*(C-H));
    - cyanide + acrolein (rank 8);
    - H3O(+) + acetone (rank 12: H3O(+)'s n(O) -> its own sigma*(O-H)).
  - **GBDT 4 / 5.** It misses cyanide + acrolein (rank 11).
  - Both scores put SN2 above E2 and 1,4 above 1,2:

    | | SN2 vs E2 | 1,4 vs 1,2 |
    |---|---|---|
    | floor | 0.127 vs 0.106 (ranks 34 and 46) | 0.414 vs 0.413 (ranks 8 and 9) |
    | GBDT | 5.38 vs 1.5 | 2.13 vs 2.06 (ranks 11 and 13) |

## 2026-10-21 — AO-level beta, site-pair filter, frozen head in eV, KS2 difference form (user decisions)

The user's seven decisions of 2026-10-20 on the previous entry, as implemented, with the
deviations and the measurements. Samples are version 6, the oriented cache version 5. Every
number is on the 576 samples (random 454 / 56 / 66, scaffold 533 / 23 / 20) at d = 2.3.

### 1. beta at AO level (the spec's form; the orbital-level mean was a spec error)

- **Formula:** beta_rs = K sum_mu,nu l_mu,r l_nu,s S^c_mu,nu (F_mu,mu + F_nu,nu) / 2.
  - S^c is the AO overlap at the oriented placement (the acceptor's LMO on its local cap).
  - F_mu,mu is each molecule's own GFN2 Fock diagonal, in eV.
  - The sum is contracted as half of (f l_r)^T S^c l_s + l_r^T S^c (f l_s), sharing the
    spline evaluations with S (`oriented._group_overlap`).
  - It is stored at K = 1 in `Contacts.beta`; `params.to_params` applies K as before.
  - The ablation kernels (table, MLP) keep their dominant-shell Fock diagonal.
- **Placement still chosen by the largest |S|.** beta is taken from that same placement,
  including the CLASH fallback.
- **Phase:** beta is stored times sign(S_rs). The LMO signs are arbitrary, and flipping one
  flips both S and the sum, so beta sign(S) does not depend on them. It is negative for bound
  AOs, like the old K |S| F.
- **Deviation, found in review: shell-averaged AO energies** (`oriented.ao_energies`).
  - GFN2's anisotropic terms split the lab-frame Fock diagonal within a p or d shell: 0.06-0.2
    eV on C and O p, 0.4 eV on Br d.
  - That split depends on the record's Cartesian orientation, which would make beta depend on
    it too, against "geometry enters only through distances".
  - The shell average is rotation-invariant. Test: a rotated single point of acetone gives the
    same averages, while its raw diagonal moves.
  - The intramolecular Wolfsberg-Helmholz B (KS1 within a fragment, the cross-centre
    coupling) uses the same averages (`ks_terms.wh_matrix`).
- **Cache: `ORIENTED_VERSION = 5`.** Version 4 was the same with the raw diagonal; its partial
  warm-up (833 blocks) was deleted unused.
  - The warm-up computed 1,210 occupied x virtual blocks at each of 2.0, 2.3 and 3.0, plus 732
    occupied x occupied at 2.3.
  - Timing on one core: 1,413 s for d = 2.3 including the occupied x occupied blocks; 2,063 s
    for 2.0 and 3.0 together.
- **The user's test, "AO-level beta at least 10x the orbital-level one for HBr's
  n(Br) -> sigma*(H-Br)", does not hold. Measured 7.5x** on the ALPB record.
  - beta_AO / (K |S|) = -12.0 eV, the Br 4p / H 1s scale.
  - The orbital-level mean is (E_r + E_s) / 2 = (-12.01 + 8.78) / 2 = -1.6 eV.
  - So this pair cannot reach 10x; the labeled pi(C=C) -> sigma*(H-Br), with a mean of
    -1.18 eV, would be about 10x.
  - On the gas-phase force-field single point of the fixtures, sigma*(H-Br) lies at +1.98 eV
    and the ratio is 2.5x.
  - Recorded as a strict xfail with these numbers, next to a passing test of what the physics
    gives: at least 5x, and |beta| / (K |S|) above 8 eV.
- **Effect on the HBr protonations** (`diagnostics/hbr_protonations.json`, the same five steps
  as before):
  - the labeled beta^2 rose from 0.013-0.030 to 0.107-0.142 (sigma(C-C) labels) and to 2.38
    (the pi(C-C) label);
  - their floor ranks moved from 492 / 502 / 499 / 79 / 94 to 103 / 117 / 118 / 33 / 49;
  - none reaches the top 5.
  - In the three large molecules the floor's top five are now intermolecular sigma(C-C) ->
    sigma*(H-Br) of other C-C bonds (beta^2 about 19).
  - In the two small ones the top three are HBr's own n(Br) -> sigma*(H-Br), which the 0.25
    site rule does not exclude (below). Ranks 4-5 are intermolecular n(Br) -> sigma*(C-H)
    (beta^2 about 19).
  - -KS2 alone ranks the labeled pair 3-25.

### 2. The site-pair filter

- **Rule (user):** exclude when sites a (donor) and b (acceptor) are identical or bonded, and
  w_a(acceptor) >= 0.25 or w_b(donor) >= 0.25.
- **Implemented at the level of the site pair, where the sites are defined**
  (`tokens.site_pairs_excluded`, bonds at 1.3x the covalent radii, `tokens.bonded_matrix`).
  - A token pair of one fragment is excluded when every one of its centre pairs is excluded
    (`tokens.site_excluded_matrix`), or when it is the donor's own antibond (the spec's rule,
    kept).
  - This is read from descriptors and geometry only, so `pair_valid` needs no kernel.
  - In a one-molecule oriented block (`intra=True`) an excluded site pair wins the
    pair-track maximum only when the token pair has no admissible site pair (key offset -100,
    against -10 for CLASH). Such a token pair is then excluded from `pair_valid` anyway.
  - The intra block has its own cache key with the threshold. Two fragments that are copies of
    one molecule use the unfiltered intermolecular block; the `same=` keyword of the
    `ContactKernel` methods selects which (`pairs` passes f == g).
- **Why per site pair.** If the rule were applied only to the argmax site pair, keeping a
  collapse would depend on whether (O, O) or (O, C) happened to win. (O, O) is always excluded,
  because a donor carries most of its weight on its own site.
- **Verification at 0.25** (critical weights in `diagnostics/site_rule_critical_weights.json`;
  a pair is excluded iff w <= w*, with w* the smallest over its centre pairs of the larger of
  the two weights):
  - **the 10 collapse / dehydration steps keep a valid label at any threshold.** Each labeled
    pair has a site pair that is neither identical nor bonded (O -> Cl, O -> leaving O, N -> O),
    so w* = -inf. No step lacks a valid label (`n_no_valid_label` 0 on both splits).
  - **n(O) -> sigma*(O-H) of the same O-H:** 650 of the 667 pairs are excluded. The 17 left
    have an O share in the sigma* of 0.163-0.243. The median w* is 0.29.
  - **HBr's own n(Br) -> sigma*(H-Br): none of the 102 is excluded.** In the ALPB record
    sigma*(H-Br) carries 0.751 on H and 0.249 on Br, just under 0.25. The gas-phase fixture
    has a larger Br share, so the unit test excludes it.
- **Margin.** Floor top-1/3/5, random split (`diagnostics/site_rule_margin.json`):

  | site weight | floor train + val | floor test | n(O) -> sigma*(O-H) left (of 667) | HBr's own pair left (of 102) | collapse steps kept |
  |---|---|---|---|---|---|
  | 0.20 | 20.0 / 37.1 / 42.4 | 21.2 / 33.3 / 43.9 | 4 | 0 | 10 / 10 |
  | **0.25 (adopted)** | 20.0 / 37.1 / 42.4 | 21.2 / 33.3 / 43.9 | 17 | 102 | 10 / 10 |
  | 0.30 | 20.0 / 37.1 / 42.4 | 21.2 / 33.3 / 43.9 | 391 | 102 | 10 / 10 |

  - **The floor does not move between 0.20 and 0.30.** Pairs do change: 329 pairs at 0.20 and
    858 at 0.30, and 66 and 16 of them rank in a step's top five. The labeled pair's rank
    changes in 69 and 53 steps. In none of them does it cross the top five, because those
    labels sit at ranks 30-3,400 (mostly the HBr protonations).
  - The difference-form KS2 already demotes most same-bond pairs: for O(-) H(+) the bond's own
    polarity becomes a cost (median KS2 of the threshold-dependent pairs +0.18 / +0.34 eV).
  - **The threshold decides which pairs are candidates, not the floor.** At 0.20 HBr's own
    pair is excluded and 4 same-bond O-H pairs remain, with nothing lost. Left at the user's
    0.25 pending their decision.

### 3. The frozen head in eV

- **Logits:** -(KS3 + KS2 + KS1) / tau, no standardization (`pairs.frozen_head`,
  `pairs.ks_energies`, `SCORERS["frozen_head"]`).
  - tau is `KSParams.tau`: 1 eV at initialization, learnable, positive. It only scales the
    logits.
  - The learned heads keep the standardized log features (`pair_feature_stats_*.json`).
- **KS1 sign.** KS1 is stored as the repulsion, >= 0 (the spec's term (1) is positive), so it
  enters with a plus sign and lowers the logit. The user wrote -(KS3 + KS2 - KS1), which reads
  KS1 as the stabilization-signed sum Sum (q + q) beta S. Read literally against our positive
  KS1, it would reward repulsion.
- **By construction** (tests):
  - the ks-rank scorer equals the head on the stored features;
  - without KS1 the ranking equals the floor's -(KS3 + KS2) exactly;
  - tau does not change the ranking.
- **Finding: KS1 dominates, and its two forms differ in scale.** Medians over the valid
  candidates (eV, `diagnostics/frozen_head_terms.json`):

  | term | intramolecular | intermolecular |
  |---|---|---|
  | \|KS3\| | 0.027 | 0.035 |
  | \|KS2\| | 0.002 | 0.008 |
  | KS1 | 7.40 | 0.51 |

  - At the floor's top pair the medians are KS3 1.17 eV and KS1 2.47 eV.
  - The within-fragment form is the full AO sum. The contact form carries the extra %Vbur
    product / 10^4 of the spec, about 0.1.
  - So the head mostly ranks by low KS1 and favours intermolecular pairs: its top pair is the
    floor's in 151 of 576 steps.
  - Top-5: 28.6% on train + val, 27.3% on test (random).
  - A common scale for the two KS1 forms is a user decision; nothing was changed.

### 4. Intramolecular KS2 in difference form

- **Formula:** KS2 = q_a q_b / eps (1 / R_contact - 1 / R_reactant) within a fragment, with
  R_contact the van der Waals sum (the distance the intermolecular form uses). Intermolecular
  KS2 is unchanged.
- It is undefined when the two main centres are one atom.
- `pair_block(..., ks2_form="geometric")` keeps the old form for the ablation.
- The atom-level `atom_coulomb` matrix is unchanged: geometric within a fragment.
- **Floor top-1/3/5, with the difference form / with the geometric form**
  (`diagnostics/ks2_forms.json`):

  | split | train + val | test |
  |---|---|---|
  | random | 20.0 / 37.1 / 42.4 vs 20.2 / 31.2 / 37.5 | 21.2 / 33.3 / 43.9 vs 19.7 / 27.3 / 37.9 |
  | scaffold | 20.3 / 36.7 / 42.6 vs 20.7 / 31.3 / 37.6 | 15.0 / 35.0 / 40.0 vs 5.0 / 15.0 / 35.0 |

  The floor's top pair is intramolecular in 53.5% of the train + val steps, against 72.7%
  with the geometric form (62% of the labeled pairs are intermolecular).

### 5-7. KS1 magnitude, cap, cache, QM batch

- **KS1 = |l_r^T Q l_r'| per orbital pair is approved** (user, 2026-10-20). It deviates from
  the spec's term-1 expression, which is written with the signed coefficients; the user will
  update the spec.
- **The contact KS1 now takes |beta|:** each AO term -(N + N) beta S is positive for bound
  AOs. So KS1 >= 0 in both forms and log|KS1| loses no sign.
- **The acceptor cap and the roll search are approved.**
- **The version-2 oriented blocks were deleted:** 7,561 files, 114 MB apparent size (the
  123 MB recorded earlier is disk usage).
  - 7,556 were identified by their recomputed version-2 keys.
  - 5 were written before the version-3 code existed (textbook molecules).
  - The 4,381 version-3 blocks (73 MB apparent) are now unused too and are kept until the user
    says.
- **The PMechDB QM batch was restarted** on 2026-10-20 at 20:58 with 2 workers, to leave room
  for the warm-up (`full_run.log`).

### Results (sample version 6, d = 2.3)

Top-1 / 3 / 5. The trained models were fitted on train, so only val and test are fair for them.

| scorer | random train + val (510) | random test (66) | scaffold train + val (556) | scaffold test (20) |
|---|---|---|---|---|
| **floor -(KS3 + KS2)** | **20.0 / 37.1 / 42.4** | **21.2 / 33.3 / 43.9** | **20.3 / 36.7 / 42.6** | **15.0 / 35.0 / 40.0** |
| frozen head (eV) | 12.7 / 26.1 / 28.6 | 9.1 / 21.2 / 27.3 | 12.4 / 25.5 / 28.6 | 10.0 / 25.0 / 25.0 |
| -KS3 | 19.6 / 30.4 / 35.7 | 22.7 / 30.3 / 39.4 | 20.3 / 30.8 / 36.5 | 10.0 / 20.0 / 25.0 |
| G | 26.7 / 36.1 / 40.6 | 25.8 / 37.9 / 40.9 | 27.0 / 36.5 / 40.8 | 15.0 / 30.0 / 35.0 |
| logistic regression (val / test) | 28.6 / 46.4 / 55.4 | 33.3 / 59.1 / 72.7 | 8.7 / 21.7 / 39.1 | 30.0 / 50.0 / 55.0 |
| GBDT ranker (val / test) | 69.6 / 89.3 / 92.9 | **72.7 / 92.4 / 93.9** | 26.1 / 47.8 / 73.9 | **55.0 / 75.0 / 75.0** |

- **Against the previous entry:**
  - the floor's top-5 is 42.4 / 43.9 (was 33.7 / 33.3 on random train + val / test) and 40.0 on
    the scaffold test (was 30.0);
  - the GBDT is unchanged within the noise of 66 and 20 test steps.
  - Model choices: logistic regression L2 0.1 (random) and 1.0 (scaffold); GBDT 170 and 100
    trees.
- **CLASH at d = 2.3:**
  - valid pairs: 24.5% intramolecular, 15.3% intermolecular;
  - teacher pairs: 4.3%;
  - the floor's top-ranked pairs: 8.5%.
- **Per class, floor top-5** (all 576), against the previous entry:
  - proton transfers recovered:
    - n(O) -> sigma*(H-O): 0% -> 63%;
    - n(N) -> sigma*(H-O): 4% -> 74%;
    - pi -> sigma*(H-O): 0% -> 36%;
    - n(O) -> sigma*(H-N): 0% -> 62%.
  - The collapse-type classes fell:
    - n(O) -> sigma*(C-O): 40% -> 6%;
    - n(O) -> delocalized pi*(C-O): 65% -> 12%.

    Their misses are outranked by intramolecular pairs (43 of 46 and 35 of 35 misses), most
    often into a pi*(C-O)-type acceptor (28 of 46, 18 of 35).
  - The HBr classes stay at 0%.
- **Textbook** (expected pair in the top 3): floor 2 / 5 (was 1 / 5), frozen head 1 / 5,
  GBDT 4 / 5.
  - The floor misses both SN2 cases (rank 28 each):
    - in MeO(-) + MeCl to an intermolecular pair, MeCl's n(Cl) -> methoxide's sigma*(C-H);
    - in MeO(-) + EtBr to geminal / vicinal hyperconjugation inside EtBr, n(Br) ->
      sigma*(C-H). The site rule does not touch it: the weights on the bonded sites are near 0.
  - It also misses cyanide + acrolein (rank 5).
  - **Orderings:**
    - the floor keeps SN2 above E2 (ranks 28 / 40) and 1,4 above 1,2 (0.544 / 0.525);
    - the GBDT keeps SN2 above E2 (5.76 / 2.08) but now puts 1,2 above 1,4 (4.02 at rank 4
      against 2.57 at rank 8; it was the other way round before);
    - the frozen head puts E2 above SN2 (ranks 0 / 38).

## 2026-10-21 — User decisions on the open points: site weight 0.20, KS1 sign, cache, QM throughput

The user's answers to the open points of the previous entry, as implemented. Samples are
version 7.

- **Site-rule weight 0.20** (`CandidateConfig.site_weight`; `OrientedConfig.site_weight`
  follows it).
  - It excludes HBr's own n(Br) -> sigma*(H-Br): 0 of 102 left, its w* is 0.249. Of the 667
    same-bond n(O) -> sigma*(O-H) pairs, 4 are left, those with an O share of 0.163-0.185.
  - The 10 collapse / dehydration steps keep a valid label: `n_no_valid_label` is 0 on both
    splits.
  - The one-molecule oriented blocks at 0.20 (cache key `intra0.2`) were computed for 2.0
    (1,001 s), 2.3 (from the margin run) and 3.0 (670 s), 478 molecules each.
  - **The floor is unchanged,** as the margin run showed: random 20.0 / 37.1 / 42.4 (train +
    val) and 21.2 / 33.3 / 43.9 (test); scaffold 20.3 / 36.7 / 42.6 and 15.0 / 35.0 / 40.0.
- **beta:**
  - the >= 5x HBr test is the criterion; the strict 10x xfail was removed;
  - the shell-averaged AO energies are approved.
- **KS1 sign: the spec had it wrong.** The spec writes the frozen head as
  l^(0) = KS3 + KS2 - KS1 (section "The frozen Klopman-Salem head"). By its own conventions
  KS3 < 0 and KS2 < 0 are stabilizing and KS1 >= 0 is the repulsion, so that logit would fall
  for favourable pairs and rise with the repulsion. The correct head is the code's,
  -(KS3 + KS2 + KS1) / tau: the logit is minus the perturbation energy, scaled.
- **KS1 is not rescaled.** It gets a difference-form definition in Prompt 5b, and the frozen
  head stays as it is until then. Its dominance by KS1 (previous entry: medians 7.4 eV within a
  fragment, 0.51 eV across) is known and accepted for now.
- **The version-3 oriented blocks were deleted:** 4,381 files, 72.7 MB. They were the files
  older than the start of the version-4 warm-up (2026-10-20 21:17:30), when no later version
  existed. `data/cache/oriented/` now holds only version 5.
  - The one-molecule blocks at site weight 0.25 (three distances) and 0.30 (2.3), kept at first
    to reproduce the margin table, were deleted at the user's request (2026-10-21; 1,912 files,
    83.5 MB). The table stays in the previous entry.

### Host load and QM throughput (user request)

- **`nproc` = 16:** a Ryzen 7 2700X, 8 cores x 2 SMT threads. There are no other users: the
  three sessions are all the user's (seat, tty, ssh). The load average (about 3.8) was this
  project's own jobs.
- **Per-stage time on 20 random pending molecules** (`diagnostics/qm_stage_profile.json`):
  - setup: one conformer, the batch's config (ALPB water, tblite L-BFGS), one at a time on
    one core next to the batch; nothing written to the cache;
  - all 20 succeeded; 11-75 atoms; 0.3-70 s each, median 14.4 s, mean 21 s.

  | stage | seconds (20 molecules) | share |
  |---|---|---|
  | geometry optimization (free; with bond walls when a bond breaks) | 393.6 | 93.9% |
  | %Vbur (morfeus) | 11.6 | 2.8% |
  | embedding (prepare + ETKDG / MMFF) | 6.0 | 1.4% |
  | N+1 / N-1 single points (finite-difference Fukui) | 5.7 | 1.4% |
  | final single point with integrals + record assembly | 2.3 | 0.5% |

  The optimization is the whole cost. It grows with size but scatters widely: 0.1-1.5 s below
  30 atoms, 11-67 s at 50-75 atoms (a 71-atom molecule took 11 s, a 52-atom one 51 s).
- **The manifest's 2,241 timed molecules:** median 3.6 s, mean 27.0 s, 90th percentile 55.4 s,
  maximum 905 s (the 900 s timeout). Molecules over 300 s take 41% of the time. Another 290 rows
  are cache-hit placeholders with no timing.
- **Projection:** the pending molecules need about 70-90 CPU-hours, roughly 17-22 h at 4
  workers (the profile's mean and the manifest's).
- **Workers: 4 (unchanged).** The user asked for nproc - 2 = 14 with OMP_NUM_THREADS=1. That
  is the configuration that hard-rebooted the host three times within 1-2 minutes on
  2026-10-19: the batch already spawned its workers with `OMP_NUM_THREADS=1` then (Batch
  section above). 8 workers rebooted it once after 34 minutes. The BLAS-oversubscription cause
  found on 2026-10-20 was `ksmech-orbitals build`, not the QM batch. The ramp was put back to the
  user before running it, and **the user chose to stay at 4 workers.**
- **Restarts were re-running the known failures.**
  - The batch retries failed molecules by default, and in SMILES order every earlier molecule
    that succeeded is already done. So the head of the queue was 67 of the 69 known failures:
    29 OPT_NOT_CONVERGED, 19 SPIN_PARITY, 12 SCF_NOT_CONVERGED, 7 TIMEOUT, 4.4 CPU-h when they
    first failed. Most of them fail deterministically.
  - The other 2 sit deeper in the queue: a 300 s-timeout pilot TIMEOUT and one SPIN_PARITY.
  - Each of tonight's restarts re-ran them from the start: 10 molecules in 28 minutes at 2
    workers, 7 in 15 minutes at 4, all of them known failures.
  - **Stopping the batch must use SIGINT.** The manifest is written every 500 molecules and on
    exit (the `finally` of `featurize_dataset`). But `systemctl --user stop` sends SIGTERM,
    which kills Python without running it, so the four stops left the manifest at its 01:01
    flush. The records themselves are safe: they are cached per molecule, and the next run
    counts them as done. Only the manifest rows (timings, warnings) of those runs are missing.
    Stop with `systemctl --user kill --signal=SIGINT <unit>.scope`.
  - **Relaunched with `--no-retry-failed`** (2026-10-21 02:45). It skips all 69 failures and
    did 102 molecules in 15 minutes at 4 workers (8.8 s a molecule), with 11,860 left: about
    29 h at this rate, 17-22 h at the mean costs above. The failures are to be retried once, at
    the end, by a plain run.

### Re-run on samples version 7 (d = 2.3)

Top-1 / 3 / 5 on the test split; the trained models' val in parentheses:

| scorer | random test (66) | scaffold test (20) |
|---|---|---|
| floor -(KS3 + KS2) | 21.2 / 33.3 / 43.9 | 15.0 / 35.0 / 40.0 |
| frozen head (eV) | 9.1 / 21.2 / 27.3 | 10.0 / 25.0 / 25.0 |
| logistic regression | 33.3 / 57.6 / 71.2 (val 26.8 / 46.4 / 55.4) | 35.0 / 55.0 / 70.0 (val 8.7 / 26.1 / 39.1) |
| GBDT ranker | 74.2 / 89.4 / 92.4 (val 76.8 / 85.7 / 92.9) | 55.0 / 75.0 / 75.0 (val 30.4 / 52.2 / 65.2) |

- **Model choices:** logistic regression L2 0.1 on both splits; GBDT 220 trees (random) and
  110 (scaffold).
- **Textbook:** floor 2 / 5, frozen head 1 / 5, GBDT 4 / 5. The GBDT misses cyanide +
  acrolein (rank 4) and again puts 1,4 above 1,2 (rank 4 against 5).
- **CLASH** on the floor's top-ranked pairs: 8.2%.

## 2026-10-21 — Domain exclusions: what removes a PMechDB step from the training target

This is the single place that lists every reason a step is not a training sample, with counts,
so that the covered fraction of PMechDB is one number. Ingest-level reasons are final. The QM
and token rows are a snapshot taken while the QM batch runs; they will be refreshed when the
batch and its retry run have finished.

### The closed-shell screen at ingest (`ksmech.data.domain`, user decisions)

- **Trigger.** A fragment, on either side of a step, whose electron count (sum Z - charge) is
  odd while RDKit marks no radical electron. GFN2 cannot compute it closed-shell as written
  (the QM layer's SPIN_PARITY).
- **How many.** A survey of all 14,382 unique PMechDB fragments found 100, not only the 20
  SPIN_PARITY failures the QM batch had reached. All but one carry a metal.
- **Rules,** applied per fragment in this order:
  1. **METAL_PLACEHOLDER** (step excluded): a metal other than an alkali or alkaline-earth
     metal with at most one neighbour, counting H. These are Lewis-acid catalysts drawn
     without their ligands (`C=[O+][Ti]`, `C[O][Fe]`, `COC(=O)[C](=[Cu])c1ccccc1`): 34
     fragments. s-block atoms are left out because O-Na and C-Li are drawn covalently in
     real molecules (a sodium phosphonate on a Cu(II) complex).
  2. **CHARGE_REPAIRED** (step kept), one change per fragment. A repair is kept only if the
     electron count becomes even and the whole repaired step still validates: charge
     balance, maps and arrow replay.
     - R1: remove a -1 on a four-bonded group-14 atom.
     - R2: read an onium -> d0 / d10 acceptor bond (Ti, Zr, Hf, Sc, Al, B, Zn, Cd; the
       acceptor's charge not negative) as dative, with the acceptor taking the -1.
     - Never for Cu, Fe, Co, Ni.
     - Every change is logged in `charge_repairs.csv` and in the step's `charge_repair`
       column (both sides as drawn).
  3. **OPEN_SHELL_METAL** (step excluded): any other odd fragment with a metal outside the
     d0 / d10 set, mostly Cu(II) and Fe(III) halide adducts: 61 fragments. No spin table;
     it is outside the closed-shell domain until the radical extension.
  4. **CHARGE_UNREPAIRED** (step excluded, added here): an odd fragment whose metals are all
     d0 / d10 and that no rule repairs consistently. It is a charge error, not an open shell:
     2 fragments.
  - The one odd fragment without a metal, `O=[P-](Cl)(Cl)Cl`, gets no code. The QM layer
    reports it (SPIN_PARITY).
- **Outcome for the molecules discussed:**
  - **#19** (the beta-silyl cation with a spurious Si(-)) is repaired by R1 on both sides of
    its step (pmechdb-c7de295e11f86278). The step validates and stays; its two repaired
    fragments enter the QM retry queue.
  - **#4 is not repaired.** It occurs only as the product of pmechdb-bec502e1804f7d61,
    whose reactant `C[O][Ti](Cl)4` carries the same missing Ti(-) on a Ti alkoxide, with no
    onium, so neither rule applies to it. Repairing the product alone leaves the step
    unbalanced (reactants +1, products 0), so the repair is dropped. Both fragments are
    CHARGE_UNREPAIRED and the step is excluded. That is the only CHARGE_UNREPAIRED step;
    a third rule for d0 alkoxides would be the user's decision.
  - **#10** (O+ -> Cu triflate) is OPEN_SHELL_METAL, as decided.
- **Implementation.**
  - `ElementaryStep` gained `domain_exclusion` and `charge_repair`.
  - `validation.build_step` runs the screen after validating the step as drawn.
  - `read_training_steps` drops resonance steps, renumbers `step_index`, then drops
    domain-excluded steps without renumbering, so a pathway keeps the gap.
  - Every builder of alignment, round-trip and sample data, and so every metric
    denominator, reads through it.
  - `stats.json` counts the steps per code under `domain`.
- **Re-ingest of 2026-10-21.**
  - Accepted and rejected records are unchanged: 13,090 and 14, rejection reasons
    identical.
  - Only pmechdb-c7de295e11f86278 changed its SMILES.
  - None of the 576 current samples is in an excluded step, so the samples did not change.
  - `n_training_steps` 12,840.
  - No QM was run.

### Where the 13,090 accepted steps are (snapshot 2026-10-21 18:20)

Each step is counted once, under the first reason that applies, in this order:

| reason | steps | level |
|---|---|---|
| RESONANCE_STEP | 147 | ingest |
| OPEN_SHELL_METAL | 78 | ingest |
| METAL_PLACEHOLDER | 24 | ingest |
| CHARGE_UNREPAIRED | 1 | ingest |
| QM failed: OPT_NOT_CONVERGED | 88 | QM (a reactant fragment has no record) |
| QM failed: SCF_NOT_CONVERGED | 51 | QM |
| QM failed: TIMEOUT | 21 | QM |
| QM failed: EMBED_FAILED | 2 | QM |
| QM pending (batch running) | 1,828 | QM |
| tokens pending (`ksmech-orbitals build` not re-run since the batch) | 10,202 | tokens |
| sample skipped: CORRESPONDENCE / UNALIGNED_ARROW | 72 | alignment |
| **sample (covered)** | **576** | |

- **Covered fraction now: 576 / 13,090 = 4.4%.** Almost all of the remainder waits for the QM
  batch and a token rebuild, so this number is not yet meaningful. The final one needs the
  batch, its retry run, `ksmech-orbitals build`, `ksmech-arrows align` and `samples`.
- **FERMI_SMEARED** marks 3 of the 648 tokenized steps (a sub-count of the last two rows).
  Today the sample builder keeps such steps as flagged samples (`fermi_smeared`), with the
  fractional tokens masked; it does not drop them. Removing them from the target is a change
  in `arrows.steps`, outside this data-layer change, and is left for the user to confirm.
- SPIN_PARITY no longer appears among the QM failures. Every step with a metal-bearing odd
  fragment is excluded at ingest. The metal-free one, `O=[P-](Cl)(Cl)Cl` (failed SPIN_PARITY
  in the batch), is only a product (pmechdb-e57aaee9d354b63c), so it does not block that
  step's sample. The table follows the reactant fragments, as the sample builder does.

### QM retry of the 69 known failures (2026-10-21)

- **The run.** After the main batch finished (FINISHED 0), a plain `ksmech-qm run --source
  pmechdb --workers 4` (failures retried) ran from 18:32 to 22:06. It had 284 molecules to
  compute: every failure, plus the two fragments repaired at ingest.
- **Of the 69 known failures, 1 recovered:** `CCCCC12CCC(OP(c3ccccc3)(c3ccccc3)(c3ccccc3)N(NC(=O)OCC)C(=O)OCC)C(OC)(CCC3C1CC3(C)C)C2` (was TIMEOUT, > 300s).
- **68 still fail.** Every reason code is unchanged from the first attempt.
  - The 20 SPIN_PARITY molecules fail the electron-parity check before any QM; their steps
    are excluded at ingest (above).
  - #19's pre-repair SMILES is listed but is no longer in `steps.parquet`.
  - The other failures are geometry optimizations still not converged after 500 L-BFGS
    iterations, SCF failures in 250 cycles, or the 900 s timeout. These are mostly
    hypervalent iodine, Ti / Sn / B Lewis-acid adducts and large silyl ethers.
- **The two repaired #19 fragments** were computed for the first time and failed too:
  reactant SCF_NOT_CONVERGED, product OPT_NOT_CONVERGED. The repaired reactant SMILES was
  already one of the known failures (it occurs elsewhere in PMechDB).
- **The main run's own failures, after the retry:** 217 molecules, SPIN_PARITY 80, OPT_NOT_CONVERGED 69, SCF_NOT_CONVERGED 46, TIMEOUT 16, EMBED_FAILED 6.
  - The SPIN_PARITY ones are the odd-electron fragments of the domain screen.
- **QM records:** 14097 of 14382 molecules in the manifest.
  The manifest still lists the two pre-repair SMILES of #19.

| reason now (first attempt) | SMILES |
|---|---|
| SPIN_PARITY | `Br[Br+][Fe-]([Br])([Br])[Br]` |
| SPIN_PARITY | `Brc1cccc([NH2+][Fe-]([Br])([Br])[Br])c1` |
| SPIN_PARITY | `C=CCC(c1cccc(Cl)c1)[N](S(=O)(=O)c1ccc(C)cc1)[Fe-]([Cl])([Cl])[Cl]` |
| SPIN_PARITY | `C=[C+]C[Si-](C)(C)(C)[O+](C)[Ti]([Cl])([Cl])([Cl])[Cl]` |
| SPIN_PARITY | `C=[NH+][Cu-]([Br])[Br]` |
| SPIN_PARITY | `C=[NH+][Cu-]([Cl])[Cl]` |
| SPIN_PARITY | `C=[NH+][Cu-]([F])[F]` |
| SPIN_PARITY | `C=[NH+][CuH2-]` |
| SPIN_PARITY | `C=[O+][Cu-]` |
| SPIN_PARITY | `C=[O+][Cu][O]S(=O)(=O)C(F)(F)F` |
| SPIN_PARITY | `C=[O+][Fe]` |
| SPIN_PARITY | `C=[O+][Sc-]` |
| SPIN_PARITY | `C=[O+][Ti]` |
| SPIN_PARITY | `C=[O+][Zn]` |
| SPIN_PARITY | `CC#[N+][Cu-]([Br])[Br]` |
| SPIN_PARITY | `CC#[N+][Cu-]([Cl])[Cl]` |
| SPIN_PARITY | `CC#[N+][Cu-]([F])[F]` |
| SPIN_PARITY | `CC#[N+][Cu-]([I])[I]` |
| SPIN_PARITY | `CC(=[C+]CC(CCc1ccccc1)[O][Ti-]([Cl])([Cl])([Cl])[Cl])[Si-](C)(C)C` |
| SPIN_PARITY | `COC=[NH+][CuH2-]` |
| OPT_NOT_CONVERGED | `C#CC=CCC1OC(CC)C=CC=CC1OP(Cl)(CCCCCCCC)(CCCCCCCC)CCCCCCCC` |
| OPT_NOT_CONVERGED | `C=C(C)C1OC2(C)CCC3C(=C2CC1O[I+]1(OC(C)=O)OC(=O)c2ccccc21)CCC1COC(=O)C13C` |
| OPT_NOT_CONVERGED | `C=C(COCc1ccccc1)CC(C)CC(O[Si](C(C)C)(C(C)C)C(C)C)C(C=[O+][Li-][O]C(=C)C(C)CC(C)CN=[N+]=[N-])O[Si](C)(C)C` |
| OPT_NOT_CONVERGED | `C=C1CC(C(=O)OC)(C(=O)OC)C[C+]1C[Si-](CC)(CC)(CC)OC(=O)C(F)(F)F` |
| OPT_NOT_CONVERGED | `C=C1CC(C=C(C)Br)OC(CC[OH+]P(c2ccccc2)(c2ccccc2)(c2ccccc2)N(NC(=O)OCC)C(=O)OCC)C1` |
| OPT_NOT_CONVERGED | `C=C1CCC2(OCOCCOC)C(=CCC3(C)C(O[Si](C)(C)C(C)(C)C)CCC23)C=C1C(OC(C)=O)C(CO[I+]1(OC(C)=O)OC(=O)c2ccccc21)OC(C)=O` |
| OPT_NOT_CONVERGED | `C=C1CCC23CC1C(O[Si-](CC)(CC)(CC)OS(=O)(=O)C(F)(F)F)=CC2CCC3C` |
| OPT_NOT_CONVERGED | `C=CC(Oc1ccccc1)=[O+][Al-]([O]c1ccccc1)([O]c1ccccc1)[N](S(=O)(=O)C(F)(F)F)S(=O)(=O)C(F)(F)F` |
| OPT_NOT_CONVERGED | `C=CCC(OC(CCc1ccc(OCc2ccccc2)cc1)OC(C)=[O+][Sn-]([Br])([Br])([Br])[Br])c1ccc(OS(=O)(=O)c2ccc(C)cc2)cc1` |
| OPT_NOT_CONVERGED | `C=CCC([OH+][Ti-]([OH])([O]C(C)C)([O]C(C)C)[O]C(C)C)C(C)=Cc1csc(C)n1` |
| OPT_NOT_CONVERGED | `C=CCC(c1ccccc1)[O+]([Zn-]([Cl])[Cl])[Sn-]([Cl])([CH2]C)([CH2]C)[CH2]C` |
| OPT_NOT_CONVERGED | `C=CCOC(=O)N1CCCC1C(=O)NC=Cc1cc(OC2CCN(C(=O)OC(C)(C)C)C2COI2(OC(C)=O)(OC(C)=O)OC(=O)c3ccccc32)ccc1OC` |
| OPT_NOT_CONVERGED | `C=CCOC(=O)N1CCCC1C(=O)NC=Cc1cc(OC2CCN(C(=O)OC(C)(C)C)C2CO[I+]2(OC(C)=O)OC(=O)c3ccccc32)ccc1OC` |
| OPT_NOT_CONVERGED | `C=Cc1cccc2c1C(C(=O)N(C)C(=O)C(N=N[N-]S(C)(=O)=O)C(=O)OC)CN2C(=O)c1ccccc1` |
| OPT_NOT_CONVERGED | `CC(=O)CCC([O-])([NH2+]c1ccc(F)cc1)c1ccc(S(C)(=O)=O)cc1` |
| OPT_NOT_CONVERGED | `CC(=O)NC1C=CC([OH+]I2(OC(C)=O)(OC(C)=O)OC(=O)c3ccccc32)CC1NC(=O)OC(C)(C)C` |
| OPT_NOT_CONVERGED | `CC(=O)OCC1OC(C=[C+]C[Si-](C)(C)(C)[O+]=C(C)O[Si](C)(C)C)C=CC1OC(C)=O` |
| OPT_NOT_CONVERGED | `CC(=O)OCC1OC(C=[C+]C[Si-](C)(C)(C)[O+]=C(C)[O][Sn-]([Cl])([Cl])([Cl])[Cl])C=CC1OC(C)=O` |
| OPT_NOT_CONVERGED | `CC(=O)OCC1OC(C[CH+]C[Si-](C)(C)(C)[O+]=C(C)O[Si](C)(C)C)C=CC1OC(C)=O` |
| OPT_NOT_CONVERGED | `CC(=O)OCC1OC=CC(OC(C)=[O+][Si-](C)(C)(C)OS(=O)(=O)C(F)(F)F)C1OC(C)=O` |
| OPT_NOT_CONVERGED | `CC(=O)OI1(OC(C)=O)([OH+]CC2CCCN2C(=O)c2ccc[nH]2)OC(=O)c2ccccc21` |
| OPT_NOT_CONVERGED | `CC(=O)OI1(OCC(=CC=C(C=O)SCC[Si](C)(C)C)SCC[Si](C)(C)C)(OC(C)=O)OC(=O)c2ccccc21` |
| OPT_NOT_CONVERGED | `CC(=O)OI1(OCC(=CC=C(CO)SCC[Si](C)(C)C)SCC[Si](C)(C)C)(OC(C)=O)OC(=O)c2ccccc21` |
| OPT_NOT_CONVERGED | `CC(=O)O[I+]1(OC(C2=CCCN3C(=O)CCC23)C2C(=O)C=CCC2C)OC(=O)c2ccccc21` |
| OPT_NOT_CONVERGED | `CC(=O)O[I+]1(OCC(=CC=C(CO)SCC[Si](C)(C)C)SCC[Si](C)(C)C)OC(=O)c2ccccc21` |
| OPT_NOT_CONVERGED | `CC(=O)O[I+]1(OCC2CCCN2C(=O)c2ccc[nH]2)OC(=O)c2ccccc21` |
| OPT_NOT_CONVERGED | `CC(=[C+]CC(C)([O][Ti-]([Cl])([Cl])([Cl])[Cl])C(C)C)[Si](C)(C)C` |
| OPT_NOT_CONVERGED | `CC(=[C+]CC1C=C([O][Ti-]([Cl])([Cl])([Cl])[Cl])CCCC1C)[Si](C)(C)C` |
| OPT_NOT_CONVERGED | `CC(C(=O)O[Si-](C)(C)(C)O[P+](CCCCc1ccccc1)(CC(=O)O[Si](C)(C)C)O[Si](C)(C)C)c1ccccc1` |
| SCF_NOT_CONVERGED | `BO` |
| SCF_NOT_CONVERGED | `Br[C-]=C(c1ccccc1)c1ccccc1` |
| SCF_NOT_CONVERGED | `C=CC1C[Br+]1` |
| SCF_NOT_CONVERGED | `C=CCC1([O+]([Si-](C)(C)(C)Cl)[Ti-]([Cl])([Cl])([Cl])[Cl])C=CCCC1` |
| SCF_NOT_CONVERGED | `C=[O+][Zn-]` |
| SCF_NOT_CONVERGED | `CC#CCC(CCc1ccccc1)[O+]([Si](C)(C)C)[Ti-]([Cl])([Cl])([Cl])[Cl]` |
| SCF_NOT_CONVERGED | `CC(=C[O-])C1CCC(C)C1C=[N+](C)c1ccccc1` |
| SCF_NOT_CONVERGED | `CC(=O)CC(C)[O+]([Si](C)(C)C)[Ti-]([Cl])([Cl])([Cl])[Cl]` |
| SCF_NOT_CONVERGED | `CC(=O)OCC1OC(C=[C+]C[Si-](C)(C)(C)OC(C)=[O+][Ti-]([Cl])([Cl])([Cl])[Cl])C=CC1OC(C)=O` |
| SCF_NOT_CONVERGED | `CC(=O)O[I+]1(OCC(=CC=C(C=O)SCC[Si](C)(C)C)SCC[Si](C)(C)C)OC(=O)c2ccccc21` |
| SCF_NOT_CONVERGED | `CC(=[C+]CC(CCc1ccccc1)[O][Ti-]([Cl])([Cl])([Cl])[Cl])[Si](C)(C)C` |
| SCF_NOT_CONVERGED | `CC(=[O+][Ti-]([Cl])([Cl])([Cl])[Cl])c1ccccc1` |
| TIMEOUT | `C=C1CCC2(OCOCCOC)C(=CCC3(C)C(O[Si](C)(C)C(C)(C)C)CCC23)C=C1C(OC(C)=O)C(C[OH+]I1(OC(C)=O)(OC(C)=O)OC(=O)c2ccccc21)OC(C)=O` |
| TIMEOUT | `C=CC1C=CCC(CC(C)CC(=C)CC(C=CCC(O[Si](C)(C)C(C)(C)C)C(C=CC2CC(C)=CCO2)O[Si](C)(C)C(C)(C)C)O[B-]2(c3cc(C(F)(F)F)cc(C(F)(F)F)c3)OC(=O)C(C(OC(=O)c3c(OC(C)C)cccc3OC(C)C)C(=O)O)O2)O1` |
| TIMEOUT | `C=CC1C=CCC(CC(C)C[C+](CC(C=CCC(O[Si](C)(C)C(C)(C)C)C(C=CC2CC(C)=CCO2)O[Si](C)(C)C(C)(C)C)O[B-]2(c3cc(C(F)(F)F)cc(C(F)(F)F)c3)OC(=O)C(C(OC(=O)c3c(OC(C)C)cccc3OC(C)C)C(=O)O)O2)C[Si-](C)(C)(C)[O+]=C(O)CC)O1` |
| TIMEOUT | `C=CC1C=CCC(CC(C)C[C+](CC(C=CCC(O[Si](C)(C)C(C)(C)C)C(C=CC2CC(C)=CCO2)O[Si](C)(C)C(C)(C)C)O[B-]2(c3cc(C(F)(F)F)cc(C(F)(F)F)c3)OC(=O)C(C(OC(=O)c3c(OC(C)C)cccc3OC(C)C)C(=O)O)O2)C[Si](C)(C)C)O1` |
| TIMEOUT | `CC(=O)OI1(OC(C)=O)([OH+]C(C(OCc2ccccc2)C(C=Cc2ccccc2)O[Si](C)(C)C(C)(C)C)C2(C(=O)OC(C)(C)C)OC3(CCCC3)OC2C(=O)OC(C)(C)C)OC(=O)c2ccccc21` |
| TIMEOUT | `CC(=O)O[I+]1(OC(C(OCc2ccccc2)C(C=Cc2ccccc2)O[Si](C)(C)C(C)(C)C)C2(C(=O)OC(C)(C)C)OC3(CCCC3)OC2C(=O)OC(C)(C)C)OC(=O)c2ccccc21` |
| TIMEOUT | `CC(=O)[O][Sn-]([Br])([Br])([Br])[Br+]C1CC(CCc2ccc(OCc3ccccc3)cc2)OC(c2ccc(OS(=O)(=O)c3ccc(C)cc3)cc2)C1` |

## 2026-10-21 — Current samples, baselines and ks-rank reports are stale (review, fix branch)

- **What is stale.** Everything under `data/processed/pmechdb/` built from samples:
  - `samples_{random,scaffold}_{train,val,test}.pt`, `samples_report.json` and
    `pair_feature_stats_{random,scaffold}.json` (2026-10-21 02:04);
  - `baseline_{lr,gbdt}_{random,scaffold}.npz` and `baseline_report_{random,scaffold}.json`
    (02:13-02:15);
  - `ks_rank_report_{random,scaffold}_oriented_d2.3.json` (02:05);
  - the untagged `ks_rank_report_{table,mlp,analytic,oriented_d*}.json` from 2026-10-20.
- **Why.**
  - The `random` files were built with `split_by_pathway`, which on PMechDB (no pathway ids) is
    a per-step split. 447 / 1,284 random-test steps have their largest reactant in train
    (docs/review_2026-10-21.md, D1).
  - `split_by_molecule` (2026-10-19 entry above) lived on `main` and was never merged into the
    line these files were built on. The merge into `fix/review-2026-10-21` brings it in. After
    the merge, `splits.default_split("pmechdb")` returns it, but `arrows/steps.split_ids` still
    maps `random` to `split_by_pathway`; fix item 2 changes that.
  - All of these files also predate the 18:15 re-ingest (12,943 → 12,840 training steps): 49 / 66
    of the old random-test steps now fall in train under the same seed.
- **Rule until the rebuild.** None of these numbers is quoted as a result. The rebuild after fix
  items 1-10 (provenance-stamped) replaces them, and the new numbers are reported next to these
  with the delta.

## 2026-10-21 — Provenance headers on every artifact (fix item 1, `ksmech.data.provenance`)

- **One module, one header.** A `Provenance` holds:
  - `kind`, `schema_version`;
  - `code_git_hash` (HEAD, `+dirty` when tracked files differ);
  - `spec_sha256` (of `docs/ksmech_architecture.tex`);
  - `inputs` {name: sha256}, `config`, `built_at`;
  - `content_sha256` (the hash of the artifact's own payload).

  Writers call `stamp()`; loaders call `verify()`, which raises `StaleArtifact` (reason
  `STALE_ARTIFACT`) with one line per change, e.g. `input changed: data/processed/pmechdb/steps.parquet`.
- **Where the header lives.** In the artifact itself:
  - `.npz`: a `__provenance__` member that `np.load` reads alone;
  - `.json`: a `provenance` key;
  - sample `.pt` files: a frame (magic, length, JSON header, then the torch payload).

  A file input's current hash is read from that file's header and recomputed from its bytes
  only when it has none (the parquet). `current_hash` memoizes on (path, mtime, size). A pure
  `touch` therefore changes nothing; changed content always does.
- **What is compared and what is not.** `schema_version`, `config` and `inputs` are compared.
  `code_git_hash`, `spec_sha256` and `built_at` are for tracing only, so a commit does not by
  itself invalidate the caches. A code change that alters outputs must bump the artifact's
  schema version (QM `SCHEMA_VERSION`, token `SCHEMA_VERSION`, `ORIENTED_VERSION`,
  `SAMPLE_VERSION` now 8, `BASELINE_VERSION`), as before.
- **Chain and stamps.**

  | artifact | inputs | config |
  |---|---|---|
  | QM record | `smiles` (sha256 of the canonical SMILES) | method, solvation, conformer id |
  | token file | its QM record (`file:`) | DescriptorConfig (tau_atom, spectral_min / max, top_centers, min_p_norm), PM tol and max sweeps |
  | contact block | `tokens:F`, `tokens:G` (content digests of the two token sets) | contact mode (`oriented_v<version>`), block, N directions, N rolls, distance, tol, clash, site weight (intra) |
  | split file `splits_<strategy>.json` (new) | steps.parquet | strategy, seed |
  | sample file | steps.parquet, the split file, every token file and contact block its samples used | split, seed, sample version, kernel description |
  | baseline model | the train and val sample files | model, feature list |

  Each sample also keeps its own inputs (`MechanismSample.inputs`), gathered by
  `provenance.collecting()` while it is built. `TokenCache` and the oriented block cache report
  every file they serve with `note_input`.
- **Loaders.**
  - Read-only consumers refuse with exit 2 and the reason: `load_samples`, `load_gbdt`,
    `load_split_file`, `load_record` (always the header; the molecule and config when the
    caller passes its identity).
  - Builders that own a cache rebuild the stale entry and count it:
    - `ksmech-orbitals build` rebuilds a stale token file and reports a stale QM record as
      `stale_qm` (`n_stale_qm`);
    - the oriented kernel recomputes a stale block (`oriented.STALE_BLOCKS`);
    - `compute_fixture_records` recomputes a stale record;
    - `textbook.ensure_tokens` rebuilds a stale token file.
  - `TokenCache` treats a stale token file as absent and keeps the reason in `.stale`, so
    MISSING_TOKENS and STALE_ARTIFACT can be told apart.
  - A file that is not even a readable `.npz` raises `CorruptArtifact`: a failure, not stale.
- **Skip flag.** Only `--allow-stale`, and only on the diagnostic commands (`ksmech-physics
  ks-rank`, `ksmech-physics textbook`). It logs the mismatch and reads anyway.
  `ksmech-arrows samples`, `ksmech-physics baseline` and training have no skip.
- **CLI.** `ksmech-data provenance <path>` prints the chain: each artifact with its config and
  status, following file inputs up to `--depth`. It exits 1 if anything is stale.
  `ksmech-data provenance --backfill qm` adds headers to existing QM records in place,
  atomically and with the arrays unchanged, marked `extra.backfilled`. Run it once, after the
  QM batch; it was not run while the batch was writing.
- **Consequence for the current caches.**
  - Every file written before this commit has no header. The QM records are backfilled.
  - Token files, contact blocks, samples and baselines are rebuilt: they would be anyway
    (fix items 2-5 change their content).
  - Until then the real-data alignment test skips with "token cache predates provenance
    headers".
- **Not stamped.** `contact_table.json` (table ablation), the MLP kernel (deleted in item 9),
  `pair_feature_stats_*.json` (read by nothing yet; it will be with the trainer) and the
  reports.

## 2026-10-22 — The molecule split is the default; the per-step split is `random_leaky` (fix item 2)

- **Strategies.** `ksmech-arrows samples` now builds `molecule`, `scaffold` and `loco` by
  default.
  - `molecule` is `splits.split_by_molecule` (2026-10-19 entry): no non-trivial molecule
    (maps stripped, canonical, more than 3 heavy atoms), reactant or product, in two of
    train / val / test.
  - The per-step split (`split_by_pathway` without pathway ids) is renamed `random_leaky`.
    It is built only when named (`--strategy random_leaky`). The old name `random` raises,
    so a script cannot fall back to it silently.
  - `ksmech-physics ks-rank` and `baseline` default to `--split molecule`. `textbook` reads
    `baseline_gbdt_molecule.npz` by default.
- **Measured on the real parquet** (12,840 training steps, seed 0, 80/10/10):

  | strategy | train / val / test | test steps whose largest reactant (by SMILES length) is in train |
  |---|---|---|
  | `molecule` | 10,272 / 1,284 / 1,284 | 0 |
  | `random_leaky` | 10,272 / 1,284 / 1,284 | 457 |

  `test_no_reactant_molecule_is_in_two_parts_of_the_real_molecule_split` (realdata) asserts
  the property on both sides of every step.
- **What the molecule test set is.** Common reagents chain about 78% of the steps into one
  group, which goes to train. Val and test are the steps that share no non-trivial molecule
  with it. They are out of distribution by construction, so numbers on them are expected to
  drop relative to the leaky split; that is the point of the change.

## 2026-10-22 — Resonance key: map-free, canonical form, set intersection (fix item 3)

- **Key.** `canonical_resonance_key` is the smallest canonical SMILES over the resonance
  structures of the molecule. The molecule first has its atom maps stripped, then is rebuilt
  from its own canonical SMILES (canonical atom order and Kekulé structure). Each atom keeps
  the H count it had, so RDKit does not add implicit H inside the enumerated structures.
  - Before, the key kept maps and depended on input atom order: phenoxide gave two keys under
    renumbering.
  - Without the H freeze, a terminal diazo N gained two H.
  - Tests: 30 random renumberings of phenoxide, a dienolate, an enone enolate and allyl give
    one key each. Acetone deprotonated at C1 or C4 gives one key.
- **Resonance step.** `is_resonance_step` is true when the two sides' map-free structure sets
  intersect, not when their two minima are equal. Under the canonical order RDKit enumerates
  the neutral C=N-X from C(+)-N(-)-X but not the reverse. With minima, 12 oxime / hydrazone
  pairs that the dataset's atom order happened to connect would have been lost.
- **Effect on PMechDB** (482 same-skeleton steps; not re-ingested yet, counted against the
  stored flags):
  - 10 steps become resonance steps and none stop being one. The 10 are a dienolate shift,
    two acetylacetonate shifts, six Zn / Sn / Sc-bound oxocarbenium ↔ carbocation pairs, and
    a proton between the two equivalent oxygens of a protonated sulfonic acid.
  - The 6 truncated enumerations are unchanged (still not counted; review D7).
  - The resonance definition question (D6: RDKit misses C=O ↔ C(+)-O(-) and lone pair →
    adjacent cation) is open and not changed here.
- **Round trip.** Dataset arrows: 12,840 / 12,840 (12,839 exact, 1 modulo resonance). It is now
  map-free, so a product formed on a symmetry-equivalent atom also matches (the user's choice;
  the review had proposed keeping a mapped key for round trips).
- **To do at the rebuild.** `steps.parquet` stores `resonance_step` from ingest, so the rebuild
  starts with `ksmech-data ingest --source pmechdb`. Provenance then marks everything downstream
  stale.

## 2026-10-22 — Degenerate canonical levels summed in f+/f-, omega and the spectral set (fix item 4)

- **The rule** (one tolerance, 0.05 eV, the anchor rule of `frontier.frontier_blocks`):
  canonical MOs within 0.05 eV of a level's anchor form one level. Inside a degenerate level
  the canonical MOs are an arbitrary basis chosen by the eigensolver, so any per-MO quantity
  is arbitrary and only the sum over the level is physical.
- **omega** (`fmo_weights`, HOMO-1 … LUMO+1) was already summed over the block; unchanged.
- **Frozen-orbital f+ / f-** (`AtomToken.f_plus_fmo` / `f_minus_fmo`, model inputs) now come from
  `frontier.frozen_orbital_fukui_blocks`: the LUMO / HOMO block population divided by the block
  size.
  - So f sums to 1 over the atoms, like a Fukui function, and benzene's six carbons get equal
    values (before: 0.273 / 0.224 / 0.002 from one MO of the e1g pair). 171 / 2,005 cached
    molecules were affected (review Q1).
  - The plain block sum would duplicate the omega HOMO / LUMO columns, hence the division.
  - The QM record's own `fukui_fmo_plus/minus` (one MO each) are no longer read by the tokens.
    They are left in the record rather than changing its schema.
- **Spectral set** (`OrbitalDescriptors.spectral_*`, set-encoder input).
  - p_rk is summed over each degenerate level before the 0.01 / top-8 cut. An entry's energy
    is the weight-averaged energy of its level, its index the level's heaviest MO.
  - Methane's four C-H tokens now carry one identical set, (-12.6 eV, 0.75), (-15.7 eV,
    0.25) at the fixture geometry. Before, it was split differently over the t2 MOs per token
    (review O1: 133 tokens in 9 of 400 cached molecules).
  - G still uses the full per-MO vectors (`full_weights`), which it needs and which are
    invariant inside a level anyway (G is linear in p).
  - The tolerance is `DescriptorConfig.degeneracy_ev`, so it is part of the token provenance
    config.
- **Rebuild.** Token `SCHEMA_VERSION` 7 → 8. Every existing token file is now stale and
  `ksmech-orbitals build` rebuilds it after the QM backfill. Nothing has been rebuilt yet.

## 2026-10-22 — Intrinsic bond orbitals are the default localization (fix item 4b, spec v7.2)

- **Method** (`localize.PMConfig(method="ibo", start="cholesky")`, the default):
  - The functional is sum_i sum_A (Q_A^i)^4 over Löwdin populations Q_A^i =
    sum_{mu in A} (S^1/2 L)_{mu i}^2 (Knizia, JCTC 9, 4834, 2013).
  - **Why Löwdin populations are the IAO populations here.** In GFN2-xTB's minimal valence basis
    the IAO reference minimal basis is the computational basis itself. The IAO projector is
    then the identity, the intrinsic atomic orbitals are the symmetrically orthogonalized AOs
    S^1/2, and IAO populations are Löwdin populations.
  - The functional is maximized by the same vectorized disjoint-pair Jacobi sweeps, with the
    same 1e-8 convergence and the same 200-sweep cap.
  - The occupied, SOMO and virtual blocks are localized separately, as before.
  - `method="pm_mulliken"` (Mulliken, exponent 2, closed-form angle) and `start="canonical"`
    stay behind flags. Both enter the token provenance config, so a token file built one way
    is stale for the other.
- **Rotation angle.** A bracketed 1-D maximization per pair, not Knizia's closed form.
  - Reason: Knizia's closed-form angle keeps only the first harmonic in 4γ, while the exact
    exponent-4 gain also has an 8γ term.
  - The exact gain 12 m²(x² − d²) + 2(x⁴ − d⁴) over atoms, with x = d cos2γ + q sin2γ, is
    evaluated on a 32-point grid over its period [−π/4, π/4).
  - The grid contains γ = 0, so no pair can lose. The best point is refined by up to 4 Newton
    steps, each kept only if it improves the gain.
  - Test: it agrees with a 20,001-point scan to 1e-9.
- **Start.** A pivoted Cholesky factorization of each block's density matrix in the Löwdin
  basis: pivot at the largest remaining diagonal, first index on ties. The resulting orbitals
  are made orthonormal symmetrically, then swept.
- **Measured** on 200 PMechDB molecules (every 70th record, single process) plus the 35 that
  hit the PM cap in the 2026-10-20 build. Data in
  `data/processed/pmechdb/diagnostics/localization_pm_vs_ibo.json`.

  | | PM (Mulliken, p=2, canonical) | IBO (Löwdin, p=4, Cholesky) |
  |---|---|---|
  | occupied sweeps median / p95 / max | 7 / 92 / 200 | 6 / 8 / 42 |
  | virtual sweeps median / p95 / max | 8 / 89 / 200 | 6 / 8 / 16 |
  | molecules at the 200-sweep cap | 2.0% | 0% |
  | the 35 PM-capped molecules: sweeps (occ / virt median, max) | 189 / 200, 200 | 8 / 7, 12 |
  | seconds per molecule, median / mean | 0.15 / 0.85 | 0.56 / 2.67 |

  - **Starts** (first 50 molecules, IBO): sweeps median / p95 / max are 6 / 8 / 19 (occupied)
    from the Cholesky start and 6 / 9 / 16 from the canonical start. The Cholesky start is
    deterministic and not slower to converge, but it is not a large gain here.
  - **The cap count.** The prompt said 18 molecules; the 2026-10-20 token build has 35 at the
    PM cap (17 occupied blocks, 26 virtual). All 35 converge under IBO, and
    `test_molecules_that_hit_the_pm_cap_converge_under_ibo` (slow, realdata) asserts it.
  - **Cost.** IBO needs far fewer sweeps but costs ~4x more per molecule today: each round
    evaluates the exponent-4 gain on the 32-point grid. Fix item 10 (Jacobi vectorization,
    target < 0.3 s per typical molecule) covers both methods.
- **Benzene: the expectation in the prompt and in spec v7.2 does not hold in this basis.**
  - IBO converges in 5 sweeps (PM: 6 at this geometry) with no cap. The three occupied π
    tokens are not two-centre Kekulé bonds; they are three-centre (0.50 on one carbon, 0.22 on
    each neighbour). The virtual π* tokens look the same.
  - This is the maximum of the functional, not a stalled sweep. On the occupied π space the
    three-centre set scores sum Q^4 = 0.202. The explicit Kekulé π sets, projected into the
    same space and orthonormalized, score 0.177. Three random rotations of the occupied block
    all converge to the same total (1.6655362), with only two-centre σ and the three-centre π.
  - So with Löwdin populations in GFN2's minimal basis the exponent-4 functional prefers the
    alternating three-centre π form; the Kekulé preference reported for IBO comes from IAOs
    built against a large basis.
  - The test pins what holds: convergence in under 60 sweeps, three π tokens (χ < 0.2) per
    block, and the e1g block weight in the occupied π tokens = 2.000 ± 1e-3. The two-centre
    expectation is kept as a strict xfail with this reason.
  - Across the 200 molecules the fraction of occupied tokens that are two-centre with χ < 0.3
    *falls* in aromatics (91 molecules: 6.2% PM → 3.5% IBO) and is flat elsewhere (3.9% →
    3.7%). Centre counts over all occupied tokens: PM 1 / 2 / 3 = 15.1 / 82.3 / 2.7%, IBO
    15.3 / 80.4 / 4.3%. The χ quantiles are unchanged (σ-dominated).
  - **The spec sentence "Benzene then converges in a few tens of sweeps to three two-centre π
    bonds" should be corrected** (the user's call).
- **Determinism.** Two builds of the same record give byte-identical array members; the
  provenance header differs only in `built_at` (`test_token_files_are_byte_identical_across_two_builds`).
  The fixture tests (acetone, chloromethane, methylamine, acetonitrile and the slow sanity
  set) pass unchanged.
- **Rebuild.** Token `SCHEMA_VERSION` 8 → 9. Nothing has been rebuilt yet.

## 2026-10-22 — The fixed-2.3 kernel is frozen as `physics/contact/legacy_fixed23.py`, rotation-invariant (fix item 5)

- **Move.** `physics/oriented.py` is now `physics/contact/legacy_fixed23.py` (git history kept),
  and `physics/contact.py` became the package `physics/contact/__init__.py`, unchanged. Imports
  were repointed. The CLI kernel name stays `oriented`; the contact mode stamped in the block
  provenance is `legacy_fixed23_v6`.
- **What depended on the lab frame** (review PK1: rotating both records by one R moved |S| by
  up to 4.8%, beta by up to 13%, and 25% of the argmax sites):
  1. the lobe of a site was the best of 200 Fibonacci directions fixed in the lab frame;
  2. the cap's roll about the approach axis was the smallest rotation taking the acceptor lobe
     onto the donor lobe. That depends on the two records' relative orientation, so rotating
     only one record changed it.
- **Fix** (spec v7.2, "Legacy fixed-distance placement": directions and rolls in a local frame
  built from the site and its bonded neighbours):
  - Each (token, site) gets `site_frame`. e3 is along the bond to the first bonded neighbour
    (by atom index); e1 is the next bond made perpendicular, else a second-shell bond, else
    the token's own p vector on the site.
    - Lab axes complete the frame only when none of these exists; such cases are counted in
      `FRAMES_WITHOUT_LOCAL_AXES`.
    - All of these vectors rotate with the molecule, so the frame does too.
  - The 200-direction grid is laid out in that frame (`frame_directions`).
  - Roll 0 maps the acceptor's (lobe, frame reference) onto the donor's (−lobe, frame
    reference) (`frame_rotation`). The six rolls are counted from there.
- **Tests** (`tests/test_legacy_fixed23.py`):
  - Rotating each fragment by its own random rotation leaves S and beta unchanged (rtol 1e-7)
    and the sites and CLASH flags identical, for C[O-] × CCl and C=C × C=O. Rotating acrolein
    leaves its intramolecular block unchanged.
  - With the lab frame patched back in, all three tests fail, so they detect the bug.
  - The outputs of the three pairs at 2.3 Å are pinned in
    `tests/fixtures/legacy_fixed23_pinned.npz` (version 6). Any change to them needs a new
    version.
  - The brute-force checks now take the placement rotation from the kernel instead of
    recomputing the smallest rotation.
- **Rebuild.** `ORIENTED_VERSION` 5 → 6, so every block on disk is stale (provenance) and is
  recomputed at the rebuild.

## 2026-10-22 — Labeled pairs that the site filter removes: counted, flagged, never the teacher (fix item 6)

- **Groups first.** The 2026-10-21 samples (PM tokens, random split) had 706 labels and 3,614
  labeled (donor, acceptor) cells; `pair_valid` rejected 115 of them. Grouped by which side
  carries at least the site weight (0.20) on the other's site:

  | cells | group |
  |---|---|
  | 41 | C → C, the acceptor on the donor site (both two-centre) |
  | 18 + 17 + 16 + 3 + 1 | O → C / O → S, the acceptor on the donor site (one-, two- and three-centre donors) |
  | 8 + 5 + 1 | O → C, the donor on the acceptor site |
  | 4 + 1 | O → C, other (weights 0.18-0.20, at the threshold) |

  - Every group is the antibonding partner of the bond the arrow forms, e.g. n(O) →
    σ*/π*(C–O) of the very C–O bond. That is the interaction the filter exists to exclude, so
    the rule is **not** narrowed.
  - No label had all of its cells removed. In 2 / 549 train labels the old teacher pair was
    one of the removed cells.
- **Rule** (`arrows.steps.filter_labels`, run on every sample):
  - A donor (acceptor) with no valid partner in the other set leaves its set, and the label is
    flagged `LABEL_FILTERED`.
  - Every rejected cell is counted by group in `MechanismSample.label_filter`. The counts reach
    the per-split reports (`n_labels_filtered`, `label_filter`) and `samples_report.json`, and
    `ksmech-arrows samples` prints the groups.
  - A label with no valid cell skips its step as `SkipReason.LABEL_FILTERED`. It is counted
    and logged, never dropped silently.
  - The teacher pair (r*, s*) is chosen among valid cells, with the alignment's tie-breaks
    (site weight, then energy), and stored on the label (`teacher_donor` / `teacher_acceptor`,
    review A9). `ranking.teacher_cell` uses it.
- `SAMPLE_VERSION` 9. The groups are reprinted on the IBO tokens at the rebuild.

## 2026-10-22 — Textbook suite: a GBDT with the five reactant pools held out (fix item 7)

- **Why.** Three of the five textbook reactant pools were in the GBDT's training samples as
  exact steps (review R1): CN- + acrolein, MeNH2 + AcCl and H3O+ + acetone. So the "GBDT 4/5"
  result and the suite's exit gate partly measured memorization.
- **Pools.** `textbook.textbook_pools()` gives the case name and the sorted map-free fragments.
  `pools_in(fragment_keys)` lists the cases whose whole pool (as a multiset) is among a step's
  fragments; half a pool is not the case.
- **`ksmech-physics baseline`** fits two GBDTs:
  - `baseline_gbdt_<split>.npz` is fitted on all training samples. Its header records
    `extra.textbook_pools_in_training`.
  - `baseline_gbdt_<split>_textbook-held-out.npz` is fitted and selected with every train and
    val sample that holds a pool removed (test untouched). Its header records `held_out` and an
    empty in-training list.
  - Both appear in the report and the table, and the report lists the pools seen in train with
    counts.
- **`ksmech-physics textbook`** runs the untrained scorer and both GBDTs. A case whose pool was
  in a model's training data is marked "seen in training" in that model's table. Its pass is
  printed separately as "not a test" and does not count. The exit code needs 4 passes among
  held-out cases from one scorer.
- The full textbook numbers come with the rebuild (new samples, molecule split).

## 2026-10-22 — Sharded sample store, valid pairs only, float16 where safe (fix item 8)

- **Layout** (`arrows.store`):
  - Every step's sample is stored once in compressed `.npz` shards of 256 samples
    (`data/processed/<source>/samples/shard_*.npz`) plus `index.json` (step id → shard).
  - Each split is a small view `samples_<split>.json`: its ids and counts.
  - Nothing is copied per split; before, random and scaffold each held a full copy, and loco
    would have added one per class.
  - Each layer has a provenance header:
    - shard: the steps parquet and every token file and contact block its samples used;
    - index: the shards;
    - view: the index and the split file.
  - `load_samples(view)` verifies the view, its split file (the parquet), the index and every
    shard it reads.
  - `build_samples` (one split, used only by tests) is gone; `ksmech-arrows samples` builds the
    store and all views.
- **What is stored per sample.** In memory a sample is unchanged: dense, float32.
  - The pair arrays (`pair_z`, `pair_defined`, `pair_gap_sign`, `pair_contact_sites`) are kept
    for the `pair_valid` cells only (89% of the cells on the old samples). Invalid cells come
    back as 0 / undefined / -1.
  - `ranking.refresh_validity` can therefore only narrow `pair_valid`, never widen it.
  - The 0/1 features `inter` and `clash` and `pair_defined` are bit-packed. Contact sites are
    int16, and the symmetric `atom_coulomb` keeps its upper triangle.
  - `pair_z` (the other 9 features) and `atom_coulomb` are float16. Their bytes are stored
    feature by feature, with the high and low bytes in separate planes. This is lossless and
    compresses about 20% better under zlib.
- **Why float16 is safe here.** Measured on the 670k valid cells of the old samples:
  - the largest absolute float16 error on any pair feature is 5e-3, on log-features spanning
    about 30 units;
  - relative errors near 1 occur only below 1e-4 (`ks2`, `cross_coupling`), where float16 is
    subnormal and the absolute error stays under 1e-7.

  All of this is far below one standard deviation of any feature. Model inputs
  (`orbital_features`, `orbital_spectra`, `atom_features`) stay float32.
- **Size.** Re-encoding the 454 old samples: 67.3 MB as torch files → 15.2 MB stored (ratio
  0.225; 0.209 on the larger half).
  - The arrows reviewer projected 4.3 GB per copy for all 13,090 steps. One store at these
    ratios is **0.90-0.97 GB**, against about 8.7 GB for the old two copies. The target was
    < 1 GB.
  - A 60-heavy-atom sample is about 0.32 MB on disk (1.8 MB before).
  - The real number is measured at the rebuild.
- `pair_feature_stats_<strategy>.json` is now fitted on the valid cells only (it never sees
  the others), as the review asked (A11).

## 2026-10-22 — Deletions and hygiene (fix item 9)

- **Vbur-scaled contact KS1 deleted** (spec v7.2: "No buried-volume factor enters").
  - The contact form is now (N_m(r) + N_m(r')) |beta_c(r, r')| a(r, r'). The V_m(r) V_m(r') /
    1e4 factor is gone, and with it review PK3: one NaN %Vbur used to void a whole KS1 block.
  - A test now checks one contact cell against that formula (review PK10).
  - The per-atom %Vbur (morfeus) stays an atom-token model input.
  - Expect the intra / inter KS1 scale gap (medians 7.40 vs 0.51 eV with the factor) to shrink;
    it is measured at the rebuild.
- **MLP contact kernel deleted:**
  - `physics/contact_mlp.py`, `ksmech-physics contact-mlp` and `--kernel mlp`;
  - `--contact-mlp` on `ks-rank` and `samples`;
  - the `ContactKernel` mlp branch, its tests and the CLAUDE.md lines.

  The data file `data/cache/contact_mlp.npz` is left on disk (data is not deleted by code
  changes); nothing reads it.
- **Dead code removed:**
  - `contact.oriented_contact_overlap` (a stub);
  - `fukui.nan_fukui`, `build.rows_to_dicts`, `OrbitalDescriptors.to_vector`;
  - `classify.OCCUPIED_TYPES` / `VIRTUAL_TYPES` (no callers);
  - the whole-molecule acceptor `_Side` built per block by the legacy kernel, of which only the
    coordinates were used (now `_Anchors`). The pinned outputs are unchanged.
- **Dead flags.** Every option of every `ksmech-*` CLI is read (checked by scanning each
  `add_argument` dest against `args.<dest>`); none was left over.
- **Function length.** No function in `src/` exceeds 60 lines; the longest is
  `orbitals.build.summarize` at 58.
- **Annotations.** Every public function in `ksmech.data` and `ksmech.arrows` is annotated
  (mypy --strict: 0 `no-untyped-def` on public functions there; 4 private helpers remain).
  - mypy --strict over all of `src/` reports 453 errors in 46 files, against 425 in 47 before
    the fix branch.
  - The rise comes from the new provenance, store, IBO and frame code (bare `dict` / `Callable`
    type parameters, untyped numpy returns). The prompt limited annotation work to the public
    functions of data/ and arrows/.

## 2026-10-22 — Jacobi sweeps: closed-form exponent-4 coefficients, reduceat atom sums (fix item 10)

- **Profile.** Before this change, 90% of an IBO localization was `_ibo_gain`: the
  exponent-4 gain evaluated over every atom at 32 grid angles plus 4 Newton angles, every
  round.
- **Closed form.** The gain of one pair is exactly a trigonometric polynomial,
  a4 (cos4γ − 1) + b4 sin4γ + a8 (cos8γ − 1) + b8 sin8γ.
  - With x² = A0 + A1 cos4γ + B1 sin4γ (A0 = (d² + q²)/2, A1 = (d² − q²)/2, B1 = d q), the
    expansion 12 m² x² + 2 x⁴ gives a4 = Σ_A (12 m² + 4 A0) A1, b4 = Σ_A (12 m² + 4 A0) B1,
    a8 = Σ_A (A1² − B1²) and b8 = Σ_A 2 A1 B1.
  - A round therefore sums over atoms once (`ibo_coefficients`). The same grid and Newton steps
    then run on four numbers per pair.
  - A test checks that the polynomial equals the direct gain to 1e-12. The angle test against
    a 20,001-point scan still passes.
- **Atom sums.** `np.add.reduceat` over the atom boundaries replaces the dense (n_atoms × nao)
  indicator product, in IBO and in PM. Tblite's AO map runs atom by atom; an unsorted map falls
  back to the indicator, and a test covers both. S^1/2 is computed once per block and shared by
  the Cholesky start and the sweeps.
- **Measured** on the same 200 PMechDB molecules, one thread:

  | seconds per molecule | median | p75 | p95 | mean |
  |---|---|---|---|---|
  | token build, IBO, now | **0.20** | 0.39 | 0.80 | 0.34 |
  | token build, IBO, before | 0.56 | 1.76 | 5.86 | 2.67 |
  | token build, PM (2026-10-20 code) | 0.15 | 0.56 | 3.03 | 0.85 |
  | localization alone, IBO, now | 0.17 | 0.34 | 0.71 | 0.30 |

  - Sweep counts are unchanged (5 / 5, 12 / 8 on the profiled molecules), and no molecule is at
    the cap.
  - Token arrays are still byte-identical across two builds
    (`test_token_files_are_byte_identical_across_two_builds`). They differ from the previous
    commit's at the last bits (summation order). That needs no schema change: no tokens were
    built with that commit.
  - The full PMechDB token build is projected at about 1.3 CPU-h, against about 4.5 before.
- **The legacy-kernel pin now holds the kernel's inputs fixed.** The first version of
  `test_outputs_are_pinned` rebuilt its tokens on every run, so it also pinned the
  localization. IBO stops when the functional changes by less than 1e-8 per sweep, which fixes
  the LMOs only to about 1e-4 in rotation angle. The new summation order therefore moved S by
  about 1e-6, against the pin's rtol of 1e-9.
  - The five fixture molecules' tokens are now stored with the pin
    (`tests/fixtures/legacy_fixed23_tokens/`, 120 KB, read with `verify=False`). The pin is
    regenerated from them.
  - The kernel code is unchanged (version 6). The rotation tests still build fresh tokens: they
    compare a record with its own rotated copy.
- `test_site_lobe_rules` computed an angle with `arccos` and no clip. An H site's lobe points
  exactly away from its single neighbour, so the cosine came out as −1 − 1 ulp → NaN. The
  helper now clips.

## 2026-10-22 — Predicted-arrow round trip: diagnosis and rules (fix-plan round-trip steps 1-4)

- **The gate.** 95%, kept. It guards inference and evaluation, not sample building (user,
  2026-10-22).
- **Dataset arrows** (`apply_arrows` on the annotated arrows, compared by the map-free
  resonance key): 12,830 / 12,830 training steps (12,829 exact, 1 modulo resonance). The
  training labels are sound, so the sample build went ahead.
- **Predicted tuples, before** (the 2026-10-20 rule, IBO tokens, all 11,865 completely aligned
  steps): 11,045 = 93.09%. All 820 failures reproduce with dataset arrows, so they come from
  turning a tuple into an edit, not from the data.
  - Grouped by (donor centres, acceptor centres, sites bonded, intramolecular, arrows,
    conjugated site, truncated):
    - **470 at a hypervalent site:** a σ* at Si / P / S / Sn, where PMechDB forms the bond
      without breaking b–d;
    - **107 at a one-centre empty Ti / Sn orbital:** the old rule broke a Ti–Cl bond;
    - **26 σ(O–C) donors treated as an O lone pair;**
    - **30 own-bond teachers:** the round trip used the unfiltered teacher, σ*(S–O⁻).
  - PM tokens give the same rate on a 1,391-step probe (92.0% vs 92.1%), so localization is
    not the cause.
- **The rule as first specified** (centres decide source and sink; tuples read on the unedited
  molecule): 67.2% (IBO), 67.9% (PM probe). It fixed 131 old failures and broke about 3,200
  steps:
  - relays (E2, enolization), because reading every tuple on the unedited molecule moves the
    C–H pair twice;
  - carbocation sinks (no open-shell guard);
  - conjugated carbanion donors (a single bond consumed).

  The old code already built one edit sanitized once; its sequential reading is what made the
  relays work.
- **The rule now in `apply_predicted`** (v4, rules only, module docstring):
  - the old reading order and pair accounting;
  - source:
    1. lone pair if r is one-centre or a is negative;
    2. else the largest-weight bonded centre of r if the bond is multiple or r is σ-like
       (χ ≥ 0.5);
    3. else a bond no occupied token spans;
    4. else a lone pair; else the flagged Kekulé fallback;
  - sink:
    5. open shell: nothing breaks;
    6. a one-centre acceptor on a period-3+ atom: nothing breaks;
    7. else a σ* breaks to its bonded centre (multiple bonds first, then weight), or the bond
       whose pair sits mostly off b;
  - the round-trip teacher is chosen among valid pairs (fix item 6).

  | rule | IBO, all 11,865 | IBO probe 1,391 | PM probe 1,391 |
  |---|---|---|---|
  | 2026-10-20 | 93.09% | 92.09% | 92.02% |
  | as first specified | 67.17% | 68.08% | 67.94% |
  | refined (no tokenless / fallback / multiple-first) | 92.63% | 91.45% | 91.37% |
  | v3 (+ those) | 93.38% | 92.52% | 92.52% |
  | **v4 (+ octet capacity)** | **94.19%** | 93.46% | 93.31% |

  The Kekulé fallback never fired.
- **What remains (689 = 5.81%):** 615 have a σ* acceptor at a site that can expand its octet,
  where the tuple (r, a, s, b) does not say whether b–d breaks:

  | site | dataset keeps b–d | dataset breaks b–d |
  |---|---|---|
  | Si | 175 | 0 |
  | P | 234 | 56 |
  | S | 140 | 171 |
  | I | 27 | 25 |
  | Br, Cl | 0 | 178 |

  The other 74 are scattered.
  - Under the current rule about 577 of these fail (P 233, Si 175, S 140, I 27, Sn 2: the
    dataset keeps the bond) and 38 more where it breaks a different bond.
  - An element rule ("Si and P keep") would bring the rate to about 97% (projection: +175 Si,
    +233 P, −49 P steps that break), but S and I stay split.
  - One more decoder output per arrow (break b–d: yes / no) resolves all 615.
  - Open: element rule or decoder pointer (user decision).
- **Decision (user, 2026-10-22): a per-arrow bond-break flag, no element rule.** `ArrowLabel`
  carries `breaks_bd` and `breaks_bd_forced` (`apply_predicted.bond_break_label`):
  - forced 0 when s is one-centre (an empty orbital of b's own) or b has an open shell
    (the open-shell case is an addition: rule 5 already decides it);
  - forced 1 when b is a full-shell atom of period 1-2 (H, C, N, O, ...; H counts as period 1);
  - learned when b is in period 3+ and s is antibonding with a centre d bonded to b: 1 when
    the annotated step breaks b–d (`sink_partners`), else 0;
  - learned too when b is in period 3+ and s is antibonding but its partner is the donor's
    own atom (the arrow makes that bond multiple: σ(O–H) → P=O, O⁻ → S=O): 1 when the
    annotated step breaks any bond at b, rule 7's weakest bond then says which. s is an
    antibonding two-centre token there, so the user's definition gives it a flag; a first
    version left it None and the rules kept it ambiguous (111 of the 206 failures at 98.26%);
  - None otherwise (s not antibonding, or b bonded to nothing but the donor's atom).

  Only learned flags drive the edit; forced flags are stored as masks and left to rules 5-7,
  which already decide them. Applying the forced flags statically gave 91.16%: in a relay a
  later arrow's sink is read on the edited state, where the forced value from the unedited
  molecule can be wrong (an open shell that the previous arrow filled).

  | rule | IBO, all 11,865 | IBO probe 1,391 | PM probe 1,391 |
  |---|---|---|---|
  | v4 + forced and learned flags applied | 91.16% | | |
  | v4 + learned flags, partner = donor atom left None | 98.26% (11,659) | 97.77% | 97.84% |
  | **v4 + learned flags** | **99.03%** (11,750) | | |

  At 98.26%, old failure groups fixed: Si 199/199, S own-bond teachers 58/58, and 33/33, 30/30, 29/29,
  84/84, 26/26, 23/23 for the next six; the two-arrow P group 23/72; the intramolecular S/Cl
  two-arrow group 1/30. The round trip passes the 95% gate.
- **What still fails (115 = 0.97%).** None reproduces with the dataset arrows.
  - **51 at a P / S sink that loses a bond the tuple does not name.** The flag says a bond
    breaks, but not which one:
    - 20: s is σ*(b–x) with x kept, and another bond at b breaks (the teacher s* is the σ*
      with the largest weight on b, not the one toward the leaving group);
    - 20: s is σ* toward the donor's atom and rule 7's weakest bond is not the leaving one
      (sulfonyl chlorides: S–OR is chosen, S–Cl leaves);
    - 11: s is one-centre (forced 0) but the step breaks a bond at S / P.

    Underdetermined by (r, a, s, b, flag). Two remedies, user decision: a teacher rule that
    picks σ*(b–leaving) as s* when it is a valid acceptor, or a pointer to d.
  - **12 N–P (Mitsunobu betaine, intramolecular):** the P–N bond the dataset breaks is not the
    one rule 7 picks.
  - **About 18 C–Cl / C–Br, intramolecular two-arrow steps:** a full-shell C sink where the
    weakest bond is not the leaving halide.
  - The rest are scattered (Li–O 3, S–F 2, single cases).

## 2026-10-22 — Provenance: a dirty-tree flag; training and evaluation refuse dirty builds

- **Header fields.** `code_dirty` and `code_diff_sha256` sit next to `code_git_hash`
  (`provenance.code_state`). The hash covers `git diff HEAD` plus the contents of untracked
  files under `src/`: a build can import a module that is not committed yet, and that counts
  as dirty too.
- **Who refuses.** `require_clean` raises STALE_ARTIFACT for a dirty build. These loaders call
  it:
  - the sample store (view, index, every shard: `store._verify`);
  - the contact sidecar;
  - the GBDT (`load_gbdt`).

  `--allow-stale` (diagnostic CLIs only: `ks-rank`, `textbook`) logs the refusal and reads on.
- **Retroactive.** Headers written before the flag carry `+dirty` in `code_git_hash`.
  `Provenance.from_dict` reads that as `code_dirty = True`, so the sample build started on
  2026-10-22 from an uncommitted tree (`--workers`) is refused like any new dirty build. Its
  oriented contact blocks stay in `data/cache/oriented/` as a cache. The samples used for
  training are rebuilt from a committed tree.
- **Tests.** Artifacts written in tests are stamped clean by an autouse fixture, since the
  commit hook runs on a tree that differs from HEAD. `code_state` itself is tested on a
  scratch repository.

## 2026-10-22 — Sample build split: stage-A samples and a legacy contact sidecar

- **Why.** The legacy oriented contact kernel dominates the sample build: about 40 h for
  PMechDB with 3 workers, almost all of it in the oriented blocks. Spec v7.2 makes the stage-A
  pair features geometry-free. Only the physics floor and the GBDT baseline read the legacy
  contact features, and they read only val, test and part of train (user, 2026-10-22).
- **Samples (`ksmech-arrows samples`, SAMPLE_VERSION 11).** The pair features are the stage-A
  set (`physics.stage_a.STAGE_A_FEATURES`):
  - G and log|G|, the fragment flag;
  - bonded, bond order and graph distance between the two centre sets on the record's Lewis
    structure (capped at 6; across fragments 0, 0, 6);
  - centroid distance and the cross-centre coupling |l_r^T B l_s| within a fragment;
  - |cos| of the two p vectors;
  - the dihedral cosine of adjacent bonds.

  They are computed at K = eps = 1 like the old features. `to_params` scales the coupling by K;
  the rest are parameter-free. `MechanismSample.pair_features` names the columns. The store
  keeps `inter`, `clash` and `bonded` as bits, and `pair_feature_stats_<strategy>.json` is
  fitted on the stage-A features.
- **Sidecar (`ksmech-arrows contacts --split <strategy>`, `arrows.contacts`).**
  - The 11 legacy features (`PAIR_FEATURES`, oriented kernel at 2.0 / 2.3 / 3.0 Å) and contact
    sites.
  - It covers every val and test step of the strategy and a training subsample of 3,000
    steps: seed 0, drawn without replacement from the sorted ids of the training part,
    recorded in `samples/contacts_<strategy>/train_subsample.json` with its provenance.
  - Storage is the store's: valid cells, float16, bit-packed `defined`.
  - Every shard is stamped with the token files and contact blocks it used, and loading refuses
    dirty or stale shards.
- **Readers.** `ksmech-physics ks-rank` and `baseline` join each sample with its sidecar entry
  (`with_legacy`), print how many samples of each part have none, and so rank and train on the
  subsample. `physical_pair_z` refuses a stage-A sample. `textbook` builds its cases with the
  legacy kernels directly (`reactant_sample(..., kernels)`). The GBDT records the sidecar index
  and the subsample file as inputs.
- **Build order.** The sample build started on 2026-10-22 from an uncommitted tree runs to the
  end as a cache warm-up for `data/cache/oriented/`; its blocks are not deleted. Its outputs
  are dirty and are not used. The stage-A samples are rebuilt from a clean committed tree,
  then the sidecar.

## 2026-10-23 — Labels: leaving-arrow refinement, WBI leaving rule, sequential masks

Two user decisions (2026-10-22 and 2026-10-23) replace the weakest-bond rule and the
"sink loses any bond" extension of the bond-break flag.

- **Acceptor labels (`align.acceptor_set`).** When another arrow of the step moves the pair of
  a bond b–x away from the sink b, the acceptor set is restricted to the antibonding tokens
  (negative overlap population) on b and x. The restriction applies only when such a token
  exists; otherwise the previous set is kept (tokens on b and x, else every token on b,
  SINK_RELAXED).
  - Such a token rarely exists where it matters. Of 143 labels with a leaving bond and no
    antibonding token on it, 140 have no virtual token on b and x at all, and 125 have no
    occupied one either.
  - The pairs are C–Cl 57, S–Cl 23, C–Br 21, S–O 13, P–Cl 5 and P–O 4: tetrahedral
    intermediates (`CN[C](C)([O-])Cl`), bromonium ions, sulfonyl and phosphoryl chlorides with
    an O⁻.
  - In those records the leaving bond is already nearly ionic: the leaving atom carries only
    one-centre lone pairs, at about 1.2 times the covalent-radius sum.
- **Wiberg bond indices (`qm.bond_index`, `ksmech-qm bond-index`).**
  WBI_ab = Σ_{μ∈a, ν∈b} (S^½ P S^½)²_μν, with P the record's total density and the Lewis bonds
  of `prepare_mol(atom_order_smiles)`. Sanity values: H2 1.00, ethane C–C 1.03, C=O 2.00,
  C≡N 2.83, benzene 1.44.
  - They are kept in a per-record sidecar (`data/cache/bond_index/<cache key>.npz`, stamped
    with the QM record) and attached to the tokens at load time (`MoleculeTokens.bond_index`).
  - They are not stored in the token file: its content hash keys the contact-block cache,
    which the warm-up build is filling.
  - `TokenCache` treats a molecule without its sidecar as missing. 14,343 records, 0 failures.
  - Stage-A pair feature `wbi`: the largest WBI over the Lewis bonds joining the two centre
    sets, 0 when they are not bonded.
- **Sink rule (`apply_predicted.sink_decision`).** The only bond that can break is b–d.
  - d is the partner of s (an antibonding s's largest-weight centre bonded to b, multiple bonds
    first, never the donor atom).
  - When s names none (one-centre, not antibonding, or its partner is the donor atom), d is the
    Lewis bond at b with the lowest WBI, excluding the bond to the donor atom. A tie within
    0.02 is flagged LEAVING_AMBIGUOUS; it never occurred on PMechDB.
  - The flag decides whether b–d breaks:
    - open shell: 0;
    - full-shell period 1–2: 1, whatever s is;
    - period 3+ with a one-centre s: 0;
    - otherwise learned: 1 iff the annotated step breaks b–d.
  - The full-shell rule now runs before the one-centre rule. A bromonium carbon whose teacher
    acceptor is a one-centre virtual must still lose a bond; this was 22 VALENCE failures.
- **Sequential masks (`break_labels`).** Forced and learned flags are labeled on the molecule
  as edited by the earlier arrows of the step, in the order `apply_predicted_arrows` reads
  them. Applying forced and learned flags together gives the same product as learned flags
  alone on all 11,865 steps (0 disagreements; the gate test asserts it). SAMPLE_VERSION 12.
- **Diagnostic: lowest-WBI bond at the sink vs the annotated leaving bond.** This covers all
  12,590 sinks with a leaving arrow, including those where s names the partner and WBI is
  never consulted. Overall 70.0%.

  | sink | agree | of | ties within 0.02 |
  |---|---|---|---|
  | H | 100% | 5,121 | 0 |
  | C | 47.1% | 6,729 | 813 |
  | N | 90.6% | 244 | 90 |
  | S | 23.8% | 181 | 21 |
  | Br | 100% | 145 | 0 |
  | P | 11.9% | 59 | 27 |
  | O | 81.6% | 49 | 7 |
  | Cl, F | 100% | 36 | 0 |
  | I | 96.0% | 25 | 0 |

  - Most C disagreements are π pairs leaving (C=O⁺ → O 1,728, C=N⁺ → N 365): a multiple bond
    has the higher WBI by construction, and there s is a π* that names its partner.
  - At S and P the lowest WBI is often S–C or P–Cl, not the leaving S–Cl or P=O. The WBI rule
    is a fallback for C-type sinks, not a general leaving-group oracle.
- **Round trip (IBO, 11,865 complete steps).**

  | rule | round trip |
  |---|---|
  | refinement + b–d only, no fallback | 98.47% |
  | + WBI fallback | 98.90% |
  | **+ full shell before one-centre** | **99.04%** (11,751) |

  The 99.03% of the 2026-10-22 extension is the baseline: 1 of its 115 failures passes now and
  none is new.
- **What still fails (114).**
  - 25 S VALENCE (sulfonyl S with an O⁻: the teacher σ* names a kept S–O, flag 0);
  - 12 Mitsunobu N–P and 7 donor-side "no bond and no lone pair" (separate diagnosis);
  - 10 C–Cl and 9 C–Br (the teacher σ* names a C–C partner);
  - 8 S–Cl, 6 P–Cl, 5 S–O;
  - the rest scattered.

## 2026-10-23 — Leaving bond: a categorical choice per arrow (spec v7.5)

User decision, 2026-10-23: the binary bond-break flag and the lowest-WBI rule are replaced by
a leaving-bond choice λ per arrow (spec v7.5, "Leaving bond"; `docs/ksmech_architecture.tex`
is now v7.5, sha256 6c6a023f66dd).

- **Choices.** λ ∈ {NONE} ∪ the Lewis bonds of the sink b other than the one being formed, as
  the molecule edited by the earlier arrows of the step has them when the arrow is read.
- **Mask (`apply_predicted.leaving_choices`).**
  - b had an open shell before the new bond: NONE only;
  - b is full-shell in period 1–2: NONE excluded;
  - otherwise learned.

  A one-centre acceptor at S, P, Si, Ti is no longer forced to NONE. That case is now learned:
  the dataset breaks a bond there in 11 steps.
- **Candidate features** (`LeavingCandidate`): whether it is the partner bond of s*, the
  reactant WBI, the bond order, the element and charge of the departing atom. WBI stays a
  feature (candidates, stage A); apply no longer reads it.
- **Labels (`leaving_labels`).** The bond b–x whose leaving arrow the step has, else NONE.
  - A partner bond that an earlier arrow of the step already consumed is not this arrow's
    leaving bond. In an enolate alkylation the C=C pair moves to the new C–C bond before O⁻
    adds to C=O; that carbon then has an open shell, and its λ is NONE.
  - Before that rule, 1,085 steps had a label outside the mask. After it, 7 do: the 7 steps the
    edit cannot read at all (below).
  - `breaks_bd` survives only as a diagnostic (λ is the partner bond of s*).
- **Apply.** λ breaks exactly that bond, and its other atom takes the pair; NONE breaks nothing.
  SAMPLE_VERSION 13.
- **Label statistics** (11,865 complete steps, 14,753 arrows): 11,324 partner bond, 3,275 NONE,
  154 another bond. 6,153 arrows have more than one choice (λ learned); 8,600 are forced by
  the mask.

  | sink | partner bond | NONE | another bond |
  |---|---|---|---|
  | C | 5,544 | 2,176 | 96 |
  | H | 5,117 | 0 | 4 |
  | S | 140 | 141 | 40 |
  | P | 46 | 234 | 13 |
  | Si | 0 | 243 | 0 |
  | B | 1 | 144 | 0 |
  | Br | 145 | 0 | 0 |
  | Li | 0 | 103 | 0 |
  | Al | 0 | 88 | 0 |
- **Round trip with labeled λ: 99.67%** (11,826 / 11,865), against 99.04% for the WBI rule:
  75 fixed, none newly failing.
- **The 39 left:**
  - 12 Mitsunobu N–P;
  - 7 donor side "no bond and no lone pair";
  - 5 C–Si, 4 C–Sn, 3 Li–O, then single cases (Li–N, C–Li, I–O, Na–O, ...).
- **Diagnosis of the Mitsunobu group (12).**
  - The source is the N–P bond of a betaine: its pair forms N=C and P leaves as P⁺.
  - In all 12 records no occupied token spans N–P (WBI 0.45–0.60), and N carries one
    one-centre lone pair, which is the teacher donor.
  - Rule 1 (one-centre donor → lone pair) runs before rule 3 (a tokenless bond gives its pair),
    so N keeps P and becomes N⁺. (r, a) cannot tell the lone pair from the N–P pair: it is the
    same token.
  - Open: a donor-side counterpart of λ, or rule 3 before rule 1 (measured on all steps first).
- **Diagnosis of the donor-side group (7).**
  - The donor is a hyperconjugated σ bond already leaning into the neighbouring cation:
    β-silyl and β-boryl carbocations, an allylborane, a cyclopropylcarbinyl cation.
  - The teacher r is a 3-centre token (C 0.43, C⁺ 0.36, Si 0.16) with χ 0.05–0.28, below the
    σ threshold of 0.5.
  - Rule 2 rejects it, the bond is not tokenless, and a has no lone pair, so the edit cannot
    read the arrow.
  - Open: before the Kekulé fallback, take the largest-weight centre of r bonded to a whatever
    its χ. This is reached only where every other rule has failed, so it cannot change a step
    that passes now.

## 2026-10-23 — Fast GBDT reference: LightGBM lambdarank, and lean sample loading

- **Why.** The numpy listwise GBDT of `ksmech-physics baseline` does not fit on the corrected
  splits. The molecule run was OOM-killed (exit 137), and the scaffold run took more than 4 h
  at about 9 GB RSS.
  - `_load_parts` held all 9.5k training samples, although only the 3,000 with a contact
    entry are used, and it densified every sidecar shard.
  - The trees saw every valid pair of every training step.
- **The slow baseline stays** as the reference that the fast one is checked against. A new
  command, `ksmech-physics gbdt --features contact|geomfree`, holds the fast one
  (`physics/lgbm.py`).
- **LightGBM** is a new dependency. The user asked for it explicitly; the "prefer numpy" rule
  gives way here because a numpy lambdarank would only re-implement it more slowly.
- **Objective.** `lambdarank`, one query per step, relevance 1 for a labeled pair of the first
  arrow and 0 otherwise. The labeled pairs of the step's other arrows are left out, as in the
  numpy GBDT.
  - Configuration: truncation level 10, learning rate 0.05, 31 leaves, at least 50 rows per
    leaf, feature and bagging fraction 0.8, `deterministic`, seed 0, 2 threads (the host rule:
    at most 3 heavy processes).
- **Negative sampling on training only.**
  - Each step keeps every positive, the 20 negatives the untrained scorer ranks highest, and
    30 drawn uniformly from the rest.
  - The draw uses a generator seeded by (0, crc32(step id)), so it does not depend on the
    order shards are read in.
  - Hard negatives come from the floor −(KS3 + KS2) for `contact`. For `geomfree` they come
    from **−G** (`gap_kernel`), because the KS floor needs the contact features that 6.7k
    training steps do not have.
  - Validation and test keep every valid pair.
- **Early stopping** uses the validation top-5, with top-1 × 1e-3 breaking ties (the numpy
  GBDT's key). Ranks follow the `label_rank` convention: non-labeled pairs scoring at least as
  well as the best labeled one. Patience is 100 rounds, at most 2,000.
- **Feature sets.**
  - `contact`: the 61 legacy columns (`FEATURE_NAMES`), training on the sidecar subsample. It
    is the like-for-like check against the slow model.
  - `geomfree`: the 11 stage-A pair features (`GEOMFREE_FEATURE_NAMES`), with **NaN where
    undefined** (LightGBM routes missing values itself; `align_cos` and `dihedral_cos` are
    often undefined), followed by the same 50 token columns, training on the whole training
    part. It is the like-for-like reference for stage A.
- **Loading.**
  - `load_samples` / `load_contacts` take `ids`. A shard holding none of them is not opened.
  - `iter_samples` yields one shard at a time.
  - `_load_parts` reads only the samples that have a contact entry. `gbdt` reduces each sample
    to its sampled rows (train) or float32 candidate set (val, test) before reading the next.
    Its token cache holds 128 molecules, because tokens are only read for the LMO energies.
- **Models.** `gbdt_lgbm_<split>_<features>[_textbook-held-out].npz`: LightGBM's text model
  with a provenance header (kind `baseline_lgbm`; the sample views, and for `contact` the
  sidecar index and subsample). Report: `gbdt_lgbm_report_<split>_<features>.json`, with
  read/stored counts, training rows, trees, load and fit seconds and peak RSS.
  `textbook --fast` uses the contact models.

## 2026-10-23 — Spec v8: the prediction target is the reaction matrix, not the arrows

- **Decision (user, 2026-10-23).** Curved arrows are no longer the model's output. Each step is
  predicted as its reaction matrix R = ΔB (Dugundji–Ugi): bond-order changes on atom pairs and
  non-bonding-electron changes on atoms, applied as B + R. Arrows and the orbitals that interact
  are a reading of R, reconstructed afterwards.

  The orbital-level target had choices the wavefunction cannot make:
  - an ionic bond's pair and its atom's lone pair are one localized orbital (Mitsunobu N–P,
    O–Li, O–Na, O–Mg; 39 round-trip failures at 99.67%);
  - relays have several readings;
  - which ligand leaves a hypervalent centre.

  Sub-decisions, user:
  - labels come from the dataset arrows, applied as one edit;
  - the target is taken modulo resonance;
  - the arrow code (alignment, tuples, λ, predicted round trip) stays as a diagnostic;
  - I draft the spec and the user approves it.
- **Measured on PMechDB (12,830 training steps).**
  - **Coverage.** 12,623 have a label (98.4%), against 11,862 samples with arrow labels. The
    207 left lack a QM record (168) or a fragment correspondence (39). The dataset edit
    reproduces the annotated product in 100% of them.
  - **Entries.** Bond-order changes are always ±1 (14,866 / 14,762), 1–8 per step, mostly 2.
    Non-bonding changes are always ±2, on 0–2 atoms.
  - **New bonds.** 8,904 form between molecules and 1,028 within one, at graph distance 2–9.
    So every atom pair is a candidate, with no radius. Pools have a median of 33 atoms with
    hydrogens, at most 411.
  - **Resonance.**
    - "Modulo resonance" taken literally is unusable: the product's resonance structures give
      more than 16 equivalent R in 2,727 steps, differing by Kekulé swaps of distant rings and
      charge-separated forms.
    - The label is the parsimonious class: the R that change the fewest entries. It has one
      member in 89.4% of steps and two in 9.3%.
    - The drawn R is parsimonious in 91.6% of steps.
    - Minimal variants touch no atom outside the drawn reaction centre in 90% of steps.
    - 3 steps have an empty minimal class (resonance interconversions) and are removed.
- **Spec changes (v8, sha256 1ad935491b71).**
  - The task, the BE section and the learning background are rewritten around R.
  - "Pointer decoder" becomes "Edit decoder":
    - an anchor pointer over atom pairs (a forming bond, or a breaking one in a heterolysis);
    - given the anchor, every other entry is classified (−1 / 0 / +1 per pair, −2 / 0 / +2 per
      atom);
    - a beam constrained by charge conservation and valence;
    - the atom-pair prior F^atom, a Boltzmann sum of the orbital-pair F_rs weighted by the centre
      weights.
  - The training objective marginalizes over the parsimonious class and the admissible anchors;
    END_STEP is gone, END_MECH stays.
  - "Label alignment" becomes "Labels, interpretation and step application"; alignment is a
    diagnostic.
  - Evaluation and the physics floor are on anchors.
- **Not changed.** QM, tokens, Wiberg indices, stage-A features, the contact sidecar, and the
  dataset-arrow round-trip gate.

## 2026-10-24 — Edit labels (spec v8, phase 1): `arrows/delta_be.py`, `ksmech-arrows edit-labels`

- **R_data.** The dataset arrows of a step are applied as one edit of the drawn reactants and
  checked against the annotated products. ReactionMatrix holds the non-zero entries on
  reactant-mol indices; `to_global` maps them onto the token atoms.
- **Parsimony, refined: fewest formal charges first, then fewest entries.** Counting entries
  alone prefers charge-separated product forms, which a drawn reactant often reaches in fewer
  changes. Measured on the 1,076 steps whose drawn R was not the entry-minimal one:
  - 441 had a minimal form with more charges than the drawn product (a hydrazone as
    N⁻–C=OH⁺, a Mg carboxylate as O⁺=C–O⁻, an arene as C⁺…C⁻);
  - 633 had as many;
  - 2 had fewer.

  Those labels would also feed the next step formal charges in an unnatural form. The cost is
  now (Σ|formal charge| of the product structure, entries): the dominant resonance contributor,
  reached with the fewest changes. This refines the approved v8 wording; the spec text is updated
  to match.
- **The key check filters the class.** Members are taken by cost among the matrices whose
  product passes `match_products`, the resonance-key check that evaluation uses. The key comes
  from a capped enumeration that is not perfectly form-invariant: from some charge-separated
  forms (a benzylic cation spread into the ring) it does not find its way back.
  - A rejected member is dropped and the next cost level is tried; R_data always passes.
  - 264 steps had a member rejected; without the filter they were 262 skips (VARIANT_MISMATCH).
- **The anchor is any rising bond order** (+1 entry: a new σ bond or a π bond on a bonded pair),
  else any falling one. The first definition, a new σ bond or a bond that disappears, left 316
  steps without an anchor: an amine closing an iminium on a carbocation, P⁺–O⁻ → P=O, C=OH⁺ →
  C⁺–OH. The spec text is updated to match.
- **Gate result (12,830 PMechDB training steps): 12,623 labeled (98.39%).** The 207 left lack a
  QM record (168) or a fragment correspondence (39).
  - Class size: 1 in 12,044 steps, 2 in 510, 3 in 61, 4 or more in 8.
  - The drawn R is parsimonious in 12,021 steps.
  - The resonance enumeration hit the cap of 256 in 942 steps.
  - Anchors: 11,838 rising, 785 falling (heterolyses).
  - Bond changes per step: 1 in 1,876, 2 in 7,332, 3 in 1,736, 4 in 1,443, 5–7 in 236.
  - The 3 steps the measurement had called resonance interconversions are now labeled: by the
    evaluation key their reactants and products differ.

## 2026-10-24 — Molecular hydrogen drawn as [HH]: correspondence with hydrogens made explicit

- **Problem.** 39 PMechDB training steps failed with CorrespondenceError ("heavy-atom counts
  differ"); their reactants include H2 drawn as `[HH]`.
  - Since 73e5f7c, `[HH]` and `[H][H]` share the key `[H][H]`, so the tokens are found.
  - But the drawn `[HH]` is one atom carrying an implicit H, while the record
    (`prepare_mol("[H][H]")`) has two explicit H atoms, so neither the canonical-order match nor
    the heavy-skeleton match applies.
- **Fix (`reactants._hydrogen_only`).** A fragment whose atoms are all hydrogens is matched with
  its hydrogens made explicit (`AddHs`; the drawn atoms keep their indices) and by count onto the
  record's hydrogens, which are equivalent. The only condition is "every atom is H".
  - For `[HH]` the drawn atom maps onto one record H; the implicit H is the other, which stays a
    token atom of the reactant side as before.
  - `[H][H]`, `[H+]` and `[H-]` map as they did.
- The orphan H2 QM record under the old `[HH]` key is kept.

## 2026-10-24 — QM retry of failed molecules: relaxed convergence, same method

- **Why.** 207 PMechDB training steps had no edit label. 39 were the `[HH]` correspondence,
  fixed in a973e5c. The other 168 come from 159 reactant molecules that the batch computed and
  that failed with a reason code; they were not skipped. Over all 283 failed molecules, by
  reason (reactant side of training steps in parentheses):
  - OPT_NOT_CONVERGED, 500 L-BFGS iterations: 98 (89);
  - SCF_NOT_CONVERGED, tblite's 250 cycles: 58 (47);
  - TIMEOUT, more than 900 s: 23 (21);
  - EMBED_FAILED, arynes: 6 (2);
  - SPIN_PARITY, Fe radicals in products: 100 (0).
- **Decision (user).** Retry with relaxed convergence only and the same method string:
  - `--timeout 3600`;
  - `--maxiter 2000`, a new CLI flag on `QMConfig.maxiter`;
  - `--scf-maxiter 1000`, new: `QMConfig.scf_maxiter`, passed to tblite's `max-iter` in every
    calculator (optimization, single point, Fukui ±1).

  No electronic temperature: smearing changes the result, iteration limits do not. Neither limit
  enters `QMConfig.method`, so retried records share the cache key and token configuration of
  the rest. Defaults are unchanged (500, 250). Arynes and Fe radicals stay out, counted by their
  reason codes.

## 2026-10-24 — The commit hook runs pytest in 14 processes (pytest-xdist)

- **Why.** The hook ran the whole suite on one core: 37 minutes a commit on a 16-thread machine
  (user, 2026-10-24: use more processes).
- **Change.** pytest-xdist 3.8.0 is a development dependency (`uv add --dev`), and the hook runs
  `uv run pytest -x -q -n 14`.
  - The "prefer numpy over dependencies" rule is about the code that runs, not test tooling.
  - Each xdist worker has its own session, so the session fixtures (QM records, tokens) are
    built once per worker. The tests write only to their tmp directories and only read `data/`.
  - The hook now takes about as long as the slowest single test: the real-data alignment and
    round trip over PMechDB.

## 2026-10-24 — The candidate filter in closed form; the real-data tests in their own hook stage

- **Why.** With xdist, the hook still took 34 minutes, because one test dominated it: the
  real-data alignment pass over the PMechDB training steps. A profile of 300 steps put 89% of
  the time in `tokens.site_excluded_matrix`, which called `site_pairs_excluded` once per donor
  centre and looked up each centre weight in a Python loop (11 million `flatnonzero` calls).
  The antibonding-partner rule took another 20% of what was left: a Python double loop over
  every token pair.
- **Closed form.** A token pair is excluded when every centre pair (a, b) is near (same atom
  or bonded) and the acceptor carries at least w on a or the donor at least w on b. So the
  pair is kept when:
  - some centre pair is far, which is the 0/1 product `C_d @ far @ C_a.T`; or
  - some donor centre is low for the acceptor and some acceptor centre is low for the donor.

  The second condition factorizes because `site_pairs_excluded` reads the two weights at the
  two sites independently. As in the loop, a NaN weight reads as low and a token without
  centres is excluded vacuously. `antibonding_partner_matrix` compares the sorted two leading
  centres and the overlap signs. The pairwise functions stay as the definitions, and the
  frozen legacy kernel still calls `site_pairs_excluded`.
- **Check.** Both matrices are identical to the loops on all 14,226 token files: 14,222
  molecules, and 4 with an empty block. The site matrix went from 1,441 s to 11.7 s in total.
  The tests compare both matrices with the pairwise rules on the ten fixture molecules, at the
  default weight and at 0.35 and 0.55. The alignment pass went from 102 s to 21 s on its first
  300 steps; what remains is mostly reading token files.
- **Parallel pass.** `evaluate_alignment(..., n_workers)` runs chunks of 64 consecutive steps
  in spawned processes, each with one BLAS thread, and merges them in step order. The report
  is identical, and a test checks it. A `TokenCache` sent to a worker starts with nothing in
  memory (`__getstate__`), so a used cache is not copied into every task. `ksmech-arrows align
  --workers N` exposes it. The dataset-arrow round trip takes 13 s for all steps and stays
  sequential.
- **Hook.** pytest now runs in two stages:
  - `pytest -x -q -n 14 -m "not realdata"`;
  - then `pytest -x -q -m realdata`, one test at a time, where a pass uses
    `KSMECH_TEST_WORKERS` processes (default 14).

  Inside an xdist run a real-data pass uses one process, because the xdist workers already
  fill the host. Either way the host never runs more than 15 heavy processes.
- **Measured.** The real-data stage takes 2 minutes: the alignment test 41 s instead of about
  30 minutes, the PMechDB load 30 s, the molecule-split check 27 s, the IBO check 18 s and the
  round trip 7 s.

## 2026-10-24 — RADICAL_DRAWN: atoms drawn with unpaired electrons are outside the domain

- **Why.** After the QM retry, one more PMechDB step was completely aligned, and its
  annotated leaving bond fell outside the physical mask: pmechdb-9346d5b585fc8087, whose
  reacting methyl is drawn `[C:20]`, without its hydrogens. RDKit reads that carbon as having
  three unpaired electrons. The odd-electron screen does not catch it because the count adds
  up, so the QM layer computed the molecule as an open-shell quartet, and the mask allows no
  bond to break at an open shell.
- **Extent.** 51 training steps have an atom that RDKit marks with unpaired electrons:
  - 25 carbons with 1–3 unpaired electrons (methyls and methylenes drawn without H);
  - 10 oxygens;
  - 8 Sn drawn without ligands;
  - 4 K or Na drawn without a charge.

  In a polar dataset these are drawing errors.
- **Rule (user decision).** RADICAL_DRAWN is domain rule 5 (`data.domain`): a fragment that is
  not odd but has an atom with radical electrons puts the step outside the closed-shell
  domain until the radical extension. It is last in the precedence, so existing codes keep
  their counts; odd fragments keep the earlier rules.
- **Re-ingest.** 13,090 accepted steps. 12,779 training steps, against 12,830 before. The
  exclusions are OPEN_SHELL_METAL 78, RADICAL_DRAWN 52, METAL_PLACEHOLDER 24 and
  CHARGE_UNREPAIRED 1.

## 2026-10-24 — QM retry and edit-label coverage: 99.62% of PMechDB training steps

- **Retry run** (`ksmech-qm run --workers 15 --timeout 3600 --maxiter 2000 --scf-maxiter 1000`).
  It recovered 130 of the 283 failed molecules, and 117 of the 159 on the reactant side of
  training steps. All of them got tokens and Wiberg sidecars: 14,226 token files, 14,473
  sidecars.
- **Still failing (153), counted by reason, not retried again.**
  - SPIN_PARITY: 98 Fe radicals, products only, outside the closed-shell domain.
  - SCF_NOT_CONVERGED in 1000 cycles: 46, of which 37 on the reactant side. Mostly TiCl4- and
    SnCl4/SnBr4-bound oxocarbenium adducts, plus bromonium and iodonium ions, `C=[O+][Zn-]`, a
    K phenoxide and `[O-]BO`.
  - TIMEOUT at more than 3600 s: 3, all on the reactant side, molecules of 150+ atoms.
  - EMBED_FAILED: 6, of which 2 on the reactant side (a Ca chelate and a fused polycycle; the
    others are arynes).
- **The run was interrupted twice.**
  - At 10 workers the host reset after 5 minutes. `stress-ng --matrix 16` alone reset it within
    a second, while `stress-ng --cpu 6` passed. The cause was bad BIOS settings, fixed by the user.
  - The rest ran at 15 workers without trouble.
- **A RAM fault remained.** On the evening of the same day the host showed a kernel oops, a
  flipped bit in a token file's page-cache copy, and a flipped bit in a btrfs checksum it
  wrote for a system library. Every one of these was bit 6 cleared. It ran three mismatched
  DIMMs, two on one channel; the user removed the odd one (now 2 x 8 GB, one per channel).
  - A full read of the project data (215,178 files, 17 GB) found no checksum error.
  - A record computed wrong in RAM before it was written would still carry a valid checksum.
    Recomputing a sample of QM records and comparing them is the check for that.
- **Edit labels after the retry, the `[HH]` fix (a973e5c) and RADICAL_DRAWN (0c62023).**
  - 12,731 / 12,779 training steps labeled (99.62%). Before RADICAL_DRAWN removed 51 training
    steps it was 12,782 / 12,830 (99.63%).
  - The only skip is MISSING_TOKENS: 48 steps that involve the 42 reactant molecules above.
    CORRESPONDENCE went from 39 to 0.
  - Class size: 1 in 12,148 steps, 2 in 513, 3 in 62, 4 or more in 8.
  - The drawn R is parsimonious in 12,115 steps. 264 steps had a resonance variant rejected
    by the key check.
  - Resonance enumeration truncated in 984 steps.
  - Anchors: 11,927 rising, 804 falling.

## 2026-10-24 — Embedding with relaxed distance bounds; `ksmech-qm run --retry-reason`

- **Why.** After the convergence retry, six molecules still failed with EMBED_FAILED: on the
  reactant side a Ca chelate (`O=C([O-])C[NH+]1CC(=O)[O][Ca]1`) and a fused polycycle, on the
  product side four arynes. ETKDG's triangle smoothing finds their distance bounds
  inconsistent (a metal in a five-membered chelate ring, a strained ring fusion, a triple bond
  in a six-membered ring), and random starting coordinates do not help.
- **Fallback.** A third attempt sets `ignoreSmoothingFailures` and switches off the
  basic-knowledge and torsion-preference terms. It embeds all six, while ethanol and the
  fixtures never reach it.
  - It runs only when the first two attempts produced nothing, so no existing record changes.
  - A record built this way carries the warning EMBED_RELAXED_BOUNDS.
  - The method string does not change: GFN2 starts its optimization from a rougher guess, and
    the connectivity check still rejects a geometry that rearranges.
- **Retry by reason.** `ksmech-qm run --retry-reason R ...` retries only the molecules whose
  recorded failure has one of those codes. The other failures stay as they are and are
  counted. The default is unchanged: every failure is retried, and `--no-retry-failed`
  retries none.
- **Result** (`ksmech-qm run --workers 6 --retry-reason EMBED_FAILED`, same relaxed convergence
  flags as the retry before it).
  - All six molecules embedded with EMBED_RELAXED_BOUNDS and passed GFN2 with their
    connectivity: the aryne C-C triple bonds come out at 1.27-1.28 Angstrom and the Ca-O at
    1.99 Angstrom.
  - QM now covers 14,232 of 14,379 molecules (98.98%). The other 147 are SPIN_PARITY 98 (Fe
    radicals), SCF_NOT_CONVERGED 46 and TIMEOUT 3.
  - Six token files and six Wiberg sidecars were added.
  - Edit labels: 12,733 / 12,779 training steps (99.64%). MISSING_TOKENS is down from 48 to 46.

## 2026-10-25 — Resonance forms with frozen hydrogens and no odd forms (resonance item 1)

- **Why.** RDKit's `ResonanceMolSupplier` returns structures that are not resonance forms of
  its input.
  - On an unbracketed atom it re-derives the implicit H in each structure: a nitro O comes out
    as [OH2+].
  - It returns structures with unpaired electrons: CCN([O+])[O-], an azide as N=N[N], a diazo
    group as C=[N][N].

  893 of the 12,779 PMechDB training products have at least one such structure. The sigma key,
  resonance-invariant by construction, differed across RDKit's structures of 84 of 1,393
  sampled products (6%) for this reason.
- **Change.**
  - `resonance.frozen_hydrogens` copies a molecule with every atom's H count frozen (explicit
    = total, no implicit H) and sanitizes it again.
  - `resonance.resonance_forms` keeps only the enumerated structures with the input's H count
    on every atom, its total charge and its radical electrons.
  - Both the resonance key (`resonance_structures`) and the edit-label class
    (`delta_be._costs`) enumerate this way.
- **No aromaticity for the edit labels.** They freeze without perceiving aromaticity.
  - Re-perceived aromaticity on a Kekule product changes which structures RDKit returns, and
    the labels read bond orders: 3,085 of 12,779 products gave a different set of reaction
    matrices.
  - Without it, the valid structures are exactly those of the raw enumeration for all 12,779,
    with the same enumeration size.
  - The key keeps its aromatic canonical form.
  - A truncated key enumeration still returns before reading any structure: reading 1,000
    per truncated molecule would slow ingestion down.
- **Effect.**
  - Sigma-key invariance over forms: 1,388 of 1,388 sampled products (100%), against 94.0%.
    The current resonance key, computed from each form, agrees across all of them in only
    825 of the 1,388 (59%).
  - Resonance-step flags are unchanged: 157 of 13,090 accepted steps.
  - Edit labels, computed without tokens over all 12,779 training steps:
    - The class changes in 53 steps. In each one a radical form had the fewest formal
      charges: an azide drawn N=N[N] or a diazo group C=[N][N], often in a spectator molecule.
      The drawn R is parsimonious again in those steps.
    - Class sizes 1 / 2 / 3 / 4+: 12,204 / 507 / 61 / 7, against 12,195 / 514 / 62 / 8.
    - Steps with a rejected variant: 260, against 265. The key check that rejects them is
      not form-invariant; item 2 replaces it with the sigma key.

## 2026-10-25 — The sigma key is the resonance identity; resonance steps leave training (resonance items 2-3)

- **Decision (user, option B).** Resonance identity is the sigma key, `data.resonance.sigma_key`:
  the map-free canonical SMILES of the sigma skeleton (every bond single, no aromatic flag), with
  every atom's H count frozen and its charge and radical electrons zeroed, then
  `|<total charge>`.
  - Two closed-shell Lewis structures with one sigma key have the same atoms, sigma bonds, H and
    total charge, so the same electron count: they are resonance forms of each other.
  - The key enumerates nothing, so it is exact and never truncated.
  - The enumerated key it replaces (the smallest SMILES over a capped enumeration) agreed across
    the forms of only 59% of sampled products, and InChI with fixed H only 58%. It is removed,
    with `ResonanceKey`, `ResonanceFlag`, `resonance_structures` and `skeleton_key`.
- **Choices inside the key.**
  - An H drawn as an atom counts on its neighbour, as an implicit H does, and H2 drawn [H][H]
    is [HH]. `RemoveHs` adds each removed H to its frozen neighbour's count.
  - Every atom is written with its H count (`allHsExplicit`). RDKit's writer decides brackets
    by valence, and wrote an O with no H (a zeroed alkoxide) as a plain `O`, like an O-H. A
    draft without this called 1,814 PMechDB steps resonance, most of them proton transfers.
  - No stereo: R carries none, and the model predicts none.
  - Radicals are zeroed with the charges, so the key assumes closed shells. Open-shell species
    are outside the domain (domain rules 3 to 5).
- **Where it decides.**
  - `is_resonance_step` at ingest: one sigma key on both sides.
  - `compare.match_products`: `resonance` when the edited molecule has the products' sigma key.
    The ingest arrow check, the round trips and the edit-label check (`delta_be.reproduces`)
    all go through it.
  - The sample's `product_key`, which step-level evaluation reads, is the products' sigma key.
    `product_key_flag` is gone and `SAMPLE_VERSION` is 14, so the existing sample stores are
    refused and must be rebuilt.
- **Degenerate exchanges.** The key is map-free, so a step that exchanges identical molecules
  has one sigma key and leaves the training target with the resonance steps. PMechDB has 3:
  - a thiolate/thiol proton swap (pmechdb-c905cbb9489419ee);
  - a thiolate/disulfide exchange (pmechdb-1dee1f33e7ecf103);
  - a degenerate carbocation shift (pmechdb-76cd41c118dfb991).
- **Resonance pairs.** Ingest writes `resonance_pairs.parquet` with the step id, the reactant
  drawing and the product drawing of every resonance step inside the closed-shell domain. They
  are equivalence examples for training. The ingest report counts them as `n_resonance_pairs`.
- **Re-ingest (PMechDB).**
  - 13,090 accepted steps and 14 rejected, both unchanged.
  - 485 resonance steps, against 157: those 157 still flagged, 328 more, none lost. 482 are
    resonance interconversions and 3 degenerate exchanges.
  - 474 of the 485 lie inside the domain and are written as pairs.
  - 12,461 training steps, against 12,779.
- **Edit labels** over the 12,461 training steps.
  - 12,415 labeled (99.63%). The only skip is MISSING_TOKENS, 46 steps.
  - Rejected variants: 0, against 264. Every class member is a resonance form of the product,
    so it has the product's sigma key.
  - Class size 1 / 2 / 3 / 4+: 11,730 / 619 / 59 / 7. Members the old key rejected now enter
    the class.
  - The drawn R is parsimonious in 11,771 steps (94.8%, against 95.2% before).
  - Enumeration truncated at 256 structures in 978 steps.
  - Anchors: 11,691 rising, 724 falling.

## 2026-10-25 — Spec v8.1: the sigma key, resonance pairs and resonance augmentation (item 4)

- **Resonance.** The spec defines resonance identity by the sigma key, and uses it wherever
  identity is decided: the label check, class acceptance, step-level exact match, beam merging
  and pool keys. Steps with one sigma key on both sides leave the training target; those inside
  the domain are kept as resonance pairs. Resonance forms, where they are needed, are RDKit's
  enumeration with frozen H and the odd structures dropped.
- **Edit labels.** Class members are accepted by the sigma key, so none is dropped on PMechDB.
  The class numbers are the new ones: one member in 94.5% of steps, two in 5.0%, drawn R
  parsimonious in 94.8%, 99.6% labeled.
- **Training objective (user decision).** Each molecule of a sample's reactant pool is drawn as
  a random valid resonance form, and the label set is recomputed for that drawing. The atom
  tokens' formal charges and the matrix B the edit is decoded on belong to a drawing, so
  without this the model would learn the dataset's choice of drawing. The resonance pairs are
  extra equivalence examples: two drawings of one pool that must get the same prediction. How
  that enters the loss (a consistency term, or extra augmentation draws) is settled with the
  training stage.
- **Removed from the spec.** "Resonance structures are never enumerated as model input."

## 2026-10-25 — Degenerate exchanges are real steps and stay in training (user decision)

- **Why.** The map-free sigma key also matched three steps whose two sides are the same
  molecules: a thiolate/thiol proton swap (pmechdb-c905cbb9489419ee), a thiol-disulfide
  exchange (pmechdb-1dee1f33e7ecf103) and a degenerate cation shift in a symmetric cage
  (pmechdb-76cd41c118dfb991). They had left training with the resonance steps. The user: they
  are actual reactions, and reagents and products are not the same because the two molecules
  are not interchangeable. The atoms change partners, and a mechanism follows the atoms.
- **Change.** `is_resonance_step` compares sigma keys with the atom maps kept
  (`sigma_key(mol, keep_maps=True)`): a resonance step changes no atom's sigma bonds or H.
  - The map-free key still compares a predicted product with the annotated one, so
    deprotonating either of two equivalent methyls gives the same product.
  - At step level, a degenerate exchange's product has its reactants' map-free key, so that
    metric cannot score these steps. The edit-level metric (R up to its class) does.
- **Numbers (PMechDB).**
  - 482 resonance steps, of which 471 inside the domain are written as pairs.
  - 12,464 training steps.
  - Edit labels: 12,418 / 12,464 labeled (99.63%); skipped
    {'MISSING_TOKENS': 46}.
  - Class size 1 / 2 / 3 / 4+: 11,733 / 619 / 59 / 7.
  - Drawn R parsimonious in 11,774 steps.
  - Rejected variants: 0.
  - Truncated enumerations: 978.

## 2026-10-25 — The samples carry the reaction-matrix target; alignment no longer gates them

- **Why.** The sample store was still shaped for the orbital-pair target of spec v7. A step was
  a sample only if every arrow aligned to a filled and an empty orbital, so the 643 lone
  heterolyses of the training set were left out. Two more steps were left out because the
  candidate mask removed their whole orbital label (LABEL_FILTERED). The spec v8 target is R,
  and its label needs no alignment (spec, "Edit labels"); `ksmech-arrows edit-labels` computed
  it for every step but stored only counts.
- **What a sample holds (`SAMPLE_VERSION` 15).**
  - `edit_label`: the parsimonious class with its anchors, on the token atoms
    (`delta_be.edit_label` + `to_global`).
  - The reactant's Lewis structure on the same atoms (`arrows.lewis`): `bonds` (i < j,
    order), `lone_pairs` and `wiberg` per bond.
    - The structure is the reactant as drawn, in its Kekulé form, since the labels are computed
      on it (user decision).
    - An H the drawing leaves implicit gets a single bond to its record neighbour.
  - `orbital_reading`: `aligned`, `LABEL_FILTERED` or the first unaligned reason, such as
    `HETEROLYSIS_ONLY`. The ArrowLabels are kept only when it is `aligned`.
  - `collate_samples` adds:
    - the per-variant targets `r_pairs`, `r_atoms` and `r_anchors`, with `variant_mask`;
    - `bond_order`, `wiberg`, `graph_distance` (-1 between molecules) and `lone_pairs`;
    - the decoder masks `can_break`, `can_form` and `can_lose_pair`.
- **What gates a sample.** Tokens; then an edit label whose atoms all have tokens; then a
  Wiberg index for every bond (NO_BOND_INDEX); then `lewis.check_edit`, which asks every
  variant to conserve electrons, keep bond orders in [0, 3] and lone pairs non-negative, and
  asks every anchor to pass the decoder's masks (R_CHECK_FAILED). The alignment no longer gates.
  The plan also asked for "B + R has the product's sigma key". That already holds by
  construction: every class member passes `reproduces` when the class is built, and
  `test_the_edit_label_takes_the_reactant_to_the_drawn_product` checks it on the token atoms.
- **Readers of the orbital labels.** `ks-rank`, `baseline` and the LightGBM ranker rank the
  first arrow's orbital pairs. They skip samples without ArrowLabels and count them as
  NO_ORBITAL_READING, so they read the same steps as before.
- **PMechDB** (a dry run of the builder over the 12,464 training steps).
  - 12,418 samples: 11,773 aligned, 643 HETEROLYSIS_ONLY and 2 LABEL_FILTERED.
  - 46 skipped as MISSING_TOKENS. No R_CHECK_FAILED, no NO_BOND_INDEX.
  - Class size 1 / 2 / 3 / 4+: 11,733 / 619 / 59 / 7. Anchors: 11,694 form, 724 break. Both
    are the edit-labels report's numbers.
- **Later (user decision).** Resonance augmentation will precompute a few forms per sample,
  each with its own Lewis structure and label class.

## 2026-10-25 — Molecule workbench (serve), backend

- **What.** `ksmech-serve` gains a display-only molecule workbench: draw (or paste) one
  molecule, featurize it, and look at its geometry, properties, orbital sets and volumetric
  surfaces. Routes under `/api/molecule`; capability `workbench`, always on.
- **Adapters, all in `src/ksmech/serve` (upstream is consumed, never modified).**

  | Module | Side | Upstream it reads |
  |---|---|---|
  | `molecule.py` | API | `data.chem.unmapped_canonical`, `qm.embed.prepare_mol`, `featurizer.preflight` |
  | `orbital_sets.py` (`wb_orbital_set`) | worker | `orbitals.localize` (METHODS, STARTS, PMConfig, mulliken_populations), `orbitals.tokens.build_molecule_tokens` / `save_tokens` / `load_tokens`, `orbitals.frontier.frontier_blocks` |
  | `densities.py` (`wb_cube`) | worker | `physics.gto.ao_values`, `serve.cubes` (grid, `on_grid`, `write_cube`) |
  | `properties.py` (`wb_properties`) | worker | `orbitals.tokens.atom_tokens`, `qm.bond_index.wiberg_matrix`, `localize.is_fermi_smeared` |
  | `workbench_service.py` | API | only the files the worker wrote, and `np.load` of the QM record for the geometry |
  | `token_rows.py` | both | the orbital-row serialization moved out of `pool_service` |

- **Version constants.** `WORKBENCH_VERSION`, `ORBITAL_SET_VERSION` and `DENSITY_VERSION`, all
  1. They key the workbench caches together with the QM record's content sha (sha256 of the
  file bytes) and the oriented fingerprint (token schema, localize.py, descriptors.py,
  tokens.py, physics/*.py). None of them enters the physics fingerprint: the workbench decides
  no score, validity or product.
- **Identity.** A drawing's key is `unmapped_canonical(mol)`, the QM / token cache key, so a
  drawn molecule shares its record and tokens with the pools. The id is
  `"m" + sha256("molecule-v1|" + key)[:20]`, so the same molecule drawn twice (or typed as a
  SMILES) is one molecule. The registry entry keeps the last featurize failure; a POST does
  not run a recorded failure again unless `retry` is set (no silent loops, no silent skips).
- **Sketch mapping by substructure match.** `prepare_mol(key).GetSubstructMatch(AddHs(drawn))`,
  with chirality first and without it as a fallback (`chirality_matched` says which). The
  drawing has the same graph as its own canonical key, so the match is an isomorphism; for a
  symmetric molecule any of the equivalent matches is equally correct. An H the drawing left
  implicit maps to its drawn heavy atom (`record_parent_sketch`). The mapping is returned with
  the POST only: the drawing is the client's, and a GET cannot know it.
- **Orbital sets.** `canonical` plus `localize.METHODS` (`ibo`, `pm_mulliken`) times
  `localize.STARTS` (`cholesky`, `canonical`). The model set is `PMConfig()` (ibo/cholesky):
  the existing token file when there is one, so the viewer shows exactly the orbitals the model
  reads. The other sets are built with `build_molecule_tokens(rec, pm_config=...)` and saved
  with upstream `save_tokens` and provenance (the QM record as input, `token_config(pm)`) under
  `data/serve/workbench/orbitals/`. No Foster-Boys: tblite exposes no dipole integrals.
- **QM records load only in the worker.** `qm.records.load_record` re-derives the AO layout
  through `qm.ao`, which imports tblite and runs a single-atom tblite call per element. The
  synthetic chemistry is read without that probe (its records were written with the current
  layout; the provenance header is still verified); real records go through `load_record`.
- **Densities are valence-only, minimal-basis.** The density is
  `rowsum((A C_occ sqrt(n))^2)`, P = C diag(n) C^T over the valence MOs of GFN2's minimal
  basis. The AOs on the grid are the STO-nG fit of GFN2's valence Slater functions
  (`physics.gto`), which is not tblite's own basis evaluation: under real chemistry the grid
  integral of the density differs a little from N_valence. Every cube therefore reports its
  integral and the exact value (comment line, sidecar, `X-Cube-Integral` / `X-Cube-Expected`).
  On the synthetic chemistry (whose S is the STO-nG overlap) |psi|^2, f+, f- integrate to 1
  and the density to N_valence within 1% at 0.2 Angstrom with 5 Angstrom of padding.
- **Fukui surfaces.** Frozen-orbital: the mean |psi|^2 over the HOMO (f-) or LUMO (f+)
  degenerate block of `frontier_blocks` (0.05 eV), so each integrates to 1 and equivalent
  atoms look alike, as `frozen_orbital_fukui_blocks` does for the condensed values. No block:
  NO_FRONTIER.
- **ESP approximation.** sum_A q_A / max(|r - R_A|, 0.2 bohr) over the Mulliken charges, in
  hartree/e. It is not the electrostatic potential of the electron density: it is a cheap,
  sign-correct picture (positive around NH4+, negative around acetate away from the nuclei,
  Q/r far away), labelled approximate in the cube and the sidecar. The floor keeps it finite
  at the nuclei.
- **Grid policy.** One whole-molecule box, `cubes.grid_around(all atoms, spacing, padding)`,
  default padding 3.0 Angstrom. The spacing is explicit or set by `quality` (draft / normal /
  fine: about 120k / 350k / 1.2M points, rounded up to 0.001 Angstrom so the cache key is
  stable); `cubes.MAX_POINTS` (4M) still applies (CUBE_TOO_LARGE). Set-independent kinds drop
  set, start and index from the request, so they share one cache entry.
- **Statuses and errors.** A molecule is missing, running, ready, qm_only (a QM record but no
  tokens, e.g. cached chemistry: geometry, properties, canonical orbitals and surfaces still
  work; the model set is then built by the workbench) or failed. New codes: INVALID_MOLFILE,
  MULTIPLE_FRAGMENTS, ORBITAL_OUT_OF_RANGE, NO_FRONTIER, LOCALIZATION_FAILED (422),
  MOLECULE_NOT_FOUND (404), MOLECULE_NOT_READY (409) and the report code MAPPING_FAILED.

## 2026-10-25 — Molecule workbench (serve), frontend

- **Page.** `apps/ui/src/features/workbench/` registers `#/workbench`, first in the nav and the
  default page while predict is off. Features never import each other, so the 3Dmol helpers
  moved to `src/chem/threed.ts` (the contact scene stays in predict as `contactScene.ts`), the
  generic hooks (`usePendingCached`, `useDebounced`, `useLatest`) to `src/api/hooks.ts` and the
  number formatting to `src/components/format.ts`.
- **Linking the sketch.** Only the POST answer carries the sketch mapping, so the page keeps it
  with the drawing that was posted. A selection highlights the drawn atom through
  `record_parent_sketch` (an implicit H highlights its heavy atom) only while the sketch's
  chemistry signature (elements, charges, bonds by atom position, orders, wedges; not the 2D
  positions) equals the posted one; otherwise a note says the sketch changed. The posted
  molblock and its id are kept in sessionStorage, so reloading `#/workbench?m=<id>` posts it
  again (cheap, cached) and gets the mapping back; a bare id from another tab loads without it.
- **Surface layers.** The hash keeps only what decides a fetch (`kind:set:start:index`); the look
  (isovalue, opacity, mesh, colours, ESP range) is client state per spec. Parsed cubes sit in a
  VolumeData LRU of 8; each layer's request has its own AbortController. The selected orbital is
  a layer that follows the list and the diagram, and "pin" copies it into the hash. Changing
  the orbital set clears the selected orbital, since an index means another orbital in another
  set; computing another molecule drops the pinned orbital layers.
- **ESP colouring.** `addIsosurface(density, {voldata: esp, volscheme: Gradient.RWB(-r, r)})`
  (fallback: a VDW surface coloured by the same volume). The default r is the 95th percentile
  of |ESP| at grid points near the density isovalue (both cubes share one grid per quality),
  and the colour bar reproduces 3Dmol's RWB formula, so it matches the surface.
- **e2e chemistry.** `e2e/serve.mjs` adds ethanol (the workbench test's molecule) to the
  `probe-cache` species: cached chemistry never computes a new molecule, so the run falls back
  to synthetic unless ethanol is cached too.

## 2026-10-25 — Workbench sketcher: 3D -> 2D projection, crossing breaks, layout fix

(Superseded the same day for the layout: the client-side projection layout and the "Fix layout"
relaxation were replaced by server-side RDKit depiction, see "Sketcher 2D layout" below; the
view convention, the depth storage, the linking rule and the crossing breaks still hold.)

- **Projection convention.** The sketcher's "3D→2D" (Shift+D) draws the record geometry of each
  drawn atom (`sketch_to_record`) as the 3D view shows it now: `v = R(q) p` with `q` the
  rotationGroup quaternion of 3Dmol's `getView()` (indices 4-7, x y z w) and R the matrix of
  3Dmol's `setRotationFromQuaternion` (the renderer's; note that 3Dmol's own
  `Vector3.applyQuaternion` rotates by the conjugate, so it is not used). View x is screen right,
  y screen up, z towards the viewer; the projection is orthographic (the translation is dropped,
  the drawing is centred on its old centroid). Checked twice: a node test against 3Dmol's matrix
  formula and the axes' handedness, and in the browser on every projection, by fitting 3Dmol's
  `modelToScreen` of the same atoms with a scale and a translation only (no rotation, no
  mirror; a negative scale counts as a failure): the RMS residual over the screen spread is
  exposed as `data-projection-residual` (0.024 for the e2e camphor view; 3Dmol's perspective,
  fov 20 deg, accounts for it). Scale: the median 3D bond becomes BOND_LENGTH (a projection is
  a uniform scale of the 3D picture; foreshortened bonds are then restored by the fix). The
  sketch module never imports 3Dmol: the page passes a `ProjectionSource` (why it is off, or
  the rotated coordinates).
- **Depth storage.** `SketchAtom.z` (model units, + towards the viewer), display data like
  explicitH: kept by edits, undo and the session store (sessionStorage now holds
  `{molblock, extras}`; a bare molblock still reads), never written to molfile coordinates
  (z = 0, header 2D, so RDKit always reads a 2D drawing; only RDKit's 3D stereo reading of a
  projection writes it, with a 3D header). RDKit's clean-up drops it (a new layout is no
  projection). `SketchModel.breakCrossings` is stored the same way.
- **Linking after a projection.** A projection changes coordinates and wedges only, so the page
  no longer compares a signature that includes wedges by bond direction: it unlinks on another
  topology (elements, charges, unordered bonds, orders), or on wedges that draw neither the
  computed drawing's isomeric SMILES nor the record's (RDKit canonical SMILES of the 3D
  geometry molblock, chirality from 3D). Wedges that redraw the record's configuration keep
  the link.
- **Stereo rule.** Centres and the target come from RDKit's reading of the projected x, y, z
  (heavy atoms with implicit H; JSON `stereo` cw/ccw plus CIP labels), which equals the
  record's isomeric SMILES in every orientation tested. A configuration is the sign of a
  volume over the drawn neighbours in id order (3: det(v1, v2, v3); 4: det(v2-v1, v3-v1,
  v4-v1)), with depth (true) or with a pseudo-depth +-|v| behind a wedge/hash whose narrow end
  is the centre (drawn). One mark per centre, on the single bond that draws the true sign and
  scores best: acyclic +4, terminal neighbour +2, neighbour not a centre +3, |V| up to +1,
  depth: +2|dz|/L and +3 when the mark follows the depth (wedge in front, hash behind) else
  -3; a degenerate volume (|V| < 0.08) or an ambiguous four-neighbour mark (the three other
  neighbours within 170 deg, where RDKit's pseudo-3D reading need not follow the volume) is
  -20. All other marks are cleared. RDKit then reads the drawing back and flips any centre
  whose CIP label differs (never needed on the six fixtures x 66 orientations of the tests and
  the sweep: alanine, CHFClBr, camphor, 2-methylcyclohexanol, menthol, cocaine).
- **Layout fix.** "Fix layout" (Shift+G, also after every projection): Gauss-Seidel position
  corrections in a fixed order, at most 300 sweeps, stopping when the net motion of a sweep is
  below 1e-4 L. Terms: bond length to L (stiffness 1; a bond under 0.05 L is first opened along
  growthAngle); angles under 40 deg open (0.3); non-bonded atoms under 0.68 L part (0.5); an
  atom under 0.36 L from a bond it is not on moves off (0.5); label boxes (drawing.labelBox:
  symbol, H count on its side, charge; 3 px padding at 25 px/unit) part from other labels,
  atoms and bonds (0.5); an anchor to the start positions, 0.02 after a projection, 0.005 for a
  hand drawing. Metrics use stricter thresholds (contacts 0.6 L, atom on bond 0.3 L, angles 35
  deg, no padding) so a converged layout reports zero. The wedge configurations drawn before
  are drawn again afterwards (the same scoring, the old mark +5). A projection also tries 12
  slightly tilted views (8 and 15 deg about the screen axes, the depth is known) when the exact
  view leaves defects, scoring 10 x label overlaps + 5 x contacts + 5 x atom-on-bond + 2 x
  small angles + 0.5 x crossings + 20 x bond RMS + 0.15 per degree of tilt: local relaxation
  cannot untangle an atom lined up behind a cage. On the six fixtures x 30 orientations this
  leaves one case (cocaine) with a contact and an atom on a bond; bond RMS <= 0.045.
- **Crossing breaks.** Render-time only (drawing.ts): where two bonds cross (no shared atom,
  ends excluded) the one behind (interpolated depth at the crossing; without depth on all four
  atoms the bond drawn later is in front) loses BREAK_GAP_PX = 3.2 px plus the front bond's
  half-width on each side, divided by the sine of the crossing angle (at least 0.35), on every
  stroke; wedges split in two, hashes lose the ticks inside. On by default after a projection,
  a toolbar toggle (Shift+X) otherwise.

## 2026-10-25 — Sketcher 2D layout: server-side RDKit depiction and perspective templates

- **Why the relaxation went.** The 3D->2D projection followed by the home-made "Fix layout"
  relaxation (bond-length, angle, contact and label terms, tilted views) was judged awful by eye:
  a projection of a cage lines atoms up behind each other, and a local relaxation only smears
  the result. It is removed (sketch/layoutFix.ts, projectLayout.ts, stereo.ts and their tests),
  with its button (Shift+G). Both layouts now come from RDKit on the server:
  `POST /api/depict2d` (serve/depiction2d.py, API process, RDKit only, no worker). Kept from the
  old code: the 3Dmol view convention (the quaternion is sent as is; the page still checks it
  against `modelToScreen`, `data-projection-residual`), the depth-based wedge choice and the
  render-time crossing breaks, now fed by the server's depth.
- **Request and answer.** `{molblock, mode: clean | match3d, style: textbook | perspective,
  molecule_id?, view?: {quaternion [x, y, z, w], convention "3dmol-rotationGroup"}}`. The
  answer keeps EXACTLY the molblock's atom and bond order: `coords[i]` (bond 1.5), `depth[i]`
  (+ towards the viewer, or null), wedges by bond index with the narrow-end atom, the molblock
  (written here: the input's Kekule form and charges, narrow end first), the canonical isomeric
  SMILES RDKit reads back, `stereo` none / kept / mismatch, the defect report and every
  candidate's score. Errors: EMPTY_INPUT, INVALID_MOLFILE, LIMIT_EXCEEDED (250 atoms),
  REQUEST_INVALID (match3d without id or view, a zero quaternion), MOLECULE_NOT_FOUND,
  MOLECULE_NOT_READY (no QM record; never featurizes), MAPPING_FAILED (the sketch is another
  molecule or does not map onto the record), DEPICTION_FAILED (new, 500: RDKit could not lay it
  out). The client applies the answer by index (sketch/depiction.ts); when the server cannot be
  reached Clean falls back to RDKit.js's CoordGen with a notice in the info bar.
- **No global RDKit state.** `SetPreferCoordGen` is process-wide and depict.py sets it, so every
  RDKit depictor call passes `forceRDKit=True` and CoordGen is called by name
  (`rdCoordGen.AddCoords`). RDKit's ring templates are global too (`SetRingSystemTemplates`
  replaces them; reloading the defaults costs 0.4 s) and RDKit never applies a template to a
  monocycle (checked: a deliberately distorted cyclohexane template is ignored, a norbornane one
  is used, and template atoms/bonds match as SMARTS, so `*` and `~` make them generic). So the
  textbook style uses RDKit's default templates (`useRingTemplates=True`), and the perspective
  library is applied by our own matcher; `RDKIT_TEMPLATE_LOCK` guards the one test that loads
  our file into RDKit to prove its format.
- **Textbook.** RDKit with its ring templates and CoordGen, the lower defect score wins, but
  CoordGen must win by 2 (the gallery favoured RDKit's templates: camphor, cubane, morphine,
  pinene; CoordGen drew camphor's gem-dimethyl bridge as a cross inside the ring).
- **Defect score** (serve/depiction_metrics.py, drawing units, bond L = 1.5): 10 label overlaps
  (a labelled atom within 0.8 L of an atom it is not bonded to) + 8 close contacts (< 0.5 L) +
  15 coincident atoms (< 0.25 L) + 6 atoms on a bond (< 0.2 L) + 10 atoms in no ring drawn
  inside a ring + w_x crossings + w_s squashed rings (principal-spread ratio < 0.6) + 2 angles
  under 35 deg + 20 bond-length RMS. By kind: flat layouts w_x 3, w_s 0; perspective templates
  w_x 1, w_s 0 (their crossings and slanted rings are the point); the view template w_x 1.5,
  w_s 6 (a slanted aromatic ring is just squashed). Calibrated on the gallery: the inside-ring
  term was added when the first score preferred CoordGen's camphor, the coincident and squashed
  terms when degenerate or slanted 3D views won.
- **Template library** (serve/data/ring_templates.smi in RDKit's format: one generic CXSMILES
  ring system with 2D coordinates and a name per line; ring_templates.json: title, depth per
  atom, examples, source, convention, derivation). Ten templates: cyclohexane chair, norbornane
  (camphor, borneol, norbornene), bicyclo[2.2.2]octane (quinuclidine, DABCO), tropane
  (bicyclo[3.2.1]), adamantane, cubane, trans- and cis-decalin, the all-chair 5alpha steroid
  core. Each is an orthographic picture of the lowest-energy MMFF structure of its parent (20
  srETKDGv3 embeddings, seed 7), turned to a hand-chosen view (up and right axes from atoms,
  then spin, tilt, roll; picked from a search over spin and tilt and checked in the gallery) and
  scaled so the median drawn bond is 1.5: parallel bonds stay parallel, axial bonds are vertical.
  A pinane template was dropped: RDKit's default pinane template already is the chemist's
  drawing, ours crossed the double bond of alpha-pinene.
- **Applying a template.** A template is a conformation, not a graph: it is matched (largest
  first) only on a whole ring system (rings sharing two or more atoms) and its 3D shape (x, y
  and the stored depth) is fitted to a 3D structure of the molecule, mirror allowed, accepted
  under 0.12 bond lengths RMSD. So benzene is never a chair, cis- and trans-decalin pick their
  own drawing and a boat is left alone. The 3D structure is the lowest-energy of 4 srETKDGv3 +
  MMFF embeddings (plain ETKDGv3 gave only twist-boats for cis-4-tert-butylcyclohexanol), or
  the record for match3d. Each exocyclic neighbour in no ring goes on the tetrahedral slot its
  3D bond points to (axial / equatorial, exo / endo), one bond length along the slot's drawn
  direction snapped onto vertical or a ring-bond direction within 8 deg (textbook chairs);
  ring neighbours (spiro, ring substituents) are left to RDKit, which lays out the rest around
  the locked atoms (`Compute2DCoords(coordMap=..., forceRDKit=True)`). The perspective drawing
  is kept unless it scores more than 15 worse than the textbook one (cocaine's two esters crowd
  on the tropane template; the spiro ketal squeezes): then the textbook layout, with a note.
- **match3d.** The record geometry (sketch atom i = record atom sketch_to_record[i], recomputed
  on the server from the posted molblock) is turned by R(q) (the same rotation as
  projection.ts rotateByQuat; a test pins it) and scaled so the median 3D bond is 1.5. The
  stereo target is read from that 3D structure (`AssignStereochemistryFrom3D`), keeping CIP
  R / S everywhere and r / s only on atoms with at most two ring bonds: a cage's bridgehead
  pseudo-stereo (adamantane, cubane) would only add wedges. Candidates:
  `GenerateDepictionMatching3DStructure` (a 3D distance-matrix mimic, so the view only orients
  it), `Compute2DCoordsMimicDistmat` on the projected distances, the view itself as a template
  made on the spot (ring atoms where the view shows them, exocyclic atoms along their projected
  bonds, RDKit for the rest), the textbook layout and, for style perspective, the template
  layout. Each is turned onto the projection (flat ones may also be mirrored; a perspective
  picture never, it would show the enantiomer) and ranked by defects + 20 x the fit residual
  (RMS distance to the projection over its spread) + 50 for a stereo mismatch. The depth is the
  view's (a template keeps its own).
- **Wedges.** One mark per stereocentre on the single bond that scores best (acyclic +4,
  terminal neighbour +2, neighbour not a centre +3; with depth +2 |dz| / L and +3 when a wedge
  is in front or a hash behind, else -3), its kind from `Chem.WedgeBond(bond, centre, conf)`
  (whose answer is for the given centre even when it is the bond's end atom; RDKit's own writer
  then puts the wrong narrow end, hence our writer). RDKit reads the result back; on a
  difference RDKit's `WedgeMolBonds` is tried, then `stereo: mismatch` is reported (never seen
  on the gallery or the tests).
- **Gallery.** `uv run --no-sync ksmech-serve gallery [--out DIR]` draws 22 hard molecules with
  the old CoordGen clean-up, textbook, perspective and match3d from two fixed views of an
  RDKit structure (no xTB) into one PNG + SVG grid per method, sheets of molecules x methods and
  gallery.json. RDKit's drawer renders them, so the sketcher's crossing breaks are not shown.

## 2026-10-25 — Perspective templates redrawn as textbook pictures

- **Why.** The ten templates were orthographic pictures of MMFF minima. By eye the cages looked
  wrong (camphor and borneol cramped; quinuclidine, adamantane and the decalins odd), and the
  pictures broke basic drawing rules: bonds drawn 0.4 to 1.4 long, angles up to 180 deg (RDKit
  prints a stray "C" there), atoms 0.1 bond from bonds they are not on.
- **Picture and shape are now separate.** `ring_templates.smi` holds a hand-made picture in the
  textbook style (Clayden, Eliel). `ring_templates.json` adds `fit`: the parent's MMFF structure
  turned to the orthographic view that best reproduces the picture (`derive_template`). The JSON
  also lists `front` atoms because a picture alone cannot tell a cage from its depth-reversed
  twin. `depth` is the z of `fit`, so the crossing breaks and wedges agree with the picture.
  Matching a molecule uses `fit`. The exocyclic slots are the tetrahedral directions of `fit`,
  drawn through a per-atom linear map onto the picture. Each slot is turned to stay 38..160 deg
  from the atom's other bonds, then snapped onto the vertical or a ring-bond direction (within
  8 deg, 30 deg for the stylised chairs). A slot can also be set by hand (`slot_overrides`: the
  V of the C7 / N8 substituents).
- **Rules** (`depiction_metrics.rule_report`, tested on every template and on 13 substituted
  examples): bonds within 15% of the median, except bonds listed as `foreshortened` (receding,
  at least 0.55); angles 30..165 deg at atoms with two or three bonds (a four-bond cross is the
  convention); no atom within 0.4 bond of a bond it is not on; atoms at least 0.6 apart; the
  stated number of crossings, and no extra crossings from substituents.
- **Conventions.**
  - Chair: three bond directions (55, -20, 10 deg), each drawn twice at one length; axial bonds
    vertical, equatorial bonds parallel to the ring bonds once removed.
  - trans-Decalin and the steroid A/B/C: the same chair turned 180 deg about each shared bond.
    The steroid is fused on the steep bond, so it reads as the natural 5alpha series with the
    angular methyls up. D is a regular pentagon.
  - cis-Decalin: a chair with the second chair rising through an axial bond.
  - Norbornane, bicyclo[2.2.2]octane and tropane: seen from the front and above, bridgeheads left
    and right, the back bridge inside and short, no crossing. The C7 / N8 bridge is a roof over
    the boat, the barrel or the half-chair.
  - Adamantane: a chair with three parallel bonds (80 deg) up to the apex, one central crossing.
  - Cubane: a cabinet cube, two crossings.
- **Symmetric fits.** A symmetric template fits a molecule equally well in several ways. Among
  fits within 0.02 bond of the best RMSD, `_crowding` picks the one that puts substituents on
  clear, drawn, front slots and keeps ring double bonds out of the back. So 1-adamantanol gets
  its OH on the apex and camphor gets its carbonyl on the front bridge.
- **Gallery.** It now also draws everything in the sketcher's own style (`app_scene`
  reimplements sketch/drawing.ts + crossings.ts: bare carbons, broken rear bonds) and adds
  templates.png / templates.json.
- **Costs.** beta-D-glucopyranose falls back to the textbook layout: the chair's adjacent
  equatorial OH labels overlap. The camphor match3d view residual bound went from 0.6 to 0.7
  because the stylised picture is further from a true view. Twistane was tried and dropped: no
  candidate view looked like a textbook figure.
- **match3d-view depth.** That candidate now takes the view's depth scaled like the other
  candidates (`view_frame`, the median bond), as documented. Its own scaling (mean bond) was 2%
  off. This showed up once the new norbornane template stopped winning one of the test views.

### 2026-10-25 — Workbench radicals, R/S, side-by-side orbitals, customizable layout

- Radicals are chemistry, not display: the sketch stores 0, 1 or 2 unpaired electrons per atom
  and writes them as V2000 `M  RAD` (2 doublet, 3 triplet; reads 1 and the old atom-block code
  4 too), so RDKit's H counts, SMILES and formula follow. Compute is disabled with the reason
  instead of left to fail, because the featurization is closed-shell (RADICAL_NOT_SUPPORTED).
  The "." key is free on every layout and collides with no shortcut (tested).
- CIP labels come from RDKit.js `get_stereo_tags` (new labeler) on the sketch's own molblock,
  so they read the drawn wedges and 2D double-bond geometry; "?" marks a possible but
  undefined stereocentre.
- The layout is a pure reducer over columns (at most 3) plus a full-width row; hidden and
  collapsed panels stay mounted. Layout and preferences live in localStorage
  (`ksmech.workbench.layout.v1`, `ksmech.workbench.prefs.v1`) and are sanitized on read. The
  default preset puts the orbitals full-width so the level diagram and the orbital list sit
  side by side. Moving a panel to another column remounts it (the sketch keeps its drawing
  but loses its undo history).

## 2026-10-25 — Sketcher drawing styles and publication export

- **Style model.** `apps/ui/src/features/workbench/sketch/style.ts`: a `DrawingStyle` holds
  what a ChemDraw document style controls, in points: fixed bond length, line width, bold
  (wedge) width, hash spacing and hash tick width, double-bond spacing (% of the fixed bond
  length) with the inner-line shortening and the terminal C=X rule (ring doubles always go to
  the ring centre), margin width, label font family / size / weight, subscript and charge size,
  superscript or circled charges, colour or black and white, crossing gap, wedge shape (classic
  wedge, or a bold bond with a tapered start), chain angle, labelled carbons and terminal
  methyls, and whether sizes follow the zoom. The fixed bond length is the length of a standard
  model bond (`BOND_LENGTH` = 1.5 units, what the tools draw and imports / Clean normalise to).
  For the publication styles `pxPerPt = zoom * 1.5 / bondLengthPt`, so every size keeps its
  proportion to the bonds at any zoom and the canvas is the exported picture scaled. Screen
  keeps fixed pixels (1 pt = 4/3 px) as the sketcher always did (labels stay readable on a big
  molecule fitted small); the font size, line width and atom-colour preferences set it, and its
  values reproduce the former constants (13 px labels, 1.5 px lines, 4.5 px double gap, 3.6 px
  wedge half-width, 3.2 px crossing gap).
- **Presets and sources.**
  - ACS Document 1996: bond 14.4 pt (0.508 cm), line 0.6 pt, bold 2.0 pt, margin 1.6 pt, hash
    spacing 2.5 pt, bond spacing 18 % of length, chain angle 120°, Arial/Helvetica 10 pt, black
    and white. Source: Revvity Signals (ChemDraw's vendor) support article 4408234173332 ("When
    I print the structures, the line thickness for chemical bonds has increased ..."), which
    lists the style sheet; the same numbers in the Middlebury science-writing guide
    (sites.middlebury.edu/middsciwriting/?p=334) and in journal author guides that quote the ACS
    settings. Not in the published list, so ours: hash tick width = line width, crossing gap =
    margin, subscript and charge at 75 %, inner line shortened 12 % at each end.
  - Total synthesis: no standard exists; the app's own preset for the bold look of synthesis
    papers and graphical abstracts, said so in the tooltip: bond 15 pt, line 1.1 pt, bold
    3.2 pt (tapered wedges), hash spacing 2.6 pt with 1.0 pt ticks, spacing 20 %, inner line
    12 %, margin 2.2 pt, crossing gap 2.2 pt, Arial 11 pt, subscript 75 %, charge 80 %, black
    and white.
  - RSC and Wiley are not offered. No publisher page gives the numbers. The only RSC values
    found are a third-party reading of ChemDraw 22's "RSC (1 Column) Document.cds" (12.2 pt
    bonds, 0.45 pt lines, 1.6 pt bold, 1.75 pt hash spacing, 1.25 pt margin, 20 %, Helvetica
    7 pt; github.com/JamesOBrien2/penzene pull 67); nothing for Wiley (Angewandte ships a
    template file instead). A user with those style sheets can enter them as a Custom style.
- **Labels.** Label widths come from Helvetica's AFM advance widths (= Arial's;
  `fontMetrics.ts`) for the publication fonts, so the canvas, the SVG and the PDF place glyphs
  identically. Bonds stop at the label's ink box (cap height, the H part on its side) plus the
  margin, along the bond direction (ChemDraw's margin width). The H count goes right or left as
  before, or above / below when bonds leave on both sides near the horizontal (an NH in a
  chain). The charge takes the first free slot of upper right, upper left, lower right, lower
  left (40° from every bond and the H).
- **Chain angle.** It shapes only new bonds (growth from a one-neighbour atom, the chain tool's
  zigzag), through one piece of module state in geometry.ts set by the Sketcher; every preset
  uses 120°, so nothing changes unless a custom style sets another angle.
- **"Apply bond length".** The ⋯ menu's existing "Reset bond length" already rescales the
  drawing so its mean bond is the standard bond, which every style draws at its fixed length.
  The editor adds "Actual size", a view change (zoom so a standard bond is the style's length
  on a 96 dpi screen) that leaves the model alone. Changing the model's unit to the style's
  bond length would desynchronise the tools, the server depiction and the molfile scale.
- **Export sizing.** `exportImage.ts` builds the scene in points (a standard bond = the fixed
  bond length) and turns it into primitives (lines, filled polygons, circles, text with
  explicit baselines and start positions: no dominant-baseline, which Word and Inkscape read
  differently). The box is the ink plus a 2 pt margin. SVG: width / height in pt and the viewBox
  in pt (1 unit = 1 pt). PNG: the SVG rasterised at `round(pt / 72 * dpi)` px (150 to 1200 dpi)
  with a pHYs chunk carrying the dpi, so Word and LaTeX place the file at its physical size.
  PDF (`pdf.ts`, no dependency): one page whose MediaBox is the drawing in pt, round-capped
  strokes, filled wedges, Bezier circles and text in the base-14 fonts (Helvetica,
  Helvetica-Bold, Helvetica-Oblique; Times-Roman for the Times family, laid out with Helvetica
  widths), WinAnsi with the minus as the en dash, nothing embedded, no dates (deterministic).
  Checked: pdfinfo reports the page size, pdftoppm renders the SVG and PDF identically, and
  pdflatex includes the PDF through graphicx.
- **LaTeX.** "Copy LaTeX" gives a figure that includes scheme.pdf at its width in cm. "Copy
  chemfig" writes chemfig code. Each bond gets its absolute angle from the 2D coordinates, plus
  a length factor beyond 5 %. Branches go in parentheses and ring closures are hooks (`?[a]` ...
  `?[a,{=^}]`). Ring doubles are offset to the ring side (`=^` / `=_`), wedges and hashes are
  Cram bonds (`<`, `>`, `<:`, `>:`) and labels carry their H and charge (`{H_3N^{+}}`). atom
  sep, widths and the font size come from the style. Compiled with chemfig 1.82 from CTAN and
  pdflatex on camphor, carvone, zwitterionic alanine, the Boc-alanine ammonium salt, cholesterol
  and benzene; the pictures match the sketch. Not supported: radical dots (dropped, with a
  comment), crossing breaks, circled charges, element colours.
- **Drag and copy: verified vs expected.** The files are rendered in the background, debounced,
  so dragstart and the copy buttons attach them synchronously.
  - Verified in Chromium (Chrome under Playwright):
    - A dragstart on the handle fills the DataTransfer with scheme.png, scheme.pdf and
      scheme.svg as File items, text/html (an `<img>` PNG data URL at the physical size) and
      DownloadURL (`image/png:scheme.png:blob:...`). The e2e test checks this on a synthetic
      DragEvent.
    - In a real drag to a drop zone in the page (Playwright's drag interception), the zone saw
      text/html, text/plain ("scheme.svg", Chromium's) and our marker type, but no Files and no
      DownloadURL.
    - "Copy image" writes image/png plus image/svg+xml (`ClipboardItem.supports` is true in
      this Chrome). A paste into a contenteditable on another page receives an image.png File
      and inserts the image. Chromium re-encodes the pasted PNG, though, and the dpi chunk is
      lost, so a pasted PNG is sized by its pixels (96 dpi) unless the target takes the SVG.
  - Expected only (no Word, Overleaf or desktop here):
    - Word desktop and Word Online: paste of the PNG works (a plain clipboard image); Microsoft
      365 may take the SVG at its true size. Otherwise set the size from the cm shown in the
      panel.
    - Word, a drop: works if Chrome hands the File items or the HTML image to the OS drag.
    - Desktop or file manager, a drop: DownloadURL saves scheme.png (Chrome only).
    - Overleaf file tree: a drop uploads only if real File items arrive, which the in-browser
      test did not show. The reliable route is the PDF download plus Overleaf's upload, then
      "Copy LaTeX", or "Copy chemfig" with no file at all.

## 2026-10-25 — Workbench point-group calculator

- **What.** `GET /api/molecule/{id}/symmetry?tolerance=` (`PointGroupOut`) gives the
  Schoenflies point group of the QM record geometry (`serve/symmetry.py`, numpy + scipy's
  kd-tree and assignment solver, no new dependency; in the API process, memoized per molecule,
  tolerance and record digest; at most 1000 atoms, LIMIT_EXCEEDED above; a numerical failure is
  SYMMETRY_FAILED). The Global tab shows the group, its order, the tolerance, the elements and
  the equivalent-atom classes; the 3D view draws the elements and colours the classes.
- **Tolerance semantics.** An operation counts when it maps every atom within the tolerance
  (Å, 0.01–0.5, default 0.1) of a distinct atom of the same element. Each operation is the
  Kabsch fit (determinant fixed) to the permutation it induces, so a slightly-off candidate
  axis is corrected before the test; products formed by the closure must pass the same test,
  so the reported group is consistent at the stated tolerance. The 0.1 default absorbs
  MMFF/GFN2 noise; a per-coordinate noise of 0.02 Å already pushes a 60-atom C60 below Ih at
  0.1 Å (pair residuals reach about 0.1 Å), which is the semantics, not a bug.
- **Algorithm.** Candidates come from the inertia axes, each equivalent-atom set's axes and
  centroid, atom directions and pair midpoints / differences / plane normals (sets above 16
  atoms use only pairs with three anchors); C_n (n <= 12), mirrors, S_2n and i are tested with
  a probe-atom kd-tree prefilter. The group is a greedy closure, best deviation first, from six
  seeds (the largest wins), keyed by (determinant, permutation), which is unique for any
  non-linear structure (a planar one has E and σh with the same permutation, told apart by the
  determinant). Linear structures (every atom within the tolerance of the least-inertia axis)
  are C∞v / D∞h by an inversion test, outside the closure. The decision tree runs on the
  closed group (counts of high-order axes, the principal axis, perpendicular C2 axes, σh,
  vertical mirrors, S_2n) and the order is checked against the name (a note if it differs).
  Element subscripts follow the conjugacy classes: in Dnh / Cnv with even n the vertical class
  holding more atoms is σv, the other σd (σv / σv′ for n = 2); perpendicular C2 classes are C2′
  and C2″; Oh mirrors normal to a C4 are σh.
- **Symmetrized structure.** The group average of each atom's images, operations refitted to
  the average three times; its RMS and largest displacement are reported. The hint repeats the
  search at 0.05, 0.1, 0.15, 0.2, 0.3, 0.4 and 0.5 Å (up to 300 atoms) and reports the highest
  group and its smallest such tolerance ("would be C2v at 0.15 Å").
- **Not done.** No icosahedral/cubic subgroup orientation conventions (D2h axes are not named
  x / y / z); C_n is capped at n = 12 (D∞h / C∞v are handled separately); the Hungarian
  matching minimizes the summed squared distance, not the largest one, so a borderline case can
  miss a bijection that a bottleneck matching would find.

## 2026-10-25: Collision-aware atom labels, tidy fragments, zoomable level diagram

- **Label placement is a render pass, not a model edit** (`sketch/labelLayout.ts`, run by
  `drawing.buildScene`, so the canvas, the SVG / PDF / PNG export and the chemfig hints agree).
  Each labelled atom starts from ChemDraw's choice; only when its ink collides with another
  label, a bond or itself are alternatives scored: overlap (ink boxes from Helvetica metrics at
  the style's size, `sketch/labelGlyphs.ts`) times 10^4 plus a small preference cost, so any
  overlap-free option beats any preference. Order of resort: H side (right, left, above, below)
  and charge slot (the four corners, then above / below / right / left), then subscript and
  charge shrunk to 85 % and 72 % (never under the style editor's 40 % of the font), then a
  render-only displacement of up to 0.9 font with a leader, for terminal or isolated atoms only
  (bonds must still meet at a symbol), else the overlap is reported (`Scene.issues`,
  `data-label-issues`). Clearances: another label at least max(margin, 0.2 font), a charge sign
  0.3 font (a sign next to a foreign symbol reads as its charge), a bond 0.75 margin.
- **Bond ends trim against the final ink** with the margin as a round-cornered band around each
  glyph box, for every line of a multiple bond and both edges of a wedge (laterals of the
  bond's half-width), instead of one ray exit from a symbol box: no stroke ever enters a label,
  and diagonal bonds are not over-trimmed by box corners.
- **The test metric** (`sceneOverlaps`): label-label (two atoms' ink boxes), label-self
  (non-adjacent runs of one label) and label-bond (a drawn stroke with its width or a wedge
  outline in an ink box), any positive area / length counts. Checked on Boc-Ala NH4+, alanine
  zwitterion, glucose (ring and open chain) and sodium tosylate in Screen, ACS 1996 and Total
  synthesis at three zooms, from RDKit.js layouts and from the server's Clean layouts recorded
  in the test.
- **Tidy fragments moves whole fragments in the model** (`sketch/fragments.ts`): a translation
  per fragment keeps every bond, wedge and depth, hence topology and stereo (the test checks
  RDKit's canonical isomeric SMILES). Clearance: half a bond length, at least one font size,
  between the default-placed ink of the fragments (labels as ChemDraw places them, not after
  the collision pass, which may have moved an H out of the way only because the ion was
  there). The largest fragment stays; the others are pushed along the nearest-point direction.
  It is measured in the current style and also in ACS 1996 and Total synthesis, so a salt
  tidied on screen does not touch in a publication export. Applied after every Clean as its own
  undo step (one undo keeps Clean's placement) and by the Tidy button, which shows only with
  two or more fragments (it would otherwise widen every toolbar).
- **chemfig hints follow the placement**: the H run goes on the drawn side (above / below fall
  back to the bonds' rule: chemfig has no such form); a charge off the upper right becomes
  `\charge{<angle>=$\scriptstyle±$}{X}` only where the H run is not, else the usual
  superscript (a `\charge` on the symbol's node would sit on the H).
- **Level diagram degeneracy is regrouped client-side** (`orbitals/levels.ts`) with the
  server's own rules: `degenerate_levels` (a new level when an MO lies more than the threshold
  above the level's first MO, ascending energy; not chaining) and `frontier_blocks` (HOMO =
  highest-index MO with occupation >= 0.5, blocks within the threshold of the anchor, HOMO−1 /
  LUMO+1 the next distinct level). At 0.05 eV both match the server. At large thresholds a
  frontier block can straddle two drawn groups (the two rules differ); the label then states
  the block's own size and mean energy. The threshold (0.001–0.5 eV) is `Prefs.degeneracyEv`.
  WorkbenchPage was not changed in this task: OrbitalPanel takes `degeneracyEv` /
  `onDegeneracy` props, and without them keeps its own state and merges the field into the
  stored preferences (the page's `usePrefs` then rewrites its own copy on its next preference
  change until the page passes `prefs.degeneracyEv` / `setPrefs`).
- **Zoom model**: the visible range is the window's until the user zooms or pans, and returns
  to it when the molecule, the orbital set or the window changes. Wheel = zoom about the cursor
  (a non-passive listener, so the page neither scrolls nor zooms), Shift+wheel and drag = pan,
  a click after a drag is swallowed so it does not select a level. The span is kept between
  0.05 eV and 4x the spectrum, the centre inside it. Occupation arrows are drawn as paths
  (not 8 px glyphs) and dropped where neighbouring levels are closer than 8 px (zoom shows
  them); frontier labels are spread at least 15 px apart with dashed leaders; below 380 px
  width the labels drop their energies.

## 2026-10-25 — Workbench metal complexes: coordination-template start geometry (serve adapter)

- **Diagnosis** (real GFN2-xTB through the workbench featurizer in a scratch data root, 33
  complexes; before: 19 right, 4 wrong, 8 failed, 2 rejected). Root causes, by weight:
  1. *Embedding.* MMFF has no transition-metal parameters and UFF types no charged metal, so
     every metal record carried EMBED_UFF_FALLBACK with raw ETKDG coordinates. ETKDG fails
     (EMBED_FAILED) on every five- and six-coordinate metal without SMILES stereo
     ([Co(NH3)6]3+, [Fe(CN)6]4-, Cr(CO)6, [Fe(H2O)6]2+, Fe(CO)5), hits an RDKit invariant
     violation on ferrocene (UNEXPECTED), and embeds every four-coordinate metal tetrahedral,
     square-planar d8 included (start angle RMS 6-33 degrees off square planar).
  2. *GFN2 from a wrong start.* GFN2 recovers square planar for small complexes, but not for
     RhCl(PPh3)3 or Pd(acac)2 (both stay tetrahedral, Wilkinson's with Rh-Cl 2.88 A and no
     warning); cisplatin and Pd(acac)2 dissociated an M-L bond (1.38x, 1.90x) and needed the
     walls (RESTRAINED_GEOM); a Grubbs-type Ru(=CHPh)Cl2(PMe3)2 ended with Cl trans to P;
     PtCl2(NH3)2 without stereo came out cis or trans by chance.
  3. *Spin.* The pipeline is closed-shell: [NiCl4]2- (a tetrahedral triplet) came out as a
     square-planar singlet. d9 Cu(II) is already SPIN_PARITY (RDKit puts no radical on metals).
  4. Not causes: dative bonds (`->`) parse and stay bonds; charge-separated forms
     (`[NH3+][Pt-2]`) work; RDKit honours @SP/@TB/@OH, so stereo-tagged Co(III) isomers were
     already right; tblite has GFN2 parameters for Z 1-86 (no UNSUPPORTED_ELEMENT). The 1.3x
     connectivity check is not too strict for M-L bonds (final M-L bonds lie at 0.81-1.03x the
     radii sum; the breaks it caught were real), and the flat-bottom walls did not distort
     (restrained cisplatin = free cisplatin from the template). Ionic multi-fragment SMILES
     (`[Fe+2].c1cc[cH-]c1.c1cc[cH-]c1`, `N.N.Cl[Pt]Cl`) fail; the workbench already refuses
     them as MULTIPLE_FRAGMENTS.
- **Adapter** (`serve/metal_geometry.py`, RDKit + numpy; `serve/metal_qm.py`, the GFN2 driver,
  `qm/` untouched). Sites from the metal's bonds (mutually bonded neighbours form one eta-n
  site); oxidation state = formal charge + covalent bond orders to non-metals (dative 0, M=C 0,
  bonds to alkali/alkaline-earth/Al -1); polyhedron by site count, SMILES stereo first, d8
  Pd/Pt/Au/Rh/Ir (Ni with C/P or all-N donors) square planar, five-coordinate d6 square
  pyramidal; isomer from @SP/@TB/@OH (tables derived from RDKit's own embedder and re-derived
  in the tests; handedness not enforced) or the stated default (chelates cis, then the most
  like-ligand pairs trans); donors at r_M + r_L - 0.05 A (sp C 0.69); ETKDG of the metal-
  stripped molecule with the template atoms in the coordMap and the bounds (metal floors
  capped), ligands moved rigidly onto their donors, monodentates turned to point their lone
  pair at the metal, eta-n substituents in the ring plane, the seed with the best
  radius-scaled closest contact. Then `GFN2Calculator.run` (inherited) with a metal-aware
  connectivity check: the M-L factor is its own knob, kept at 1.3 (walls 1.2) on the evidence
  above, and walls only if a bond breaks. Mononuclear connected complexes with CN <= 6 only;
  otherwise the standard embed and METAL_TEMPLATE_SKIPPED. METAL_POLYHEDRON_CHANGED /
  METAL_ISOMER_CHANGED (best vertex assignment of the final geometry) and FERMI_SMEARED are
  added to the record warnings.
- **Open shells refused at preflight** with RADICAL_NOT_SUPPORTED (the workbench's
  closed-shell code): Werner complexes (no C, P, Si, H or eta-n donor) with octahedral d2/d4/d8,
  high-spin 3d d6 (not Co(III), not six aromatic N), or tetrahedral d2-d8. Deliberately
  conservative: organometallic ligand fields are never refused.
- **Method tag.** Records and tokens of the metal path use `<GFN2 method>+metal-geom-v1`,
  applied by `metal_geometry.method_for` only to keys with a d-block metal and only to the GFN2
  method (non-metal keys and the synthetic method are untouched). The cache key and the
  provenance config differ (solvation unchanged), so they never collide with the batch's
  records; the shared layer is still read first under the tagged method, so an upstream
  adoption with the same tag is picked up. `orbital_sets.load_qm` verifies the record against
  `method_for(key)`.
- **After**: 27 right (every connected closed-shell case: polyhedron and isomer, no
  RESTRAINED_GEOM), 4 refused (2 d9, [Fe(H2O)6]2+, [NiCl4]2-), 2 failed (the two ionic
  multi-fragment SMILES). Wilkinson's catalyst: 167 s instead of 339 s. On 22 PMechDB metal
  species (read-only comparison with the batch manifest): 6 of 8 SCF_NOT_CONVERGED TiCl4
  carbonyl adducts now converge (square pyramidal, flagged METAL_POLYHEDRON_CHANGED from the
  TBP start), bent ZnCl2 becomes linear, the clean ones are unchanged; the RESTRAINED_GEOM ones
  are one-coordinate fragments the template cannot help.
- **Upstream proposal** (not applied, for the user to decide): move `metal_geometry` to `qm/`;
  `QMConfig.metal_template` (default off) and `QMConfig.method_for(smiles)`, used for the cache
  path in `batch._compute/_is_done/build_manifest`, `orbitals.build.process_molecule` and
  `arrows.reactants.TokenCache`; `embed()` dispatches metal molecules to the template; the M-L
  factor in `GFN2Calculator._relax`; a failure code OPEN_SHELL_LIKELY. `provenance.verify`
  compares kind, schema, config and inputs only (code hashes are tracing), so the change stales
  no existing record; switching it on recomputes the 302 PMechDB metal molecules (179 records,
  46 RESTRAINED_GEOM; 123 failures, 99 of them SPIN_PARITY) under the new method and re-runs
  the samples of their steps. It must not be edited into the checkout while a batch runs
  (records stamped `code_dirty`, refused by `require_clean`).
## 2026-10-25 — Phase-2 commands in parallel: sidecar, ks-rank, LightGBM; torch out of `arrows.steps`

- **Why.** The phase-2 rebuild left the CPU mostly idle, and the user asked why it was slower
  than before. Only the sample build used workers.
  - The sidecar ran as one process per split, about 2 hours each.
  - ks-rank ran in one process.
  - LightGBM fitted on its default 2 threads, and it scored val and test one step at a time on
    one thread (the commands ran with `OMP_NUM_THREADS=1`). For the scaffold geometry-free
    ranker that was 31 of 49 minutes.
- **Sidecar.** `ksmech-arrows contacts --workers N` runs chunks of 16 consecutive steps (in
  step-id order) in spawned processes, as the sample build does.
  - The parent writes the shards in order, so they are the same for any N: same entries,
    arrays and recorded inputs. A test checks this.
  - A cold oriented-kernel cache costs 9.9 s per step in one process (1.2 GB). With 6 workers
    on a 240-step slice it was 2.9 s per step, at 838 MB per worker before torch left the
    import (see below). The slice had only 15 tasks, and two workers can compute one block at
    once. With a warm cache the same slice took 4 s.
- **ks-rank.** `ksmech-physics ks-rank --workers N`. Each part's step ids, in view order (the
  views list them sorted, as the store's shards), are cut into chunks of 64 (`physics.parallel`).
  - Each worker loads its own samples and contact entries, refreshes their validity and
    computes their components for every distance. Ranking, failure classes and the class table
    stay in the parent, and the report is identical (a test compares the JSON).
  - Measured: the scaffold val part in 36 s with 8 workers, against about 123 s in one process.
- **LightGBM.**
  - `gbdt --workers N` loads the samples the same way. The training rows are identical,
    because negatives are drawn per step id; so are the val / test candidate sets. A test
    checks both feature sets; undefined geometry-free features are NaN on both sides.
    - Measured: the scaffold geometry-free val read in 17 s with 8 workers, against 50 s.
  - `--n-jobs` defaults to 14 (was 2), and `LGBMRanker.score_sets` predicts val and test in
    batches of about a million rows on `--n-jobs` threads.
    - Measured on 300 scaffold geometry-free val steps: per-step prediction 193 s, batched on
      one thread 193 s, batched on 12 threads 23 s. Scores identical.
  - ~~With `deterministic` and `force_row_wise` the trees do not depend on the thread count,
    so `n_jobs` stays out of the configuration a model records.~~ Wrong on real data; see the
    correction below.
  - The pool is closed before fitting starts, so the loading workers and the LightGBM threads
    never run at the same time.
- **torch left `arrows.steps`.** `SampleDataset`, `collate_samples`, `make_loader` and the
  batch tensors moved to `arrows.batches`. Importing `arrows.steps` pulled in torch, about
  500 MB per worker process for nothing: the physics CLI went from 683 to 205 MB at import.
  - With 8 workers the largest ks-rank worker went from 1,051 to 570 MB (9.4 GB to 5.1 GB for
    all processes).
  - The largest LightGBM loading worker went from 1,208 to 698 MB.
- **Settings for 16 GB.** `--workers 8` for `contacts`, `ks-rank` and `gbdt`, and
  `--n-jobs 14`. The default stays 1, as for `samples` and `align`.
- **A script that spawns workers needs an `if __name__ == "__main__":` guard.** A spawned
  worker re-imports the main module. An unguarded script started a pool inside every worker,
  which died during bootstrap and was respawned without end, and the parent waited forever.
  The console scripts are guarded.

## 2026-10-25 — Correction: LightGBM's trees depend on its thread count; the parallel runs timed

- **What happened.** The scaffold runs were repeated with `--workers 8`, and with `--n-jobs 14`
  for LightGBM.
  - ks-rank: report identical; 489 s → 301 s.
  - LightGBM, contact features: report, trees and learning history identical; 1,718 s → 392 s
    (load 220 → 126 s, fit 416 → 163 s).
  - LightGBM, geometry-free: 2,938 s → 574 s (load 331 → 87 s, fit 745 → 296 s), but the trees
    differ: 1,295 / 1,258 against 1,273 / 1,082. Test top-1/3/5 is 45.3/64.3/75.3% against
    45.3/64.7/75.6%.
  - The same run with `--workers 8 --n-jobs 2` reproduces the 2-thread run exactly: results,
    trees and history.
- **So.** The parallel loading is exact. LightGBM's `deterministic` and `force_row_wise` make a
  model reproducible for one thread count only. The 1-against-4-thread toy test, and the
  156k-row contact set, did not show it; the 521k-row geometry-free set does.
- **Change.** `n_jobs` is back in the configuration a model records, and the toy test now
  checks only that batched scoring equals scoring one step per call. `--n-jobs` stays 14 by
  default: models fitted with different thread counts are told apart by their configuration.
  The geometry-free scaffold model on disk is the 2-thread one, identical to the morning run.
- **Still sequential in ks-rank.** After the parallel part, about 2.5 minutes of the 5 run in
  the parent. Its failure-class step reads each sample's token files again, because the
  workers, not the parent, warmed the token cache.

## 2026-10-25 — git in provenance and its tests ignores the hook's GIT_* variables

- **Why.** Two `git commit --only` attempts failed with "files were modified by this hook",
  though every test passed.
  - For `git commit --only` (and `git commit <paths>`), git builds the commit in a temporary
    index and exports its absolute path to the hooks as GIT_INDEX_FILE.
  - `test_code_state_sees_tracked_edits_and_untracked_sources` runs `git init / add / commit`
    in a scratch repository. It inherited the variable and wrote the scratch repository into
    that index. A copy of the real index went from 154 files to 1. pre-commit noticed the
    change and refused the commit; unnoticed, the commit would have held the wrong files.
  - `provenance.code_state`, pointed at the scratch repository, read the same index.
  - A plain `git commit` was safe, apparently because the path it exports is relative and
    resolves inside the scratch repository.
- **Change.** `provenance.git_env()` is the environment without GIT_* variables. `code_state`
  and the tests' git helper run git with it. A test sets GIT_INDEX_FILE to a sentinel file and
  checks that it is left untouched and that `code_state` still reads the scratch repository.

## 2026-10-25 — Review fixes before training: worker exceptions, pair statistics, pair tensors, skip codes

From the code review of the branch (five findings fixed together).

- **Exceptions from spawned workers.** `StaleArtifact(path, changes)` could not be rebuilt in the
  parent: pickle calls the class with the message alone. A worker that met a stale shard
  (`ks-rank --workers`, `gbdt --workers`) killed `multiprocessing.Pool`'s result thread, and the
  parent waited forever (reproduced). `StaleArtifact`, `ArrowApplyError`, `StepRejected`,
  `QMError` and `MissingTokens` now define `__reduce__` from their own fields. A test pickles
  each one, and one raises `StaleArtifact` in a spawned worker and catches it in the parent.
- **Pair statistics over the valid cells.** The fix-item-8 entry (2026-10-22) says the
  statistics are fitted on the valid cells only. The code fitted the samples just built, whose
  stage-A blocks define every (occupied, virtual) cell: on the molecule split, 48,802,340
  entries per feature against 43,950,761 valid cells, the only ones a loaded sample holds.
  `pair_stats` now masks with `pair_valid` and rounds the values to float16 as the store keeps
  them. The statistics of the samples just built then equal a refit on the loaded samples
  exactly (tested).
- **A provenance header on `pair_feature_stats_<strategy>.json`.** The file was plain text, so
  the trainer could not refuse a stale one. It is now a stamped JSON (kind `pair_stats`,
  inputs the training view and the store index, config the features, `cells: valid` and K,
  eps). `arrows.steps.load_pair_stats` verifies it and refuses a dirty build, through
  `provenance.verify_clean` (the store's verify helper, now shared, rather than another copy).
  Existing files (including the retired `pair_feature_stats_random.json`) have no header and
  are refused until `ksmech-arrows samples` is re-run.
- **Pair tensors in `collate_samples`.** The loader had none of the pair inputs. They now
  stay in each sample's (occupied, virtual) layout, padded to the batch's largest counts, with
  the batch token position of every row and column (`pair_row_pos`, `pair_col_pos`). The model
  scatters them into its N x N pair track. A dense (B, N, N, F) block would cost N^2 per
  sample (N up to ~1,000 on the largest PMechDB pools) to store values that exist only on
  occupied x virtual cells. The values stay at K = eps = 1 and unstandardized: K and eps are
  learnable, so `to_params` and the training statistics are applied in the model.
  `pair_contact_sites` is not collated: stage-A samples hold -1 there.
- **Skip codes for unusable token files.** `TokenCache` kept the reason for a stale token file
  or a missing Wiberg sidecar in `stale`, which nothing read (and a worker's copy dropped), so
  such steps were counted as MISSING_TOKENS. The cache now logs each one and gives its code
  (`missing_reason`: STALE_ARTIFACT, NO_BOND_INDEX). `load_fragments` puts the codes on
  `MissingTokens`, and the sample build, the edit-label report, the contact sidecar and the
  alignment pass count the step under `MissingTokens.reason`. When several fragments lack
  tokens, the step's code is the first of STALE_ARTIFACT, NO_BOND_INDEX, MISSING_TOKENS that
  any of them has, so a stale file is never hidden behind a missing one. On the current build
  all 46 MISSING_TOKENS skips are genuinely missing files.

## 2026-10-25 — Cleanup after the review: one pool helper, one verify helper, one candidate rule

No behaviour change: every output is the same, and the samples, sidecars and models on disk
stay valid.

- **`ksmech.processes`** holds `single_threaded_children`, `chunked` and `in_processes` (a
  spawned pool, results in task order). Five hand-written pools used to repeat that pattern:
  the token build, the alignment pass, the sample build, the contact sidecar and
  `ksmech-physics`. The module imports nothing from ksmech, so every layer can use it; the
  physics copy could not be reused by `arrows` because `physics` imports `arrows`.
  - The sample build now receives its chunks in order (`imap`) rather than as they finish. It
    keeps every sample until the store is written, sorted by step id, so the store is the same.
  - The serial paths run the same chunks in this process.
- **`provenance.verify_clean`**: verify the header, refuse a dirty build, or only log the
  mismatch under `--allow-stale`. It replaces four copies: the sample store, the contact
  sidecar, the LightGBM ranker and the GBDT.
- **One candidate rule.** `ReactantTokens.donor_ok` / `acceptor_ok` now read
  `MoleculeTokens.donor_candidates` / `acceptor_candidates`, the methods that
  `physics.pairs.valid_pairs` uses. Before, the rule was written twice, and only a test tied the
  copies together (cf733dc).
- **Dead code removed:**
  - `MechanismSample.to_dict` / `from_dict`, from the torch sample files;
  - `store.stored_bytes`;
  - `provenance.write_framed` / `read_framed`. Old framed `.pt` files still have their
    headers read, for `ksmech-data provenance`.
- **Edit-label gate.** `ksmech-arrows edit-labels` tested for VARIANT_MISMATCH, a code that
  has not existed since class members failing the key check are dropped and counted
  (`steps_with_rejected_variants`). The test is gone; the gate is the labeled fraction, as
  before.
- `label_report` and `split_report` share `delta_be.class_shape`: the class sizes and the
  first anchor's kind.

## 2026-10-25 — Stage B leaves main: one training stage on geometry-free pair features (user decision, spec v9)

- **Why.** User decision, after the training discussion. Untrained, the contact terms ranked a
  step's first arrow worse than the gap factor alone. On the molecule test split, top-1 was
  about 3% for the frozen head, 14.3% for KS3 + KS2 and 20.4% for G. Stage B also needed
  physics that was never built (the configurational contact of spec v8.1) and a contact
  sidecar for every training step.
- **Where it is.** The local branch `archive/stage-b` at 8811e57 holds:
  - spec v8.1;
  - the contact kernels and the legacy oriented kernel;
  - the sidecar, `ks-rank`, the numpy logistic regression and GBDT, and the frozen head;
  - their tests.
  The branch is not pushed.
- **Model (spec v9).**
  - The pair track gets the geometry-free pair features only, and all H heads are learned.
  - Gone: the Klopman-Salem attention bias, the frozen head, tau, kappa, F^atom and eps as a
    model constant.
  - K is fixed at 1.75. A learnable K would only rescale one standardized feature, the
    cross-centre coupling.
  - The step query drops E_step[t]. On PMechDB, which has no pathways, the stop head is not
    trained.
- **Code removed.**
  - Modules: `physics/contact/` (with `legacy_fixed23`), `contact_kernel`, `contact_table`,
    `repulsion`, `gto` (with `physics/data/sto_ng.json`), `ranking`, `parallel` and
    `arrows/contacts`.
  - Commands: `ksmech-arrows contacts`, `ksmech-orbitals contact-table`,
    `ksmech-physics ks-rank`, `ksmech-physics baseline` and `gbdt --features contact`.
  - From `ks_terms`: the KS3 forms. From `pairs`: the legacy pair block and kernels,
    `ks_energies` and `frozen_head`. From `params`: `tau`, `shell_fock_ev` and the legacy
    scalings.
  - The kernel argument of the sample build.
- **Kept and moved.**
  - `ao_energies` moved to `ks_terms`. It reads the shell layout from `qm.ao.gfn2_shells`
    rather than the STO-nG tables. The values are bit-identical on 400 PMechDB token files
    (elements up to Ti, Zn, Sn, Br and I); no GFN2 element has two shells of one l.
  - `vdw_sum` moved to `coulomb`.
  - `first_label`, `label_pairs`, `label_rank` and `summarize_ranks` moved to `baseline`, with
    the candidate sets. `token_class` moved to `textbook`, and the read chunk to `physics.cli`.
  - `baseline.FEATURE_NAMES` is the geometry-free list (the old `GEOMFREE_FEATURE_NAMES`).
- **Commands now.**
  - `gbdt` is geometry-free only. Its files keep the `_geomfree` tag, so the models on disk
    stay valid.
  - `textbook` runs the untrained gap kernel and the two geometry-free LightGBM models of the
    molecule split. It used to run KS3 + KS2 and the contact GBDTs.
- **Samples.**
  - Schema 15 is unchanged, so there is no rebuild. The samples, `pair_feature_stats_*.json`
    and the geometry-free models on disk stay valid: provenance compares inputs, and the code
    hash is for tracing only.
  - `pair_contact_sites` (always -1) and `atom_coulomb` stay in the store. Neither is a model
    input, and `collate_samples` no longer emits `atom_coulomb`. Both leave at the next schema
    bump, the resonance augmentation.
- **Data left on disk, readable only from the archive branch.**
  - `data/cache/oriented/` (6.1 GB) and `data/cache/contact_table.json`.
  - The sidecars `samples/contacts_molecule/` and `samples/contacts_scaffold/` (0.7 GB).
  - The `ks_rank_report_*`, `baseline_*` and `gbdt_lgbm_*_contact*` reports and models.
  Nothing was deleted.
- **Other branches.** `claude/workbench` and `claude/serve-ui` import the contact code (KS
  ranking, physics view, predictors). They must drop it when they merge main.
- **Open, for the anchor-level references.**
  - Spec v9 pools the standardized orbital-pair features onto atom pairs as a weighted mean.
  - G^atom sums over every occupied x virtual pair, the donor's own antibond included, which
    can inflate it on bonded pairs. Decide whether it should run over `pair_valid` only before
    the floor is built.

## 2026-10-25 — Serve on spec v9: merge of main, the vendored Gaussian basis, the contact view removed

- **Merge.** `claude/predict` merges main at spec v9 (d566010). Main removed `physics/gto.py`,
  `physics/ranking.py`, `physics/contact/`, `physics/contact_table.py`, `physics/repulsion.py`
  and `arrows/contacts.py`; serve imported several of them.
- **`serve/gto.py`.** The workbench's orbital and density surfaces (`cubes`, `densities`) and
  the synthetic EHT chemistry (`synthetic`) evaluate AOs and overlaps with the GFN2 Gaussian
  basis, so the module and its STO-nG table are vendored from 46adf6b into serve, bit-identical
  below the docstring (`serve/data/sto_ng.json`, same sha256). `tests/test_serve_gto.py` pins
  stored values (float.hex) and compares the code with the original. `cap_atoms` (the cube
  grid's cap) is vendored into `cubes` from `physics/contact/legacy_fixed23`.
- **Removed.** The contact placement replay (`serve/placement.py`, `serve/upstream.py`, the
  contact routes and cubes, the oriented-block cache and its token-digest bookkeeping in the
  worker), the site-pair view, and the spec-v3 KS panels (`serve/physics_view.py`,
  `serve/candidates.py`, `/api/pool/{id}/candidates`, `/physics`, `/site_pairs`). The capabilities
  `candidates` and `physics` stay listed, off, with the reason; `contact` and `site_pairs` are
  gone. The `oriented` fingerprint became the `token` fingerprint (tokens, QM and basis code).

## 2026-10-25 — Serve Predict and Correct on the reaction-matrix target (spec v9)

- **Edges are reaction matrices.** A network edge is R on the pool atoms of its source state
  (`parse_mapped(mapped_smiles)` indices, every H an atom), with its anchor, the arrow reading,
  the orbital readings of the forming bonds, the product SMILES and sigma keys. B + R is
  applied with `arrows.apply.apply_edit` and `sanitize_product`; an extra octet check (more than
  8 valence electrons on H-Ne) catches what RDKit's valence table allows for charged atoms. A
  product with the parent's pool key is a resonance interconversion (RESONANCE_ONLY_STEP),
  never an edge. Networks are format v2; spec-v3 networks on disk are refused with a reason.
- **Pooled gap in serve.** `serve/gatom.py` computes G_atom = sum over the valid (r, s) of
  (w_i^r w_j^s + w_j^r w_i^s) G(r, s), G in float64 from the full spectra, w the token centre
  weights; only the pairs `physics.pairs.valid_pairs` allows contribute (the donor's own
  antibond is excluded, which answers the open question of the stage-B entry for the floor).
  Move upstream when `model.inputs.pooled_gap` lands. The pool analysis npz now holds the
  pair block and the weights instead of the KS components.
- **The floor.** Anchors are atom pairs i < j ranked by |G_atom|: "form" while the order stays
  at most 3, "break" on a bonded pair (both for a bonded pair below 3). The top `beam` anchors
  are completed: a forming anchor from its two best channels in each direction through upstream
  `apply_predicted_arrows` (private `_step` is used only to read `leaving_choices` on the state
  after the arrow's source and new bond), one candidate per allowed leaving bond; a breaking
  anchor as the heterolysis towards either atom. Every dropped completion is counted
  (`COMPLETION_FAILED:<reason>`, `ANCHOR_NOT_COMPLETED`, `APPLY_FAILED:<issue>`) and shows in
  the node's `apply_failures` and the network's `skipped`.
- **Scores are not probabilities.** The floor reports |G_atom| (1/eV), the GBDT its lambdarank
  score; `logp` is null for both and `scores_are_probabilities` is false. The beam and the
  pathways use a log weight: the log of a child's share of the kept children's scores (floor)
  or a softmax of the scores (GBDT); a pathway's `probability` is null unless the predictor
  gives probabilities. Only the network adapter (`NetworkModel.decode -> (R, logp)`) does.
- **GBDT.** The default file is `<shared>/data/processed/pmechdb/gbdt_lgbm_molecule_geomfree.npz`
  (`ksmech-physics gbdt`), loaded with `physics.lgbm.load_lgbm` (provenance verified: a stale
  model is unavailable with its reason). It ranks the valid orbital pairs; the best `beam`
  are completed at the centres (a, b) with the largest w_a^r w_b^s.
- **Network.** Available only through a `NetworkModel` adapter, which needs `ksmech.model.net`
  and a checkpoint `data/models/<source>/<run>/best.pt`. Neither exists on this branch, so it is
  listed unavailable with the reason; a fake-model test exercises the adapter.
- **Arrow reading.** `serve/reading.py` decomposes R as a perfect matching of the pair sources
  (lone-pair losses, breaking bonds) onto the sinks (forming bonds, lone-pair gains). Where
  several exist, the spec asks for the largest |G_atom| of the forming bonds, but G_atom is
  symmetric and the same in every reading, so the reading uses the directed part G_dir[a, b]
  (donor on a, acceptor on b; G_atom = G_dir + G_dir^T) of each forming bond at the end the
  pair comes from. Ties keep the first matching in sorted order. The orbital reading of a
  forming bond is the channel with the largest term of G_atom, as the spec says.
- **Correct.** `POST /api/step/check` validates R with issue codes and atoms and returns the
  readings when the pool's analysis is ready (a 1 s wait; otherwise `readings_status`
  "pending"). A user step (`override`) is synchronous and adds an edge of predictor "user"
  (kind "corrected", weight 0) and a FRONTIER child; expansion of a node is an `expand` job.
  `predictor: "user"` on `POST /api/predict` builds a root-only network for Correct.
- **Curation.** StepDocs are format v2: R (`MatrixIO` and `ReactionMatrix.to_list()`), the
  anchor, the reading (pool atoms, and map-number `dataset_arrows` for Browse), the orbital
  readings and the sigma keys. A re-check ranks the curated product among the predictor's
  distinct products by pool key (sorted sigma keys); the Recheck columns keep their names
  (`first_arrow_rank` is that rank). A user's mechanism is re-checked with the floor and never
  goes predictor-stale.
- **Export.** The PMechDB-format export is a download only (`GET /api/curate/export/pmechdb`,
  `ksmech-serve export --out`): mapped reactions and the reading's arrow code, each record
  checked by parsing it back with `data.pmechdb.parse_arrow_code` and applying it. The
  `user_curated` ingest export is gone: curated steps are never training data.
- **Wiberg sidecars.** The geometry-free pair features need each fragment's Wiberg indices.
  The serve token loader attaches the layer's sidecar (`data/cache/bond_index` for the shared
  tokens, `data/serve/bond_index` for serve-built ones), and the synthetic and real
  featurizers write the sidecar when they build tokens.

## 2026-10-26 — Serve verifies the GBDT model against the shared root

- `physics.lgbm.load_lgbm` resolves the model's relative `file:` inputs (its sample views)
  against the checkout it runs from, so a serve worktree saw the shared
  `gbdt_lgbm_molecule_geomfree.npz` as stale (input missing). Serve now runs the same checks
  (`predictors.gbdt_problems`: kind, version, features, every input re-hashed, a clean build)
  with the inputs resolved against `KSMECH_SHARED_CACHE_ROOT`, as the token loader does, and
  reads the booster without upstream's verify. On the host the model is available and ranks
  the MeO- + MeCl SN2 product first.

## 2026-10-26 — Predict UI: network and step views (user choice among previews)

Three versions of each viewport were built and compared on real jobs before anything was
committed; the user kept two network views and all three step views.

- Network: **Scheme** (default) and **Pathways**. The force-directed graph was dropped: past about
  20 states its cycle edges tangle. Scheme is hand-rendered (HTML cards plus SVG curves), no
  longer cytoscape: card sizes are computed first, each depth column is stacked with fixed gaps
  in barycentre order (split into sub-columns past 7 states), so no card or caption can overlap;
  unit tests assert it on two real networks and a synthetic 200-state one. Cycle back-edges are
  hidden behind a "↺ n" chip unless hovered or toggled. Pathways merges shared prefixes into one
  trunk and places cards depth-first at their parent's height.
- Step: **Textbook** (default), **Animated**, **Orbitals**. Textbook uses ACS-1996 lines and fonts
  with a 17 pt bond instead of 14.4 pt, because at 14.4 pt a bond between two labels shrinks to a
  stub. Animated keeps atom identity by pool atom and relaxes the reactant drawing into the
  product with springs instead of morphing to RDKit's product layout, so substituents do not swap
  sides mid-animation; its last frame can therefore differ slightly from Textbook's product.
  Orbitals shows the dominant channel's donor and acceptor LMOs over the pool geometry with the
  canonical pair in a two-level diagram.
- The step type colour (substitution, addition, proton transfer, ...) in Pathways is a rough
  classification of the arrow reading for colour only, never a label.
## 2026-10-25 — Training preparation: ROCm torch, the model, the trainer and the anchor floor (spec v9)

- **Device (user decision).** Train on the AMD RX 6600 XT (8 GB, gfx1032) through ROCm, and
  keep the CUDA path for NVIDIA cards.
  - torch left the main dependencies for two conflicting uv groups:
    - `rocm`, the default: torch 2.14.1+rocm7.2, with triton-rocm 3.8.0, from
      download.pytorch.org/whl/rocm7.2. PyPI has only triton-rocm 3.0.0rc1, so the lock failed
      until triton-rocm got the same index.
    - `cuda`: cu130. An NVIDIA host runs `uv sync --no-group rocm --group cuda`.
  - The lock covers linux x86_64 only (`tool.uv.environments`). The Windows fork failed on the
    ROCm wheels.
  - gfx1032 has no rocBLAS kernels of its own: run with HSA_OVERRIDE_GFX_VERSION=10.3.0 (the
    gfx1030 kernels). Ollama on this host already ran ROCm 7.2 on the card that way.
    `select_device` runs one small product, so a missing override fails at start with a hint
    rather than mid-run.
  - Mixed precision:
    - ROCm: float16 with a GradScaler (RDNA2 has no fast bfloat16);
    - CUDA: bfloat16 where supported;
    - CPU: none.
    - The attention logits, softmax and loss stay in float32.
- **G^atom over the valid pairs (the user left it open; chosen here).** Eq. (Gatom) and the
  pooled means of y_ij run over `pair_valid`:
  - the store keeps no other cells, so pooling over all pairs would need the token files or a
    schema bump;
  - the donor's own antibond, which `pair_valid` drops, would inflate every bonded pair.
  - The spec's equation now names the set.
- **Model choices the spec leaves open** (`ksmech.model`):
  - Token inputs are standardized with training statistics (`FeatureStats`, stored in every
    checkpoint). A missing atom descriptor is 0 with its own flag.
  - z enters the pair track on both cells of an (occupied, virtual) pair with an orientation
    channel, so a virtual token's attention also sees its donors. Further channels: the centre
    weight between an orbital and its atoms, and a one-hot pair kind (the spec's pair-track
    paragraph now lists them).
  - MLP_U has hidden width d_p, its first layer split over [p_ij, h_i, h_j].
  - Each encoder layer, and MLP_P, is recomputed in the backward pass (`checkpoint`):
    activation memory becomes about one layer's.
  - The decoder reads the symmetrized atom-pair vectors (p_ij + p_ji) / 2, since R is
    symmetric. The anchor context c holds:
    - h_a + h_b;
    - the sum of the pair vectors joining the entry's atoms to a and b;
    - an embedding of the graph distances of i and j to the nearer anchor atom (0..6; 7 for
      another molecule), as an unordered pair.
  - y_ij = pooled standardized means (11), log1p(pooled weight), asinh(G^atom), WBI.
  - The stop head is built after all (the entry "Three encoders behind one interface, and the
    stop head", below); lambda_stop = 0 on PMechDB, which has no stop labels.
- **Loss.** The exact marginal over the class variants and their admissible anchors (one
  logsumexp per sample over the (variant, anchor) rows). A test recomputes it for every sample
  with a Python loop over all entries. The anchor-only NLL and the greedy R (top anchor,
  argmax entries, no beam) are reported beside it.
- **Batches by a cell budget** (`train.batching`):
  - Samples are pooled by 1,024, sorted by size inside a pool, and packed while
    B x (N_max + 1)^2 <= `--cell-budget`; the batches are then shuffled. The order depends only
    on (seed, epoch).
  - Samples over `--max-tokens` (default 1,024) are skipped as TOO_LARGE and counted in
    `run.json` and in every metrics line.
- **Loader.**
  - `lewis.graph_distances` is a breadth-first search (scipy csgraph). The integer matrix
    product per level took about 0.8 s per collation of the 411-atom pool; the search takes
    9 ms.
  - The DataLoader workers fork, so the 4.4 GB of loaded training samples are shared. Spawned
    workers would each reload them for 163 s.
- **Distances.** `torch.cdist`'s matrix-product form lost about 2e-3 A near zero distance once
  the molecule sat away from the origin; the rotation-invariance test caught it. The pair
  channels use the direct difference.
- **Physics floor** (`ksmech-eval floor`, molecule split). Each candidate pair is proposed
  once, with the sign form when its order may rise and break otherwise.

  | part | n | top-1 | top-3 | top-5 | median rank | unreachable |
  |---|---|---|---|---|---|---|
  | val | 1,239 | 15.2% | 25.8% | 30.8% | 26 | 39 |
  | test | 1,182 | 17.1% | 27.2% | 31.5% | 26 | 77 |

  `unreachable` counts steps whose only admissible anchors break a bond that could also form;
  the floor never proposes them. These are anchor accuracies over atom pairs and cannot be
  compared with the orbital-pair rankers of the first arrow (G 20.4% top-1), which rank
  another candidate set.
- **On the card (RX 6600 XT, ROCm 7.2, gfx1030 kernels).**
  - Matrix products run at 5.6 TFLOPS in float32 and 8.0 in float16.
  - Without the override, `torch.cuda.is_available()` is True but the first kernel segfaults
    (exit 139), which no exception handler catches. So `ksmech-train` sets
    HSA_OVERRIDE_GFX_VERSION=10.3.0 itself (`train.loop.rocm_override`), but only when:
    - the build is ROCm;
    - the variable is unset;
    - a KFD node reports an RDNA2 target without kernels (gfx_target_version 100301-100306;
      this card is 100302).
- **Memory and speed, default model** (7.34 M parameters, d = 256, d_p = 64, L = 8, H = 8,
  float16, per-layer recomputation). Molecule split, 9,994 training samples. 3 samples are
  over 1,024 tokens and skipped as TOO_LARGE; the largest kept has 966 tokens (935k cells).

  | cell budget | batches / epoch | peak, fullest batch | epoch (12-batch estimate) |
  |---|---|---|---|
  | 1.0 M | 460 | 3.2 GiB | ~7.7 min |
  | 1.6 M | 319 | 4.9 GiB | ~6.7 min |
  | 2.4 M | 248 | 7.1 GiB | ~4.9 min |

  - From 1.6 M up the caching allocator logged failed allocations and retried; 2.4 M sits at
    the 8 GiB limit.
  - The default `--cell-budget` is 1.1 M: about 7.5 min per epoch, about 4 h for 30 epochs.
- **Gradient scaler.** float16 starts at a scale of 2^10. At the default 2^16, the first
  losses (~700 nats, a sum over every entry) overflowed the first ten backward passes, and
  those steps were skipped.
- **Smoke run** (`--limit 256 --epochs 2`). The forked loader workers, validation,
  `metrics.jsonl`, `last.pt` / `best.pt` with provenance and the `step n / total` log all ran
  on the card. A real run should start from a committed tree: `load_model` refuses a checkpoint
  trained from a dirty one.

## 2026-10-25 — Three encoders behind one interface, and the stop head (spec v9)

- **One interface (user decision: keep the dense model, add flex and sparse).** The encoder is
  `ModelConfig.encoder` (`--encoder dense|flex|sparse`, default `dense`). Every encoder takes
  the same prepared inputs: the token inputs, z-hat and its defined flags. It returns:
  - the token vectors (B, N + 1, d);
  - the atom-pair vectors (B, A, A, d_p);
  - a dict of auxiliary losses, which the trainer adds to the loss.

  The decoder, the loss and the metrics are shared.
  - `dense` is spec v9 with its pair track.
  - `flex` and `sparse` keep no pair state. Their decoder input comes from `AtomPairHead`: a
    linear map of the static atom-pair channels, plus h_a * h_b and h_a + h_b, then GELU to
    d_p. The channels are RBF times the same-fragment flag, bonded, order, capped graph
    distance, Wiberg index and the same-fragment flag.
- **flex** (`model.flex`, FlexAttention). Each layer and head adds a learned bias, linear in
  the pair features, inside `score_mod`:
  - a radial function of the distance, within a fragment;
  - on atom pairs, tables over bond order and graph distance, and a Wiberg weight;
  - a same-fragment weight;
  - z on the (occupied, virtual) cells, projected per head. This (B, R, C, H) block is the
    only one materialized.

  Padding is a block mask.
  - *Tabulated radial function.* FlexAttention cannot index a tensor that requires gradients
    more than once in a `score_mod`, and the 32-term sum over the RBF weights did. So each
    head's sum_k w_hk RBF_k(d) is tabulated from the weights, 50 points per A up to 10 A, and
    interpolated linearly from two tensors (the lower and the upper neighbour of each bin). The
    table is built from the weights, so it stays differentiable.
  - *An integer grid constant.* FlexAttention is compiled with dynamic shapes, one compile for
    every batch shape. Under it, dynamo traced the float grid step 0.02 as a symbolic float, and
    Inductor failed to lower the `score_mod` (`LoweringException: unbacked_bindings`).
    - A term-by-term bisect showed that only the radial term failed. Every term passed with
      `dynamic=False`, but then every new batch shape would recompile.
    - The grid is now 50 bins per A, an int, and it compiles.
  - *GPU only.* FlexAttention has no backward on the CPU. `FlexBias.dense` builds the same bias
    as a (B, H, N + 1, N + 1) tensor for SDPA. It is a test oracle, not a training path (user:
    "use only gpu"). The test compares, on the card, the outputs and the gradient of every
    encoder parameter between the kernel and the reference.
- **sparse** (`model.sparse`, after DeepSeek Sparse Attention). It selects single tokens, not
  blocks, since the tokens are a set with no order.
  - An indexer scores every key: I_ij = sum_h w_ih ReLU(q_ih . k_j), 4 heads of width 32, on
    the token vectors detached. Each query attends to MECH, itself and its top-64 other real
    keys.
  - The pair bias is MLP_P of `inputs.pair_channels_at` on the selected cells only, projected
    per head. These are the dense track's input channels, computed per cell, so no N^2 tensor
    is built.
  - The indexer is trained only by KL(main attention, averaged over heads and detached ||
    indexer softmax over the same keys), returned as the auxiliary loss `indexer_kl`.
  - *Dense warm-up.* During the LR warm-up (the first `--warmup` steps) the attention runs over
    every key, with the bias on the full grid, and the indexer learns from it. Validation runs
    in the mode of the last training step. With k >= N the two paths agree (a test).
  - *Per-query copies of the small tables.* The first flex run took 7.6 s per batch, against
    0.9 s for dense; its backward took 1.2-3.1 s against a 0.1 s forward. The term-by-term
    timing on the card (6 batches with N >= 128) gave these backward times:

    | score_mod term | backward |
    |---|---|
    | none | 0.09 s |
    | radial table | 0.32 s |
    | order, distance, Wiberg and fragment tables, read directly | 2.28 s |
    | the same, one copy per (sample, head, query) | 0.30 s |
    | z | 0.12 s |

    - Read directly, every cell of a head adds its gradient to the same few addresses, and the
      atomic adds serialize.
    - The four small tables are now copied per (sample, head, query). That is
      B x H x N x 14 numbers, and the copies' backward is a plain sum.
    - The radial table gained little from copies (0.32 s to 0.26 s) and would cost
      B x H x N x 500 numbers, so it is read directly.
  - *Compiled variants.* Each of these compiles its own variant, 60-100 s on the card on a cold
    Inductor cache: a batch of one, fewer than 128 queries (FlexAttention's decoding kernel),
    and the forward without gradient used in validation.
    - That makes 8 variants, dynamo's default limit; past it, dynamo would fall back to the slow
      eager path. `compiled_flex` raises the limit to 16.
    - A probe run hit 5 variants.
- **Trainer wiring.** `train_epoch` adds the auxiliary losses and passes
  `dense_warmup = step < --warmup`. Validation reports the same metrics for every encoder.
- **Memory and speed per encoder** (default sizes, d = 256, L = 8, H = 8; float16, per-layer
  recomputation, the stop head included). Molecule split, `--cell-budget` 1.1 M (425 batches
  per epoch). The fullest batch is B = 24, N = 213. The time per batch is the mean of 12
  random batches after the compiles.

  | encoder | parameters | peak, fullest batch | s / batch | epoch |
  |---|---|---|---|---|
  | dense | 7.37 M | 3.14 GiB | 0.87 | ~6.2 min |
  | flex | 7.11 M | 0.59 GiB | 0.54 | ~3.8 min |
  | sparse, top-64 | 7.52 M | 1.23 GiB | 0.70 | ~5.0 min |
  | sparse, dense warm-up | 7.52 M | 1.34 GiB | 0.52 | ~3.7 min |

  - At PMechDB's sizes a query has a few hundred keys, so the top-64 selection saves memory,
    not time. Its gathers cost more than the dense product.
  - Flex adds the one-off compiles (above) to each run.
- **The stop head (user request relayed by the review session; spec, "Stop head").** It
  replaces "the stop head is not built" in the entry above.
  - P(END_MECH) = sigmoid(MLP(h_MECH)). The MLP is d to `decoder_hidden` to 1, in
    `EditDecoder`, so every encoder shares it.
  - L = L_step + lambda_stop BCE, masked per sample. The optional batch key `stop_label` (B,)
    is 1 when the mechanism ends after the step, 0 when it goes on, and -1 when unknown.
  - The masked samples' targets are clamped to 0 before the BCE. Otherwise a NaN target would
    leak a NaN gradient through the mask.
  - lambda_stop is `--stop-weight`. Below 0 (the default) the source decides: 0.5, or 0 on
    PMechDB, whose steps are isolated.
  - With no label in the batch, or lambda_stop = 0, the term is not added, so the loss equals
    L_step bit for bit (a test). A hand-computed case and a masked batch are tested too.
  - P(END_MECH) is returned in the loss parts (`p_stop`). Its validation mean is `val_p_stop`
    in `metrics.jsonl`.
  - No source has stop labels yet. The sample store and `collate_samples` get the field with
    the first source that has pathways.
- **Smoke runs on the card**, one per encoder (`--limit 256 --epochs 2 --warmup 5`, so the
  sparse encoder leaves its dense warm-up). Each ran training, validation, `metrics.jsonl`
  (with `val_p_stop`), `last.pt` / `best.pt` and the log, with lambda_stop resolved to 0.
  - Wall time was about 100 s each, mostly loading the samples; flex took 207 s with its
    compiles. Peak RSS was 6 GB.
  - The run directories were deleted.

## 2026-10-26 — The anchor NLL counts an anchor once, however many variants share it

- `anchor_nll` (and so `val_anchor_nll`) is -log P(any admissible anchor). `step_loss` took the
  logsumexp over the (variant, anchor) rows, so an anchor that several variants of the label
  class share was counted once per variant. Its probability could then add up past 1, and one
  logged step of the first flex run showed an anchor NLL of -0.045.
- The rows are now deduplicated on (sample, i, j, sign) for this metric only. The loss keeps one
  path per (variant, anchor), since each path is a different R. A hand-computed test with two
  variants on one anchor gives -log 1/5 (it gave -log 2/5 before).
- Nothing else read it: the top-k ranks take the best anchor, and `best.pt` is chosen by
  top-1, then by the loss. The run `flex_first` (pod, code 7b090cb) logged `val_anchor_nll` the
  old way, slightly low wherever variants share an anchor.

## 2026-10-26 — The first run: flex on the molecule split, on a RunPod RTX 3090 Ti

- **Run (user decision: flex first, on a pod).** `ksmech-train fit --run flex_first --encoder
  flex` with every default:
  - 30 epochs, cell budget 1.1 M, batches of at most 64;
  - AdamW at 3e-4 with weight decay 0.01, 2,000 warm-up steps, gradient clip 1;
  - bfloat16 on CUDA.

  The model has 7.11 M parameters. 9,994 training samples (3 skipped as TOO_LARGE) and 1,239
  validation samples make 12,750 steps.
  - Code 7b090cb, a clean clone on the pod, so the run's provenance is clean.
  - The run directory is copied back to `data/models/pmechdb/flex_first/`, with the pod's
    stdout as `stdout.log`; `load_model` reads `best.pt` here.
- **Speed.** The RTX 3090 Ti (24 GB, driver 580.65, the cu130 build) ran 27 min wall for the
  30 epochs.
  - About 48 s of training per epoch at about 220 samples/s, plus a few seconds of validation.
  - The first epoch took 111 s, of which about 60 s was the FlexAttention compiles.
  - GPU memory stayed at 3.3 GB.
  - On the RX 6600 XT the probe estimated 3.8 min per epoch, so this is about 4.7 times
    faster.
- **Validation per epoch** (anchor top-k over the candidate atom pairs with set
  marginalization; R exact = the greedy R is a variant of the label class). The physics floor
  is 15.2 / 25.8 / 30.8% (top-1 / 3 / 5).

  | epoch | top-1 | top-3 | top-5 | R exact | val loss | train loss |
  |---|---|---|---|---|---|---|
  | 1 | 27.8% | 47.9% | 57.9% | 14.7% | 7.93 | 126.66 |
  | 2 | 40.8% | 60.6% | 69.5% | 23.0% | 5.60 | 8.51 |
  | 3 | 44.1% | 64.5% | 73.7% | 28.2% | 5.23 | 5.92 |
  | 4 | 38.3% | 62.7% | 73.2% | 25.2% | 5.15 | 4.77 |
  | 5 | 46.5% | 70.7% | 79.3% | 36.2% | 3.82 | 3.92 |
  | 6 | 50.1% | 71.7% | 81.0% | 40.7% | 3.72 | 3.14 |
  | 7 | 50.3% | 72.5% | 81.4% | 41.5% | 3.37 | 2.59 |
  | 8 | 53.2% | 74.0% | 82.1% | 46.7% | 3.33 | 2.19 |
  | 9 | 54.8% | 77.8% | 86.4% | 44.6% | 2.85 | 1.89 |
  | 10 | 50.9% | 73.8% | 80.1% | 44.5% | 3.37 | 1.60 |
  | 11 | 57.1% | 78.0% | 84.1% | 49.6% | 2.74 | 1.45 |
  | 12 | 55.7% | 75.8% | 83.1% | 49.6% | 2.85 | 1.20 |
  | 13 | 57.0% | 78.2% | 87.2% | 49.2% | 2.87 | 1.11 |
  | 14 | 57.1% | 76.6% | 86.5% | 51.8% | 3.02 | 0.90 |
  | 15 | 55.3% | 77.6% | 88.5% | 49.9% | 2.81 | 0.81 |
  | 16 | 56.7% | 75.0% | 85.6% | 50.9% | 3.04 | 0.67 |
  | 17 | 56.4% | 80.5% | 87.3% | 50.6% | 2.97 | 0.61 |
  | 18 | 57.7% | 77.3% | 87.3% | 52.3% | 2.79 | 0.54 |
  | 19 | 57.0% | 80.7% | 87.2% | 51.7% | 2.99 | 0.43 |
  | 20 (best) | **60.9%** | **79.9%** | **87.3%** | **55.6%** | 2.93 | 0.40 |
  | 21 | 58.2% | 79.7% | 86.3% | 53.3% | 3.25 | 0.34 |
  | 22 | 58.4% | 79.3% | 86.4% | 52.9% | 3.30 | 0.31 |
  | 23 | 59.0% | 80.3% | 86.8% | 53.8% | 3.41 | 0.26 |
  | 24 | 58.8% | 78.5% | 86.4% | 53.8% | 3.53 | 0.23 |
  | 25 | 58.8% | 79.3% | 86.4% | 53.4% | 3.77 | 0.20 |
  | 26 | 59.2% | 79.6% | 86.0% | 53.8% | 3.89 | 0.19 |
  | 27 | 59.6% | 79.3% | 85.7% | 54.6% | 4.13 | 0.17 |
  | 28 | 58.6% | 79.0% | 85.5% | 53.4% | 4.29 | 0.16 |
  | 29 | 58.6% | 79.0% | 85.6% | 53.8% | 4.36 | 0.15 |
  | 30 | 58.7% | 78.8% | 85.7% | 53.8% | 4.37 | 0.14 |

  - The dip at epoch 4 is the end of the warm-up, when the LR reaches its peak.
  - The `val_anchor_nll` of this run uses the old definition (the entry above), so it reads
    slightly low; it is left out of the table.
- **It overfits from about epoch 11.**
  - Validation loss is lowest at epoch 11 (2.74) and climbs to 4.37 by epoch 30, while the
    training loss falls to 0.14.
  - Top-1 still rises to 60.9% at epoch 20, then stays at 58-59%.
  - `best.pt` is epoch 20 (by top-1, then loss).
  - The next levers, none decided yet: the resonance augmentation (user decision: after a first
    run), a shorter schedule or stopping on validation loss, and regularization. The model has
    no dropout.
  - The test split is not evaluated yet.
- **Pod lessons.**
  - *Check CUDA first.* Two RTX 3080 Ti pods, very likely the same host (same bus id), had
    CUDA broken at host level: `nvidia-smi` fine, `cuInit` 999, `/dev/nvidia-uvm` failing to
    open with EIO. That needs a host reboot, so the fix was another machine. On a new pod,
    open `/dev/nvidia-uvm` and call `cuInit` before installing anything.
  - *RunPod's proxy SSH* (`<pod>@ssh.runpod.io`) takes only PTY sessions. Commands work with
    `ssh -tt` and stdin; scp and rsync do not. base64 through the PTY moved 18 KiB/s. Data
    needs "SSH over exposed TCP".
  - *uv on a CUDA host.* `uv run` re-synced to the default `rocm` group and began
    reinstalling the ROCm build. `UV_NO_SYNC=1` after the CUDA sync (README, CLAUDE.md).
  - *Threads on a 112-core host.* The unit suite with `-n 32` reached a load of 120 (each
    worker's BLAS threads) and was stopped halfway. The run capped OMP, OpenBLAS and MKL at 8
    threads.
- **Transfer lessons.**
  - *What the pod needs.* A run needs 20,642 files, 3.2 GiB:
    - the views, split file, steps parquet and the 49 shards, 0.53 GB;
    - 10,293 token files, 2.75 GB;
    - 10,293 bond-index files, 16 MB.

    The loaders verify every shard's inputs, so the token files must travel although training
    reads only the shards.
  - *Not through GitHub's public repo.* `rhoMech` is public, so the data went to a private
    `rhoMech-data` repo, pushed with the account key and cloned on the pod with a read-only
    deploy key, removed from the pod after use. GitHub refuses a deploy key that is already an
    account key; "Hi user!" (account) against "Hi user/repo!" (deploy key) tells which.
  - *Rates.*
    - Uploads from here reached GitHub at 1.9 MB/s.
    - The route to the pod's datacenter (RTT 135 ms, about 0.7% retransmissions) collapsed
      each TCP stream to a congestion window of 2-3 packets. One rsync gave 0.8-1.4 MB/s,
      four 1.8 MB/s at first and 0.13 MB/s once the losses grew, sixteen 0.39 MB/s.
    - GitHub to the pod was just as lossy for one connection (0.3 MB/s). Eight parallel
      partial clones (`--filter=blob:none` and a sparse checkout per eighth of the missing
      files) reached 2.5 MB/s.
    - In the end the direct rsync supplied 1.5 GB, and GitHub the remaining 1.7 GB (one push
      of the missing tokens on top of the views and shards).
  - Left to clean up: the private `rhoMech-data` repo and the 3.2 GB staging repository
    `/home/jacopo/Documents/rhoMech-data-push.git`.

## 2026-10-26 — Resonance augmentation: each pool drawn in its resonance forms, the label recomputed (schema 16)

- **Why.** The formal charges of the atom tokens, the matrix B on which R is decoded and the
  pair feature `bond_order` belong to one drawing of the reactant pool, while the wavefunction
  does not. Without augmentation the model learns the dataset's choice of drawing (spec v9,
  "Resonance augmentation"; user decision 2026-10-25: after a first run).
- **Which forms (user decision).** A molecule is drawn in its resonance forms
  (`resonance_forms`: H frozen, odd structures dropped) with no more formal charges (the sum
  of |q|) than the molecule as drawn: an enolate's charge on C or O, a carboxylate's on either
  O, the positions of a benzyl or allyl ion. The charge-separated minor forms are never drawn.
  - On 600 random training steps (1,369 reactant molecules, 875 distinct), RDKit returns 50 or
    more structures for 76 molecules, nearly all charge-separated, so uniform draws over every
    form would mostly be zwitterions.
  - With the charge rule, 18% of the molecules have another form, mostly 2 or 3.
  - The drawn form had the fewest charges in all but one molecule.
- **What follows the drawing** (`arrows.drawings`, `MechanismSample.drawn(k)`):
  - the formal charge of every atom token;
  - the bond orders, hence the decoder masks `can_break` and `can_form`;
  - the lone pairs, hence `can_lose_pair`;
  - the pair feature `bond_order`;
  - the label class.

  The sigma bonds, H counts, Wiberg indices, graph distances and every orbital token are
  shared by all drawings.
- **`bond_order` is read on the drawn B, for the dataset's drawing too** (`lewis.pair_bond_orders`,
  in `reactant_sample`). It used to be read on the QM record's own Kekulé structure (RDKit's
  kekulization of the aromatic cache key), which differed from the drawn B in 94 of 398
  sampled steps. The textbook samples get the same rule.
- **The label of a drawing** (`delta_be.drawn_label`). It is the parsimonious class of the spec
  with the drawing's B in place of the drawn one, built as for the drawn step. The candidates
  are:
  - the product's resonance forms, enumerated once per step;
  - the drawn edit's product, which the drawing reaches with `matrix_between` (it plays the
    part of R_data and always passes);
  - the structures that the drawn label's members reach from the drawing. A drawing that
    differs far from the reaction centre then keeps the drawn class, even where the capped
    enumeration lacks the matching product structure.
- **Each form keeps the drawn Kekulé structure wherever it can** (`kekule_closest`).
  - RDKit's structure of a delocalized cation also swaps rings the charge never reaches: in
    three of the four forms of a 3-phenylbenzyl cation it swaps the phenyl, and in a trityl
    cation it swaps the two other rings.
  - The label then has to swap them back. On 398 sampled steps the label of a drawing grew by
    up to 24 entries.
  - A swap is an alternating cycle of ±1 pi-bond changes. Undoing it keeps every atom's bond
    total, charge and lone pairs, so the structure stays valid (`without_kekule_cycles`;
    bonds that form or break never take part, so a Diels-Alder ring is not undone).
  - After the fix a drawing's label differs from the drawn one by -4 to +14 entries. The
    largest are real: a cation drawn in the far ring of a naphthalene has to come back.
- **Not used on the label candidates.** Undoing Kekulé cycles in RDKit's product structures
  would change 22 of 1,000 dataset labels.
  - 8 gain an equal-cost member.
  - 14 are replaced by a smaller label, mostly an iminium (C=N+) written as its carbocation
    form (C+ next to a neutral N), which has as many formal charges and two entries fewer.
  - The label's cost (formal charges, then entries) cannot tell a sextet from an octet. Today
    those steps keep the iminium only because RDKit's carbocation structure carries a ring
    swap. That is left as it is, and it is an open question whether the cost should prefer full
    octets.
- **How many drawings.** Up to 8 per sample (`MAX_DRAWINGS`), the dataset's own included:
  every combination of one form per molecule when there are no more, else 7 random ones, the
  generator seeded by the step id, so a build is reproducible. A drawing whose label cannot
  be recomputed or fails `check_edit` is dropped and counted (`drawing_skips`, the
  `drawings` block of the reports).
- **Measured** on 398 random training steps (`sample_for_step` on the token cache):
  - 132 samples (33%) have other drawings, 274 in all.
  - Drawings per sample 1 / 2 / 3 / 4 / 5 / 6 / 8: 266 / 75 / 18 / 21 / 4 / 7 / 7.
  - None was dropped.
  - Label size, drawing minus drawn: 0 in 120 drawings, +2 in 66, +4 in 31, -2 in 30, +6 in 16,
    and 11 others between -4 and +14.
  - Building a sample took 0.38 s per step (p95 1.6 s, max 4.5 s).
- **Schema 16** (`SAMPLE_VERSION`). A sample stores its drawings as stacked int8 arrays
  (`drawing_charges`, `drawing_orders`, `drawing_lone_pairs`) with their labels in the JSON.
  The two fields kept only for schema 15, `pair_contact_sites` and `atom_coulomb`, are gone.
- **Training** (`--resonance-draws`, on by default).
  - Each epoch every training sample is drawn in a uniformly random one of its drawings, from
    the seed and the epoch.
  - The choice is made in the batch sampler, in the main process, so persistent DataLoader
    workers see each epoch's draw.
  - Validation and test always read the dataset's drawing.
- **Resonance pairs: later (user decision).** The 471 pairs need a consistency term of their
  own, since they have no step label. They wait until a run with the augmentation shows
  whether drawing invariance is still lacking.
- **Rebuild** (code d172272, 14 workers, 590 s, store 1.2 GB).
  - 12,418 samples from 12,464 training steps, 46 skipped as MISSING_TOKENS, as before.
  - The dataset's label classes are unchanged: 11,733 / 619 / 59 / 7.
  - 3,622 samples (29%) have other drawings, 7,864 in all, and none was dropped.
  - Drawings per sample 1 / 2 / 3 / 4 / 5 / 6 / 7 / 8: 8,796 / 1,932 / 636 / 474 / 136 / 201
    / 12 / 231.
  - In `molecule_train`, 3,111 of 9,997 samples have other drawings, 6,955 in all.
- **Downstream, re-run on the new samples.**
  - *Anchor floor:* unchanged, val 15.2 / 25.8 / 30.8% and test 17.1 / 27.2 / 31.5%. It reads
    the dataset's drawing and no bond order.
  - *LightGBM,* `--n-jobs 2` as before. The only input change is `bond_order` read on the
    drawn B.

    | split, part | top-1 / 3 / 5 before | after | trees before | after |
    |---|---|---|---|---|
    | molecule val | 44.1 / 60.5 / 69.6% | 44.1 / 62.0 / 70.8% | 467 | 638 |
    | molecule test | 46.6 / 67.0 / 73.8% | 48.8 / 67.7 / 75.1% | | |
    | scaffold val | 40.3 / 59.9 / 69.7% | 40.8 / 58.2 / 68.3% | 1,273 | 1,202 |
    | scaffold test | 45.3 / 64.7 / 75.6% | 45.1 / 66.1 / 75.2% | | |

    Loading the samples took 72 s instead of 300 s.
  - *Textbook* (expected pair in the top 3): untrained gap kernel 3 / 5; the molecule model
    2 / 2 on the cases it did not see; the textbook-held-out model 4 / 5.
  - The `predict` worktree's server checks its LightGBM model against these files, so it saw
    the model as stale between the sample build and the `gbdt` run.
- **Smoke runs** (`--limit 64 --epochs 2`, RX 6600 XT).
  - 17 of the 64 training samples have other drawings, 40 in all.
  - Dense: training loss 48.8 to 36.2, validation loss 39.8 to 24.4.
  - Flex in float16 overflows in its first steps (`grad inf` or `nan`, losses in the
    thousands, steps skipped by the scaler). It does the same without the draws: validation
    loss 516.96 after the first epoch in both, 19.46 against 19.51 after the second. So this
    is the float16 start of the flex encoder, not the augmentation. The pod trains in
    bfloat16.
- **To do.** Push the schema-16 views and shards to `rhoMech-data` before a pod run; then a
  full run with the augmentation.

## 2026-10-26 — A drawing's label must be an elementary step (the first augmented run crashed)

- **What happened.** `flex_resonance` on a RunPod RTX 5080 crashed early in its second epoch
  with a CUDA device-side assert: an index out of range in the gather of `_entry_sums` (the
  loss). The first epoch had ended with val top-1 31.2%.
- **Why.** The label of a resonance drawing can change a bond order by 2: a drawing with a
  triple bond that the product needs single (pmechdb-023f2470184b934b: one bond -2). The
  decoder writes {-1, 0, +1} per pair and {-2, 0, +2} per atom, so `r_pairs + 1` indexed
  outside the three classes.
  - `check_edit` checked conservation and ranges, not this.
  - 5 of the 7,864 drawings of the schema-16 store have such a class (2 in `molecule_train`).
    None of the dataset's own labels has.
  - The first epoch drew neither of the two; the second drew one.
- **Fix.**
  - A class holds only elementary steps (`ReactionMatrix.is_elementary`: every bond order
    changes by 1, every atom's non-bonding electrons by 2), at the cheapest level that has one.
    R_data is always accepted only when it is elementary. A class left empty is
    NOT_ELEMENTARY: the drawing is dropped and counted, or the step is skipped.
  - `check_edit` refuses a non-elementary variant (NOT_ELEMENTARY), so no sample or drawing
    reaches the decoder with one.
  - The trainer drops such drawings from a store built before this
    (`drop_undecodable_drawings`), logs each one, and counts them in the run's `skipped`. So
    the schema-16 store, here and on `rhoMech-data`, needs no rebuild. The next rebuild drops
    them at the source.

## 2026-10-26 — QMugs for encoder pretraining: GFN2-xTB single points at its own geometries

- **Why.** The spec names QMugs for encoder pretraining. The tokens are built from the GFN2-xTB
  wavefunction (S, C, the orbital energies, the N+-1 charges for the Fukui features), and no
  public dataset found ships GFN2 matrices. QMugs has GFN2-optimized geometries, GFN2 and DFT
  properties, and DFT wavefunctions.
  - The survey: QMugs, nablaDFT, QCML, OMol25, SPICE, GEOM, PubChemQC, QM7-X, Transition1x and
    Halo8.
  - User decision: download QMugs and run single points at its geometries, not new
    optimizations.
- **Data.** Downloaded `summary.csv` and `structures.tar.gz` (9.2 GB, sha256 in
  `data/raw/qmugs/SHA256SUMS`). The 7 TB of wavefunctions were not downloaded. The layout and
  the SD tags are in `docs/data_schema.md` ("QMugs").
  - 664,303 molecules, 20,350 with a charge in their SMILES.
  - Median 30 heavy atoms, at most 100.
- **Method** (`QMConfig(given_geometry=True)`, `qm.given`, `ksmech-qm qmugs`).
  - The QMugs geometry is taken as it is; a bond of the Lewis structure stretched past 1.3
    times the covalent radii is CONNECTIVITY_CHANGED.
  - Then the same single point in ALPB water as the PMechDB records, the N+-1 Fukui and %Vbur.
  - The method string is `GFN2-xTB/ALPB-water/given`, so these records never share a key with
    optimized ones. They have their own cache, `data/cache/qmugs/qm`.
  - The molfile's atoms are put in the record order of the canonical SMILES with the
    reactants' `record_correspondence`.
  - QMugs's energies are gas-phase (CHEMBL1: -117.4268 Ha, against -117.4473 Ha in ALPB water
    here). Its geometries are therefore gas-phase minima, not ALPB ones; the electronic
    structure is in water like PMechDB's.
- **Pilot** (500 conformers, 14 workers, from an uncommitted tree, deleted afterwards).
  - 500 / 500 ok, no warnings.
  - 3.9 s per conformer on one thread (median; p95 9.8 s, max 39 s), so 2.94 conformers/s
    with 14 workers.
  - 564 KB per record.
- **Scope (user decision).** Conformer 0 of the first 100,000 molecules in archive order
  (ChEMBL ids sorted as strings): about 56 GB and 9.5 h here. Kept local.
  - All 664k would be about 375 GB as records are now, or about 186 GB without `H_core` (read
    by nothing downstream) and `fock` (rebuilt from S, C and the orbital energies). 93 GB were
    free here.
  - GitHub takes 2 GB per push and wants repositories under 5 GB, so later only compact
    pretraining samples, or a subset, can go to `rhoMech-data`.
- **Code.**
  - `batch.execute` runs any worker on the spawned pool (the QMugs driver reuses it).
  - Two new reason codes: SDF_UNREADABLE and RECORD_ORDER.
- **Not settled yet.** The pretraining objective (the spec does not define one). The QMugs
  path for tokens and bond indices: `ksmech-orbitals build` and `ksmech-qm bond-index` read
  the molecules of a steps parquet and the main cache.

## 2026-10-26 — Label classes closed under the pool's symmetry; a tie no longer counts as a hit (schema 17)

- **Why (the analysis of the ~60% top-1 plateau).** Per-sample predictions of
  `flex_resonance/best.pt` (val top-1 63.4% in training) and `flex_first/best.pt` on the 1,239
  molecule-split val samples:
  - In 267 val samples (21.5%) the label names one of several atoms that are equivalent by
    symmetry: one H of H3O+, of a CH3 or of an NH3+, an atom of either of two identical molecules.
    A dataset step names its atoms by atom maps; the model reads the pool. The equivalence was
    checked with real automorphisms of the pool graph (96 of 99 rank-based calls confirmed; the
    other 3 hit the enumeration cap).
  - The model learns the symmetry: it gives each of the three H of H3O+ P = 0.34. So top-1 on
    those samples is a coin toss: 18% in float32, against 71% on the other samples. In 99 of the
    505 wrong val answers the top-1 is an equivalent atom. Counted up to symmetry, top-1 is 67.2%.
    The loss also keeps a floor of log k on such samples, in training as well.
  - `anchor_ranks` counted only strictly higher candidates. In bfloat16 equivalent atoms tie
    exactly, so 5.1% of val samples (63, all but one of them symmetric) were hits only through a
    tie: the same `best.pt` scores 63.4% in bfloat16 and 59.2% in float32, and 67.3% / 67.2% up
    to symmetry. The training metric, and the choice of `best.pt`, depended on the precision.
  - Not a cause: identical reactant pools with different answers. One answer per pool would still
    allow 99.1% of val.
  - What remains (406 val errors, 33%) is mostly near misses and generalization:
    - 68% share an atom with the true anchor, 17% have the right molecules and other atoms, 15%
      one wrong molecule;
    - top-1 is 80% on non-symmetric pools of one molecule, 71% of two, about 50% of three or
      four;
    - train top-1 86% (top-5 99.5%) against val 59%;
    - 65% of val pools have Tanimoto < 0.4 to the nearest training pool, and top-1 there is 56%
      against 64-79% on closer pools.
- **The closure** (`arrows.symmetry.symmetric_closure`, in `steps.edit_target` for the dataset's
  label and in `drawings.drawing_of` for each drawing's).
  - Every automorphism of the pool's Lewis structure on the token atoms (elements, formal
    charges, bond orders, every H an atom; no stereo) maps each variant to an equivalent one, and
    the images join the class with their anchors.
  - The structure is the drawing's: a Kekulé ring keeps only the symmetry its alternation has
    (protonating Kekulé benzene: 6 attacks, not 12), and a localized charge breaks what it breaks
    in B, as for the label.
  - The automorphism group is not enumerated. Every H with a single heavy neighbour is folded
    into it; `placements` enumerates the distinct images of the label's atoms by backtracking
    within RDKit's symmetry classes (`CanonicalRankAtoms` without tie breaking), each completed
    once to a full automorphism. The folded H's then go to every choice of distinct H's on the
    image of their atom. Enumerating whole groups had capped 9 of 398 pools (TBS groups,
    phenyls) and taken 1.2 s a step; placements cap none and take 0.6 s a step (0.38 s
    before the closure).
  - Limits: 1,000 placements, 200,000 candidate checks, 32 members a class; a closure cut short
    is flagged `symmetry_capped`. `EditLabel` gains `n_symmetric` (members added) and
    `symmetry_capped`. The loss already marginalizes over the class, so it needs no change.
- **The metric** (`model.loss.anchor_ranks`, `eval.floor.floor_ranks`). The rank counts the
  candidates outside the label that score at least as high, so a tie with such a candidate is a
  miss. With symmetric classes, equivalent anchors are all in the label and their ties cannot
  matter.
- **Measured on 400 random training steps.**
  - 92 of 398 classes (23%) gain members, 165 in all.
  - Class sizes after the closure 1 / 2 / 3 / 4 / 6 / 8 / 9: 281 / 73 / 26 / 12 / 4 / 1 / 1.
  - No closure was cut short.
  - Drawings' classes: 1 / 2 / 3 / 4 / 6: 209 / 51 / 6 / 5 / 2.
- **Schema 17** (`SAMPLE_VERSION`): the classes change, so a schema-16 store is refused. Any
  checkpoint's val metrics must be recomputed on schema 17 to compare.
- **Rebuild** (code 81aeaff, 14 workers, 660 s).
  - 12,418 samples, 46 skipped as MISSING_TOKENS, as before.
  - 3,132 classes (25.2%) gained members, 4,968 in all, and no closure was cut short. On val,
    280 of 1,239.
  - Class sizes 1 / 2 / 3 / 4+: 8,737 / 2,514 / 682 / 485 (before: 11,733 / 619 / 59 / 7).
    26 classes have 9 or more members.
  - The 5 drawings whose label is not elementary are now dropped at the source
    (NOT_ELEMENTARY).
- **`flex_first/best.pt` on the schema-17 val, without retraining.**

  | labels | top-1 | top-3 | top-5 | R exact |
  |---|---|---|---|---|
  | schema 16, float32 | 55.9% | 79.3% | 87.2% | 51.8% |
  | schema 17, float32 | 63.9% | 81.3% | 88.0% | 59.1% |
  | schema 17, float16 | 63.9% | 81.3% | 88.0% | 59.1% |

  float32 and float16 agree on every sample. In float16, 40 hits have tied admissible anchors,
  which are equivalent and all in the label.
- **Downstream, re-run on the new samples.**
  - *Anchor floor:* val 15.6 / 26.0 / 31.1% (was 15.2 / 25.8 / 30.8%), test 17.8 / 27.3 / 31.6%
    (was 17.1 / 27.2 / 31.5%). Its |G^atom| now also gets credit for an equivalent atom.
  - *LightGBM:* the same results and trees on both splits. It ranks orbital pairs on the
    ArrowLabels, which the closure does not touch; only its inputs, the views, changed.
  - *Textbook:* unchanged (3 / 5, 2 / 2, 4 / 5).
- **To do.**
  - Push the schema-17 views and shards to `rhoMech-data` before a pod run. The runs queued on
    the pod read schema 16, so their numbers compare only with schema-16 evaluations.
  - A batch's label tensors are (B, V, A, A) and V now reaches 32: watch memory in the first
    training run.
