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
