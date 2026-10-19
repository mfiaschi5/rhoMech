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
