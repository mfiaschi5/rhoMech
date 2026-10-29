# Reaction networks: searching towards a product and without one, several valid outcomes, a thermodynamic stop

Status: proposed 2026-10-29; nothing implemented. To start after P0b of `docs/open_shell_plan.md`.
Nothing here changes the loss or the training presets, so the P0b comparison (GENERAL against
LEGACY) stays paired. Written on branch `claude/reaction-network-plan` from main f6aa758: the
open-shell plan, `eval/paired.py`, the closed_base cohort and the v18 store it cites are on
`claude/open-shell-p0b` until that branch is merged.

## Contents

1. The problem
2. Decisions taken
3. What the code and data hold today
4. Work items
5. Evaluation
6. Later (needs training)
7. Order
8. Not doing

## 1. The problem

The same reactants can take several correct elementary steps, in reality and not only in a
dataset. Methoxide and acetone add to the carbonyl (the hemiketal alkoxide, fast and reversible)
and deprotonate the alpha carbon (the enolate). Which one matters is decided by what follows:
with excess acetone the enolate leads to the aldol, while the hemiketal reverts. It also depends
on the solvent and the temperature: in water the deprotonation is uphill by about 4 pKa units
(acetone 19.3, methanol 15.5), in DMSO it is downhill (26.5 against 29.0). The aldol addition of
acetone lies on the acetone side at room temperature, and heating drives the dehydration
(water released, dS > 0) towards mesityl oxide.

So the answer for one step is a distribution over steps, and the outcome comes from the network
of steps, not from one step. Two questions have to be kept apart:

- explain a reaction (reactants and product known): the pathway is nearly unique, the one that
  reaches the product;
- predict what happens (reactants and conditions only): the answer is a branching network with a
  probability on each branch and an equilibrium over its end states.

Measured on the v18 stores (map-free canonical reactants, products compared map-free):

- PMechDB has 474 reactant pools recorded with more than one product: 973 steps, 7.7% of the
  steps, up to 6 outcomes per pool. Ethyl acetate + ethoxide is recorded both as the enolate
  (deprotonation) and as the tetrahedral adduct (addition); beta-keto esters are deprotonated at
  either alpha position. The ReactMech pilot has none (each pool occurs once).
- closed_base val holds 36 samples in such pools. The general runs (3 seeds) hit their own label
  at top-1 on 44-50% of them and some recorded outcome of the pool on 72-83%. Over the whole val
  part that is about +1 point (65.6 -> about 66.6).

The visible penalty is small. The larger problem cannot be seen in the data: for most pools the
dataset records one channel (the one leading to the observed product), and the other valid
channel counts as a negative.

## 2. Decisions taken

- **D1. Maximum likelihood stays the training objective for distinct outcomes.** Each recorded
  outcome is a sample, so the model learns the split between branches (the annotators' split,
  which the network then corrects with thermodynamics, item F). A loss that accepts any member of
  the outcome set, -log sum_{y in Y} p(y), is kept only for sets defined by a law, where the split
  means nothing: resonance drawings and symmetry-equivalent edits (both already in the label
  class, `arrows/symmetry.py`), and the order of independent steps.
  - For "any": no conflicting gradients; no dilution of the valid mass over k outcomes (one
    wrong candidate can beat each of them at top-1); annotation noise ignored; a smaller beam
    finds a valid outcome.
  - Against "any": the split inside Y becomes arbitrary (seed and initialization), so outcome
    probabilities and p(pathway | reaches target) lose their meaning; the model may collapse on
    one channel, which the network then never explores (methoxide + acetone collapsed on the
    addition never reaches the aldol, and detailed balance cannot restore an edge the model does
    not propose); it generalizes "at least one" instead of "both"; more variance between seeds;
    a store rebuild to express sibling outcomes on each sample's atoms; and a gain of about one
    point that the multi-reference metric (item A) already recovers without training.
  - If dilution proves to matter (top-1 "any" far below top-5 "any" on multi-outcome pools), the
    candidate is a hybrid behind a flag whose default an ablation sets:
    L = -(1 - beta) log sum_{y in Y} p(y) - beta (1/|Y|) sum_{y in Y} log p(y).
- **D2. The stop is thermodynamic, not kinetic.** Thermodynamics decides where the system ends,
  not which steps occur, so it acts on states, never by cutting an uphill step (item F).
  Kinetics would only bound how far the search climbs; without it that bound is a search flag.
- **D3. Packages are allowed where they are the better tool** (user, 2026-10-29: the early
  "prefer plain numpy" instruction is superseded). Here: the xtb program for thermochemistry,
  CREST for conformers, an atom mapper (RXNMapper or similar), scipy.sparse.

## 3. What the code and data hold today

- **The network exists in the server** (`serve/predict/network.py`, pure: no file I/O, no RDKit).
  Nodes are pool states merged by pool key (the multiset of sigma keys, so up to resonance);
  edges are elementary steps with the predictor's log p. `NetParams`: beam 5, `max_steps` 8,
  `max_nodes` 200, turnover cycles detected, pathways ranked by summed log weights. Terminal
  reasons: NO_CANDIDATES, NO_APPLICABLE_PROPOSAL, TURNOVER, MAX_STEPS, NODE_BUDGET.
  `NetParams.temperature` is the predictor's softmax temperature, not a physical one.
- **No target, no stop, no reverse steps.** The network has no goal. END_MECH is not read
  (`serve/predict/network_predictor.py`), and no source has stop labels, so the stop head has
  never been trained, on ReactMech either (`val_p_stop` is an untrained output). Only the
  `network` predictor gives probabilities; floor and gbdt give scores.
- **The engine never imports `ksmech.serve`**, so a headless evaluation on benchmarks cannot use
  the server's network as it stands.
- **QM records** hold the GFN2 energy (`energy_ha`) with ALPB water for every calculation and one
  conformer. There is no Hessian, so no free energy, and no other solvent.
- **No conditions.** `conditions` is null in PMechDB and in the ReactMech pilot. The PMechRP human
  benchmark (`data/raw/pmechrp/pathway_datasets/human_benchmark/pathways/350_HumanBenchmark.csv`,
  350 pathways from Clayden, 2-7 steps, mean 3.0) gives a temperature on 168 rows ("n/a" on the
  others), the final product and each step's target; SMILES are unmapped and there are no arrows.
- **No ReactMech pathways ingested.** `pathway_id` and `step_index` are null; the step chains are
  in the ID_CRM mechanism files, unused (`docs/data_schema.md`).
- **Scoring a given matrix**: `model/decode.py` has `exact_logp`, but no entry point that starts
  from a reaction matrix.

## 4. Work items

### A. Multi-reference evaluation (small)

- `eval/decode.py`, `eval/cli.py`: for each sample, the set of products the source records for
  the same reactant pool; the hits files get a second rank, `rank_any`. Compared by product pool
  key, not by matrix, since sibling samples number their atoms differently.
- `eval/paired.py`: `--metric own|any`.
- Siblings come from the whole source; the report counts how many come from another part.

### B. Free energy of a pool (medium, plus a QM batch of hours)

- Per-molecule thermochemistry from the xtb program (`--hess`, quasi-RRHO), several temperatures
  in one run, with the frequencies stored so G(T) is evaluated at any T afterwards. A sidecar
  beside the records, like the Wiberg indices: `data/cache/thermo/`, provenance headers, a
  `ksmech-qm thermo` command run in the background over the cached molecules, rows in the SQLite
  index.
- The solvent becomes a parameter. ALPB covers methanol, DMSO, THF and others; the method string
  is already in the cache key, so another solvent writes new records beside the water ones.
- Conformers: CREST ensembles give the lowest conformer and the conformational entropy, which a
  single conformer misses. Hours over thousands of molecules, so only for the species a network
  reaches, not the whole cache.
- G(pool) = sum of the molecules' G + RT ln(c_i / c0) per species. The standard-state convention
  is fixed once and tested. Concentrations and reservoir species (solvent, reagents in excess)
  are parameters, since the data have none.

### C. The network core leaves the server (medium)

- Move the pure core of `serve/predict/network.py` into the engine (for instance
  `ksmech.network`); the server imports it. Update `tests/test_serve_layers.py` and
  `docs/serve_architecture.md`.
- An engine entry point for "log p of a given matrix R at a state", needed to score reverse
  edges.

### D. Search towards a product (medium)

- `NetParams`: a target (SMILES) and the mode.
- Arrival test on two levels: the exact sigma key, or the heavy-atom skeleton without H and
  charges (up to proton transfers, since datasets often draw the worked-up neutral product).
  Terminal reason TARGET_REACHED with its level.
- Guidance from the start: the atom map reactants -> target (an atom mapper) gives
  dB = B_target - B_current; best-first search with cost -log p plus a weight times the remaining
  dB (the weight a flag). Detours (protonate to activate, then deprotonate) cost but are allowed;
  this is bookkeeping, not a chemical rule. Where mapping fails, a map-free similarity to the
  target, counted under MAPPING_FAILED.
- Best-first replaces the layered beam, which prunes unlikely but productive branches.
- Output: the pathways that reach the target with p(pathway | reaches) = p(pathway) /
  sum over reaching pathways, i.e. the alternative mechanisms of one transformation with their
  relative weights.

### E. Prediction without a product (medium)

- Reverse edges by microscopic reversibility: each edge R implies -R, scored by the model at the
  child state.
- Detailed balance: the transition weight is s_ij exp(-dG_ij / 2RT), with s_ij symmetric from the
  model (the geometric mean of p forward and p back). The stationary distribution is then
  Boltzmann over the pools: the model shapes the pathways, QM fixes where they end.
- The explored graph as a Markov chain: absorption probabilities by a sparse linear solve
  (scipy.sparse). Absorbing states: sinks at T (item F), turnover, no outgoing step.
- Exploration by probability mass instead of a beam: expand the unexpanded node most likely to be
  reached, stop when the unexplored mass is below a flagged tolerance, and report that mass as
  the uncertainty. The enolate, unlikely but the only way to the aldol, is no longer pruned.
- Reservoir species: always available, never consumed, with their chemical potential in the G
  balance. Without them a pool runs out of reactants after one bimolecular step.
- Network predictor only: floor and gbdt scores are not probabilities.

### F. Thermodynamic stop (small, after B and E)

- A physical temperature in kelvin in `NetParams`, under a name distinct from `temperature`.
- Every edge records dG(T) and K(T) = exp(-dG/RT); no edge is cut for being uphill, since an
  uphill step happens and reverts (the enolate in water).
- Sink: when every proposed step from a pool climbs more than
  delta(T) = RT ln(1/epsilon) + margin, the node is terminal with reason SINK_AT_T
  (epsilon = 1%: 2.7 kcal/mol at 25 C, 3.5 at 110 C). Before declaring it, the search rescans
  with more candidates and reports how many it examined: "every proposed step climbs" is not
  "no step descends".
- Outcome NO_NET_REACTION at T: the starting pool is a sink, or the reactants hold more than
  1 - epsilon of the equilibrium population over the explored pools.
- Inversion temperature T* = dH / dS for edges and outcomes ("favoured above T*" or "below T*"):
  associations (aldol addition, Diels-Alder) lose with heat, releases of a small molecule
  (H2O, CO2, N2) gain.
- The cumulative climb allowed during exploration is a flag: a state 15 kcal/mol up has a
  population near 1e-11 but may lead to a deeper sink, and thermodynamics does not say how far
  to look.
- epsilon and the margin are flags. GFN2 errs by several kcal/mol on reaction energies (more for
  charged species and acid-base equilibria), RRHO overstates the entropy cost of association in
  solution, so the margin will exceed delta itself. It is calibrated so that no recorded
  reaction comes out unfavourable at its temperature (false sinks about zero), except reactions
  driven by precipitation, gas evolution or work-up, which the pool cannot see.

### G. Data for pathway evaluation (medium)

- The human benchmark as an evaluation-only set (QM and tokens for its species), after checking
  its molecule overlap with PMechDB training: it comes from the same release.
- The ReactMech ID_CRM chains: `pathway_id`, `step_index`, and stop labels (the last step of a
  chain is 1), which also make the stop head trainable.
- ORD_USPTO (133 transformations) and mech-USPTO-31K (never trained on) as final tests.
- Every new file inspected first and documented in `docs/data_schema.md`.

### H. Documents and tests

- The spec (`docs/ksmech_architecture.tex`), "Inference": the target mode, the Markov chain and
  the thermodynamic stop. Edited only while no build runs (tracked edits make artifacts stale).
- `docs/decisions.md`: D1-D3 and every later choice, with dates.
- Tests: thermochemistry on the fixed molecule set and on the OPEN_SHELL list; network, chain,
  reverse edges and sinks with the server's synthetic chemistry; detailed balance as a law test
  (the stationary distribution equals Boltzmann).

## 5. Evaluation

- With a product: arrival rate (exact and up to protons), step targets found, rank of the
  recorded pathway compared as the set of steps plus the final pool (spec v9, "Inference").
- Without a product: top-k final pool against the recorded outcomes (multi-reference),
  calibration of the outcome probabilities.
- Stop: false sinks on recorded reactions (about zero); true sinks on a small set of negative
  controls (pools known not to react at a given temperature), to be built, since the datasets
  hold only reactions that happened.

## 6. Later (needs training)

- The stop head, trained on the labels of item G.
- A target-conditioned policy: dB as a pair-track channel, trained on the ReactMech and FlowER
  chains, with every later state of a pathway used as a target.
- Conditions as inputs, once a source has them.
- Kinetics (barriers), if the climb bound needs it.

## 7. Order

1. A: small and useful at once.
2. G and B in parallel: data work, and the thermochemistry batch runs for hours.
3. C, then D, then E, then F.
4. Evaluation (section 5) and calibration of the flags.
5. Section 6, with new training runs.

## 8. Not doing

- A loss over any valid outcome for chemically distinct outcomes (D1).
- A kinetic stop (D2).
- Cutting edges because they are uphill.
