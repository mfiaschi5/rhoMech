# Reaction networks: searching towards a product and without one, several valid outcomes, a thermodynamic stop, the previous step, graph search, joining branches, the search from a product alone

Status: proposed 2026-10-29, extended the same day with the previous-step model (item H), the
graph search (item I), joining branches on a species graph (item J) and the search from a
product alone (item K). Listed in `docs/todo.md` on 2026-11-09, when it came to main.
Implemented on 2026-11-09 on the branch `claude/network` (code and tests; every choice in
`docs/decisions.md` under that date): A, B, C, D with I1 and I2, E, F, G, H (code and a pilot),
I3, I4, J, K, and the spec's draft section (L, v10.1, "The reaction network", not yet approved).
What waits is compute and data, not code: the thermochemistry batch over the species a network
reaches, the ReactMech chains ingested in full with their QM, QM for the human benchmark's
species, the backward store and its 3-seed training, the calibration of the flags (section 5),
and section 6. Section 3 describes the code before the implementation. Nothing here changes the
forward model's loss or the training presets. The measurements of section 1 were taken on the
v18 stores (2026-10-29); main now holds Release B (v19), the open-shell plan, `eval/paired.py`
and the closed_base cohort.

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
  `max_nodes` 200, turnover cycles detected. The graph grows by a layered beam that does not
  know any target; `rank_pathways` then lists simple paths best-first by summed log weights
  with a priority queue. Terminal reasons: NO_CANDIDATES, NO_APPLICABLE_PROPOSAL, TURNOVER,
  MAX_STEPS, NODE_BUDGET. `NetParams.temperature` is the predictor's softmax temperature, not a
  physical one.
- **The graph is implicit and expanding a node costs seconds**: xTB for new molecules (cached by
  canonical SMILES), tokens, and the model (0.6-0.8 s per sample on CPU in the P0b decodes). The
  measure of a search algorithm is the number of nodes it expands.
- **No target, no stop, no reverse steps.** The network has no goal. END_MECH is not read
  (`serve/predict/network_predictor.py`), and no source has stop labels, so the stop head has
  never been trained, on ReactMech either (`val_p_stop` is an untrained output). Only the
  `network` predictor gives probabilities; floor and gbdt give scores.
- **The engine never imports `ksmech.serve`**, so a headless evaluation on benchmarks cannot use
  the server's network as it stands.
- **QM records** hold the GFN2 energy (`energy_ha`) with ALPB water for every calculation and one
  conformer. There is no Hessian, so no free energy, and no other solvent. The batch computes the
  fragments of both sides of every step (`qm/batch.collect_molecules` reads `reactants_smiles`
  and `products_smiles`), so product species have records.
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
- Arrival test on two levels: the exact identity key (`data.resonance.identity_key`: the sigma
  key for a closed shell, with the unpaired counts for an open shell), or the heavy-atom
  skeleton without H and charges (up to proton transfers, since datasets often draw the
  worked-up neutral product).
  Terminal reason TARGET_REACHED with its level.
- The atom map reactants -> target (an atom mapper) gives dB = B_target - B_current, the change
  still to make. Detours (protonate to activate, then deprotonate) cost but are allowed; this is
  bookkeeping, not a chemical rule. Where mapping fails, a map-free similarity to the target,
  counted under MAPPING_FAILED.
- The search itself is item I1 (A* with the remaining dB, then bidirectional with item H); it
  replaces the layered beam, which prunes unlikely but productive branches.
- Output: the pathways that reach the target with p(pathway | reaches) = p(pathway) /
  sum over reaching pathways, i.e. the alternative mechanisms of one transformation with their
  relative weights (items I2 and I3).

### E. Prediction without a product (medium)

- Reverse edges by microscopic reversibility: each edge R implies -R, scored by the model at the
  child state.
- Detailed balance: the transition weight is s_ij exp(-dG_ij / 2RT), with s_ij symmetric from the
  model (the geometric mean of p forward and p back). The stationary distribution is then
  Boltzmann over the pools: the model shapes the pathways, QM fixes where they end.
- The explored graph as a Markov chain: absorption probabilities by a sparse linear solve
  (scipy.sparse). Absorbing states: sinks at T (item F), turnover, no outgoing step.
- Exploration by probability mass instead of a beam (item I4): expand the unexpanded node most
  likely to be reached, stop when the unexplored mass is below a flagged tolerance, and report
  that mass as the uncertainty. The enolate, unlikely but the only way to the aldol, is no longer
  pruned.
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

### H. The previous step (backward model; medium, with training)

- **The object is the same.** If B_child = B_parent + R, the previous step seen from the child is
  -R, an elementary step by microscopic reversibility, on the same atoms and under the same laws.
  Its anchor is a bond that forms in -R, i.e. one the forward step broke; when the forward step
  broke none (a pure addition), -R forms none and its -1 entries are the anchors, as the loss
  already allows.
- **A separate model, not the forward model run on -R.** The forward model scoring -R at the
  child gives the probability that the child reverts (the reversibility used for the reverse
  edges of item E), not where it came from. At equilibrium the two coincide (detailed balance);
  the datasets are trajectories directed from reactants to products, and the backward model
  learns exactly that direction. Example: from a tetrahedral intermediate the forward model says
  what it does next (expel the leaving group or revert), the backward model says where it came
  from (the reverse of the addition). Going back is more ambiguous (an alkoxide can come from a
  deprotonation, an addition to a carbonyl or an epoxide opening), so its top-1 will be lower
  than the forward one.
- **Data.** QM already covers the product species (section 3); token coverage of every product
  species is checked first. Labels: the edit-label build with the sides swapped, the class of -R
  on the product pool with its resonance drawings and the symmetry closure. A separate backward
  store; the molecule split already groups the molecules of both sides, so no leakage.
- **Model.** The same architecture, first as a separate model (a direction option of
  `ksmech-train fit`), 3 seeds; then one encoder with the direction as an input, so every
  recorded step gives two examples, if the paired comparison says it helps.
- **Decode and evaluation.** The same decoder, with B_parent = B_child - R checked by the same
  laws (octet, charges); backward top-k on the molecule split, compared with the P0b tools.
- **The full pool.** Going back starts from the child's whole pool, the product with its
  by-products (Cl-, water, salts), or the atoms do not balance. Dataset pools are complete. With
  the reactants known, the atom balance reactants - product gives the by-products' formula, the
  atom map says which atoms they hold, and their protonation states are enumerated at fixed total
  charge. From a product alone, without reactants, it becomes mechanistic retrosynthesis:
  item K.
- **Uses**, in order of value: the bidirectional search (I1); the round-trip check, a forward
  step trusted more when the backward model, from the child, finds the parent again (I5); more
  signal for a shared encoder; completing a mechanism with a missing step, from a known
  intermediate (the step targets of the human benchmark).

### I. Graph search over the network (medium)

The network is an implicit graph that the models generate on demand: nodes are pool states,
edges elementary steps. In the pool graph a step that joins two branches (A -> B, C -> D,
B + D -> E) is still one simple path, because the pool holds every molecule, but every
interleaving of the two branches is a path of its own; item J keeps such branches apart on a
species graph. On the pool graph each question gets its algorithm:

- **I1. One pathway from reactants to product.** A* with edge cost -log p and the heuristic
  "remaining dB divided by the most entries one step can change", a lower bound on the steps
  left, so A* stays exact. Then bidirectional search: forward from the reactants with the
  forward model, backward from the full product pool with the backward model (item H), meeting
  on the pool key. With b candidates per state and d steps the cost falls from about b^d to
  about 2 b^(d/2) (b = 20, d = 6: about 16,000 expansions instead of 64 million), and a step
  that is unlikely but needed (the enolate on the way to the aldol) only has to appear among the
  candidates of one state instead of surviving the beam for several layers. A pathway found from
  two sides is rescored with one criterion (I5) before it is ranked.
- **I2. Several alternative mechanisms.** The k best pathways: Yen and Eppstein on the explored
  graph, K* (lazy, A*-driven) on the implicit one. Two pathways that differ only in the order of
  independent steps (two proton transfers) are one mechanism: the dependency DAG of spec v9
  defines the classes, and the enumeration yields classes, not permutations (partial-order
  reduction). Output: k distinct mechanisms with p(pathway | reaches).
- **I3. All pathways, each with its weight.** Transition path theory on the Markov chain of item
  E: the forward committor q+(x), the probability of reaching the product before returning to
  the reactants, by a sparse linear solve; the reactive flux on each edge; its decomposition
  into dominant pathways by repeatedly extracting the largest-bottleneck path, each with its
  share of the flux. The committor of each intermediate also says how productive it is. This is
  the probabilistic form of I1: the ensemble of mechanisms, weighted, instead of one pathway.
- **I4. What forms, without a product.** Best-first by probability mass (item E), deterministic
  and with the unexplored mass as its error bound. Option behind a flag: Monte Carlo tree search,
  rollouts with the forward model, the end states valued by their thermodynamic stability (item
  F) or by END_MECH once trained, UCT keeping improbable branches explored; for deep networks
  where the mass expansion is too slow.
- **I5. How the models enter the search.**
  - Edge cost -log p_forward(R | parent) - lambda log p_backward(-R | child), lambda a flag whose
    default an ablation sets (the two models normalize differently). Disagreement between them
    marks the steps to review in curation and to weigh less in the network.
  - Thermodynamics as a constraint (sinks at T, the climb budget of item F), never as an
    additive cost, which would mix two units.
  - Heuristic towards the target: the remaining dB (exact A*), or a learned value function
    (faster, not exact; section 6).
  - Cost: frontier nodes expanded in batches in one model pass on the GPU, QM in parallel
    workers over the existing cache; budgets (nodes, depth) are flags and every cut is counted.

### J. Branches that join: the species graph and AND-OR search (medium to large)

- **Why a second graph.** When two branches made independently meet (A -> ... -> B,
  C -> ... -> D, then B + D -> E), the pool graph holds one path per interleaving of the two
  branches, so its states multiply with the number of independent branches. The species graph
  keeps them apart:
  - OR nodes are species: a species can come from any of the steps that make it;
  - AND nodes are elementary steps: a step needs all of its reactant species, and a step
    between species of different branches is the join;
  - a pathway becomes a hyperpath: the set of steps that makes the target from the starting
    species, a derivation DAG whose joins are AND nodes with several inputs.
- **Same models, two graphs.** The model's input is already a pool of one or two species (plus
  the reservoir), i.e. one AND node; QM and G are per species (cache, item B). The pool graph
  stays for one-pot mechanisms where the counter-ions and proton shuttles of one pool matter;
  the species graph serves mixtures, independent branches and the product-only search of item K.
  A hyperpath linearized in any order the DAG allows is a pool-graph pathway, and the partial-
  order reduction of I2 maps it back, so results compare across the two.
- **Expansion without a product.** Species by species: every new species reacts with itself, the
  core species and the reservoir; it joins the core when the probability mass reaching it
  exceeds a flagged tolerance, with partners at fixed concentration (pseudo-first order), as in
  I4. Once rates exist (section 6) the same loop runs on flux from a microkinetic simulation,
  as RMG does.
- **One algorithm per question, as in item I:**
  - one hyperpath to the target: the shortest-B-hyperpath procedure of Gallo, Longo, Pallottino
    and Nguyen (a Dijkstra for additive costs, a step costing -log p), or AO* with an admissible
    heuristic when the target is mapped;
  - several alternative mechanisms: the k shortest hyperpaths of Nielsen, Andersen and
    Pretolani, classes taken up to the order of independent steps;
  - all hyperpaths with their weights: the model's branching ratios with partners at fixed
    concentration, and atom-flux analysis. Every step carries its atom map, so the flux of each
    target atom traces back through the network as an ordinary graph; the dominant pathways are
    extracted per atom and merged. Hyperpath weights are not products of step probabilities
    once bimolecular steps compete for partners, so these weights are the fixed-concentration
    approximation until a microkinetic simulation with rates replaces them; the end state
    always comes from the thermodynamic equilibrium of item F;
  - what forms without a product: the expansion above, with the end species and their shares.

### K. The previous step from a product alone (mechanistic retrosynthesis; large)

- **Start:** the target molecule only; the reactants and the by-products are unknown.
- **By-products from a reservoir.** Going back, a step that released a by-product forward
  (C -> D + X) needs X to join D. The reservoir is the set of small species that leave pools in
  the recorded steps (leaving groups, ions, water, solvent), counted from the datasets, not a
  hand list. At each backward expansion the backward model (item H) scores the current pool
  alone and the pools current + x for the most frequent x (how many: a flag); a step on an
  enlarged pool counts only if it uses x. A step that splits a species going back (an addition
  forward) needs no by-product.
- **AND-OR search on the species graph (item J).** A backward step that splits C into A + B
  makes two subgoals, each solved on its own (AND); a species with several possible origins is
  an OR node. The remaining dB is not available (no map to unknown reactants), so the search is
  Retro* (best-first on the AND-OR tree with a learned value), proof-number search, or Monte
  Carlo tree search; budgets are flags and every cut is counted.
- **Where to stop.** A leaf is a plausible starting material:
  - a building-block catalogue (eMolecules, ZINC or similar: a new data dependency);
  - a species that is thermodynamically stable on its own at T (item F: a sink by itself);
  - later, a START head trained on the first pools of pathways (the ReactMech chains and the
    human benchmark reactants, item G).
  The first two from the start, the third in section 6.
- **Forward verification.** From the reactants found, the forward search (I1, I3) with
  thermodynamics must reach the target; only verified retro-mechanisms are reported, ranked by
  the forward criterion.
- **Scope:** one reaction (one pot), up to a flagged number of elementary steps (the human
  benchmark goes to 7). Several pots in sequence would chain this with a reaction-level
  retrosynthesis tool (AiZynthFinder or similar): section 6.

### L. Documents and tests

- The spec (`docs/ksmech_architecture.tex`), "Inference": the target mode, the Markov chain, the
  thermodynamic stop, the backward model, the search algorithms, the species graph and the
  product-only search. Edited only while no build runs (tracked edits make artifacts stale).
- `docs/decisions.md`: D1-D3 and every later choice, with dates.
- Tests: thermochemistry on the fixed molecule set and on the OPEN_SHELL list; backward labels on
  the fixed set (applying -R to the product pool gives the reactant pool back); network, chain,
  reverse edges and sinks with the server's synthetic chemistry; laws as tests: detailed balance
  (the stationary distribution equals Boltzmann), flux conservation in transition path theory
  and in the atom-flux analysis, A* with the dB heuristic and the shortest-B-hyperpath procedure
  returning the same best result as exhaustive search on small synthetic graphs, the k-pathway
  classes invariant under permuting independent steps, a hyperpath linearized and reduced back
  giving the same class, and the reservoir counted from the data, never listed by hand.

## 5. Evaluation

- With a product: arrival rate within a node budget (exact and up to protons), step targets
  found, rank of the recorded pathway compared as the set of steps plus the final pool (spec v9,
  "Inference"), nodes expanded. A*, bidirectional search and the k-pathway classes are compared
  on the same budgets.
- Without a product: top-k final pool against the recorded outcomes (multi-reference),
  calibration of the outcome probabilities.
- All pathways: the share of the reactive flux (I3) on the recorded pathway's class.
- Backward model: backward top-k on the molecule split (3 seeds, paired); the round-trip check
  as a reranker of the forward beam, forward top-1 on val with and without it.
- Species graph: on reactions whose pathways have independent branches, the nodes expanded by
  the species graph against the pool graph for the same pathway found, and the same pathway
  class from both.
- From a product alone: given only the product (human benchmark, ReactMech chains,
  mech-USPTO-31K), top-k recovery of the recorded reactants (map-free) and of the mechanism
  class, the share of results that pass the forward verification, nodes expanded.
- Stop: false sinks on recorded reactions (about zero); true sinks on a small set of negative
  controls (pools known not to react at a given temperature), to be built, since the datasets
  hold only reactions that happened.
- Benchmarks: the 350 human pathways with their step targets, the ReactMech chains once
  ingested (item G), mech-USPTO-31K.

## 6. Later (needs training)

- The stop head, trained on the labels of item G.
- A target-conditioned policy: dB as a pair-track channel, trained on the ReactMech and FlowER
  chains, with every later state of a pathway used as a target.
- A learned value function for the search (I5), without pathway data: solve the reactant ->
  product pairs of USPTO (millions, no mechanism; those of mech-USPTO-31K excluded, it stays the
  test) with the search, train the value function on the pathways found, then the step models,
  and repeat (expert iteration). The risk is the model reinforcing its own errors, so the found
  pathways are filtered by thermodynamics and by the round-trip check, and measured only on
  benchmarks the loop never used.
- The START head of item K, trained on the first pools of pathways (item G).
- Several pots in sequence: item K chained with a reaction-level retrosynthesis tool.
- Conditions as inputs, once a source has them.
- Kinetics (barriers), if the climb bound needs it; with rates, the species graph of item J is
  expanded and weighted by microkinetic flux.

## 7. Order

1. A: small and useful at once.
2. G and B in parallel: data work, and the thermochemistry batch runs for hours.
3. C, then D with I1 as A* on the forward model, and I2: no new training.
4. H: backward labels and store, 3-seed training, backward top-k, then the round-trip reranker.
5. I1 bidirectional.
6. E and F, then I3 and I4 on the Markov chain.
7. J: the species graph, its expansion and the hyperpath algorithms, checked against the pool
   graph on the same pathways.
8. K: the reservoir counted from the data, the AND-OR search with the backward model, the
   stopping rules and the forward verification.
9. Evaluation (section 5) and calibration of the flags.
10. Section 6, with new training runs.

## 8. Not doing

- A loss over any valid outcome for chemically distinct outcomes (D1).
- A kinetic stop (D2).
- Cutting edges because they are uphill.
- A by-product list written by hand: the reservoir of item K is counted from the data.
