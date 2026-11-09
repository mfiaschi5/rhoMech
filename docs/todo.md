# To do

Open work, newest last. Each item says why, what done looks like, and where it came from. An item
leaves this file when it is done (its outcome goes to `docs/decisions.md`).

## Do the orbital token columns carry anything the model needs? (2026-11-09)

**Why.** The permutation importance of pm_v19_s0 (`ksmech-eval importance`, molecule val, 1,241
samples, 3 repeats; `data/models/pmechdb/pm_v19_s0/token_importance_molecule_val.json`) found the
model relying on the atom tokens only: element (dNLL +4.57, top-1 -43.6%), formal charge (+1.98,
-30.2%), the Fukui block (+0.79, -18.3%), the Mulliken charge (+0.71, -13.9%), %Vbur (+0.37,
-5.3%), the frontier weights (+0.19, -4.3%). Every orbital column, and the spectral set, moves
the anchor NLL by less than 0.03 and top-1 by at most 0.5%. The likely reason: the orbital
information already reaches the model through the pair features (G, g2, the cross-centre
coupling) and through the embedding each orbital inherits from its centre atoms, so the orbital
columns are redundant. Permutation cannot tell redundant from useless: a retrained model can
shift the work onto another input, so only a retrained ablation settles it.

**What.**
- A `ModelConfig` switch that removes the orbital token columns at batch time (like
  `drop_pair_features`: the orbital MLP input zeroed and masked, or a `drop_token_features` by
  name), with the spectral set as its own switch; tests on the fixture molecules.
- Three arms on the molecule split, `closed_base` val, 3 seeds each, general preset: (a) as
  pm_v19_s0; (b) no orbital columns; (c) no orbital columns and no spectral set. Optionally (d) no
  inherited atom embeddings, to see whether the orbital token then needs its own columns.
- Decode the arms and pair them with `ksmech-eval compare` (A fails when its mean top-k is below
  B's by more than max(2 x pooled SD, 1 point)).

**Done when.** The comparison is in `docs/decisions.md`: either the orbital columns stay
(an arm without them is worse beyond the tolerance), or they are redundant with the pair track
and the inherited atom embeddings, and the spec's orbital token is revisited.

## Reaction networks: towards a product, without one, and from a product alone (2026-11-09)

**Why.** The same reactants take several correct elementary steps, in reality and not only in a
dataset (methoxide + acetone: addition to the carbonyl and deprotonation), and the outcome comes
from the network of steps, not from one step. PMechDB records 474 pools with more than one
product (973 steps, 7.7%, v18); on the 36 such samples of closed_base val the general runs hit
their own label at top-1 on 44-50% and some recorded outcome on 72-83%. Today the network lives
in the server (`serve/predict/network.py`) and grows by a beam that knows no target; it has no
stop (END_MECH is not read, and no source has stop labels) and no reverse steps; QM records hold
no free energy. The full plan, with the reasons and the code facts, is
`docs/reaction_network_plan.md` (proposed 2026-10-29, nothing implemented).

**Decisions taken.**
- D1: maximum likelihood stays the loss for chemically distinct outcomes. A loss over any
  outcome only for the sets a law defines (resonance, symmetry, the order of independent steps);
  a hybrid behind a flag only if dilution shows up.
- D2: the stop is thermodynamic, and acts on states, never by cutting an uphill step.
- D3: packages are allowed (the xtb program, CREST, an atom mapper, scipy.sparse, a building-block
  catalogue).

**What**, in this order (letters are the plan's items):
1. A. Multi-reference evaluation: a second rank `rank_any` in the hits files, by product pool
   key; `ksmech-eval compare --metric own|any`.
2. G and B in parallel. G: pathway data (the ReactMech ID_CRM chains, the 350 human pathways,
   stop labels). B: free energy of a pool (xtb `--hess`, quasi-RRHO, a sidecar under
   `data/cache/thermo/`, the solvent as a parameter, CREST for the species a network reaches).
3. C: the network core moves into the engine, with log p of a given R. D: search towards a
   product (arrival on the identity key or on the skeleton up to protons, the atom map giving the
   change still to make), with I1 as A* and I2 the k best pathways.
4. H: the backward model (-R from the child pool, its own store, 3-seed training), then the
   round-trip check as a reranker of the forward beam.
5. I1 bidirectional, meeting on the pool key.
6. E: prediction without a product (reverse edges, detailed balance, the Markov chain,
   exploration by probability mass, reservoir species). F: the thermodynamic stop (SINK_AT_T,
   NO_NET_REACTION, the inversion temperature T*). Then I3 (transition path theory) and I4.
7. J: branches that join, on the species graph (AND-OR): shortest and k shortest hyperpaths,
   atom-flux analysis, checked against the pool graph on the same pathways.
8. K: the previous step from a product alone: a by-product reservoir counted from the data, AND-OR
   search (Retro*, proof-number search or MCTS) with the backward model, stopping on building
   blocks or species stable at T, every result verified forward.
9. Evaluation (plan section 5) and calibration of the flags; L: the spec's "Inference", D1-D3 in
   `docs/decisions.md`, tests of the laws (detailed balance, flux conservation, A* and hyperpaths
   against exhaustive search on small graphs).

**Done when.** Each item meets its measure in the plan's section 5: with a product, the arrival
rate and the recorded pathway's rank within a node budget on the human benchmark and the
ReactMech chains; without a product, top-k final pool against the recorded outcomes with
calibrated probabilities; the stop with about zero false sinks on recorded reactions; from a
product alone, top-k recovery of the recorded reactants and the share verified forward. Each
item's outcome goes to `docs/decisions.md` as it lands; the plan's section 6 (stop head,
target-conditioned policy, value function, START head) needs training and comes after.
