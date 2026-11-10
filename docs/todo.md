# To do

Open work, newest last. Each item says why, what done looks like, and where it came from. An item
leaves this file when it is done (its outcome goes to `docs/decisions.md`).

## Which orbital token columns to drop, if any (2026-11-10)

**Why.** The orbital-token ablation (`docs/decisions.md`, 2026-11-10) kept the 25 orbital
columns for now, the user's call: on PMechDB closed_base val no arm without them is worse, but on
ReactMech val the arms without them lose 1-1.7 points (McNemar p 0.003-0.06), and the full model
reads the s/p/d shares and the centre weights there (12 of 12 block-seed cases raise the NLL)
while it ignores them on PMechDB. The energy, two-centre and s-share blocks showed no use on
either set. The user decides later which columns, if any, leave the orbital token.

**What.** An arm with only the chosen columns removed (`--drop-token-features NAME...`), 3 seeds,
judged on ReactMech val as well as PMechDB closed_base val, by the tolerance rule and by McNemar
(the rule alone cannot see an effect below about 3 points).

**Done when.** The user's choice and its comparison are in `docs/decisions.md`.

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

**Status (2026-11-10).** The code of every item is on main (merged 2026-11-10), with its tests
and `docs/decisions.md` entries: A (`rank_any`, `compare --metric any`), B
(`ksmech-qm thermo`), C, D/I1/I2 (`ksmech-network search`, `evaluate`), E/F/I3/I4 (`ksmech-network
predict`, `--free-energy qm`), G (`ksmech-data pathways`, `human-benchmark`), H (`--direction
backward`, pilot only), J (`ksmech-network species`), K (`ksmech-network retro`), L (spec v10.1,
approved 2026-11-10). What is left is compute, data and calibration:
- the thermochemistry of the species a network reaches (`ksmech-qm thermo`; the whole cache is
  about 208 CPU-h) and the default for imaginary modes of loose minima;
- QM and tokens for the ReactMech chain steps outside the pilot (17,595), so the benchmarks of
  section 5 can run (the human benchmark's species: done 2026-11-10, 1,056 of 1,072, 19 tombstones);
- the backward model is trained (3 seeds, 2026-11-10, `docs/decisions.md`); the round-trip
  reranker (I5) is `ksmech-eval roundtrip`, its evaluation (QM of the beam's child species, the
  lambda grid on both val sets) and I1 bidirectional come next;
- the calibration of the flags (`step_cost`, epsilon and margin, the climb budget, the mass
  tolerance) and the measures of "Done when".

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
