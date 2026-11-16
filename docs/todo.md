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
  reranker (I5) is `ksmech-eval roundtrip` and the bidirectional search (I1) `ksmech-network
  search --backward-run`; left: the round-trip evaluation (QM of the beam's child species, the
  lambda grid on both val sets) and the bidirectional search against A* on the ReactMech chains;
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

## A model that holds out of distribution (2026-11-15)

**Why.** A model trained on ReactMech ranks PMechDB's anchors barely above the untrained floor,
and the measurements of `docs/decisions.md`, 2026-11-15 (the `ksmech-eval ood` reference
matrix) say why: the ranking reads the Lewis-structure channels and the element, not the
orbital-pair physics (permuting G moves the top anchor of about one sample in ten), it depends on
the order the reactants are written in, and nothing in the architecture confines a candidate's
score to the molecules that react. The user's go-ahead of 2026-11-15 ("procedi con il piano")
covers the plan below as proposed; the pair locality of phase 2 was put to the user as the one
choice hard to undo and is confirmed again before its build.

**What.**

- Phase 1, the cheap arms on the model as it is (switches of 2026-11-15, each with `--probe` for
  the out-of-distribution curve of every epoch): the control; `--no-fragment-embedding`; that and
  `--element-dropout 0.5`; that and `--element-dropout 1`; then the joint PMechDB + ReactMech
  run with `--balance-power 0.5`. They say how much data and regularization recover alone.
  Open: on which held-out set `best.pt` is chosen (a set that is neither the training
  distribution nor the test: a ReactMech class left out, `loco`), and all recorded products of a
  pool as positives of the anchor loss.
  Found on the way (decisions.md, 2026-11-16): every fp16 run starts with a loss of thousands of
  nats, a sum over every entry of an untrained head, and its loss scale falls from 2^10 to about
  1 in fifteen steps and stays near there. An arm of its own: the entry heads initialized
  towards "no change", so the first loss is tens of nats and the scale holds.
- Phase 2, the local selector: every molecule encoded alone (order invariance by construction,
  encodings cached per species); the anchor logit a sum over the (occupied, virtual) orbital
  pairs of the two atoms, initialized at the |G^atom| floor, so no atom-only path exists; an
  intermolecular candidate scored from its two molecules only, the pool entering through the
  softmax; the site selector's inputs physical and size-consistent (no fragment index, E_Z and
  the formal charge dropped or masked, the charge term q_i q_j of Klopman-Salem as a feature);
  the completion of R on the Lewis structure as now. Ablations: a low-capacity context channel,
  E_Z, the canonical frontier inputs.
- Phase 3, after phase 2: QM pretraining (QMugs), and, if the completion still limits the decode,
  R generated as an alternating path, valid by construction.

**Done when.** Phase 1: the arms' `ood_report` files and their comparison are in
`docs/decisions.md`, with the choice of what phase 2 starts from. Phase 2: its
out-of-distribution top-1 is above the best phase-1 arm on the same sets (or the thesis that the
localized orbitals transfer is recorded as not holding). Risks, to measure and not to assume:
PMechDB in distribution may fall (part of its 66% is the source's conventions), and a
molecule-pair-local score has no third-body effect.
