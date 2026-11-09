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
