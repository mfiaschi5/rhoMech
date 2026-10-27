+++
title = "The reaction matrix"
summary = "B, R = ΔB, elementary steps, anchors, and resonance modulo the sigma key."
tags = ["representation", "steps"]
order = 10
+++
# The reaction matrix

## The bond–electron matrix B

A Lewis structure of *n* atoms is a symmetric integer matrix **B** (Dugundji and Ugi):

- off the diagonal, `B[i][j]` is the formal bond order between atoms *i* and *j*;
- on the diagonal, `B[i][i]` is the number of non-bonding valence electrons on atom *i*.

The formal charge of atom *i* follows from it: its valence electrons, minus `B[i][i]`, minus
the bond orders of its bonds. No separate rule for charges is needed.

## A step is R = ΔB

A reaction is the **reaction matrix** `R = B(product) − B(reactant)`. An elementary step in
ChemRho is one such matrix with small entries:

- bond orders change by **±1** on atom pairs;
- lone pairs change by **±2** electrons on atoms.

The product is `B + R`. Mass is conserved by construction, and charge conservation is a single
linear constraint on R. Each curved arrow is the smallest such edit:

| Arrow | Entries of R |
|---|---|
| lone pair on *i* → atom *j* | `R[i][i] = −2`, `R[i][j] = R[j][i] = +1` |
| bond (*i*, *k*) → atom *j*, the new bond at *i* | `R[i][k] = −1`, `R[i][j] = +1` |
| bond (*i*, *k*) → lone pair on *k* (a leaving group) | `R[i][k] = −1`, `R[k][k] = +2` |

A step is the sum of its arrows, applied at once.

## Why R and not the arrows

The model predicts R itself, for three reasons:

1. **One R, several arrow readings.** The decomposition of R into arrows is not unique. Some
   ambiguities are physical. An ionic bond and a lone pair of its atom can be the same
   localized orbital, and a relay can be drawn in more than one way.
2. **R is what the step does.** Its product is `B + R`, so mass is conserved by construction
   and charge by one linear check.
3. **R is sparse.** Most entries of every step are zero.

Arrows stay the chemist's language. They are read back from R after the prediction, together
with the orbital pairs behind each forming bond (see
[[klopman-salem|the Klopman–Salem pair features]]).

## Anchors and the step decoder

Every step raises at least one bond order or, in a heterolysis, lowers one. The decoder first
points at an **anchor**:

- a `+1` entry: a new bond, or a π bond on a pair that is already bonded;
- a `−1` entry, in a step that forms nothing.

Given the anchor, it classifies every other entry: −1, 0 or +1 on atom pairs, and −2, 0 or +2
on atoms. Masks rule out impossible values. A completion that breaks charge conservation or an
atom's valence is discarded. A mechanism is a sequence of steps: each intermediate `B + R` is
featurized again (3D embedding, xTB, localization, new tokens) and fed back in, until a stop
head says the mechanism is complete.

## Resonance and the sigma key

Resonance forms are different **B** matrices of one molecule, so two matrices R can describe
the same step. ChemRho compares structures with the **sigma key**. It is the canonical SMILES
of the σ skeleton, with:

- every bond single, and no aromaticity or stereochemistry;
- every atom's H count frozen;
- every formal charge and unpaired electron set to zero;
- the total charge appended.

Two closed-shell structures with one sigma key differ only in where their π bonds, lone pairs
and charges sit. They are resonance forms of one molecule. The key needs no enumeration, so it
is exact.

The sigma key decides what a step is:

- **A resonance interconversion.** The two sides share the sigma key with the atom maps kept,
  so no atom changes its σ bonds or H. This is not a step to predict. It is kept as a
  *resonance pair*: two drawings of one pool that should get the same prediction.
- **A degenerate exchange.** The same molecules are on both sides, but atoms change partners.
  It is a real step.
- **A label.** Among the matrices R that reach a resonance form of the product, the label is
  the *parsimonious* class: the fewest formal charges in the product, then the fewest changed
  entries.

In training, the reactants are drawn as random valid resonance forms and the labels are
recomputed for each drawing. The model learns the step, not the drawing.
