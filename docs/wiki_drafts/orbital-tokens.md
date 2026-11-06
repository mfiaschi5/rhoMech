+++
title = "Atom and orbital tokens"
summary = "The two token families the model reads, and their continuous descriptors."
tags = ["representation", "orbitals"]
order = 20
+++
# Atom and orbital tokens

A molecule enters the model as one sequence that holds two families of tokens: **atom tokens**
and **localized-orbital tokens**. Both come from a GFN2-xTB calculation of the molecule. The
model sees no absolute coordinates: geometry enters only through distances.

## Localized orbitals

The canonical molecular orbitals are Pipek–Mezey localized in two blocks, the occupied and the
virtual, so a filled orbital never mixes with an empty one. The singly occupied orbitals of a
radical form a third block. The localization starts deterministically from the canonical
orbitals, so one QM record always gives the same tokens.

## What an orbital token carries

An orbital token carries continuous physical descriptors only. They are its whole input to the
model:

- its **occupation**: 2, 0, or 1 for a singly occupied orbital;
- its **centres**: how many atoms carry at least 15% of its population, and the four largest
  centre weights;
- the **s / p / d shares** of the orbital on its four leading centres;
- for the two leading centres:
  - the |cos| between each centre's p vector and the axis between them, so σ versus π
    character is a number;
  - the signed overlap population, positive for bonding and negative for antibonding;
  - the distance between them;
- its **radius of gyration**;
- its **spectral set**: the canonical orbitals it is made of, as (energy, weight) pairs. Pairs
  with a weight of at least 0.01 are kept, at most eight, renormalized, and read by a set
  encoder.

An orbital also inherits its atoms' embeddings through its centre weights.

## What an atom token carries

- the atomic number and the formal charge;
- the Mulliken charge;
- the frozen-orbital Fukui functions f⁺ and f⁻, and the finite-difference dual descriptor;
- the frontier weights for HOMO−1, HOMO, LUMO and LUMO+1, with near-degenerate levels (within
  0.05 eV) summed;
- the buried volume %Vbur;
- the chirality sign, only where the SMILES specifies the stereocentre.

## No chemist's taxonomy as input

Labels such as lone pair, σ, π, σ\*, π\*, empty p, hybridization, aromaticity or
functional-group ids are computed and stored. They are never model inputs. They serve the
tests, the diagnostics and the interface.

The reason: orbital character must generalize across elements. σ\*(C–Cl), σ\*(C–Br) and
σ\*(C–I) should look to the model like the same kind of acceptor at different energies. A
nitrogen lone pair and a carbanion should look like the same kind of donor. A label would hide
that; the descriptors expose it.

The labels do not decide which interactions are allowed either. That decision uses the
descriptors:

- a **donor** is a doubly occupied token;
- an **acceptor** is an empty token, other than a d-type polarization function and other than
  the antibonding partner of the donor's own bond.

A token with a fractional occupation (between 0.2 and 1.8) is neither, with one exception: the
open-shell tokens of a radical.

## Radicals: open-shell tokens and their holes

In a radical computed as an open shell, the singly occupied orbitals (one SOMO at occupation 1,
or a degenerate set such as the three 4p orbitals of a bromine atom at 5/3 each) are
**open-shell tokens**. A Fermi-smeared level of a small-gap molecule is not one.

- An open-shell token is a **donor**: it can give one electron.
- Each open-shell token gets a twin at sample assembly, its **hole**: a virtual token with the
  same coefficients, centres, descriptors and spectrum. The hole is an **acceptor**: it can
  take one electron into the vacancy.
- Inside one molecule an open-shell token and a hole never pair (that would be an excitation
  inside the open shell). Between two radicals the same pair is a recombination.
- Token files never change: the hole exists only in the sample.
- An open-shell token on a metal d shell, as in Cu(II) or Fe(III), takes no role by default.

Two inputs describe the open shell, and a model chooses which to read:

- a **spin role** on each orbital: +1 for an open-shell token in its donor role, −1 for a hole,
  0 otherwise;
- the **SOMO population** of each atom, how much of the unpaired electron density sits there.
  It sums to the number of unpaired electrons (1 for Br·, 2 for O₂), is 0 for a closed shell,
  and does not depend on which resonance structure is drawn.

How pairs of these tokens interact is the subject of
[[klopman-salem|the Klopman–Salem pair features]].
