+++
title = "The Klopman–Salem pair features"
summary = "The gap factor G, the cross-centre coupling, and the geometry-free pair features."
tags = ["physics", "orbitals"]
order = 30
+++
# The Klopman–Salem pair features

## The Klopman–Salem equation

When two molecules approach, perturbation theory splits their interaction energy into three
terms:

1. **filled–filled repulsion**: always repulsive, it grows with the overlap of filled orbitals.
   This is the quantitative content of steric hindrance.
2. **Coulomb attraction or repulsion** between atomic charges. This is *charge control*, which
   wins for hard reagents.
3. **filled–empty stabilization**: for each filled orbital *r* on one side and empty orbital
   *s* on the other, a coupling squared over the energy gap `E_r − E_s`. This is *orbital
   control*, which wins for soft reagents. Restricted to the HOMO and the LUMO, it is Fukui's
   frontier-orbital theory.

Term 3 is pairwise in exactly the objects a curved arrow names: one filled orbital and one
empty one. That is why it is the backbone of the model. The donor and acceptor orbitals are the
tokens of [[orbital-tokens|atom and orbital tokens]].

## The gap factor G

With localized orbitals, term 3 of a pair splits into a coupling times a factor that depends
only on the two orbitals' canonical spectra:

```
G(r, s) = Σ_k Σ_m  2 · p(r, k) · p(s, m) / (ε_k − ε_m)
```

Here *k* runs over the occupied canonical orbitals, *m* over the virtual ones, `p(r, k)` is the
weight of canonical orbital *k* in localized orbital *r*, and ε are the canonical energies.

- G is negative.
- A larger |G| means a smaller effective gap.
- The dominant canonical pair (*k*, *m*) in the sum is the physical explanation of a predicted
  bond.

G is defined for every occupied–virtual pair, between molecules and within one. The model gets
G and log|G|.

## Why no contact geometry

The coupling factor of term 3 depends on how two orbitals overlap at contact, in the pose the
reaction adopts. No calculation on the separate reactants gives that pose. The expansion is in
small overlaps, so its repulsion and its stabilization are both of second order in the overlap,
and their ratio has no minimum along the approach. The contact distance, and the strain of
reaching it, lie outside the equation.

ChemRho therefore computes **no contact geometry**. It keeps the factor that is intrinsic to
the two orbitals, G, and leaves the approach to be learned from the token descriptors.

## Inside one molecule: the cross-centre coupling

Within one molecule the reactant pose is a geometry the two orbitals share. The
Wolfsberg–Helmholz approximation gives the resonance integral between two atomic orbitals μ
and ν:

```
β(μ, ν) = K · S(μ, ν) · (F(μ, μ) + F(ν, ν)) / 2,    K = 1.75
```

The **cross-centre coupling** is `|l_rᵀ β l_s|`, with β restricted to the blocks between the
centres of *r* and those of *s*. It measures pre-organization: between bonded centres it
carries the antiperiplanar requirement of negative hyperconjugation. Its sign is an arbitrary
orbital phase, so only the magnitude is used.

Two more features come from the same pose:

- the **centroid distance** of the two orbitals: how far a chain must fold;
- the **stereoelectronic alignment**:
  - the |cos| between the two lobe axes;
  - when the orbitals sit on adjacent bonds, the cosine of the dihedral. A value of −1 is
    antiperiplanar, as eliminations and migrations require.

## The pair feature vector

For an ordered (donor, acceptor) pair the model gets:

| Group | Features |
|---|---|
| every pair | G, log\|G\|, whether the two orbitals are on different molecules |
| on the Lewis structure | bonded flag, bond order, Wiberg bond index, graph distance (capped at 6) |
| within one molecule | centroid distance, cross-centre coupling, the two alignment cosines |

Undefined entries are zero and masked. Every entry is standardized with training-set
statistics.

These features initialize the transformer's **pair track**, which every attention head reads
as a learned bias. All heads are learned.

## From orbital pairs to atom pairs

The decoder works on atom pairs (see [[reaction-matrix|the reaction matrix]]). The pooled
gap factor of an atom pair (*i*, *j*) sums G over the valid orbital pairs, in both directions
of donation, weighted by the orbitals' centre weights on *i* and *j*.

Ranked by |G| alone, with no learned parameter, these atom pairs give the **physics floor**,
the baseline every trained model must beat. The orbital pair with the largest term is the
orbital reading of a forming bond.
