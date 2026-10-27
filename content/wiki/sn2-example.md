+++
title = "Worked example: an SN2 step"
summary = "Methoxide and chloromethane: the reaction matrix, its arrows and its orbital reading."
tags = ["example", "steps"]
order = 40
+++
# Worked example: an SN2 step

Methoxide attacks chloromethane: CH₃O⁻ + CH₃Cl → CH₃OCH₃ + Cl⁻.

```scene
scenes/sn2-step.json
```

The figure is a step computed by ChemRho from the two SMILES `C[O-]` and `CCl`, with GFN2-xTB
and the physics floor predictor (no trained model). It is the floor's first-ranked step. It is
not taken from any reaction database.

## The matrices

Take the three heavy atoms in the order (O, C, Cl), where C is the carbon of chloromethane.
Hydrogens are left out for brevity, and so is the O–CH₃ bond of methoxide, which does not
change.

Reactant **B**:

| | O | C | Cl |
|---|---|---|---|
| **O** | 6 | 0 | 0 |
| **C** | 0 | 0 | 1 |
| **Cl** | 0 | 1 | 6 |

Product **B**:

| | O | C | Cl |
|---|---|---|---|
| **O** | 4 | 1 | 0 |
| **C** | 1 | 0 | 0 |
| **Cl** | 0 | 0 | 8 |

The step **R = ΔB**:

| | O | C | Cl |
|---|---|---|---|
| **O** | −2 | +1 | 0 |
| **C** | +1 | 0 | −1 |
| **Cl** | 0 | −1 | +2 |

The entries of R sum to zero. Oxygen gives up a lone pair, chlorine gains one, the O–C bond
forms and the C–Cl bond breaks. The formal charges follow from **B** alone: O goes from −1 to
0, and Cl from 0 to −1.

## The anchor

The decoder's anchor is the forming O–C bond, the `+1` entry. Given it, the remaining entries
are the breaking C–Cl bond and the two lone-pair changes. See
[[reaction-matrix|the reaction matrix]] for how a step is decoded.

## The arrows

R decomposes into the two textbook arrows:

1. **lone pair O → C**: `R[O][O] = −2`, `R[O][C] = +1`;
2. **bond C–Cl → Cl**: `R[C][Cl] = −1`, `R[Cl][Cl] = +2`.

Their sum is R. The model never predicts these arrows. They are read back from R.

## The orbital reading

The forming O–C bond is explained by the orbital pair with the largest term in its pooled gap
factor (see [[klopman-salem|the Klopman–Salem pair features]]). Here that pair is the oxygen
lone pair of methoxide (donor) and the σ\*(C–Cl) orbital of chloromethane (acceptor). Its
dominant canonical pair is the pool's HOMO and LUMO. This is the frontier-orbital picture of an
SN2 attack, recovered from the localized orbitals rather than assumed.
