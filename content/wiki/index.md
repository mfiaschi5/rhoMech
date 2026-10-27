+++
title = "ChemRho"
summary = "What ChemRho does, and the sections of the platform."
tags = ["overview"]
order = 0
+++
# ChemRho

ChemRho predicts organic reaction mechanisms as sequences of **elementary steps**. Each step
is predicted as the change it makes to the Lewis structure: which bonds form, which break, and
where electron pairs end up. Curved arrows are not predicted directly. They are a reading of
that change, reconstructed afterwards together with the orbital pairs that drive it.

The model reads molecules as two kinds of tokens: atoms, and localized molecular orbitals
computed with GFN2-xTB. Between an occupied and an empty orbital it starts from the
Klopman–Salem picture of orbital interactions, instead of learning that picture from scratch.

## Sections

| Section | What it is for |
|---|---|
| **ChemPtah** | Drawing: sketch or import molecules, and look at their orbitals and properties. |
| **ClotoMech** | Mechanisms. **Predict** proposes steps for a set of reactants, **Correct** replaces a predicted step with your own and checks it, **Curate** keeps reviewed mechanisms. |
| **ChemThot** | Browsing the reaction database behind the model. |
| **Wiki** | These pages. |

## Start here

- [[reaction-matrix|The reaction matrix]]: how a step is written, and what counts as the same
  step under resonance.
- [[orbital-tokens|Atom and orbital tokens]]: what the model sees of a molecule.
- [[klopman-salem|The Klopman–Salem pair features]]: the physics between two orbitals.
- [[sn2-example|A worked example]]: methoxide and chloromethane, one SN2 step.
