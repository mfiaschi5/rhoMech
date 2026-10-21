"""Dataset statistics over a list of `ElementaryStep`."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from rdkit import Chem

from ksmech.data import chem
from ksmech.data.domain import PRECEDENCE
from ksmech.data.schema import ElementaryStep

NO_CLASS = "<none>"


@dataclass(frozen=True)
class StepFacts:
    n_arrows: int
    intermolecular: bool | None  # None when the step has no arrows
    heavy_atoms: int
    reactant_smiles: tuple[str, ...]  # unmapped canonical SMILES, one per fragment
    reactant_charges: tuple[int, ...]
    reactant_heavy_atoms: tuple[int, ...]
    elements: frozenset[str]


def step_facts(step: ElementaryStep) -> StepFacts:
    mol = chem.parse_mapped(step.reactants_smiles)
    frags = chem.fragments(mol)
    frag_of = chem.map_to_fragment(mol)
    inter = None
    if step.arrows:
        inter = any(len({frag_of[a] for a in arrow.atoms}) > 1 for arrow in step.arrows)
    elements = {a.GetSymbol() for a in mol.GetAtoms()}
    if chem.hydrogen_count(mol):
        elements.add("H")
    return StepFacts(
        n_arrows=len(step.arrows),
        intermolecular=inter,
        heavy_atoms=chem.heavy_atom_count(mol),
        reactant_smiles=tuple(chem.unmapped_canonical(f) for f in frags),
        reactant_charges=tuple(Chem.GetFormalCharge(f) for f in frags),
        reactant_heavy_atoms=tuple(chem.heavy_atom_count(f) for f in frags),
        elements=frozenset(elements),
    )


def distribution(values: Sequence[int], bin_width: int = 10) -> dict:
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return {"n": 0}
    q = np.percentile(v, [0, 25, 50, 75, 100])
    edges = np.arange(0, v.max() + bin_width + 1, bin_width)
    counts, _ = np.histogram(v, bins=edges)
    hist = {
        f"{int(lo)}-{int(lo) + bin_width - 1}": int(c)
        for lo, c in zip(edges[:-1], counts, strict=True)
        if c
    }
    keys = ("min", "p25", "median", "p75", "max")
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        **dict(zip(keys, q.tolist(), strict=True)),
        "hist": hist,
    }


def pathway_stats(steps: Sequence[ElementaryStep]) -> dict:
    """Pathway-level metrics over the steps that carry a `pathway_id`. Steps without one are
    counted and skipped. With no pathway id at all (PMechDB today), every pathway metric is None,
    so filling the ids later changes the numbers, not the code."""
    sizes = Counter(s.pathway_id for s in steps if s.pathway_id is not None)
    return {
        "n_pathways": len(sizes) if sizes else None,
        "n_steps_without_pathway": len(steps) - sum(sizes.values()),
        "steps_per_pathway": distribution(list(sizes.values()), bin_width=5) if sizes else None,
    }


def compute_stats(steps: Sequence[ElementaryStep]) -> dict:
    facts = [step_facts(s) for s in steps]
    n = len(steps)
    with_arrows = [f for f in facts if f.intermolecular is not None]
    charge_of: dict[str, int] = {}
    heavy_of: dict[str, int] = {}
    for f in facts:
        charge_of.update(zip(f.reactant_smiles, f.reactant_charges, strict=True))
        heavy_of.update(zip(f.reactant_smiles, f.reactant_heavy_atoms, strict=True))
    element_steps = Counter(e for f in facts for e in f.elements)
    classes = Counter(s.mechanism_class or NO_CLASS for s in steps)
    return {
        "sources": sorted({str(s.source) for s in steps}),
        "n_steps": n,
        "n_resonance_steps": sum(s.resonance_step for s in steps),
        "n_training_steps": sum(not s.resonance_step and s.domain_exclusion is None for s in steps),
        "domain": domain_counts(steps),
        **pathway_stats(steps),
        "steps_per_class": dict(classes.most_common()),
        "arrows_per_step": {
            str(k): v for k, v in sorted(Counter(f.n_arrows for f in facts).items())
        },
        "reactant_molecules": {
            "total_fragments": sum(len(f.reactant_smiles) for f in facts),
            "unique": len(charge_of),
            "mean_fragments_per_step": sum(len(f.reactant_smiles) for f in facts) / max(n, 1),
        },
        "heavy_atoms_per_step": distribution([f.heavy_atoms for f in facts]),
        "heavy_atoms_per_unique_reactant": distribution(list(heavy_of.values())),
        "intermolecular_fraction": _frac(
            sum(f.intermolecular for f in with_arrows), len(with_arrows)
        ),
        "n_steps_with_arrows": len(with_arrows),
        "elements": {e: element_steps[e] for e in sorted(element_steps)},
        "charged": {
            "steps_with_charged_reactant": _frac(sum(any(f.reactant_charges) for f in facts), n),
            "unique_reactants_charged": _frac(
                sum(q != 0 for q in charge_of.values()), len(charge_of)
            ),
        },
        "definitions": DEFINITIONS,
    }


def domain_counts(steps: Sequence[ElementaryStep]) -> dict:
    """Steps removed from the training target at ingest, by reason (a step can carry both a
    resonance flag and a domain code; each is counted), and the steps whose SMILES were
    charge-repaired (they stay)."""
    codes = Counter(s.domain_exclusion for s in steps if s.domain_exclusion)
    return {
        "RESONANCE_STEP": sum(s.resonance_step for s in steps),
        **{str(c): codes.get(str(c), 0) for c in PRECEDENCE},
        "CHARGE_REPAIRED (kept)": sum(s.charge_repair is not None for s in steps),
    }


def _frac(num: int, den: int) -> float | None:
    return num / den if den else None


DEFINITIONS = {
    "n_pathways": "distinct pathway_id values; null when no step has one",
    "steps_per_pathway": "steps per pathway_id, over steps that have one; null when none do",
    "reactant_molecules": "connected fragments on the reactant side, incl. spectators and "
    "counter-ions; 'unique' compares map-free canonical SMILES",
    "heavy_atoms_per_step": "heavy atoms summed over all reactant fragments of a step",
    "intermolecular_fraction": "steps with >=1 arrow whose atoms lie in >=2 reactant fragments, "
    "over steps that have arrows",
    "elements": "number of steps whose reactants contain the element (H incl. implicit)",
    "n_training_steps": "steps that are neither resonance steps nor outside the closed-shell "
    "domain (`domain`)",
    "domain": "steps removed from the target at ingest, by reason; QM-level reasons (failed "
    "records, Fermi smearing) are in docs/decisions.md, 'Domain exclusions'",
    "charged.steps_with_charged_reactant": "steps with >=1 reactant fragment of nonzero net charge",
    "charged.unique_reactants_charged": "unique reactant molecules with nonzero net charge",
}
