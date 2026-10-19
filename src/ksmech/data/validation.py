"""Step validation shared by all loaders: reason codes and the record -> ElementaryStep pipeline."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum

from rdkit import Chem

from ksmech.data import chem
from ksmech.data.schema import Arrow, ElementaryStep, Source, SourceKind


class RejectReason(StrEnum):
    MALFORMED_RECORD = "MALFORMED_RECORD"  # record cannot be split into its fields
    ARROW_PARSE = "ARROW_PARSE"  # arrow code does not follow the grammar
    ARROW_INVALID = "ARROW_INVALID"  # arrow shape has no electron-pushing meaning
    SMILES_PARSE = "SMILES_PARSE"  # RDKit cannot read the SMILES
    SANITIZE_FAILED = "SANITIZE_FAILED"  # RDKit rejects valence / kekulization
    UNBALANCED = "UNBALANCED"  # heavy-atom element counts differ between sides
    H_UNBALANCED = "H_UNBALANCED"  # hydrogen counts differ
    CHARGE_UNBALANCED = "CHARGE_UNBALANCED"  # total formal charge differs
    MAP_DUPLICATE = "MAP_DUPLICATE"  # a map number is used twice on one side
    MAP_NOT_BIJECTIVE = "MAP_NOT_BIJECTIVE"  # reactant and product map sets differ
    ARROW_ATOM_UNMAPPED = "ARROW_ATOM_UNMAPPED"  # arrow names a map number absent from reactants
    BOND_NOT_FOUND = "BOND_NOT_FOUND"  # a bond-source arrow names two atoms that are not bonded
    DUPLICATE_STEP = "DUPLICATE_STEP"  # same step_id seen before


class StepRejected(Exception):
    def __init__(self, reason: RejectReason, detail: str = ""):
        super().__init__(f"{reason}: {detail}")
        self.reason = RejectReason(reason)
        self.detail = detail


@dataclass(frozen=True)
class Rejection:
    step_id: str
    reason: RejectReason
    detail: str
    raw: str


def make_step_id(source: Source | str, raw_record: str) -> str:
    """Stable id from the raw record text, independent of file order."""
    digest = hashlib.sha1(raw_record.encode("utf-8")).hexdigest()[:16]
    return f"{Source(source)}-{digest}"


def split_reaction(rxn_smiles: str) -> tuple[str, str]:
    parts = rxn_smiles.split(">>")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise StepRejected(
            RejectReason.MALFORMED_RECORD, f"not 'reactants>>products': {rxn_smiles}"
        )
    return parts[0], parts[1]


def parse_side(smiles: str, side: str) -> Chem.Mol:
    try:
        return chem.parse_mapped(smiles)
    except chem.SmilesParseError as exc:
        raise StepRejected(RejectReason.SMILES_PARSE, f"{side}: {exc}") from exc
    except chem.SanitizeError as exc:
        raise StepRejected(RejectReason.SANITIZE_FAILED, f"{side}: {exc}") from exc


def check_balance(reactants: Chem.Mol, products: Chem.Mol) -> None:
    r, p = chem.heavy_composition(reactants), chem.heavy_composition(products)
    if r != p:
        diff = {k: p[k] - r[k] for k in sorted(r.keys() | p.keys()) if p[k] != r[k]}
        raise StepRejected(RejectReason.UNBALANCED, f"heavy atoms products-reactants {diff}")
    rh, ph = chem.hydrogen_count(reactants), chem.hydrogen_count(products)
    if rh != ph:
        raise StepRejected(RejectReason.H_UNBALANCED, f"H reactants={rh} products={ph}")
    rq, pq = chem.formal_charge(reactants), chem.formal_charge(products)
    if rq != pq:
        raise StepRejected(RejectReason.CHARGE_UNBALANCED, f"charge reactants={rq} products={pq}")


def check_maps(reactants: Chem.Mol, products: Chem.Mol) -> None:
    for side, mol in (("reactants", reactants), ("products", products)):
        dup = sorted(k for k, n in Counter(chem.map_numbers(mol)).items() if n > 1)
        if dup:
            raise StepRejected(RejectReason.MAP_DUPLICATE, f"{side} repeat maps {dup}")
    r, p = set(chem.map_numbers(reactants)), set(chem.map_numbers(products))
    if r != p:
        detail = f"only in reactants {sorted(r - p)}, only in products {sorted(p - r)}"
        raise StepRejected(RejectReason.MAP_NOT_BIJECTIVE, detail)
    if not r:
        raise StepRejected(RejectReason.MAP_NOT_BIJECTIVE, "no mapped atoms")


def check_arrow_atoms(reactants: Chem.Mol, arrows: list[Arrow]) -> None:
    idx = chem.map_to_index(reactants)
    for arrow in arrows:
        missing = sorted(arrow.atoms - idx.keys())
        if missing:
            raise StepRejected(RejectReason.ARROW_ATOM_UNMAPPED, f"{arrow} uses {missing}")
        if arrow.source_kind is SourceKind.BOND:
            a, b = (idx[m] for m in arrow.source_atoms)
            if reactants.GetBondBetweenAtoms(a, b) is None:
                raise StepRejected(RejectReason.BOND_NOT_FOUND, f"no bond {arrow.source_atoms}")


def apply_arrows_stub(reactants: Chem.Mol, products: Chem.Mol, arrows: list[Arrow]) -> None:
    """Placeholder for the check 'reactants + arrows == products'.

    The bond-electron-matrix edit lands in the arrow phase (`ksmech.arrows`). Until then this
    check does nothing and accepts every step.
    """
    return None


def validate(
    reactants: Chem.Mol, products: Chem.Mol, arrows: list[Arrow], require_arrows: bool = True
) -> None:
    if require_arrows and not arrows:
        raise StepRejected(RejectReason.ARROW_PARSE, "no arrows")
    check_balance(reactants, products)
    check_maps(reactants, products)
    check_arrow_atoms(reactants, arrows)
    apply_arrows_stub(reactants, products, arrows)


def build_step(
    *,
    step_id: str,
    source: Source,
    rxn_smiles: str,
    arrows: list[Arrow],
    pathway_id: str | None = None,
    step_index: int | None = None,
    mechanism_class: str | None = None,
    conditions: str | None = None,
    require_arrows: bool = True,
) -> ElementaryStep:
    """Parse, validate and canonicalize one step. Raises StepRejected.

    `require_arrows=False` is for sources that ship mapped steps without arrows (ReactMech).
    Their `arrows` stay empty until the arrow-alignment phase derives them.
    """
    r_smi, p_smi = split_reaction(rxn_smiles)
    reactants, products = parse_side(r_smi, "reactants"), parse_side(p_smi, "products")
    validate(reactants, products, arrows, require_arrows)
    return ElementaryStep(
        step_id=step_id,
        source=source,
        pathway_id=pathway_id,
        step_index=step_index,
        reactants_smiles=chem.canonical_mapped_smiles(reactants),
        products_smiles=chem.canonical_mapped_smiles(products),
        arrows=list(arrows),
        mechanism_class=mechanism_class,
        conditions=conditions,
    )
