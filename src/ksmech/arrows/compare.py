"""Compare an edited molecule with the annotated products of a step.

Light on purpose (RDKit and `ksmech.data` only), so the ingest validation can use it without
importing the token and QM layers. A match is `exact` (same canonical Kekulé SMILES with maps)
or `resonance` (same canonical resonance key); a failure is the `ApplyReason` of the edit or
`MISMATCH` with a pattern naming which properties of which elements differ.
"""

from __future__ import annotations

from rdkit import Chem

from ksmech.arrows.apply import ApplyResult
from ksmech.data import chem
from ksmech.data.resonance import canonical_resonance_key

MATCH_STATUSES = ("exact", "resonance")


def _atom_state(mol: Chem.Mol) -> dict[int, tuple]:
    return {
        a.GetAtomMapNum(): (
            a.GetSymbol(),
            a.GetFormalCharge(),
            a.GetTotalNumHs(),
            a.GetNumRadicalElectrons(),
        )
        for a in mol.GetAtoms()
        if a.GetAtomMapNum()
    }


def _bond_orders(mol: Chem.Mol) -> dict[tuple[int, int], float]:
    out = {}
    for b in mol.GetBonds():
        i, j = b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum()
        if i and j:
            out[(min(i, j), max(i, j))] = b.GetBondTypeAsDouble()
    return out


def difference_pattern(got: Chem.Mol, want: Chem.Mol) -> str:
    """Which mapped-atom properties (charge, H, radical) and mapped bonds differ, by element.
    `other` means the difference lies only on unmapped atoms."""
    sa, sb = _atom_state(got), _atom_state(want)
    kinds: set[str] = set()
    for m in sa.keys() & sb.keys():
        for name, k in (("charge", 1), ("H", 2), ("radical", 3)):
            if sa[m][k] != sb[m][k]:
                kinds.add(f"{name}:{sa[m][0]}")
    ba, bb = _bond_orders(got), _bond_orders(want)
    for pair in ba.keys() | bb.keys():
        if ba.get(pair, 0) != bb.get(pair, 0):
            sym = sorted(sa.get(m, ("?",))[0] for m in pair)
            kinds.add(f"bond:{sym[0]}-{sym[1]}")
    return ";".join(sorted(kinds)) or "other"


def match_products(
    result: ApplyResult, products: Chem.Mol, products_smiles: str | None = None
) -> tuple[str, str]:
    """(status, pattern): `exact` / `resonance` with an empty pattern, or the failure code and
    its pattern. `products_smiles` is the products' canonical Kekulé SMILES when known."""
    if not result.ok:
        return str(result.reason), f"{result.reason}"
    want = products_smiles or chem.canonical_mapped_smiles(products)
    if result.smiles == want:
        return "exact", ""
    if result.key.key == canonical_resonance_key(products).key:
        return "resonance", ""
    return "MISMATCH", f"MISMATCH {difference_pattern(result.mol, products)}"
