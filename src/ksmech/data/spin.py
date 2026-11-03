"""The spin of a drawn species (open-shell plan P3 step 1, P4 step 1): the single source of the
unpaired-electron count for the domain laws, the QM records and the token readers.

`assign` gives one connected fragment its unpaired count, the source that decided it and, where
it is defined, the count per atom:

- 'metal': an odd fragment with an open-shell-capable metal (`domain.fallthrough_code` gives
  OPEN_SHELL_METAL) reads the metal table (`spin_tables/metal_v1.json`): one row per element
  and oxidation state, with its literature basis. A row holds for a species only where the
  species table's rule C confirms it (A and B agree on the row's count, case 'agree');
  otherwise, and for any fragment without a row (no open-shell centre, two of them, radicals
  drawn beside the metal, a REDOX_INCOMPATIBLE row), the count is MULTIPLICITY_AMBIGUOUS, or
  MULTIPLICITY_UNTABULATED when the species table lacks it. A ligand-dropped metal
  (`domain.is_placeholder`, odd or even) is no species: MULTIPLICITY_AMBIGUOUS, whatever its
  drawing;
- 'rdkit': RDKit's radical count when it is 0 or 1 (a closed shell or a doublet has one
  multiplicity);
- 'species': the species table (`data.spin_table`, spin_table_v2) when RDKit counts 2 or more
  (carbenes, CO drawn [C]=O, O2, diradicals: the drawings whose pairing is ambiguous), keyed by
  the map-free canonical SMILES. A species the table lacks is MULTIPLICITY_UNTABULATED and one
  whose entry is ambiguous MULTIPLICITY_AMBIGUOUS; the remedy for either is a new table
  version, never a rule here.

No chemist's clause is code here: the dihalocarbene, vinylidene and CO singlets are entries of
the species table, and tests check them (tests/test_data_spin.py).

`spin_config` is what every artifact that reads this module stamps into its provenance config:
the rule version and both tables' versions and content hashes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache, lru_cache
from pathlib import Path

from rdkit import Chem

from ksmech.data import chem, spin_table

# v1 (P3): RDKit for u <= 1, the species table for u >= 2, a table count other than RDKit's
# pending; v2 (P4): the metal table, and every assigned count stands (tagged QM records)
SPIN_RULE_VERSION = "spin-rule-v2"
UNTABULATED = "MULTIPLICITY_UNTABULATED"
AMBIGUOUS = spin_table.AMBIGUOUS
METAL_TABLE = spin_table.TABLE_DIR / "metal_v1.json"
METAL_VERSION = "metal_v1"
RDKIT, SPECIES, METAL = "rdkit", "species", "metal"  # the sources of a count
# why a metal fragment has no count (`Assignment.reason`)
NO_ROW = "NO_ROW"  # no single open-shell centre, or no row for it
REDOX_INCOMPATIBLE = "REDOX_INCOMPATIBLE"  # a row that rejects the species outright
NOT_CONFIRMED = "NOT_CONFIRMED"  # the species table's rule C does not give the row's count
PLACEHOLDER = "PLACEHOLDER"  # a ligand-dropped metal (`domain.is_placeholder`)
LIGAND_RADICALS = "LIGAND_RADICALS"  # radical electrons drawn beside the metal


@dataclass(frozen=True)
class Assignment:
    """The spin of one fragment: `unpaired` is a count, UNTABULATED or AMBIGUOUS; `per_atom`
    is the count per atom where it is defined (None otherwise); `row` is the metal row."""

    unpaired: int | str
    source: str
    per_atom: tuple[int, ...] | None = None
    reason: str | None = None
    row: str | None = None

    @property
    def assigned(self) -> bool:
        return isinstance(self.unpaired, int)


def rdkit_unpaired(mol: Chem.Mol) -> int:
    """RDKit's radical electrons summed over the molecule's atoms."""
    return sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())


def _rdkit_atoms(mol: Chem.Mol) -> tuple[int, ...]:
    return tuple(a.GetNumRadicalElectrons() for a in mol.GetAtoms())


@cache
def species_table() -> dict:
    """The species table (`spin_table.load`: v2), validated against its theory table (loaded
    once per process)."""
    return spin_table.load()


def load_metal_table(path: Path = METAL_TABLE) -> dict:
    """The metal table, checked against its version and content hash."""
    table = json.loads(Path(path).read_text())
    if table.get("version") != METAL_VERSION:
        raise spin_table.SpinTableError(f"{path}: not a {METAL_VERSION} table")
    if spin_table.with_hash(table)["content_hash"] != table.get("content_hash"):
        raise spin_table.SpinTableError(f"{path}: content_hash does not match the table")
    return table


@cache
def metal_table() -> dict:
    return load_metal_table()


def _row_matches(row: dict, atom: Chem.Atom, os_: int) -> bool:
    if (row["element"], row["oxidation_state"]) != (atom.GetSymbol(), os_):
        return False
    donors = {n.GetSymbol() for n in atom.GetNeighbors()} | (
        {"H"} if atom.GetTotalNumHs() else set()
    )
    if row.get("donors_any") and not donors & set(row["donors_any"]):
        return False
    return not donors & set(row.get("donors_none", ()))


def _open_centres(mol: Chem.Mol) -> list[Chem.Atom]:
    from ksmech.data import domain  # domain reads this module: imported where it is used

    return [
        a
        for a in mol.GetAtoms()
        if a.GetAtomicNum() in domain.GROUP and domain.d_count(a) not in (0, 10)
    ]


def _metal_assignment(mol: Chem.Mol, table: dict, metals: dict) -> Assignment:
    """The 'metal' source (module docstring) for an odd open-shell-metal fragment."""
    from ksmech.data import domain

    centres = _open_centres(mol)
    if len(centres) != 1:
        return Assignment(AMBIGUOUS, METAL, reason=NO_ROW)
    if rdkit_unpaired(mol):
        return Assignment(AMBIGUOUS, METAL, reason=LIGAND_RADICALS)
    centre, os_ = centres[0], domain.oxidation_state(centres[0])
    row = next((r for r in metals["rows"] if _row_matches(r, centre, os_)), None)
    if row is None or row.get("redox_incompatible"):
        reason = NO_ROW if row is None else REDOX_INCOMPATIBLE
        return Assignment(AMBIGUOUS, METAL, reason=reason, row=row and row["id"])
    entry = table["entries"].get(chem.unmapped_canonical(mol))
    if entry is None:
        return Assignment(UNTABULATED, METAL, reason=NOT_CONFIRMED, row=row["id"])
    if entry["unpaired"] != row["unpaired"] or entry["case"] != spin_table.AGREE:
        return Assignment(AMBIGUOUS, METAL, reason=NOT_CONFIRMED, row=row["id"])
    per_atom = [0] * mol.GetNumAtoms()
    per_atom[centre.GetIdx()] = row["unpaired"]
    return Assignment(row["unpaired"], METAL, tuple(per_atom), row=row["id"])


def assign(mol: Chem.Mol, table: dict | None = None, metals: dict | None = None) -> Assignment:
    """The spin of one fragment (module docstring)."""
    from ksmech.data import domain

    table = table or species_table()
    if domain.is_placeholder(mol):  # a ligand-dropped metal is no species to give a count
        return Assignment(AMBIGUOUS, METAL, reason=PLACEHOLDER)
    if domain.is_odd(mol) and domain.fallthrough_code(mol) == domain.DomainCode.OPEN_SHELL_METAL:
        return _metal_assignment(mol, table, metals or metal_table())
    u = rdkit_unpaired(mol)
    if u <= 1:
        return Assignment(u, RDKIT, _rdkit_atoms(mol))
    value = spin_table.lookup(table, chem.unmapped_canonical(mol))
    if value is None:
        return Assignment(UNTABULATED, SPECIES)
    if value == u:
        return Assignment(value, SPECIES, _rdkit_atoms(mol))
    return Assignment(value, SPECIES, (0,) * mol.GetNumAtoms() if value == 0 else None)


@lru_cache(maxsize=65536)
def assign_smiles(smiles: str) -> Assignment:
    """`assign` of a map-free canonical SMILES (a QM or token cache key), memoized."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"cannot parse {smiles!r}")
    return assign(mol)


def species_u(mol: Chem.Mol, table: dict | None = None) -> int | str:
    """Unpaired electrons of one species, or UNTABULATED / AMBIGUOUS (module docstring)."""
    return assign(mol, table).unpaired


def reassigned(a: Assignment, drawn: int) -> bool:
    """The assignment gives a count other than the drawing's (RDKit's): its QM record is
    computed at the assigned count under a tagged method (`qm.spin`)."""
    return a.assigned and a.unpaired != drawn


def spin_config(table: dict | None = None, metals: dict | None = None) -> dict:
    """The provenance config of an artifact that reads `assign`."""
    t, m = table or species_table(), metals or metal_table()
    return {
        "spin_rule_version": SPIN_RULE_VERSION,
        "spin_table": t["version"],
        "spin_table_digest": t["content_hash"],
        "metal_table": m["version"],
        "metal_table_digest": m["content_hash"],
    }
