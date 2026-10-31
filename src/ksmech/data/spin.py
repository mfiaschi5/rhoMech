"""The spin of a drawn species (open-shell plan P3 step 1; P4 adds the metal table and the QM
routing).

`species_u` is the number of unpaired electrons of one connected fragment:

- RDKit's radical count when it is 0 or 1: a closed shell or a doublet has one multiplicity;
- spin_table_v1 (`data.spin_table`, P2's QM-derived species table) when RDKit counts 2 or more
  (carbenes, CO drawn [C]=O, O2, diradicals: the drawings whose pairing is ambiguous), keyed by
  the map-free canonical SMILES. A species the table lacks is MULTIPLICITY_UNTABULATED and one
  whose entry is ambiguous MULTIPLICITY_AMBIGUOUS; the remedy for either is a new table version,
  never a rule here.

`spin_config` is what every artifact that reads this module stamps into its provenance config:
the rule version and the table's version and content hash.
"""

from __future__ import annotations

from functools import cache

from rdkit import Chem

from ksmech.data import chem, spin_table

SPIN_RULE_VERSION = "spin-rule-v1"  # RDKit for u <= 1, spin_table_v1 for u >= 2
UNTABULATED = "MULTIPLICITY_UNTABULATED"
AMBIGUOUS = spin_table.AMBIGUOUS


def rdkit_unpaired(mol: Chem.Mol) -> int:
    """RDKit's radical electrons summed over the molecule's atoms."""
    return sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())


@cache
def species_table() -> dict:
    """spin_table_v1, validated against spin_theory_v1 (loaded once per process)."""
    return spin_table.load()


def species_u(mol: Chem.Mol, table: dict | None = None) -> int | str:
    """Unpaired electrons of one species, or UNTABULATED / AMBIGUOUS (module docstring)."""
    u = rdkit_unpaired(mol)
    if u <= 1:
        return u
    value = spin_table.lookup(table or species_table(), chem.unmapped_canonical(mol))
    return UNTABULATED if value is None else value


def spin_config(table: dict | None = None) -> dict:
    """The provenance config of an artifact that reads `species_u`."""
    t = table or species_table()
    return {
        "spin_rule_version": SPIN_RULE_VERSION,
        "spin_table": t["version"],
        "spin_table_digest": t["content_hash"],
    }
