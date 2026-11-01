"""spin_table_v2: the species multiplicity table of the open-shell plan (P2 step 4; v2 after P3).

Every multiplicity is derived twice, independently, and combined by one rule (the user's choice,
2026-10-30):

- (A) spGFN2-xTB at every allowed multiplicity, each state optimized (scripts/probes/spin_scan.py).
  A is conclusive when its lowest state lies below every other by more than the method's
  published error bar (`ERROR_BARS`); a species with one allowed state is conclusive by count.
  Since v2, a state of a carbene (one centre holding the drawing's two radical electrons, the
  species kind of AC12, the aryl-carbene set the bar was measured on) whose every start formed
  a bond the drawing lacks takes its single-point energy at the production geometry: an upper
  bound of the state's energy in the drawing's bonding. A bounded state can make A conclusive
  as the lowest state (its true gap is at least the measured one), never as another state (its
  true energy may be lower), so A is inconclusive when a state other than the lowest is
  bounded. Other species keep such a state failed: for biradicals with distant centres
  spGFN2's singlet reads 0.4-0.8 eV below the triplet even at the drawn geometry, outside the
  bar's domain (docs/decisions.md, 2026-11-01).
- (B) the theory rules of `data.spin_theory` (spin_theory_v2): a predicted unpaired count or
  undetermined (None).
- (C) `combine`: A and B give the same lowest state -> assigned; B undetermined -> A decides
  when conclusive, else MULTIPLICITY_AMBIGUOUS; every disagreement -> MULTIPLICITY_AMBIGUOUS,
  a lowest state of A that differs from B's inside A's error bar included. A with a failed
  state has no lowest state and assigns nothing.

The table maps a canonical map-free SMILES to its unpaired count (multiplicity u + 1) or
MULTIPLICITY_AMBIGUOUS, with every state's energy, both verdicts and the case of C. It is
tracked and versioned: `content_hash` covers every field but itself, and `theory_digest` must
equal the theory table's digest. Any change is a new version and a rebuild; v1 (P2) stays in
`spin_tables/` as the record of what P3 read.

v2 (2026-10-31, docs/decisions.md): A's bounded states (above) and spin_theory_v2's B (its
geometry off symmetric saddles, the unsaturated-carbene and pi-donor rules).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

TABLE_DIR = Path(__file__).parent / "spin_tables"
SPECIES_V1 = TABLE_DIR / "species_v1.json"
THEORY_V1 = TABLE_DIR / "theory_v1.json"
SPECIES_V2 = TABLE_DIR / "species_v2.json"
THEORY_V2 = TABLE_DIR / "theory_v2.json"
VERSION = "spin_table_v2"
SCHEMA = 1
AMBIGUOUS = "MULTIPLICITY_AMBIGUOUS"

# The cases of rule C, in the order the plan lists them, plus A without a lowest state.
AGREE = "agree"
A_DECIDES = "b_undetermined_a_conclusive"
A_INCONCLUSIVE = "b_undetermined_a_inconclusive"
DISAGREE = "disagree"
A_FAILED = "a_failed"
CASES = (AGREE, A_DECIDES, A_INCONCLUSIVE, DISAGREE, A_FAILED)

# A's error bars (eV), the method's published errors, not tuned. The organic bar is spGFN2's
# singlet-triplet error on AC12; the metal bar its mean absolute deviation on the 3d subset of
# TM90S against TPSSh-D4/def2-QZVPP (docs/decisions.md, 2026-10-30).
EV_PER_KCAL = 1 / 23.0605
ERROR_BARS = {
    "organic": {
        "ev": 7.0 * EV_PER_KCAL,
        "citation": "Moradi, Tomann, Hendrix, Head-Gordon, Stein, J. Comput. Chem. 45, 2786 "
        "(2024), arXiv 2405.05761: spGFN2-xTB error on AC12 about 7 kcal/mol",
    },
    "transition_metal": {
        "ev": 17.9 * EV_PER_KCAL,
        "citation": "Neugebauer, Baedorf, Ehlert, Hansen, Grimme, J. Comput. Chem. 44, 2120 "
        "(2023), doi 10.1002/jcc.27185: spGFN2-xTB spin-state splittings, TM90S 3d subset, "
        "MAD 17.9 kcal/mol against TPSSh-D4/def2-QZVPP",
    },
}


@dataclass(frozen=True)
class ScanVerdict:
    """A's reading of one species: the lowest state (None when a state has no energy), the gap
    to the next state (eV; None with one state) and whether that gap exceeds the error bar."""

    lowest: int | None
    gap_ev: float | None
    conclusive: bool


def scan_verdict(
    energies_ev: dict[int, float | None], bar_ev: float, bounded: Collection[int] = ()
) -> ScanVerdict:
    """A from the energy of every allowed state (None for a failed state); `bounded` holds the
    states whose energy is an upper bound (module docstring), which keep A from being
    conclusive unless the lowest state is the only one among them."""
    if not energies_ev or any(e is None for e in energies_ev.values()):
        return ScanVerdict(None, None, False)
    order = sorted(energies_ev, key=energies_ev.get)
    if len(order) == 1:
        return ScanVerdict(order[0], None, True)
    gap = energies_ev[order[1]] - energies_ev[order[0]]
    return ScanVerdict(order[0], gap, gap > bar_ev and not set(order[1:]) & set(bounded))


def combine(a: ScanVerdict, b_unpaired: int | None) -> tuple[int | str, str]:
    """Rule C: (unpaired count or AMBIGUOUS, the case that decided)."""
    if a.lowest is None:
        return AMBIGUOUS, A_FAILED
    if b_unpaired is None:
        return (a.lowest, A_DECIDES) if a.conclusive else (AMBIGUOUS, A_INCONCLUSIVE)
    if a.lowest == b_unpaired:
        return a.lowest, AGREE
    return AMBIGUOUS, DISAGREE


def digest(payload: dict) -> str:
    """sha256 of the canonical JSON of `payload`."""
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode()).hexdigest()


def with_hash(table: dict) -> dict:
    """`table` with its content_hash over every other field."""
    body = {k: v for k, v in table.items() if k != "content_hash"}
    return {**body, "content_hash": digest(body)}


class SpinTableError(ValueError):
    """The species table does not validate (schema, content hash or theory digest)."""


REQUIRED = ("version", "schema", "method", "theory_digest", "error_bars", "entries")
ENTRY_KEYS = ("unpaired", "case", "states", "a", "b")


def validate(table: dict, theory: dict | None = None, version: str = VERSION) -> None:
    """Raise SpinTableError unless the table has its fields, its content hash and, when the
    theory table is given, that table's digest."""
    missing = [k for k in REQUIRED if k not in table]
    if missing or table.get("version") != version or table.get("schema") != SCHEMA:
        raise SpinTableError(f"not a {version} schema {SCHEMA} table (missing {missing})")
    if with_hash(table)["content_hash"] != table.get("content_hash"):
        raise SpinTableError("content_hash does not match the table")
    if theory is not None and table["theory_digest"] != theory.get("digest"):
        raise SpinTableError("theory_digest differs from the theory table's digest")
    for smiles, entry in table["entries"].items():
        bad = [k for k in ENTRY_KEYS if k not in entry]
        if bad or entry["case"] not in CASES:
            raise SpinTableError(f"{smiles}: entry missing {bad} or unknown case")
        if (entry["unpaired"] == AMBIGUOUS) != (entry["case"] in CASES[2:]):
            raise SpinTableError(f"{smiles}: case {entry['case']} with {entry['unpaired']!r}")


def load(path: Path = SPECIES_V2, theory_path: Path = THEORY_V2) -> dict:
    """The validated species table, checked against its theory table (whose own digest is
    verified; the theory rules themselves are never run here)."""
    from ksmech.data.spin_theory import load_theory_table

    if not Path(theory_path).exists():
        raise SpinTableError(f"{theory_path} is missing: the table cannot be checked")
    table = json.loads(Path(path).read_text())
    validate(table, load_theory_table(Path(theory_path)))
    return table


def lookup(table: dict, smiles: str) -> int | str | None:
    """The table's unpaired count or AMBIGUOUS for a canonical SMILES; None when untabulated."""
    entry = table["entries"].get(smiles)
    return None if entry is None else entry["unpaired"]
