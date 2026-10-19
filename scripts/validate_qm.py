"""Run the QM layer on the ten fixture molecules and check the chemistry.

    uv run python scripts/validate_qm.py [--cache-dir DIR]

Exit code 0 when every check passes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `tests` importable

from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402
from tests.fixtures.molecules import MOLECULES  # noqa: E402

from ksmech.qm.records import DEFAULT_CACHE_DIR  # noqa: E402
from ksmech.qm.validation import compute_fixture_records, gap_ev, run_checks  # noqa: E402
from ksmech.qm.xtb_calc import QMConfig  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    args = parser.parse_args()
    console = Console()
    records = compute_fixture_records(MOLECULES, QMConfig(cache_dir=args.cache_dir))

    info = Table(title="Fixture molecules")
    for col in ("molecule", "atoms", "gap (eV)", "energy (Eh)"):
        info.add_column(col, justify="left" if col == "molecule" else "right")
    for name, rec in records.items():
        info.add_row(name, str(rec.n_atoms), f"{gap_ev(rec):.3f}", f"{rec.energy_ha:.6f}")
    console.print(info)

    results = run_checks(records)
    table = Table(title="Checks")
    for col in ("check", "result", "detail"):
        table.add_column(col)
    for r in results:
        table.add_row(r.name, "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]", r.detail)
    console.print(table)
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
