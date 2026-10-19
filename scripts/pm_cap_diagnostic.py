"""Which orbitals are still rotating when a Pipek-Mezey block hits the sweep cap?

    uv run python scripts/pm_cap_diagnostic.py [--tokens-dir DIR] [--qm-cache-dir DIR]

For every token file with `pm_converged == False`, each capped block is re-run for the cap
(deterministic, so it reproduces the stored tokens), then one more sweep is run from there. The
orbitals that this extra sweep rotates (sin of the rotation angle of their column of U) are
classified; a block is "pi only" when every orbital rotating by at least half the largest
rotation is a pi-type orbital. Exit code 1 if any block rotates a sigma or lone-pair orbital.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

from ksmech.orbitals.classify import PI_TYPES, classify_block
from ksmech.orbitals.localize import PMConfig, pipek_mezey_stats
from ksmech.orbitals.tokens import load_tokens
from ksmech.qm.fukui import OCC_THRESHOLD
from ksmech.qm.records import DEFAULT_CACHE_DIR, cache_path, load_record

CAP = PMConfig().max_sweeps


def rotating_orbitals(rec, mask, cap: int) -> tuple[list[tuple[float, str, str]], float]:
    """(rotation, type, parents) of the orbitals that one extra sweep moves, largest first."""
    C = rec.C[:, mask]
    C_cap, _, _ = pipek_mezey_stats(C, rec.S, rec.ao_atom, PMConfig(max_sweeps=cap))
    _, U1, _ = pipek_mezey_stats(C_cap, rec.S, rec.ao_atom, PMConfig(tol=0.0, max_sweeps=1))
    rot = np.sqrt(np.clip(1.0 - np.diag(U1) ** 2, 0.0, None))
    occupied = bool(rec.occupations[mask][0] >= OCC_THRESHOLD)
    classes = classify_block(
        C_cap, rec.S, rec.ao_atom, rec.ao_l, rec.coords_ang, occupied, None, rec.atomic_numbers
    )
    rows = [(float(rot[k]), str(c.type), str(c.parent_atoms)) for k, c in enumerate(classes)]
    rows.sort(key=lambda r: -r[0])
    return rows, float(rot.max())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens-dir", type=Path, default=Path("data/cache/tokens"))
    parser.add_argument("--qm-cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    args = parser.parse_args()
    console = Console(width=140)
    table = Table(title=f"Blocks at the {CAP}-sweep cap")
    for col in (
        "smiles",
        "block",
        "n",
        "max rot",
        "rotating orbitals (rot, type, parents)",
        "verdict",
    ):
        table.add_column(col)
    n_sigma = 0
    for path in sorted(args.tokens_dir.glob("*.npz")):
        t = load_tokens(path)
        if t.pm_converged:
            continue
        rec = load_record(cache_path(args.qm_cache_dir, t.smiles, t.conformer_id, t.method))
        blocks = (("occ", t.pm_sweeps_occ, rec.occupations >= OCC_THRESHOLD),)
        blocks += (("virt", t.pm_sweeps_virt, rec.occupations < OCC_THRESHOLD),)
        for label, sweeps, mask in blocks:
            if sweeps < CAP:
                continue
            rows, top = rotating_orbitals(rec, mask, CAP)
            moving = [r for r in rows if r[0] >= 0.5 * top]
            pi_only = all(r[1] in PI_TYPES for r in moving)
            n_sigma += not pi_only
            shown = "; ".join(f"{r[0]:.1e} {r[1]} {r[2]}" for r in moving[:6])
            verdict = "[green]pi only[/green]" if pi_only else "[red]sigma / lone pair[/red]"
            table.add_row(t.smiles[:45], label, str(int(mask.sum())), f"{top:.1e}", shown, verdict)
    console.print(table)
    console.print(f"{n_sigma} capped block(s) rotate a sigma or lone-pair orbital")
    return 1 if n_sigma else 0


if __name__ == "__main__":
    sys.exit(main())
