"""Open-shell plan P7 step 9: |g| and |g2| by cell kind, and the intramolecular |beta_WH| from
which BETA_INTER is stated. Diagnostics, not gates; read-only.

    uv run python scripts/probes/open_shell_g_table.py [--slice 500] [--workers 3] --out DIR

For every reactant pool of a set (a closed_base slice of PMechDB: the first `--slice` ids of
`cohort_closed_base.json` in sorted order; every PMechDB training step with an open-shell
`domain_class`), its tokens from the token cache, with the P7 roles and holes
(`ReactantTokens`), give every (occupied, virtual) cell: its kind (doubly->empty, SOMO->empty,
doubly->hole, SOMO->hole, and own pair: a bonding token with its own antibond), intra or inter,
flagged or not (any of P0a's three exclusion flags), |g| (where its weight is not 0), |g2| at the
BETA_INTER the code holds, and within a fragment the coupling |beta| = K x cross_coupling
(the beta of g2) with the graph distance of the two centre sets. The report gives quantiles
(p50, p90, p99, p99.9, max) by kind, location and flag, with and without the flagged cells
and with and without the intramolecular open-shell cells, and |beta| by graph distance.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

QUANTILES = (0.5, 0.9, 0.99, 0.999)


def cell_rows(smiles: str, loader) -> dict[str, np.ndarray]:
    """Per-cell arrays of one reactant pool (module docstring)."""
    from ksmech.arrows.reactants import ReactantTokens
    from ksmech.physics.pairs import PAIR_FLAGS, pair_flags
    from ksmech.physics.params import KSParams
    from ksmech.physics.stage_a import OPEN_SHELL_BLOCK, A, open_shell_block, stage_a_block

    rt = ReactantTokens.from_smiles(smiles, loader)
    tok = [f.tokens for f in rt.fragments]
    block = stage_a_block(tok)
    flags = pair_flags(tok, rt.candidate)
    extra, extra_d = open_shell_block(tok, block, flags["own_antibond"])
    occ = [o for t in tok for o in t.occupied()]
    virt = [o for t in tok for o in t.virtual()]
    row = np.array([o.spin_role for o in occ])[:, None] * np.ones((1, len(virt)), int)
    col = np.ones((len(occ), 1), int) * np.array([o.spin_role for o in virt])[None, :]
    k = {n: i for i, n in enumerate(OPEN_SHELL_BLOCK)}
    return {
        "kind": np.select(
            [flags["own_antibond"], (row == 1) & (col == -1), row == 1, col == -1],
            ["own_pair", "somo_hole", "somo_empty", "doubly_hole"],
            "doubly_empty",
        ),
        "inter": block.z[..., A["inter"]] > 0,
        "flagged": np.any([flags[n] for n in PAIR_FLAGS], axis=0),
        "g": np.where(block.defined[..., A["g"]], np.abs(block.z[..., A["g"]]), np.nan),
        "g2": np.where(extra_d[..., k["g2"]], np.abs(extra[..., k["g2"]]), np.nan),
        "beta": np.where(
            block.z[..., A["inter"]] > 0, np.nan, block.z[..., A["cross_coupling"]] * KSParams().K
        ),
        "graph": block.z[..., A["graph_distance"]],
        "open": (row != 0) | (col != 0),
    }


def _quantiles(x: np.ndarray) -> dict:
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0}
    q = {f"p{100 * p:g}": float(np.quantile(x, p)) for p in QUANTILES}
    return {"n": int(len(x)), **q, "max": float(x.max())}


def summarize(rows: list[dict[str, np.ndarray]]) -> dict:
    """Quantiles of |g| and |g2| by (kind, intra/inter, flagged), the two whole-set views, and
    the intramolecular |beta| by graph distance (0, 1, 2, 3+)."""
    cat = {k: np.concatenate([r[k].ravel() for r in rows]) for k in rows[0]}
    groups: dict = defaultdict(dict)
    for kind in np.unique(cat["kind"]):
        for where, inter in (("intra", False), ("inter", True)):
            for flag in (False, True):
                sel = (cat["kind"] == kind) & (cat["inter"] == inter) & (cat["flagged"] == flag)
                if sel.any():
                    name = f"{kind}/{where}/{'flagged' if flag else 'unflagged'}"
                    groups[name] = {
                        "g": _quantiles(cat["g"][sel]),
                        "g2": _quantiles(cat["g2"][sel]),
                    }
    views = {
        "all": np.ones_like(cat["inter"]),
        "unflagged": ~cat["flagged"],
        "without_intra_open": ~(cat["open"] & ~cat["inter"]),
        "unflagged_without_intra_open": ~cat["flagged"] & ~(cat["open"] & ~cat["inter"]),
    }
    whole = {
        v: {"g": _quantiles(cat["g"][m]), "g2": _quantiles(cat["g2"][m])} for v, m in views.items()
    }
    beta = {}
    intra = ~cat["inter"]
    for name, sel in (("0", cat["graph"] == 0), ("1", cat["graph"] == 1), ("2", cat["graph"] == 2),
                      ("3+", cat["graph"] >= 3)):  # fmt: skip
        beta[name] = _quantiles(cat["beta"][intra & sel & ~cat["flagged"]])
    return {"groups": dict(groups), "whole": whole, "beta_by_graph_distance": beta}


def pools(slice_size: int) -> dict[str, list[str]]:
    """Reactant SMILES of the closed_base slice and of PMechDB's open-shell training steps."""
    from ksmech.data.io import read_training_steps, steps_path
    from ksmech.eval.cohorts import load_cohort

    steps = {s.step_id: s for s in read_training_steps(steps_path("pmechdb"))}
    ids = sorted(load_cohort("pmechdb", "closed_base") & steps.keys())[:slice_size]
    open_ids = sorted(
        i for i, s in steps.items()
        if s.domain_class and any(c in s.domain_class for c in ("RADICAL", "CARBENE", "OSM"))
    )  # fmt: skip
    return {
        "closed_base": [steps[i].reactants_smiles for i in ids],
        "open_shell": [steps[i].reactants_smiles for i in open_ids],
    }


def _task(smiles: str):
    from ksmech.arrows.reactants import MissingTokens, TokenCache

    try:
        return cell_rows(smiles, TokenCache())
    except (MissingTokens, ValueError) as exc:
        return str(exc)[:120]


def main(argv: list[str] | None = None) -> None:
    from ksmech.processes import in_processes

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--slice", type=int, default=500)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = {}
    for name, smiles in pools(args.slice).items():
        out = list(in_processes(_task, smiles, args.workers))
        rows = [r for r in out if isinstance(r, dict)]
        report[name] = {"n_pools": len(smiles), "n_skipped": len(out) - len(rows)}
        report[name] |= summarize(rows) if rows else {}
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "open_shell_g_table.json"
    path.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    for name, r in report.items():
        print(name, json.dumps({k: r[k] for k in ("n_pools", "n_skipped")}))
        print(" beta", json.dumps(r.get("beta_by_graph_distance"), indent=None))
        print(" whole", json.dumps(r.get("whole"), indent=None)[:600])
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
