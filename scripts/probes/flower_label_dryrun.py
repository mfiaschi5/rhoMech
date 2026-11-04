"""Open-shell plan P6 step 8: a dry run of FlowER's radical lines (RS/RC) and of the v18 PMechDB
and ReactMech labels through the P6 rules. Read-only; it writes one report and nothing else.

    uv run python scripts/probes/flower_label_dryrun.py [--source flower|pmechdb|reactmech]
        [--limit N] [--workers 3] [--out DIR]

flower: every line of data/raw/flower/flower_new_dataset/{train,val,test}.txt tagged RS or RC,
streamed, as one step of the one-electron source (`domain.open_shell_labels`, rmechdb): the
flow kind of its sides (`resonance.classify_step`: NULL_STEP, RESONANCE, NONCONJUGATED_SHIFT
or a step), then the label class over the reactant drawings x the product forms
(`drawings.redrawn_label`) or its skip code (SET_STEP, DISSOCIATIVE_ET_STEP, NOT_ELEMENTARY,
...). pmechdb / reactmech: every training step of the v18 steps file, labelled as the sample
builder labels it. For every labelled step, the drawn R (R_data) and the class are checked
against each mask and screen of P9: (a) on a pool with u = 0 every +1 atom entry sits on an
end of a broken bond; (b) no -1 atom entry when u = 0; (c) an atom entry -1 needs N >= 1 and -2
needs N >= 2; (d) the 5-class atom alphabet with bond entries of +-1 (NOT_ELEMENTARY); (e) the
legacy decoder caps (8 bond, 2 atom entries). A class violates when every variant does. The
report gives the counts by code, by R shape and by violation, and lists the NONCONJUGATED_SHIFT,
DISSOCIATIVE_ET_STEP and violating lines one by one; u comes from `spin.spin_of_drawing`.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

RAW = Path("data/raw/flower/flower_new_dataset")
FILES = ("train", "val", "test")
TAGS = ("RS", "RC")
MAX_BONDS, MAX_ATOMS = 8, 2  # DecodeConfig's legacy caps
SHAPES = {  # (sigma form, pi form, sigma break, pi break, atom +1, atom -1, +2, -2) -> name
    (0, 0, 1, 0, 2, 0, 0, 0): "homolysis",
    (0, 0, 0, 1, 2, 0, 0, 0): "pi_homolysis",
    (1, 0, 0, 0, 0, 2, 0, 0): "recombination",
    (1, 0, 0, 1, 1, 1, 0, 0): "addition",
    (1, 0, 1, 0, 1, 1, 0, 0): "transfer (HAT, SH2)",
    (0, 1, 1, 0, 1, 1, 0, 0): "beta_scission",
    (1, 1, 1, 0, 0, 2, 0, 0): "disproportionation",
}


def radical_lines(raw: Path = RAW, limit: int | None = None) -> Iterator[tuple]:
    """(file, line number, tag, reaction) of every RS/RC line, streamed in file order."""
    n = 0
    for name in FILES:
        with open(raw / f"{name}.txt") as fh:
            for k, line in enumerate(fh, 1):
                rxn, _, tag = line.rstrip("\n").rpartition("|")
                if tag in TAGS:
                    yield name, k, tag, rxn
                    n += 1
                    if limit is not None and n >= limit:
                        return


def shape(mol, r) -> str:
    """The R shape's name (SHAPES) from its entries, sigma or pi by the reactant's orders."""
    from ksmech.arrows.delta_be import _order

    sig_f = sum(d > 0 and _order(mol, i, j) == 0 for i, j, d in r.bonds)
    pi_f = sum(d > 0 and _order(mol, i, j) > 0 for i, j, d in r.bonds)
    sig_b = sum(d < 0 and _order(mol, i, j) == 1 for i, j, d in r.bonds)
    pi_b = sum(d < 0 and _order(mol, i, j) > 1 for i, j, d in r.bonds)
    atoms = Counter(d for _, d in r.lone)
    key = (sig_f, pi_f, sig_b, pi_b, atoms[1], atoms[-1], atoms[2], atoms[-2])
    return SHAPES.get(key, "other: " + ",".join(map(str, key)))


def violations(mol, r, pool_u: int) -> list[str]:
    """The masks and screens of P9 that R breaks on the reactant drawing `mol` (docstring)."""
    from ksmech.arrows.apply import nonbonding_electrons

    out = []
    broken = {a for i, j, d in r.bonds if d < 0 for a in (i, j)}
    if pool_u == 0 and any(d == 1 and i not in broken for i, d in r.lone):
        out.append("a_closed_screen")
    if pool_u == 0 and any(d == -1 for _, d in r.lone):
        out.append("b_minus1_filter")
    n = [nonbonding_electrons(a) for a in mol.GetAtoms()]
    if any(n[i] + d < 0 for i, d in r.lone if d < 0):
        out.append("c_atom_mask")
    if not r.is_elementary(open_shell=True):
        out.append("d_alphabet")
    if len(r.bonds) > MAX_BONDS or len(r.lone) > MAX_ATOMS:
        out.append("e_caps")
    return out


def _pool_u(mol) -> int:
    from ksmech.data import chem, spin

    return sum(spin.spin_of_drawing(f)[0] for f in chem.fragments(mol))


def _labelled(row: dict, step, edit, out) -> dict:
    """The row of a step whose class passed: shape, drawing, violations, label maxima."""
    from ksmech.arrows.delta_be import smallest_variant

    k, mol, label = out
    drawn = violations(edit.mol, edit.r_data, _pool_u(edit.mol))
    each = [set(violations(mol, v, _pool_u(mol))) for v in label.variants]
    small = smallest_variant(label)
    return row | {
        "code": "LABEL_OK",
        "drawing": k,
        "shape": shape(mol, small),
        "drawn_shape": shape(edit.mol, edit.r_data),
        "drawn_violations": drawn,
        "class_violations": sorted(set.intersection(*each)),
        "maxima": [len(small.bonds), len(small.lone)],
        "drawn_maxima": [len(edit.r_data.bonds), len(edit.r_data.lone)],
        "u": _pool_u(edit.mol),
    }


def _step_row(step, row: dict, redraw: bool) -> dict:
    from ksmech.arrows.delta_be import EditSkip, edit_label, step_edit
    from ksmech.arrows.drawings import redrawn_label
    from ksmech.arrows.hydrogens import explicit_moving_hydrogens

    moved = explicit_moving_hydrogens(step)
    if moved is None:
        return row | {"code": "H_TRANSFER_AMBIGUOUS"}
    edit = step_edit(moved)
    if isinstance(edit, EditSkip):
        return row | {"code": str(edit)}
    out = redrawn_label(moved, edit) if redraw else edit_label(moved, edit)
    if isinstance(out, EditSkip):
        drawn = violations(edit.mol, edit.r_data, _pool_u(edit.mol))
        return row | {"code": str(out), "drawn_shape": shape(edit.mol, edit.r_data),
                      "drawn_violations": drawn}  # fmt: skip
    return _labelled(row, moved, edit, out if redraw else (0, edit.mol, out))


def flower_row(task: tuple) -> dict:
    from ksmech.data import chem
    from ksmech.data.resonance import StepKind, classify_step
    from ksmech.data.schema import ElementaryStep, Source

    name, k, tag, rxn = task
    row = {"id": f"{name}:{k}", "tag": tag}
    try:
        r, p = (chem.parse_mapped(s) for s in rxn.split(">>"))
    except (chem.SmilesParseError, chem.SanitizeError, ValueError) as exc:
        return row | {"code": "PARSE", "detail": str(exc)[:200]}
    kind = classify_step(r, p)
    if kind is not StepKind.STEP:
        return row | {"code": str(kind), "rxn": rxn}
    sides = (chem.canonical_mapped_smiles(r), chem.canonical_mapped_smiles(p))
    step = ElementaryStep(row["id"], Source.RMECHDB, None, None, *sides)
    out = _step_row(step, row, redraw=True)
    return out | ({"rxn": rxn} if out["code"] != "LABEL_OK" or out["class_violations"] else {})


def source_row(step) -> dict:
    from ksmech.data import chem

    row = {"id": step.step_id}
    out = _step_row(step, row, redraw=False)
    mol = chem.parse_mapped(step.reactants_smiles)
    return out | {"closed": _pool_u(mol) == 0}


def _summary(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["code"] == "LABEL_OK"]
    steps = [r for r in rows if r["code"] not in ("RESONANCE", "NULL_STEP")]
    listed = ("NONCONJUGATED_SHIFT", "DISSOCIATIVE_ET_STEP", "SET_STEP", "NOT_ELEMENTARY")
    return {
        "n_lines": len(rows),
        "codes": dict(Counter(r["code"] for r in rows).most_common()),
        "label_ok_fraction_of_non_resonance": len(ok) / len(steps) if steps else 0.0,
        "redrawn": sum(r.get("drawing", 0) > 0 for r in ok),
        "shapes": dict(Counter(r["shape"] for r in ok).most_common()),
        "drawn_shapes": dict(
            Counter(r.get("drawn_shape") for r in rows if "drawn_shape" in r).most_common()
        ),  # fmt: skip
        "drawn_violations": dict(Counter(v for r in rows for v in r.get("drawn_violations", ()))),
        "class_violations": dict(Counter(v for r in ok for v in r["class_violations"])),
        "drawn_maxima": [max((r["drawn_maxima"][i] for r in ok), default=0) for i in (0, 1)],
        "label_maxima": [max((r["maxima"][i] for r in ok), default=0) for i in (0, 1)],
        "listed": {c: [r for r in rows if r["code"] == c] for c in listed},
        "class_violating": [r for r in ok if r["class_violations"]],
    }


def run(source: str, limit: int | None, workers: int) -> dict:
    from ksmech.processes import in_processes

    if source == "flower":
        tasks = list(radical_lines(limit=limit))
        rows = list(in_processes(flower_row, tasks, workers))
        return {"source": source, **_summary(rows)}
    from ksmech.data.io import read_training_steps, steps_path

    steps = read_training_steps(steps_path(source))[:limit]
    rows = list(in_processes(source_row, steps, workers))
    closed = [r for r in rows if r["closed"]]
    return {"source": source, "closed": _summary(closed),
            "open": _summary([r for r in rows if not r["closed"]])}  # fmt: skip


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", default="flower", choices=("flower", "pmechdb", "reactmech"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True, help="a scratch directory")
    args = parser.parse_args(argv)
    report = run(args.source, args.limit, args.workers)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"label_dryrun_{args.source}.json"
    path.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    parts = [report] if args.source == "flower" else [report["closed"], report["open"]]
    for part in parts:
        print(json.dumps({k: v for k, v in part.items() if k not in ("listed", "class_violating")},
                         indent=1))  # fmt: skip
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
