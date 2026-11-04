"""Open-shell plan P6 exit checks on the v18 data: the edit labels, resonance flags and keys of
PMechDB and ReactMech under the code that runs this script, so a run under the P5 code and one
under the P6 code compare by content.

    uv run python scripts/p6_checks.py dump --source pmechdb --out FILE [--workers 3]
    uv run python scripts/p6_checks.py compare --old FILE --new FILE [--out FILE]

`dump` reads `data/processed/<source>/steps.parquet` (read-only) and writes, per step: the
mol-level edit label of every training step (`delta_be.edit_label` after
`hydrogens.explicit_moving_hydrogens`, as the sample builder labels it) or its skip code; the
resonance flag the code gives the two sides (`resonance.classify_step` under P6,
`resonance.is_resonance_step` before) beside the stored one; the identity key of each side
(`resonance.identity_key`, `sigma_key` before) and whether the pool is a closed shell (no atom
with odd N, no Hund-row species). It uses only functions both versions have, looked up by name.
`compare` lists every step whose label, flag or key differs between two dumps, split into
closed-shell and open-shell pools, and the sha256 over every closed-shell key of each dump.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

PROCESSED = Path("data/processed")


def _key(mol) -> str:
    from ksmech.data import resonance

    return getattr(resonance, "identity_key", resonance.sigma_key)(mol)


def _closed(mol) -> bool:
    """No atom with odd N, and (under P6) no species the table raises."""
    from ksmech.arrows.apply import nonbonding_electrons

    if any(nonbonding_electrons(a) % 2 for a in mol.GetAtoms()):
        return False
    try:
        from ksmech.data.spin import spin_of_drawing
    except ImportError:
        return True
    from ksmech.data import chem

    return not any(spin_of_drawing(f)[0] for f in chem.fragments(mol))


def _flag(r, p):
    from ksmech.data import resonance

    if hasattr(resonance, "classify_step"):
        return str(resonance.classify_step(r, p))
    return "RESONANCE" if resonance.is_resonance_step(r, p) else "STEP"


def _label(step) -> dict | str:
    from ksmech.arrows.delta_be import EditLabel, edit_label
    from ksmech.arrows.hydrogens import explicit_moving_hydrogens

    moved = explicit_moving_hydrogens(step)
    if moved is None:
        return "H_TRANSFER_AMBIGUOUS"
    out = edit_label(moved)
    return out.to_dict() if isinstance(out, EditLabel) else str(out)


def _row(task: tuple) -> tuple[str, dict]:
    from ksmech.data import chem

    step, training = task
    r, p = chem.parse_mapped(step.reactants_smiles), chem.parse_mapped(step.products_smiles)
    row = {
        "stored_resonance": bool(step.resonance_step),
        "flag": _flag(r, p),
        "keys": [_key(r), _key(p)],
        "closed": _closed(r) and _closed(p),
        "exclusion": step.domain_exclusion,
    }
    if training:
        row["label"] = _label(step)
    return step.step_id, row


def dump(source: str, out: Path, workers: int) -> None:
    from ksmech.data.io import read_steps, read_training_steps, steps_path

    path = steps_path(source, PROCESSED)
    training = {s.step_id for s in read_training_steps(path)}
    tasks = [(s, s.step_id in training) for s in read_steps(path)]
    with mp.get_context("spawn").Pool(workers) as pool:
        rows = dict(pool.imap(_row, tasks, chunksize=32))
    out.write_text(json.dumps({"source": source, "steps": rows}, sort_keys=True) + "\n")
    print(f"wrote {out}: {len(rows)} steps, {len(training)} training")


def _digest(rows: dict) -> str:
    keys = sorted(k for r in rows.values() if r["closed"] for k in r["keys"])
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()


def compare(old: Path, new: Path) -> dict:
    a, b = json.loads(old.read_text())["steps"], json.loads(new.read_text())["steps"]
    out: dict = {"n_steps": len(b), "only_old": sorted(a.keys() - b.keys())}
    for kind in ("closed", "open"):
        ids = sorted(i for i in a.keys() & b.keys() if b[i]["closed"] == (kind == "closed"))
        labels = [i for i in ids if "label" in b[i] and a[i].get("label") != b[i]["label"]]
        flags = [i for i in ids if (a[i]["flag"] == "RESONANCE") != (b[i]["flag"] == "RESONANCE")]
        stored = [i for i in ids if b[i]["stored_resonance"] != (b[i]["flag"] == "RESONANCE")]
        keys = [i for i in ids if a[i]["keys"] != b[i]["keys"]]
        out[kind] = {
            "n": len(ids),
            "n_labelled": sum("label" in b[i] for i in ids),
            "label_diffs": {i: [a[i].get("label"), b[i]["label"]] for i in labels},
            "resonance_flips": {i: [a[i]["flag"], b[i]["flag"]] for i in flags},
            "flag_vs_stored": stored,
            "key_diffs": {i: [a[i]["keys"], b[i]["keys"]] for i in keys},
            "codes": _codes(b, ids),
        }
    out["closed_key_digest"] = {"old": _digest(a), "new": _digest(b)}
    return out


def _codes(rows: dict, ids: list[str]) -> dict:
    out: dict[str, int] = {}
    for i in ids:
        out[rows[i]["flag"]] = out.get(rows[i]["flag"], 0) + 1
    return dict(sorted(out.items()))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--source", required=True)
    d.add_argument("--out", type=Path, required=True)
    d.add_argument("--workers", type=int, default=3)
    c = sub.add_parser("compare")
    c.add_argument("--old", type=Path, required=True)
    c.add_argument("--new", type=Path, required=True)
    c.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.command == "dump":
        dump(args.source, args.out, args.workers)
        return
    report = compare(args.old, args.new)
    text = json.dumps(report, indent=1, sort_keys=True)
    if args.out:
        args.out.write_text(text + "\n")
    summary = {k: v for k, v in report.items() if k not in ("closed", "open")}
    for kind in ("closed", "open"):
        r = report[kind]
        summary[kind] = {k: (len(v) if isinstance(v, dict | list) else v) for k, v in r.items()}
        summary[kind]["codes"] = r["codes"]
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
