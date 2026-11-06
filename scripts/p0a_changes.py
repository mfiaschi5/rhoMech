"""Every step the P0a build changes, removes, returns or flags, by rule (open-shell plan P0a,
step 7): `data/processed/<source>/p0a_changes.json`.

    uv run python scripts/p0a_changes.py --source reactmech [--old-dir data/processed_v17]

Run after the P0a re-ingest and before the QM, token and sample builds. The steps parquet of
`--old-dir` (the v17 copy) is compared step by step with the new one:

- DATIVE: the dative law re-charged the step (`charge_repair` rule 'D dative onium->M');
  `changed` were kept before and are kept now (their re-charged fragments need QM:
  `qm_steps`), `returned` were excluded before and are admitted now;
- PLACEHOLDER: the placeholder law excludes the step now (METAL_PLACEHOLDER): `removed` were
  kept, `relabelled` (by the old code) were excluded under another code; `returned` were
  METAL_PLACEHOLDER and are kept now without a dative repair;
- KEKULE_CLOSEST: a kept step without arrows whose R_data lost Kekule swaps
  (`delta_be.dataset_matrix`, after the moving hydrogens are written as atoms);
- METAL_SUSPECT: flagged in `domain_class` (the sample is unchanged: a slice, not a change);
- OTHER: any other difference in the sides, the exclusion code or the resonance flag (none is
  expected; a non-empty list means a change no rule explains).

`changed_steps` is every step whose sample can differ from its v17 one (every rule but
METAL_SUSPECT); the legacy reproduction gate compares the other samples per sample. Counts are
given per rule and per frozen molecule-split part ('-' for a step the snapshot does not hold).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import RDLogger

from ksmech.arrows.delta_be import EditSkip, dataset_matrix
from ksmech.arrows.hydrogens import explicit_moving_hydrogens
from ksmech.arrows.steps import load_frozen_split
from ksmech.data import provenance as prov
from ksmech.data.domain import DomainCode, domain_classes
from ksmech.data.io import DEFAULT_PROCESSED_DIR, read_steps, source_dir, steps_path

SCHEMA = 1
DATIVE_RULE = "D dative onium->M"
SAMPLE_RULES = ("DATIVE", "PLACEHOLDER", "KEKULE_CLOSEST", "OTHER")  # change a sample


def _fields(step) -> tuple:
    return (step.reactants_smiles, step.products_smiles, step.domain_exclusion, step.resonance_step)


def _domain_rule(old, new) -> tuple[str, str] | None:
    """(rule, kind) of the domain laws for one step, None when they leave it as it was."""
    was_kept, kept = old.domain_exclusion is None, new.domain_exclusion is None
    if DATIVE_RULE in (new.charge_repair or ""):
        return "DATIVE", "changed" if was_kept and kept else "returned" if kept else "excluded"
    if new.domain_exclusion == DomainCode.METAL_PLACEHOLDER:
        if was_kept:
            return "PLACEHOLDER", "removed"
        if old.domain_exclusion != DomainCode.METAL_PLACEHOLDER:
            return "PLACEHOLDER", f"relabelled from {old.domain_exclusion}"
        return None
    if old.domain_exclusion == DomainCode.METAL_PLACEHOLDER and kept:
        return "PLACEHOLDER", "returned"
    return None


def kekule_changed(step) -> bool:
    """A kept step without arrows whose R_data the Kekule-closest rule changed."""
    if step.arrows or step.domain_exclusion is not None or step.resonance_step:
        return False
    explicit = explicit_moving_hydrogens(step)
    out = dataset_matrix(explicit) if explicit is not None else EditSkip.EDIT_FAILED
    return not isinstance(out, EditSkip) and out[3] > 0


def classify(old_steps, new_steps) -> dict[str, dict[str, list[str]]]:
    """rule -> kind -> step ids, over the steps of both files (matched by step id)."""
    old = {s.step_id: s for s in old_steps}
    out: dict = defaultdict(lambda: defaultdict(list))
    for s in new_steps:
        o = old.get(s.step_id)
        if o is None:
            out["OTHER"]["new step"].append(s.step_id)
            continue
        rule = _domain_rule(o, s)
        if rule is not None:
            out[rule[0]][rule[1]].append(s.step_id)
        elif _fields(o) != _fields(s):
            out["OTHER"]["unexplained"].append(s.step_id)
        if "METAL_SUSPECT" in domain_classes(s.domain_class):
            out["METAL_SUSPECT"]["flagged"].append(s.step_id)
        if kekule_changed(s):
            out["KEKULE_CLOSEST"]["changed"].append(s.step_id)
    gone = sorted(set(old) - {s.step_id for s in new_steps})
    if gone:
        out["OTHER"]["step gone"] = gone
    return {r: {k: sorted(v) for k, v in kinds.items()} for r, kinds in out.items()}


def _parts(source: str, processed_dir: Path) -> dict[str, str]:
    frozen = load_frozen_split(source, "molecule", processed_dir) or {}
    return {i: name.split("_")[-1] for name, ids in frozen.items() for i in ids}


def counts(rules: dict, parts: dict[str, str]) -> dict:
    return {
        rule: {
            kind: dict(sorted(Counter(parts.get(i, "-") for i in ids).items())) | {"n": len(ids)}
            for kind, ids in kinds.items()
        }
        for rule, kinds in rules.items()
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", required=True)
    parser.add_argument("--old-dir", type=Path, default=Path("data/processed_v17"))
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    args = parser.parse_args(argv)
    RDLogger.DisableLog("rdApp.*")
    old_path = steps_path(args.source, args.old_dir)
    new_path = steps_path(args.source, args.processed_dir)
    rules = classify(read_steps(old_path), read_steps(new_path))
    changed = sorted({i for r in SAMPLE_RULES for ids in rules.get(r, {}).values() for i in ids})
    body = {
        "counts": counts(rules, _parts(args.source, args.processed_dir)),
        "qm_steps": rules.get("DATIVE", {}).get("changed", []),
        "changed_steps": changed,
        "rules": rules,
    }
    inputs = prov.file_input(old_path) | prov.file_input(new_path)
    content = prov.sha256_text(json.dumps(body, sort_keys=True))
    header = prov.stamp("p0a_changes", SCHEMA, inputs, {"source": args.source}, content)
    out = source_dir(args.source, args.processed_dir) / "p0a_changes.json"
    prov.write_json(out, body, header)
    print(json.dumps(body["counts"], indent=1))
    print(f"{len(changed)} steps whose sample can change; wrote {out}")
    return 1 if rules.get("OTHER") else 0


if __name__ == "__main__":
    raise SystemExit(main())
