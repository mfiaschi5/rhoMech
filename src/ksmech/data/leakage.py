"""The rmechdb leakage audit (open-shell plan P8 step 6), in both directions, by the molecule
split's species rule (`splits.step_molecules`: non-trivial means more than 3 heavy atoms) with
species matched through the identity key v2 (`resonance.identity_key`), so a species matches
however it is drawn (O=O and [O][O], a radical's resonance forms).

`ksmech-data leakage-audit` writes `leakage_audit.json` under rmechdb's folder. Held out on
the rmechdb side: the RC val and test parts of rc_molecule. Held out on the PMechDB side: the
val and test parts of closed_base (the pinned molecule-split cohort, P0b). The report lists
every overlapping step id:

- RC val/test steps sharing a non-trivial species with PMechDB (training part, and any step),
  ReactMech (any step) and mech-USPTO (any reaction of its raw file); and those whose
  map-free reaction key (v2) is a PMechDB or ReactMech step;
- rmechdb training steps (RC train and RS) sharing a species with closed_base val/test: they
  leave joint training (LEAK_GUARD, the cohort `leak_guard_joint`);
- PMechDB training steps sharing a species with RC val/test: a slice, never removed, so the
  PMechDB training set is the same in pm_control and joint (the cohort `rc_heldout_shared`);
- closed_base val/test steps sharing a species with rmechdb training: closed_base is
  reported with and without them (the cohort `closed_base_rmechdb_overlap`); RC val/test with
  and without its overlap with PMechDB training (`rc_heldout_overlap`).
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping
from functools import cache
from pathlib import Path

from rdkit import Chem, rdBase

from ksmech.data import chem
from ksmech.data.schema import ElementaryStep

MECH_USPTO_RAW = Path("data/raw/mech_uspto/mech-USPTO-31k.csv")
SpeciesSets = Mapping[str, frozenset[str]]  # step id -> its non-trivial species keys (v2)


@cache
def species_key(smiles: str) -> str | None:
    """The identity key v2 of one map-free species, None when it is trivial (3 heavy atoms or
    fewer) or does not parse."""
    from ksmech.data.resonance import identity_key

    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumHeavyAtoms() <= chem.TRIVIAL_MAX_HEAVY_ATOMS:
        return None
    return identity_key(mol)


def side_species(smiles: str) -> frozenset[str]:
    """The non-trivial species keys (v2) of one mapped side."""
    out = set()
    for frag in chem.fragments(chem.parse_mapped(smiles)):
        key = species_key(chem.unmapped_canonical(frag))
        if key is not None:
            out.add(key)
    return frozenset(out)


def step_species(steps: Iterable[ElementaryStep]) -> dict[str, frozenset[str]]:
    return {
        s.step_id: side_species(s.reactants_smiles) | side_species(s.products_smiles) for s in steps
    }


@cache
def _any_key(smiles: str) -> str:
    """The identity key v2 of any map-free species (its SMILES when RDKit cannot read it)."""
    from ksmech.data.resonance import identity_key

    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
    return smiles if mol is None else identity_key(mol)


def reaction_key(step: ElementaryStep) -> str:
    """The map-free reaction key v2: each side's species keys (trivial ones included) sorted."""
    sides = []
    for smi in (step.reactants_smiles, step.products_smiles):
        frags = chem.fragments(chem.parse_mapped(smi))
        sides.append(".".join(sorted(_any_key(chem.unmapped_canonical(f)) for f in frags)))
    return ">>".join(sides)


AUDIT_SCHEMA = 1
COHORTS = {  # cohort name -> (its source, the report block that holds its ids)
    "leak_guard_joint": ("rmechdb", "rmechdb_train_shares_closed_base_heldout"),
    "rc_heldout_overlap": ("rmechdb", "rc_heldout_shares_pmechdb_train"),
    "rc_heldout_shared": ("pmechdb", "pmechdb_train_shares_rc_heldout"),
    "closed_base_rmechdb_overlap": ("pmechdb", "closed_base_heldout_shares_rmechdb_train"),
}


def write_audit(processed_dir: Path, seed: int = 0, cohorts: bool = False) -> tuple[Path, dict]:
    """`leakage_audit.json` under rmechdb's folder (`run_audit`), and with `cohorts` the four
    cohorts of `COHORTS` (each written once, `eval.cohorts`)."""
    from ksmech.data import provenance as prov
    from ksmech.data.io import source_dir, steps_path
    from ksmech.eval.cohorts import write_cohort

    report = run_audit(processed_dir, seed)
    inputs: dict[str, str] = {}
    for source in ("rmechdb", "pmechdb", "reactmech"):
        inputs |= prov.file_input(steps_path(source, processed_dir))
    content = prov.sha256_text(json.dumps(report, sort_keys=True))
    header = prov.stamp("leakage_audit", AUDIT_SCHEMA, inputs, {"seed": seed}, content)
    out = prov.write_json(source_dir("rmechdb", processed_dir) / "leakage_audit.json", report,
                          header)  # fmt: skip
    if cohorts:
        for name, (source, block) in COHORTS.items():
            note = {"from": "leakage_audit.json", "block": block}
            write_cohort(source, name, report[block]["ids"], note, processed_dir)
    return out, report


def sharing(ids: Iterable[str], of: SpeciesSets, against: frozenset[str]) -> list[str]:
    """The ids whose species meet `against`."""
    return sorted(i for i in ids if of[i] & against)


def union(of: SpeciesSets, ids: Iterable[str]) -> frozenset[str]:
    return frozenset().union(*(of[i] for i in ids))


def mech_uspto_species(path: Path = MECH_USPTO_RAW) -> frozenset[str] | None:
    """Every non-trivial species key of mech-USPTO's `updated_reaction` column; None when the
    raw file is absent. Read only: the benchmark is never trained on."""
    if not path.is_file():
        return None
    out: set[str] = set()
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            for side in row["updated_reaction"].split(">>"):
                try:
                    out |= side_species(side)
                except (chem.SmilesParseError, chem.SanitizeError, ValueError):
                    continue
    return frozenset(out)


def _block(ids: list[str]) -> dict:
    return {"n": len(ids), "ids": ids}


def audit(
    rc: Mapping[str, list[str]],
    rm_train: list[str],
    closed: Mapping[str, list[str]],
    species: SpeciesSets,
    others: Mapping[str, frozenset[str] | None],
    reactions: Mapping[str, str] | None = None,
    other_reactions: Mapping[str, frozenset[str]] | None = None,
) -> dict:
    """The report of the module docstring from the parts (step ids) and the species sets:
    `rc` RC val/test, `rm_train` rmechdb training, `closed` closed_base train/val/test,
    `others` the species of every other source (name -> keys; a missing source is None),
    `reactions` the RC val/test steps' reaction keys and `other_reactions` every other
    source's."""
    held = rc["val"] + rc["test"]
    closed_held = closed["val"] + closed["test"]
    out = {
        "rc_heldout": {p: len(rc[p]) for p in ("val", "test")},
        "rc_heldout_shares_pmechdb_train": _block(
            sharing(held, species, union(species, closed["train"]))
        ),
        "rmechdb_train_shares_closed_base_heldout": _block(
            sharing(rm_train, species, union(species, closed_held))
        ),
        "pmechdb_train_shares_rc_heldout": _block(
            sharing(closed["train"], species, union(species, held))
        ),
        "closed_base_heldout_shares_rmechdb_train": _block(
            sharing(closed_held, species, union(species, rm_train))
        ),
    }
    for name, keys in others.items():
        found = None if keys is None else _block(sharing(held, species, keys))
        out[f"rc_heldout_shares_{name}"] = found
    for name, keys in (other_reactions or {}).items():
        out[f"rc_heldout_reaction_in_{name}"] = _block(
            sorted(i for i in held if (reactions or {}).get(i) in keys)
        )
    return out


def _parts(ids: Mapping[str, list[str]], strategy: str) -> dict[str, list[str]]:
    return {p: ids[f"{strategy}_{p}"] for p in ("train", "val", "test")}


def run_audit(processed_dir: Path, seed: int = 0) -> dict:
    """`audit` over the stored steps: rmechdb's rc_molecule parts (its frozen snapshot when
    there is one), closed_base's parts of PMechDB's frozen molecule split, ReactMech's steps and
    mech-USPTO's raw file."""
    from ksmech.arrows.steps import load_frozen_split, split_ids
    from ksmech.data.io import read_steps, read_training_steps, steps_path
    from ksmech.eval.cohorts import load_cohort

    rm = read_training_steps(steps_path("rmechdb", processed_dir))
    frozen_rm = load_frozen_split("rmechdb", "rc_molecule", processed_dir)
    rc = _parts(split_ids(rm, "rc_molecule", seed, frozen_rm), "rc_molecule")
    rm_train = list(rc["train"])
    base = load_cohort("pmechdb", "closed_base", processed_dir)
    frozen = load_frozen_split("pmechdb", "molecule", processed_dir)
    closed = {p: sorted(set(frozen[f"molecule_{p}"]) & base) for p in ("train", "val", "test")}
    pm = read_steps(steps_path("pmechdb", processed_dir))
    rx = read_steps(steps_path("reactmech", processed_dir))
    wanted = set(rm_train) | set(rc["val"]) | set(rc["test"]) | base
    species = step_species([s for s in rm + pm if s.step_id in wanted])
    held = {s.step_id: s for s in rm if s.step_id in set(rc["val"]) | set(rc["test"])}
    others = {
        "pmechdb_any": union(step_species(pm), [s.step_id for s in pm]),
        "reactmech_any": union(step_species(rx), [s.step_id for s in rx]),
        "mech_uspto": mech_uspto_species(),
    }
    reactions = {i: reaction_key(s) for i, s in held.items()}
    other_reactions = {
        "pmechdb": frozenset(map(reaction_key, pm)),
        "reactmech": frozenset(map(reaction_key, rx)),
    }
    out = audit(rc, rm_train, closed, species, others, reactions, other_reactions)
    return {"seed": seed, "rule": "non-trivial species (> 3 heavy atoms), identity key v2"} | out
