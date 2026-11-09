"""The previous step (reaction-network plan, item H): a step seen from its child.

If B_child = B_parent + R, the previous step seen from the child is -R, an elementary step on the
same atoms under the same laws. A backward sample is the forward sample builder
(`arrows.steps.sample_for_step`) run on the step with its two sides swapped (`backward_step`):

- input: the child pool as the dataset draws it, with its tokens, Lewis structure, pair features
  and resonance drawings (each with its label recomputed, `arrows.drawings`);
- target: the edit-label class of the swapped step (`arrows.delta_be`), so of -R up to
  resonance: the matrices that take the drawn child to a resonance form of the parent, fewest
  formal charges in that form first, then fewest entries, closed under the child pool's
  symmetry (`arrows.symmetry`). The forward arrows are dropped: they draw the forward electron
  flow, so the swapped step takes the path of a source without arrows (`delta_be.mapped_closest`
  on the two mapped sides, KEKULE_CLOSEST); there is no second label path;
- product key: the identity key v2 of the parent pool.

Its anchors are the bonds -R forms, i.e. those the forward step broke; when the forward step
broke none (a pure addition), -R forms none and its breaking bonds are the anchors
(`ReactionMatrix.anchors`), as the loss and the decoder already allow for a heterolysis.

The class is not always the negated forward class: the forward class picks the product form its
own rule prefers, the backward one starts from the drawn product (`label_agreement` counts how
often they agree, `docs/decisions.md`, 2026-11-09).

A backward store lives in its own directory, `data/processed/<source>/backward/`
(`direction_dir`): shards, index, views, split files, pair statistics and reports under the
same names as the forward ones, so a forward artifact is never overwritten or made stale. Its
parts are the forward parts step by step (the same split functions on the forward steps, the
same frozen snapshots), so one split serves both directions; the molecule split groups the
molecules of both sides, which `molecule_leakage` checks on the backward inputs.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import replace
from pathlib import Path

from ksmech.data.io import DEFAULT_PROCESSED_DIR, source_dir
from ksmech.data.schema import ElementaryStep

log = logging.getLogger(__name__)

FORWARD, BACKWARD = "forward", "backward"
DIRECTIONS = (FORWARD, BACKWARD)


def check_direction(direction: str) -> str:
    """`direction` when it is one of DIRECTIONS; ValueError otherwise."""
    if direction not in DIRECTIONS:
        raise ValueError(f"direction {direction!r}: one of {DIRECTIONS}")
    return direction


def backward_step(step: ElementaryStep) -> ElementaryStep:
    """The step seen from its child: the two sides swapped and no arrows (they draw the forward
    flow), the step id and every other field kept."""
    return replace(
        step,
        reactants_smiles=step.products_smiles,
        products_smiles=step.reactants_smiles,
        arrows=[],
    )


def oriented(steps: Iterable[ElementaryStep], direction: str) -> list[ElementaryStep]:
    """The steps as the samples of `direction` read them (forward: unchanged)."""
    steps = list(steps)
    return steps if check_direction(direction) == FORWARD else [backward_step(s) for s in steps]


def direction_dir(
    source: str, processed_dir: Path = DEFAULT_PROCESSED_DIR, direction: str = FORWARD
) -> Path:
    """Where the samples of `direction` live: the source's directory (forward), or its
    `backward/` subdirectory."""
    root = source_dir(source, processed_dir)
    return root if check_direction(direction) == FORWARD else root / BACKWARD


def direction_config(direction: str, limit: int = 0) -> dict:
    """The provenance config a store, view or statistics file of `direction` adds: nothing for
    the forward samples (bit for bit as before), the direction (and a pilot's step limit) for
    the backward ones."""
    if check_direction(direction) == FORWARD:
        return {}
    return {"direction": direction} | ({"limit": limit} if limit else {})


def view_direction(path: Path) -> str:
    """The direction a sample view (or any artifact of `direction_config`) was written for:
    the forward ones carry none."""
    from ksmech.data import provenance as prov

    header = prov.read_header(path)
    return (header.config if header else {}).get("direction", FORWARD)


def require_direction(path: Path, direction: str) -> Path:
    """`path` when it was written for `direction` (or does not exist: its reader reports
    that); ValueError otherwise, so the samples of the two directions are never read one for
    the other."""
    if not Path(path).exists():
        return path
    found = view_direction(path)
    if found != check_direction(direction):
        raise ValueError(f"{path} holds {found} samples, not {direction} ones")
    return path


def pilot_steps(steps: list[ElementaryStep], limit: int, seed: int = 0) -> list[ElementaryStep]:
    """`limit` training steps drawn at random (seeded), in their parquet order: a pilot's
    subset. All of them when `limit` is 0 or not below their number."""
    import numpy as np

    if not limit or limit >= len(steps):
        return list(steps)
    keep = set(np.random.default_rng(seed).choice(len(steps), size=limit, replace=False).tolist())
    return [s for k, s in enumerate(steps) if k in keep]


def molecule_leakage(
    steps: list[ElementaryStep], ids: Mapping[str, list[str]], direction: str = BACKWARD
) -> dict:
    """Per held-out part of one strategy's split (`<strategy>_val`, `<strategy>_test`), the
    steps whose input pool (the products when backward, the reactants when forward) holds a
    non-trivial molecule (`data.splits.step_molecules`' grouping: more than 3 heavy atoms) that
    a step of the training part holds on either side. Zero for the molecule split by
    construction."""
    from ksmech.data import chem
    from ksmech.data.splits import step_molecules

    by_id = {s.step_id: s for s in steps}
    train_key = next((k for k in ids if k.endswith("_train")), None)
    seen: set[str] = set()
    for sid in ids.get(train_key, []):
        s = by_id[sid]
        seen |= step_molecules(s.reactants_smiles, s.products_smiles)
    out = {}
    for name, members in ids.items():
        if name.endswith("_train"):
            continue
        sides = [by_id[i].products_smiles if direction == BACKWARD else by_id[i].reactants_smiles
                 for i in members]  # fmt: skip
        leaks = [i for i, side in zip(members, sides, strict=True)
                 if set(chem.nontrivial_species(side)) & seen]  # fmt: skip
        out[name] = {"n_steps": len(members), "n_leaking": len(leaks), "examples": leaks[:5]}
    return out


AGREEMENT = ("equal", "overlap", "disjoint", "unmapped")


def _by_maps(label, smiles: str, sign: int) -> set | None:
    """The members of `label` (on the mol atoms of `smiles`) as sets of map-numbered entries
    times `sign`; None when a member touches an atom without a map number."""
    from ksmech.data import chem

    maps = {a.GetIdx(): a.GetAtomMapNum() for a in chem.parse_mapped(smiles).GetAtoms()}
    out = set()
    for v in label.variants:
        atoms = [a for i, j, _ in v.bonds for a in (i, j)] + [i for i, _ in v.lone]
        if not all(maps[a] for a in atoms):
            return None
        bonds = frozenset((*sorted((maps[i], maps[j])), sign * d) for i, j, d in v.bonds)
        out.add((bonds, frozenset((maps[i], sign * d) for i, d in v.lone)))
    return out


def label_agreement(forward: tuple, backward: tuple) -> str:
    """How a step's backward label class compares with its forward class (`delta_be.mol_label`
    of the step and of its `backward_step`: (step, label), each label on its own reactant
    side's mol atoms), through the atom maps: `equal` (the backward class is the negated
    forward class), `overlap`, `disjoint`, or `unmapped` when a member touches an atom without
    a map number."""
    fwd = _by_maps(forward[1], forward[0].reactants_smiles, -1)
    bwd = _by_maps(backward[1], backward[0].reactants_smiles, 1)
    if fwd is None or bwd is None:
        return "unmapped"
    return "equal" if fwd == bwd else "overlap" if fwd & bwd else "disjoint"


def backward_agreement(steps: Iterable[ElementaryStep]) -> dict:
    """`label_agreement` over `steps` on the mol atoms (no tokens): the count of each kind,
    `unlabelled` when a direction has no label, and up to five step ids of each kind."""
    from ksmech.arrows.delta_be import mol_label

    counts: Counter = Counter()
    examples: dict[str, list[str]] = {}
    for s in steps:
        fwd, bwd = mol_label(s), mol_label(backward_step(s))
        kind = "unlabelled" if isinstance(fwd, str) or isinstance(bwd, str) else None
        kind = kind or label_agreement(fwd, bwd)
        counts[kind] += 1
        if len(examples.setdefault(kind, [])) < 5:
            examples[kind].append(s.step_id)
    return {"agreement": dict(counts.most_common()), "agreement_examples": examples}


def _step_keys(steps: list[ElementaryStep], side: str) -> list[frozenset]:
    """The fragment keys (`reactants._fragment_keys`; None for a fragment that cannot be
    canonicalized) of each step's `side`."""
    from ksmech.arrows.reactants import _fragment_keys
    from ksmech.data import chem

    out = []
    for s in steps:
        smiles = s.products_smiles if side == "products" else s.reactants_smiles
        out.append(frozenset(key for *_, key in _fragment_keys(chem.parse_mapped(smiles))))
    return out


UNCANONICALIZABLE = "UNCANONICALIZABLE"  # a fragment without a cache key: never tokenized
CHUNK = 64  # keys per task of the coverage check


def _token_worker(task: tuple) -> dict[str, str | None]:
    from ksmech.arrows.reactants import MISSING_TOKENS

    keys, loader = task
    reason = getattr(loader, "missing_reason", lambda key: MISSING_TOKENS)
    return {k: None if loader(k) is not None else reason(k) for k in keys}


def token_status(keys: Iterable[str], loader, n_workers: int = 1) -> dict[str, str | None]:
    """Key -> None when the token loader (`reactants.TokenCache`) serves its tokens with their
    Wiberg sidecar, else its reason (MISSING_TOKENS, STALE_ARTIFACT, NO_BOND_INDEX), over
    `n_workers` spawned processes when above 1."""
    from ksmech.processes import chunked, in_processes

    tasks = [(chunk, loader) for chunk in chunked(sorted(keys), CHUNK)]
    out: dict[str, str | None] = {}
    for part in in_processes(_token_worker, tasks, n_workers):
        out.update(part)
    return out


def qm_status(keys: Iterable[str], manifest: Path | None) -> dict[str, str]:
    """Key -> `ok` when the QM manifest has a record of it, `failed:<reason>` when its rows all
    failed, `absent` when it has no row (or there is no manifest)."""
    import pandas as pd

    rows: dict[str, str] = {}
    if manifest is not None and Path(manifest).exists():
        df = pd.read_parquet(manifest, columns=["smiles", "status", "reason"])
        for smiles, status, reason in df.itertuples(index=False):
            if rows.get(smiles) != "ok":
                rows[smiles] = "ok" if status == "ok" else f"failed:{reason}"
    return {k: rows.get(k, "absent") for k in keys}


def species_coverage(
    steps: list[ElementaryStep], loader, manifest: Path | None = None, n_workers: int = 1
) -> dict:
    """Token and QM coverage of the species of each side of `steps` (reaction-network plan, H):
    per side the unique fragment keys, the token loader's reasons for those it does not serve,
    their QM status (`qm_status`), and the steps whose side it fully serves; then the product
    species no reactant side holds, which only a backward sample reads."""
    keys = {side: _step_keys(steps, side) for side in ("reactants", "products")}
    every = {k for side in keys.values() for ks in side for k in ks if k is not None}
    tokens = token_status(every, loader, n_workers)
    qm = qm_status(every, manifest)
    status: dict = {k: tokens[k] or "ok" for k in every} | {None: UNCANONICALIZABLE}
    out: dict = {"n_steps": len(steps)}
    for side, per_step in keys.items():
        out[side] = _side_report(set().union(*per_step), status, qm)
        out[side]["n_steps_served"] = sum(all(status[k] == "ok" for k in ks) for ks in per_step)
    reactants = set().union(*keys["reactants"])
    only = set().union(*keys["products"]) - reactants
    out["products_only"] = _side_report(only, status, qm)
    return out


def _side_report(keys: set, status: dict, qm: dict) -> dict:
    """The token and QM coverage of a set of fragment keys (`species_coverage`)."""
    missing = {k: status[k] for k in keys if status[k] != "ok"}
    no_qm = {k: qm.get(k, "absent") for k in keys if k is not None and qm.get(k) != "ok"}
    return {
        "n_species": len(keys),
        "n_with_tokens": len(keys) - len(missing),
        "missing_tokens": dict(Counter(missing.values()).most_common()),
        "n_without_qm": len(no_qm),
        "without_qm": dict(Counter(no_qm.values()).most_common()),
        "examples_missing": sorted(str(k) for k in missing)[:10],
    }
