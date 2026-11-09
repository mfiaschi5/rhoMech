"""Permutation importance of the token inputs (`ksmech-eval importance`): which part of an atom
or orbital token a trained model relies on, measured on a whole part of a split.

Each group of token input columns is permuted across the real tokens of its kind in a batch
(atoms with atoms, orbitals with orbitals, mixing molecules), every other input left as it is,
and the part is scored again with `train.loop.evaluate`; the importance of a group is how much
the anchor NLL rises and the anchor top-1 / top-3 and the greedy-R match fall against the
unperturbed baseline, averaged over `repeats` permutations. A group is one column, or a block
of related columns (BLOCKS) permuted together with one permutation, so a block keeps its joint
distribution. Permuting rather than zeroing keeps every input inside its training distribution.

What a group can reach: an atom group changes the atom token and, through the centre weights,
the embedding every orbital inherits from it; an orbital group changes the orbital token only.
The pair features (G, the cross-centre coupling, the Lewis-structure channels) are never
touched: their ablation is `ksmech-train fit --drop-pair-features`. The atomic number and the
formal charge are the categorical groups `element` and `formal_charge`; the spectral set is
`spectral_set` (whole sets permuted between orbitals). `spin_role` is no input column here: it
also decides which pair cells are open-shell cells (`model.inputs.open_cells`).

Batches are the cell-budget batches of training, in a fixed order, so a seed gives the same
permutations on every run.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import torch

from ksmech.train.loop import evaluate

IMPORTANCE_VERSION = 1
METRICS = ("anchor_nll", "top1", "top3", "r_exact")
NOT_PERMUTED = ("spin_role",)  # also a mask: `model.inputs.open_cells`
ATOM_CATEGORICAL = {"element": 0, "formal_charge": 1}
SPECTRAL = "spectral_set"
BLOCKS = {
    "atom": {
        "fukui": (
            "f_plus_fmo",
            "f_minus_fmo",
            "f_plus_fmo_exact",
            "f_minus_fmo_exact",
            "f_plus",
            "f_minus",
            "dual",
        ),  # fmt: skip
        "frontier": tuple(
            f"fmo{x}_{lv}"
            for x in ("", "_exact")
            for lv in ("homo_minus_1", "homo", "lumo", "lumo_plus_1")
        ),  # fmt: skip
    },
    "orbital": {
        "centre_weights": ("n_centers", *(f"center_weight_{k}" for k in range(4))),
        "spd_shares": tuple(f"center_{k}_{x}" for k in range(4) for x in "spd"),
        "s_shares": tuple(f"center_{k}_s" for k in range(4)),
        "p_shares": tuple(f"center_{k}_p" for k in range(4)),
        "d_shares": tuple(f"center_{k}_d" for k in range(4)),
        "two_centre": ("cos_a", "cos_b", "overlap_population", "distance_ab"),
        "energy": ("energy_ev", "retained_fraction"),
    },
}


@dataclass(frozen=True)
class Group:
    """Token inputs permuted together: `kind` atom | orbital | atom_categorical | spectral and
    the column names (atom and orbital columns) or the categorical column's index."""

    name: str
    kind: str
    columns: tuple = ()


def groups(atom_names: Iterable[str], orbital_names: Iterable[str], scope: str = "both"):
    """The groups of a model's inputs: with scope `columns` each column alone, the categorical
    atom inputs and the spectral set; `blocks` the BLOCKS (columns the model does not read
    left out, a block with fewer than two left dropped); `both` all of them."""
    atoms, orbitals = tuple(atom_names), tuple(n for n in orbital_names if n not in NOT_PERMUTED)
    out = []
    if scope in ("columns", "both"):
        out += [Group(n, "atom_categorical", (k,)) for n, k in ATOM_CATEGORICAL.items()]
        out += [Group(n, "atom", (n,)) for n in atoms]
        out += [Group(n, "orbital", (n,)) for n in orbitals]
        out.append(Group(SPECTRAL, "spectral"))
    if scope in ("blocks", "both"):
        for kind, names in (("atom", atoms), ("orbital", orbitals)):
            for block, cols in BLOCKS[kind].items():
                kept = tuple(c for c in cols if c in names)
                if len(kept) >= 2:
                    out.append(Group(f"{kind}:{block}", kind, kept))
    if scope not in ("columns", "blocks", "both"):
        raise ValueError(f"scope {scope!r}, expected columns, blocks or both")
    return out


def _real_tokens(batch: dict, kind: str) -> torch.Tensor:
    """(B, A) the real atoms, or (B, O) the real orbitals, of a padded batch."""
    n_atoms = batch["atom_categorical"].shape[1]
    if kind in ("atom", "atom_categorical"):
        return batch["is_atom"][:, :n_atoms].bool()
    return batch["is_orbital"][:, n_atoms:].bool()


def _permute_rows(x: torch.Tensor, real: torch.Tensor, cols, gen: torch.Generator):
    """A copy of x (B, T, ...) with the entries `cols` of the last axis (all when None) moved
    between the real tokens by one random permutation."""
    out = x.clone()
    rows = out[real]
    perm = torch.randperm(len(rows), generator=gen)
    if cols is None:
        out[real] = rows[perm]
    else:
        rows[:, cols] = rows[perm][:, cols]
        out[real] = rows
    return out


def permute_group(batch: dict, group: Group, gen: torch.Generator) -> dict:
    """The batch with the group's inputs permuted across the real tokens of its kind; the
    tensors it does not touch are shared with `batch`. An atom column moves with its missing
    flag."""
    out = dict(batch)
    real = _real_tokens(batch, group.kind)
    if group.kind == "atom_categorical":
        cat = batch["atom_categorical"]
        out["atom_categorical"] = _permute_rows(cat, real, list(group.columns), gen)
    elif group.kind == "spectral":
        out["orbital_spectra"] = _permute_rows(batch["orbital_spectra"], real, None, gen)
    elif group.kind == "atom":
        cols = [tuple(batch["atom_columns"]).index(c) for c in group.columns]
        state = gen.get_state()
        out["atom_features"] = _permute_rows(batch["atom_features"], real, cols, gen)
        gen.set_state(state)  # the same permutation for the missing flags
        out["atom_missing"] = _permute_rows(batch["atom_missing"], real, cols, gen)
    elif group.kind == "orbital":
        cols = [tuple(batch["orbital_columns"]).index(c) for c in group.columns]
        out["orbital_features"] = _permute_rows(batch["orbital_features"], real, cols, gen)
    else:
        raise ValueError(f"group kind {group.kind!r}")
    return out


class Perturbed:
    """A loader whose every batch is passed through `fn(batch, gen)`, one generator seeded
    once per pass, so a pass is reproducible."""

    def __init__(self, loader, fn: Callable[[dict, torch.Generator], dict], seed: int):
        self.loader, self.fn, self.seed = loader, fn, seed

    def __iter__(self):
        gen = torch.Generator().manual_seed(self.seed)
        for batch in self.loader:
            yield self.fn(batch, gen)

    def __len__(self) -> int:
        return len(self.loader)


def _deltas(scores: list[dict], base: dict) -> dict:
    """Mean and SD over the repeats of each metric's change against the baseline (NLL up and
    hit rates down are both reported as they are: perturbed minus baseline)."""
    out = {}
    for m in METRICS:
        d = np.array([s[m] - base[m] for s in scores])
        out[f"d_{m}"] = float(d.mean())
        out[f"d_{m}_sd"] = float(d.std(ddof=1)) if len(d) > 1 else 0.0
    return out


def importance(model, loader, group_list, repeats: int, seed: int, device, dtype, log=None):
    """The baseline metrics of `evaluate` and, per group, the change of METRICS when it is
    permuted (`_deltas`), the groups sorted by the rise of the anchor NLL."""
    base = evaluate(model, loader, device, dtype)
    rows = []
    for i, g in enumerate(group_list):
        fn = lambda b, gen, g=g: permute_group(b, g, gen)  # noqa: E731
        runs = [Perturbed(loader, fn, seed + 1000 * r) for r in range(repeats)]
        scores = [evaluate(model, run, device, dtype) for run in runs]
        row = {"group": g.name, "kind": g.kind, "columns": list(g.columns)}
        rows.append(row | _deltas(scores, base))
        if log:
            log(i + 1, len(group_list), rows[-1])
    rows.sort(key=lambda r: -r["d_anchor_nll"])
    keep = ("n", "loss", *METRICS, "top5")
    return {"baseline": {k: base[k] for k in keep}, "groups": rows}
