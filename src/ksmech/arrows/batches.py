"""Batches of training samples for torch: the dataset over one split view, the padded
collation (`collate_samples`) and the DataLoader (`make_loader`).

Kept apart from `arrows.steps` so that the processes that only build, read or score samples
(the sample build, `ksmech-physics`) do not import torch (about 500 MB of resident memory per
process).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ksmech.arrows.lewis import graph_distances
from ksmech.arrows.steps import MechanismSample, load_samples


class SampleDataset(torch.utils.data.Dataset):
    def __init__(self, path: Path):
        self.samples = load_samples(path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> MechanismSample:
        return self.samples[i]


def _pad(arrays: list[np.ndarray], length: int, fill=0) -> torch.Tensor:
    """Stack arrays along a new batch axis, padding axis 0 to `length` with `fill`."""
    out = np.full((len(arrays), length, *arrays[0].shape[1:]), fill, dtype=arrays[0].dtype)
    for i, a in enumerate(arrays):
        out[i, : len(a)] = a
    return torch.from_numpy(out)


def _pad2(arrays: list[np.ndarray], rows: int, cols: int, fill=0) -> torch.Tensor:
    """Stack arrays along a new batch axis, padding axes 0 and 1 to (rows, cols) with `fill`."""
    out = np.full((len(arrays), rows, cols, *arrays[0].shape[2:]), fill, dtype=arrays[0].dtype)
    for i, a in enumerate(arrays):
        out[i, : a.shape[0], : a.shape[1]] = a
    return torch.from_numpy(out)


def _pair_tensors(batch: list[MechanismSample], n_atoms: int) -> dict:
    """The pair track's inputs, padded in each sample's (occupied r, virtual s) layout: z_rs at
    K = 1 in `pair_features` order (unstandardized: `params.to_params` at the physical K and
    the training statistics apply in the model), its defined mask, the validity filter, the
    sign of G, and the batch token position of every row and column (-1 on padding), for
    scattering into the N x N pair track. The stored atom-atom Coulomb matrix is no model input
    and is not collated."""
    names = batch[0].pair_features
    if any(s.pair_features != names for s in batch):
        raise ValueError("a batch mixes samples with different pair features")
    occ = [np.flatnonzero(s.orbital_occupied) for s in batch]
    virt = [np.flatnonzero(~s.orbital_occupied) for s in batch]
    for s, o, v in zip(batch, occ, virt, strict=True):
        if s.pair_valid.shape != (len(o), len(v)):
            raise ValueError(
                f"{s.step_id}: pair block {s.pair_valid.shape}, tokens {len(o)} x {len(v)}"
            )
    rows, cols = max(len(o) for o in occ), max(len(v) for v in virt)
    return {
        "pair_features": tuple(names),
        "pair_z": _pad2([s.pair_z for s in batch], rows, cols),
        "pair_defined": _pad2([s.pair_defined for s in batch], rows, cols, False),
        "pair_valid": _pad2([s.pair_valid for s in batch], rows, cols, False),
        "pair_gap_sign": _pad2([s.pair_gap_sign for s in batch], rows, cols),
        "pair_row_pos": _pad([n_atoms + o for o in occ], rows, -1),
        "pair_col_pos": _pad([n_atoms + v for v in virt], cols, -1),
    }


def _token_masks(batch: list[MechanismSample], n_atoms: int, n_orb: int) -> dict:
    """Per-token masks in the batch layout: atoms at [0, A), orbitals at [A, A + O)."""
    b, n = len(batch), n_atoms + n_orb
    m = {
        k: torch.zeros(b, n, dtype=torch.bool)
        for k in (
            "token_mask",
            "is_atom",
            "is_orbital",
            "occupied",
            "virtual",
            "donor_mask",
            "acceptor_mask",
        )
    }
    fragment = torch.full((b, n), -1, dtype=torch.long)
    for i, s in enumerate(batch):
        na, no = s.n_atoms, s.n_orbitals
        orb = slice(n_atoms, n_atoms + no)
        occ = torch.from_numpy(s.orbital_occupied)
        m["token_mask"][i, :na] = m["token_mask"][i, orb] = True
        m["is_atom"][i, :na] = True
        m["is_orbital"][i, orb] = True
        m["occupied"][i, orb], m["virtual"][i, orb] = occ, ~occ
        m["donor_mask"][i, orb] = torch.from_numpy(s.donor_mask)
        m["acceptor_mask"][i, orb] = torch.from_numpy(s.acceptor_mask)
        fragment[i, :na] = torch.from_numpy(s.atom_fragment)
        fragment[i, orb] = torch.from_numpy(s.orbital_fragment)
    real = m["token_mask"]
    m["fragment"] = fragment
    m["same_fragment"] = (
        (fragment[:, :, None] == fragment[:, None, :]) & real[:, :, None] & real[:, None, :]
    )
    return m


def _incidence_matrix(batch: list[MechanismSample], n_atoms: int, n_orb: int) -> torch.Tensor:
    """(B, N, A) token-to-centre weights: an orbital row holds w_a over its centres, an atom
    row is one-hot on itself (the site pointer's support)."""
    out = torch.zeros(len(batch), n_atoms + n_orb, n_atoms)
    for i, s in enumerate(batch):
        idx = torch.arange(s.n_atoms)
        out[i, idx, idx] = 1.0
        if len(s.incidence):
            j = torch.from_numpy(s.incidence[:, 0]).long() + n_atoms
            g = torch.from_numpy(s.incidence[:, 1]).long()
            out[i, j, g] = torch.from_numpy(s.incidence[:, 2])
    return out


def _label_tensors(batch: list[MechanismSample], n_atoms: int, n_orb: int) -> dict:
    """Multi-hot donor / acceptor sets over batch token positions and the site atoms."""
    b, n = len(batch), n_atoms + n_orb
    k = max(1, max(len(s.labels) for s in batch))
    out = {
        "donor_sets": torch.zeros(b, k, n, dtype=torch.bool),
        "acceptor_sets": torch.zeros(b, k, n, dtype=torch.bool),
        "donor_site": torch.full((b, k), -1, dtype=torch.long),
        "acceptor_site": torch.full((b, k), -1, dtype=torch.long),
        "label_mask": torch.zeros(b, k, dtype=torch.bool),
    }
    for i, s in enumerate(batch):

        def pos(t: int, s=s) -> int:
            return t if t < s.n_atoms else n_atoms + t - s.n_atoms

        for j, lab in enumerate(s.labels):
            out["donor_sets"][i, j, [pos(t) for t in lab.donor_ids]] = True
            out["acceptor_sets"][i, j, [pos(t) for t in lab.acceptor_ids]] = True
            out["donor_site"][i, j] = lab.donor_site
            out["acceptor_site"][i, j] = lab.acceptor_site
            out["label_mask"][i, j] = True
    return out


def _reactant_tensors(batch: list[MechanismSample], n_atoms: int) -> dict:
    """The reactant's Lewis structure on the padded atoms (bond orders, Wiberg indices, graph
    distances, -1 between molecules and on padding, lone pairs) and the decoder's masks: a bond
    may break, a pair of distinct atoms may gain an order while it stays at most 3, an atom
    with a lone pair may lose one (spec v8, Edit decoder)."""
    b = len(batch)
    order = torch.zeros(b, n_atoms, n_atoms, dtype=torch.int8)
    wiberg = torch.zeros(b, n_atoms, n_atoms)
    dist = torch.full((b, n_atoms, n_atoms), -1, dtype=torch.int16)
    lone = torch.zeros(b, n_atoms, dtype=torch.int8)
    real = torch.zeros(b, n_atoms, dtype=torch.bool)
    for k, s in enumerate(batch):
        i, j = (torch.from_numpy(s.bonds[:, c].astype(np.int64)) for c in (0, 1))
        order[k, i, j] = order[k, j, i] = torch.from_numpy(s.bonds[:, 2].astype(np.int8))
        wiberg[k, i, j] = wiberg[k, j, i] = torch.from_numpy(s.wiberg.astype(np.float32))
        na = s.n_atoms
        dist[k, :na, :na] = torch.from_numpy(graph_distances(s.bonds, na))
        lone[k, :na] = torch.from_numpy(s.lone_pairs)
        real[k, :na] = True
    pair = real[:, :, None] & real[:, None, :] & ~torch.eye(n_atoms, dtype=torch.bool)
    return {
        "bond_order": order,
        "wiberg": wiberg,
        "graph_distance": dist,
        "lone_pairs": lone,
        "can_break": order > 0,
        "can_form": pair & (order < 3),
        "can_lose_pair": lone > 0,
    }


ANCHOR_SIDE = {"form": 0, "break": 1}  # the last axis of `r_anchors`


def _edit_tensors(batch: list[MechanismSample], n_atoms: int) -> dict:
    """The reaction-matrix targets on the padded atoms, one slot per variant of the label
    class: the pair entries (-1, 0, +1, symmetric), the atom entries (-2, 0, +2) and the
    admissible anchors (last axis ANCHOR_SIDE); `variant_mask` marks the real slots."""
    b = len(batch)
    v = max([1] + [len(s.edit_label.variants) for s in batch if s.edit_label is not None])
    out = {
        "r_pairs": torch.zeros(b, v, n_atoms, n_atoms, dtype=torch.int8),
        "r_atoms": torch.zeros(b, v, n_atoms, dtype=torch.int8),
        "r_anchors": torch.zeros(b, v, n_atoms, n_atoms, 2, dtype=torch.bool),
        "variant_mask": torch.zeros(b, v, dtype=torch.bool),
    }
    for k, s in enumerate(batch):
        if s.edit_label is None:
            continue
        slots = zip(s.edit_label.variants, s.edit_label.anchors, strict=True)
        for x, (variant, anchors) in enumerate(slots):
            for i, j, d in variant.bonds:
                out["r_pairs"][k, x, i, j] = out["r_pairs"][k, x, j, i] = d
            for i, d in variant.lone:
                out["r_atoms"][k, x, i] = d
            for i, j, kind in anchors:
                c = ANCHOR_SIDE[kind]
                out["r_anchors"][k, x, i, j, c] = out["r_anchors"][k, x, j, i, c] = True
            out["variant_mask"][k, x] = True
    return out


def collate_samples(batch: list[MechanismSample]) -> dict:
    """Pad a batch. Token positions: atoms at [0, A), orbitals at [A, A + O), where A and O
    are the largest atom and orbital counts in the batch; sites index atoms directly. The
    pair features keep their (occupied, virtual) layout with the token positions of their rows
    and columns (`_pair_tensors`). The reaction-matrix targets and the reactant's structure
    are on the atoms (`_edit_tensors`, `_reactant_tensors`); the ArrowLabel tensors are empty
    for samples without them."""
    n_atoms = max(s.n_atoms for s in batch)
    n_orb = max(s.n_orbitals for s in batch)
    out = {
        "step_id": [s.step_id for s in batch],
        "product_key": [s.product_key for s in batch],
        "orbital_reading": [s.orbital_reading for s in batch],
        "atom_categorical": _pad([s.atom_categorical for s in batch], n_atoms),
        "atom_features": _pad([s.atom_features for s in batch], n_atoms),
        "atom_missing": _pad([s.atom_missing for s in batch], n_atoms, False),
        "atom_positions": _pad([s.atom_positions for s in batch], n_atoms),
        "orbital_features": _pad([s.orbital_features for s in batch], n_orb),
        "orbital_spectra": _pad([s.orbital_spectra for s in batch], n_orb),
        "orbital_centroids": _pad([s.orbital_centroids for s in batch], n_orb),
        "incidence": _incidence_matrix(batch, n_atoms, n_orb),
    }
    out.update(_token_masks(batch, n_atoms, n_orb))
    out.update(_label_tensors(batch, n_atoms, n_orb))
    out.update(_pair_tensors(batch, n_atoms))
    out.update(_reactant_tensors(batch, n_atoms))
    out.update(_edit_tensors(batch, n_atoms))
    return out


def make_loader(
    path: Path, batch_size: int = 16, shuffle: bool = True, **kwargs: Any
) -> torch.utils.data.DataLoader:
    """DataLoader over one samples file with `collate_samples`."""
    return torch.utils.data.DataLoader(
        SampleDataset(path),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_samples,
        **kwargs,
    )
