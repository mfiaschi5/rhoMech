"""Model inputs from a collated batch (`arrows.batches.collate_samples`), none of them learned:
the standardized token features, the channels of the pair track on the (N + 1) x (N + 1) layout
(the MECH token at 0, then the batch tokens: atoms, then orbitals), and the atom-pair physics
y_ij of the decoder with the pooled gap factor G^atom (spec v9, "Edit decoder").

The pair features z_rs are stored at K = 1 (`physics.params.REDUCED`); here they are taken to
the fixed K and standardized with the training statistics (`pair_feature_stats_<split>.json`).
The token features are standardized with `FeatureStats`, fitted on the training samples.
G^atom and the pooled means run over the valid (occupied, virtual) pairs only (`pair_valid`):
the store keeps no others, and the donor's own antibond is not among them (decisions.md,
2026-10-25).
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch

from ksmech.physics.pairs import PairStats
from ksmech.physics.params import SCALING

MAX_Z = 118
CHARGE_MAX = 4  # formal charges are clipped to [-4, 4]
GRAPH_CAP = 6  # graph distances are capped at 6; another molecule counts as 6
DIST_BUCKETS = GRAPH_CAP + 2  # 0..6, and 7 for an atom of another molecule
KINDS = 5  # pair kind: atom-atom, atom-orbital, orbital-atom, orbital-orbital, MECH
Y_EXTRA = 3  # y_ij beyond the pooled features: log1p(pooled weight), asinh(G^atom), WBI


@dataclass(frozen=True)
class FeatureStats:
    """Mean and standard deviation of the continuous token inputs over a training set: atom
    features (missing entries left out), orbital features and spectral energies (entries with
    weight > 0)."""

    atom_mean: np.ndarray
    atom_std: np.ndarray
    orbital_mean: np.ndarray
    orbital_std: np.ndarray
    energy_mean: float
    energy_std: float

    def to_dict(self) -> dict:
        return {k: np.asarray(v).tolist() for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict) -> FeatureStats:
        arrays = {k: np.asarray(v, dtype=np.float64) for k, v in d.items()}
        return cls(**{k: (float(v) if v.ndim == 0 else v) for k, v in arrays.items()})


def _mean_std(x: np.ndarray, keep: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = keep.sum(0)
    s1 = np.where(keep, x, 0.0).sum(0)
    s2 = np.where(keep, x * x, 0.0).sum(0)
    mean = np.divide(s1, n, out=np.zeros_like(s1), where=n > 0)
    var = np.divide(s2, n, out=np.zeros_like(s2), where=n > 0) - mean**2
    std = np.sqrt(np.clip(var, 0.0, None))
    return mean, np.where(std > 0, std, 1.0)


def fit_feature_stats(samples: Iterable) -> FeatureStats:
    samples = list(samples)
    atoms = np.concatenate([s.atom_features for s in samples]).astype(np.float64)
    missing = np.concatenate([s.atom_missing for s in samples])
    orbitals = np.concatenate([s.orbital_features for s in samples]).astype(np.float64)
    spectra = np.concatenate([s.orbital_spectra.reshape(-1, 2) for s in samples])
    energies = spectra[spectra[:, 1] > 0, 0].astype(np.float64)
    am, asd = _mean_std(atoms, ~missing)
    om, osd = _mean_std(orbitals, np.ones_like(orbitals, dtype=bool))
    em, esd = _mean_std(energies[:, None], np.ones((len(energies), 1), dtype=bool))
    return FeatureStats(am, asd, om, osd, float(em[0]), float(esd[0]))


def to_device(batch: dict, device: torch.device) -> dict:
    return {
        k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }


def _t(x, like: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=like.device)


def token_inputs(batch: dict, stats: FeatureStats) -> dict:
    """Standardized atom inputs with their missing flags, orbital inputs, and spectral sets
    (energy standardized, weight as is) with the mask of their real entries."""
    a = batch["atom_features"]
    missing = batch["atom_missing"]
    ax = torch.where(missing, 0.0, (a - _t(stats.atom_mean, a)) / _t(stats.atom_std, a))
    n_atoms = a.shape[1]
    orb_real = batch["is_orbital"][:, n_atoms:]
    o = batch["orbital_features"]
    ox = (o - _t(stats.orbital_mean, o)) / _t(stats.orbital_std, o) * orb_real[..., None]
    e, p = batch["orbital_spectra"][..., 0], batch["orbital_spectra"][..., 1]
    real = p > 0
    e = torch.where(real, (e - stats.energy_mean) / stats.energy_std, 0.0)
    return {
        "atom_x": torch.cat([ax, missing.float()], -1),
        "orbital_x": ox,
        "spectra": torch.stack([e, p], -1),
        "spectra_mask": real,
    }


def standardized_pair_z(batch: dict, pair_stats: PairStats, dropped: Iterable[str], K: float):
    """(z_hat, defined), both (B, R, C, F): z_rs at the fixed K, standardized with the training
    statistics, 0 where undefined; the `dropped` features (ablation) are undefined everywhere."""
    names = tuple(batch["pair_features"])
    if names != tuple(pair_stats.names):
        raise ValueError(f"pair features {names}, statistics for {pair_stats.names}")
    z = batch["pair_z"]
    scale = [K ** SCALING[n][1] if SCALING[n][0] == "K" else 1.0 for n in names]
    defined = batch["pair_defined"].clone()
    for n in dropped:
        defined[..., names.index(n)] = False
    z = (z * _t(scale, z) - _t(pair_stats.mean, z)) / _t(
        np.where(pair_stats.std > 0, pair_stats.std, 1.0), z
    )
    return torch.where(defined, z, 0.0), defined


def _rows(incidence: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """(B, R, A) centre weights of the tokens at batch positions `pos` (-1: zero row)."""
    idx = pos.clamp(min=0)[..., None].expand(-1, -1, incidence.shape[-1])
    return torch.gather(incidence, 1, idx) * (pos >= 0)[..., None]


def _pool(w_occ: torch.Tensor, x: torch.Tensor, w_virt: torch.Tensor) -> torch.Tensor:
    """sum_rs (w_i^r w_j^s + w_j^r w_i^s) x_rs on atom pairs: (B, A, A, F)."""
    t = torch.einsum("bra,brcf->bacf", w_occ, x)
    t = torch.einsum("bacf,bcd->badf", t, w_virt)
    return t + t.transpose(1, 2)


def centre_weights(batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """(w_occ (B, R, A), w_virt (B, C, A)): the centre weights of the occupied and the virtual
    orbital tokens, in the layout of the pair block."""
    inc = batch["incidence"].float()
    return _rows(inc, batch["pair_row_pos"]), _rows(inc, batch["pair_col_pos"])


def pooled_gap(batch: dict, weights=None) -> torch.Tensor:
    """G^atom (B, A, A), Eq. (Gatom) over the valid (occupied, virtual) pairs: the gap factor G
    of every valid orbital pair, weighted by w_i^r w_j^s + w_j^r w_i^s."""
    w_occ, w_virt = weights or centre_weights(batch)
    g = batch["pair_z"][..., tuple(batch["pair_features"]).index("g")]
    g = torch.where(batch["pair_valid"], g.float(), 0.0)
    return _pool(w_occ, g[..., None], w_virt)[..., 0]


def atom_pair_physics(batch: dict, z_hat: torch.Tensor, defined: torch.Tensor):
    """(y, g_atom): y_ij (B, A, A, F + 3) holds the weighted mean of the standardized pair
    features over the valid orbital pairs the two atoms carry (spec, "Atom-pair physics"),
    log1p of the pooled weight, asinh(G^atom) and the Wiberg index; g_atom is `pooled_gap`."""
    w_occ, w_virt = centre_weights(batch)
    valid = batch["pair_valid"]
    g_atom = pooled_gap(batch, (w_occ, w_virt))
    m = (defined & valid[..., None]).float()
    num, den = _pool(w_occ, z_hat * m, w_virt), _pool(w_occ, m, w_virt)
    mean = torch.where(den > 0, num / den.clamp(min=1e-12), 0.0)
    weight = _pool(w_occ, valid[..., None].float(), w_virt)
    extra = [torch.log1p(weight), torch.asinh(g_atom)[..., None], batch["wiberg"][..., None]]
    return torch.cat([mean, *extra], -1), g_atom


def _kind_channels(batch: dict, n: int) -> torch.Tensor:
    """(B, n + 1, n + 1, KINDS) one-hot pair kind on the MECH layout."""
    atom, orb = batch["is_atom"], batch["is_orbital"]
    kinds = [
        atom[:, :, None] & atom[:, None, :],
        atom[:, :, None] & orb[:, None, :],
        orb[:, :, None] & atom[:, None, :],
        orb[:, :, None] & orb[:, None, :],
    ]
    out = torch.zeros(atom.shape[0], n + 1, n + 1, KINDS, device=atom.device)
    out[:, 1:, 1:, :4] = torch.stack(kinds, -1).float()
    out[:, 0, :, 4] = out[:, :, 0, 4] = 1.0
    return out


@contextlib.contextmanager
def frozen(device_type: str):
    """No gradient and no autocast: the inputs that are not learned, in float32."""
    with torch.no_grad(), torch.autocast(device_type, enabled=False):
        yield


def distances(pos: torch.Tensor) -> torch.Tensor:
    """(B, N, N) distances by the direct difference: cdist's matrix-product form loses ~1e-3 A
    near zero distance once a molecule sits away from the origin."""
    return torch.cdist(pos, pos, compute_mode="donot_use_mm_for_euclid_dist")


def rbf(r: torch.Tensor, n_rbf: int, r_max: float) -> torch.Tensor:
    """Gaussians of width r_max / (n_rbf - 1) centred on [0, r_max]: (..., n_rbf)."""
    mu = torch.linspace(0.0, r_max, n_rbf, device=r.device)
    return torch.exp(-(((r[..., None] - mu) / (r_max / (n_rbf - 1))) ** 2))


def pair_channel_count(n_rbf: int, n_features: int) -> int:
    return n_rbf + 2 * n_features + 1 + 1 + 3 + 1 + KINDS


def pair_channels(batch: dict, z_hat: torch.Tensor, defined: torch.Tensor, n_rbf: int, r_max):
    """(B, N + 1, N + 1, C) input of MLP_P (spec, "Pair track initialization"): RBF(distance)
    within a fragment; z_hat, its defined mask and the orientation (+1 occupied -> virtual, -1
    back) on the (occupied, virtual) orbital pairs, both orientations; the same-fragment flag;
    on atom pairs the bonded flag, bond order and min(graph distance, 6) / 6 (another molecule
    6); between an orbital and its centre atoms the centre weight; the pair kind."""
    same = batch["same_fragment"]
    b_sz, n = same.shape[:2]
    a = batch["atom_features"].shape[1]
    f = z_hat.shape[-1]
    out = torch.zeros(b_sz, n + 1, n + 1, pair_channel_count(n_rbf, f), device=same.device)
    pos = torch.cat([batch["atom_positions"], batch["orbital_centroids"]], 1).float()
    out[:, 1:, 1:, :n_rbf] = rbf(distances(pos), n_rbf, r_max) * same[..., None]
    c = n_rbf
    row, col = batch["pair_row_pos"], batch["pair_col_pos"]
    b, r, s = ((row[:, :, None] >= 0) & (col[:, None, :] >= 0)).nonzero(as_tuple=True)
    i, j = row[b, r] + 1, col[b, s] + 1
    vals = torch.cat([z_hat[b, r, s], defined[b, r, s].float()], -1)
    out[b, i, j, c : c + 2 * f] = vals
    out[b, j, i, c : c + 2 * f] = vals
    out[b, i, j, c + 2 * f], out[b, j, i, c + 2 * f] = 1.0, -1.0
    c += 2 * f + 1
    out[:, 1:, 1:, c] = same.float()
    order, dist = batch["bond_order"].float(), batch["graph_distance"].float()
    out[:, 1 : a + 1, 1 : a + 1, c + 1] = (order > 0).float()
    out[:, 1 : a + 1, 1 : a + 1, c + 2] = order
    capped = torch.where(dist < 0, float(GRAPH_CAP), dist.clamp(max=GRAPH_CAP)) / GRAPH_CAP
    out[:, 1 : a + 1, 1 : a + 1, c + 3] = capped
    w = batch["incidence"][:, a:, :].float()
    out[:, a + 1 :, 1 : a + 1, c + 4] = w
    out[:, 1 : a + 1, a + 1 :, c + 4] = w.transpose(1, 2)
    out[..., c + 5 :] = _kind_channels(batch, n)
    return out


@dataclass
class PairContext:
    """Per-token lookups on the MECH layout (0 = MECH, then the batch tokens) from which the
    pair inputs of any cell can be built without the N x N grid (`pair_channels_at`)."""

    pos: torch.Tensor  # (B, N + 1, 3) atom positions and orbital centroids, 0 for MECH
    frag: torch.Tensor  # (B, N + 1) fragment of a real token, -1 for MECH and padding
    is_atom: torch.Tensor  # (B, N + 1) real atom tokens
    is_orbital: torch.Tensor  # (B, N + 1) real orbital tokens
    occ_row: torch.Tensor  # (B, N + 1) row of an occupied token in the pair block, else -1
    virt_col: torch.Tensor  # (B, N + 1) column of a virtual token, else -1
    z: torch.Tensor  # (B, R, C, 2F): z_hat and its defined mask
    n_atoms: int  # atom slots are 1..n_atoms


def _block_index(pos: torch.Tensor, n: int) -> torch.Tensor:
    """(B, n + 1) the pair-block row (or column) of each MECH-layout token, -1 elsewhere."""
    out = torch.full((pos.shape[0], n + 1), -1, dtype=torch.long, device=pos.device)
    b, r = (pos >= 0).nonzero(as_tuple=True)
    out[b, pos[b, r] + 1] = r
    return out


def pair_context(batch: dict, z_hat: torch.Tensor, defined: torch.Tensor) -> PairContext:
    mask = batch["token_mask"]
    b_sz, n = mask.shape

    def mech(x, fill):
        return torch.cat([torch.full_like(x[:, :1], fill), x], 1)

    pos = torch.cat([batch["atom_positions"], batch["orbital_centroids"]], 1).float()
    return PairContext(
        pos=mech(pos, 0.0),
        frag=mech(torch.where(mask, batch["fragment"], -1), -1),
        is_atom=mech(batch["is_atom"], False),
        is_orbital=mech(batch["is_orbital"], False),
        occ_row=_block_index(batch["pair_row_pos"], n),
        virt_col=_block_index(batch["pair_col_pos"], n),
        z=torch.cat([z_hat, defined.float()], -1),
        n_atoms=batch["bond_order"].shape[1],
    )


def _z_at(ctx: PairContext, b, i, j):
    """z channels and orientation of cells (i, j): +1 from an occupied to a virtual token,
    -1 back, 0 (and zero z) elsewhere."""
    ri, cj, rj, ci = ctx.occ_row[b, i], ctx.virt_col[b, j], ctx.occ_row[b, j], ctx.virt_col[b, i]
    fwd, back = (ri >= 0) & (cj >= 0), (rj >= 0) & (ci >= 0)
    zf = ctx.z[b, ri.clamp(min=0), cj.clamp(min=0)]
    zb = ctx.z[b, rj.clamp(min=0), ci.clamp(min=0)]
    z = torch.where(fwd[..., None], zf, torch.where(back[..., None], zb, 0.0))
    return z, fwd.float() - back.float()


def pair_channels_at(ctx: PairContext, batch: dict, b, i, j, n_rbf: int, r_max: float):
    """The `pair_channels` vector of the cells (b, i, j) (index tensors of one shape S):
    (*S, C), equal to `pair_channels(...)[b, i, j]`."""
    a = ctx.n_atoms
    same = (ctx.frag[b, i] == ctx.frag[b, j]) & (ctx.frag[b, i] >= 0)
    r = (ctx.pos[b, i] - ctx.pos[b, j]).norm(dim=-1)
    z, orient = _z_at(ctx, b, i, j)
    slot_i, slot_j = (i >= 1) & (i <= a), (j >= 1) & (j <= a)
    aa = slot_i & slot_j
    ia, ja = (i - 1).clamp(0, a - 1), (j - 1).clamp(0, a - 1)
    order = torch.where(aa, batch["bond_order"][b, ia, ja].float(), 0.0)
    dist = batch["graph_distance"][b, ia, ja].float()
    capped = torch.where(dist < 0, float(GRAPH_CAP), dist.clamp(max=GRAPH_CAP)) / GRAPH_CAP
    inc = batch["incidence"].float()
    n = inc.shape[1]
    w_ij = inc[b, (i - 1).clamp(0, n - 1), ja]
    w_ji = inc[b, (j - 1).clamp(0, n - 1), ia]
    w = torch.where((i > a) & slot_j, w_ij, torch.where(slot_i & (j > a), w_ji, 0.0))
    at_i, at_j, orb_i, orb_j = (
        ctx.is_atom[b, i],
        ctx.is_atom[b, j],
        ctx.is_orbital[b, i],
        ctx.is_orbital[b, j],
    )
    kinds = [at_i & at_j, at_i & orb_j, orb_i & at_j, orb_i & orb_j, (i == 0) | (j == 0)]
    scalars = [orient, same, (order > 0) & aa, order, torch.where(aa, capped, 0.0), w, *kinds]
    return torch.cat(
        [rbf(r, n_rbf, r_max) * same[..., None], z, torch.stack([x.float() for x in scalars], -1)],
        -1,
    )


def atom_pair_static(batch: dict, n_rbf: int, r_max: float) -> torch.Tensor:
    """(B, A, A, n_rbf + 5) atom-pair inputs of the encoders without a pair track: RBF of the
    distance within a molecule, bonded flag, bond order, min(graph distance, 6) / 6 (another
    molecule 6), Wiberg index, same-molecule flag."""
    a = batch["bond_order"].shape[1]
    same = batch["same_fragment"][:, :a, :a]
    order, dist = batch["bond_order"].float(), batch["graph_distance"].float()
    capped = torch.where(dist < 0, float(GRAPH_CAP), dist.clamp(max=GRAPH_CAP)) / GRAPH_CAP
    scalars = [order > 0, order, capped, batch["wiberg"].float(), same]
    r = distances(batch["atom_positions"].float())
    return torch.cat(
        [rbf(r, n_rbf, r_max) * same[..., None], torch.stack([x.float() for x in scalars], -1)],
        -1,
    )
