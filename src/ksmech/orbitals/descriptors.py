"""Continuous physical descriptors of one localized orbital: the model input of an orbital token.

Nothing here is categorical. An orbital is described by its occupation, how its Mulliken
population spreads over atoms (centres), the s/p/d composition on each centre, the geometry of
its two main centres (p-vector alignment with the axis, overlap population, distance), its
spatial extent, and its spectral decomposition over the canonical MOs of its block:
p_rk = U_kr^2, the weight of canonical MO k in localized orbital r. The model sees the 8
largest spectral entries plus three scalars of the full spectrum (mean, variance, retained
fraction); the full spectral vector is kept as bookkeeping for the gap kernel. sigma / pi /
lone-pair labels are diagnostics computed elsewhere and never enter `to_vector`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

import numpy as np

from ksmech.orbitals.classify import ClassifyConfig, overlap_population, p_vector

MODEL_INPUT = "model_input"  # dataclass field metadata key


@dataclass(frozen=True)
class DescriptorConfig:
    tau_atom: float = 0.05  # an atom is a centre when it carries at least this population
    top_centers: int = 4  # centres kept in the feature vector (zero-padded)
    spectral_min: float = 0.01  # spectral entries below this weight are dropped
    spectral_max: int = 8  # at most this many spectral entries per token, renormalized
    min_p_norm: float = ClassifyConfig.min_p_norm  # an atom with less p character has no axis


def _in(default=None, **kw):
    return field(default=default, metadata={MODEL_INPUT: True}, **kw)


def _diag(default=None, **kw):
    return field(default=default, metadata={MODEL_INPUT: False}, **kw)


@dataclass
class OrbitalDescriptors:
    """Model-input fields carry `model_input=True` metadata; the others are bookkeeping."""

    occupation: float = _in(0.0)  # spectral average of the canonical occupations (2, 0, ~1)
    n_centers: int = _in(0)  # atoms with population >= tau_atom
    center_weights: np.ndarray = _in(None)  # (top_centers,) sorted populations, zero-padded
    center_spd: np.ndarray = _in(None)  # (top_centers, 3) s/p/d composition on each centre
    cos_a: float = _in(0.0)  # |cos| between the p vector on centre a and the a-b axis
    cos_b: float = _in(0.0)  # the same on centre b; 0 for one-centre orbitals or no p
    n_axis_centers: int = _in(0)  # how many of a, b carry p character (cos defined)
    overlap_sign: float = _in(0.0)  # sign of the a-b Mulliken overlap population
    overlap_magnitude: float = _in(0.0)
    distance_ab: float = _in(0.0)  # Angstrom
    radius_gyration: float = _in(0.0)  # population-weighted, centres about the centroid
    spectral_energies: np.ndarray = _in(None)  # (spectral_max,) eV, zero-padded
    spectral_weights: np.ndarray = _in(None)  # (spectral_max,) renormalized, zero-padded
    spectral_mean: float = _in(0.0)  # sum_k p_rk E_k over the full spectrum (eV) = E_r
    spectral_variance: float = _in(0.0)  # sum_k p_rk (E_k - mean)^2 (eV^2)
    retained_fraction: float = _in(0.0)  # weight of the kept entries before renormalization
    # bookkeeping, not model input
    center_atoms: np.ndarray = _diag(None)  # (top_centers,) atom indices, -1 padded
    center_s_coeff: np.ndarray = _diag(None)  # (top_centers,) |s AO coefficient| per centre
    center_p_norm: np.ndarray = _diag(None)  # (top_centers,) norm of the p vector per centre
    spectral_indices: np.ndarray = _diag(None)  # (spectral_max,) canonical MO indices, -1
    full_energies: np.ndarray = _diag(None)  # (n_block,) canonical energies of the block, eV
    full_weights: np.ndarray = _diag(None)  # (n_block,) p_rk = U_kr^2, sums to 1
    energy_ev: float = _diag(0.0)  # E_r = sum_k p_rk E_k, a diagnostic scalar
    centroid: np.ndarray = _diag(None)  # (3,) Angstrom

    def cos_character(self) -> float:
        """Mean |cos| over the centres that carry p character: 1 is axis-aligned (sigma-like),
        0 pi-like; 0 when neither centre has p (an s-s bond or a one-centre orbital)."""
        if not self.n_axis_centers:
            return 0.0
        return (self.cos_a + self.cos_b) / self.n_axis_centers

    def n_spectral(self) -> int:
        return int(np.count_nonzero(self.spectral_weights))

    def to_vector(self) -> np.ndarray:
        """Flat float vector of the model-input fields, in `feature_names()` order."""
        parts = []
        for f in fields(self):
            if not f.metadata.get(MODEL_INPUT):
                continue
            parts.append(np.asarray(getattr(self, f.name), dtype=float).ravel())
        return np.concatenate(parts)


def feature_names(config: DescriptorConfig | None = None) -> list[str]:
    """One name per entry of `OrbitalDescriptors.to_vector()`."""
    config = config or DescriptorConfig()
    k, m = config.top_centers, config.spectral_max
    names: list[str] = []
    for f in fields(OrbitalDescriptors):
        if not f.metadata.get(MODEL_INPUT):
            continue
        if f.name == "center_weights":
            names += [f"center_weight_{i}" for i in range(k)]
        elif f.name == "center_spd":
            names += [f"center_{i}_{ell}" for i in range(k) for ell in "spd"]
        elif f.name in ("spectral_energies", "spectral_weights"):
            names += [f"{f.name[:-1]}_{i}" for i in range(m)]
        else:
            names.append(f.name)
    return names


def spectral_entries(
    u_column: np.ndarray, energies_ev: np.ndarray, mo_indices: np.ndarray, config: DescriptorConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """(indices, energies, weights, retained) of the canonical MOs that make up one LMO.

    Weights p_k = U_k^2 are kept when >= `spectral_min`, at most `spectral_max` of them (largest
    first), renormalized to sum 1 and zero-padded to `spectral_max`. `retained` is their total
    weight before renormalization.
    """
    p = u_column**2
    order = np.argsort(-p, kind="stable")
    keep = [k for k in order[: config.spectral_max] if p[k] >= config.spectral_min] or [order[0]]
    retained = float(p[keep].sum())
    w = p[keep] / retained
    m = config.spectral_max
    idx = np.full(m, -1, dtype=np.int32)
    e = np.zeros(m)
    weights = np.zeros(m)
    idx[: len(keep)] = mo_indices[keep]
    e[: len(keep)] = energies_ev[keep]
    weights[: len(keep)] = w
    return idx, e, weights, retained


def center_composition(c, S, ao_atom, ao_l, atom: int) -> np.ndarray:
    """s, p, d composition of the orbital's gross population on one atom (sums to 1).

    Gross AO populations c_mu (S c)_mu use the full S, so tails on other atoms count through
    the overlap, exactly as in the atom populations.
    """
    gross = c * (S @ c)
    rows = ao_atom == atom
    total = float(gross[rows].sum())
    if abs(total) < 1e-12:
        return np.zeros(3)
    return np.array([float(gross[rows & (ao_l == ell)].sum()) / total for ell in (0, 1, 2)])


def center_coefficients(c, ao_atom, ao_l, atom: int) -> tuple[float, float]:
    """(|s coefficient|, p-vector norm) of the orbital on one atom; d functions are ignored."""
    s_idx = np.flatnonzero((ao_atom == atom) & (ao_l == 0))
    s_coeff = abs(float(c[s_idx[0]])) if len(s_idx) else 0.0
    vec = p_vector(c, ao_atom, ao_l, atom)
    return s_coeff, 0.0 if vec is None else float(np.linalg.norm(vec))


def _axis_terms(c, S, ao_atom, ao_l, coords, a: int, b: int, config: DescriptorConfig):
    """cos_a, cos_b, axis centres, overlap sign, overlap magnitude, distance for (a, b)."""
    axis = coords[b] - coords[a]
    dist = float(np.linalg.norm(axis))
    axis = axis / dist
    cos, n_axis = [], 0
    for atom in (a, b):
        vec = p_vector(c, ao_atom, ao_l, atom)
        norm = 0.0 if vec is None else float(np.linalg.norm(vec))
        has_p = norm >= config.min_p_norm
        n_axis += has_p
        cos.append(abs(float(vec @ axis)) / norm if has_p else 0.0)
    op = overlap_population(c, S, ao_atom, a, b)
    return cos[0], cos[1], n_axis, float(np.sign(op)), abs(op), dist


def radius_of_gyration(weights: np.ndarray, coords: np.ndarray, centroid: np.ndarray) -> float:
    w = np.clip(weights, 0.0, None)
    if w.sum() <= 0:
        return 0.0
    return float(np.sqrt((w * ((coords - centroid) ** 2).sum(axis=1)).sum() / w.sum()))


def _center_arrays(c, S, ao_atom, ao_l, populations, top, k):
    """Zero-padded per-centre weights, s/p/d composition, |s| coefficient and p norm."""
    center_w, s_coeff, p_norm = np.zeros(k), np.zeros(k), np.zeros(k)
    spd = np.zeros((k, 3))
    center_w[: len(top)] = populations[top]
    for i, atom in enumerate(top):
        spd[i] = center_composition(c, S, ao_atom, ao_l, atom)
        s_coeff[i], p_norm[i] = center_coefficients(c, ao_atom, ao_l, atom)
    atoms = np.full(k, -1, dtype=np.int32)
    atoms[: len(top)] = top
    return center_w, spd, s_coeff, p_norm, atoms


def build_descriptors(
    c: np.ndarray,
    populations: np.ndarray,
    S: np.ndarray,
    ao_atom: np.ndarray,
    ao_l: np.ndarray,
    coords: np.ndarray,
    u_column: np.ndarray,
    block_energies_ev: np.ndarray,
    block_occupations: np.ndarray,
    block_indices: np.ndarray,
    config: DescriptorConfig | None = None,
) -> OrbitalDescriptors:
    """Descriptors of one LMO `c` whose block rotation column is `u_column`."""
    config = config or DescriptorConfig()
    order = [int(i) for i in np.argsort(-populations, kind="stable")]
    centers = [i for i in order if populations[i] >= config.tau_atom] or order[:1]
    top = centers[: config.top_centers]
    w_clip = np.clip(populations, 0.0, None)
    centroid = (w_clip @ coords) / w_clip.sum() if w_clip.sum() > 0 else coords.mean(axis=0)
    center_w, spd, s_coeff, p_norm, atoms = _center_arrays(
        c, S, ao_atom, ao_l, populations, top, config.top_centers
    )
    axis = (0.0, 0.0, 0, 0.0, 0.0, 0.0)
    if len(centers) >= 2:
        axis = _axis_terms(c, S, ao_atom, ao_l, coords, centers[0], centers[1], config)
    idx, e, p, retained = spectral_entries(u_column, block_energies_ev, block_indices, config)
    full_p = u_column**2
    mean = float(full_p @ block_energies_ev)
    return OrbitalDescriptors(
        occupation=float(full_p @ block_occupations),
        n_centers=len(centers),
        center_weights=center_w,
        center_spd=spd,
        cos_a=axis[0],
        cos_b=axis[1],
        n_axis_centers=int(axis[2]),
        overlap_sign=axis[3],
        overlap_magnitude=axis[4],
        distance_ab=axis[5],
        radius_gyration=radius_of_gyration(populations[centers], coords[centers], centroid),
        spectral_energies=e,
        spectral_weights=p,
        spectral_mean=mean,
        spectral_variance=float(full_p @ (block_energies_ev - mean) ** 2),
        retained_fraction=retained,
        center_atoms=atoms,
        center_s_coeff=s_coeff,
        center_p_norm=p_norm,
        spectral_indices=idx,
        full_energies=np.asarray(block_energies_ev, dtype=float).copy(),
        full_weights=full_p.copy(),
        energy_ev=mean,
        centroid=centroid,
    )
