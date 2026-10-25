"""The empirical constants of the pair features, and how a feature depends on them.

    beta_mu,nu = K S_mu,nu (F_mu,mu + F_nu,nu) / 2   (Wolfsberg-Helmholz, GFN2 Fock diagonal
                                                       averaged per shell, `ks_terms.ao_energies`;
                                                       `ks_terms.resonance_integral`)
    KS2        = q_k q_l / (eps R_kl)                (the stored atom-atom Coulomb matrix only)

K enters the cross-centre coupling, eps the stored atom-atom Coulomb matrix (no model input).
The features are computed at K = eps = 1 (`REDUCED`) and each one depends on the parameters in
a fixed way (`SCALING`): the cross-centre coupling scales as K, every other feature is
parameter-free. `to_params` applies other values. Standardization statistics are fitted at the
physical values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

COULOMB_EV_ANG = 14.399645  # e^2 / (4 pi eps0) in eV Angstrom


@dataclass(frozen=True)
class KSParams:
    K: float = 1.75  # Wolfsberg-Helmholz constant
    eps: float = 4.0  # dielectric of the Coulomb term


REDUCED = KSParams(K=1.0, eps=1.0)

# pair feature -> (parameter, power):
#   feature(params) = feature(REDUCED) * params.<parameter> ** power
SCALING = {
    "g": ("K", 0),
    "log_abs_g": ("K", 0),
    "inter": ("K", 0),
    "bonded": ("K", 0),
    "bond_order": ("K", 0),
    "graph_distance": ("K", 0),
    "wbi": ("K", 0),
    "centroid_distance": ("K", 0),
    "cross_coupling": ("K", 1),
    "align_cos": ("K", 0),
    "dihedral_cos": ("K", 0),
}


def to_params(z_reduced: np.ndarray, params: KSParams, names) -> np.ndarray:
    """Features computed at `REDUCED` (last axis in the order of `names`) at `params`."""
    out = np.array(z_reduced, dtype=np.float64, copy=True)
    for k, name in enumerate(names):
        param, power = SCALING[name]
        out[..., k] *= float(getattr(params, param)) ** power
    return out
