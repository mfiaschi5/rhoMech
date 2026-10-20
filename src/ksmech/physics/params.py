"""The two empirical constants of the Klopman-Salem terms, and the resonance integral.

    beta_mu,nu = K S_mu,nu (F_mu,mu + F_nu,nu) / 2   (Wolfsberg-Helmholz, GFN2 Fock diagonal;
                                                       `ks_terms.resonance_integral`)
    KS2        = q_k q_l / (eps R_kl)

K and eps are shared learnable scalars initialized at their textbook values (1.75 and 4.0):
the prior is the functional form, not two numbers. Every pair feature is a fixed power of
them (`SCALING`), so features are computed in numpy at any `KSParams` and the model can apply a
learned K and eps as `reduced * scale(params, name)`, with `reduced` computed at K = eps = 1.
Standardization statistics must be frozen at the initial values: statistics recomputed at the
current K would divide the learned scale out again.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ksmech.qm.fukui import HARTREE_TO_EV

COULOMB_EV_ANG = 14.399645  # e^2 / (4 pi eps0) in eV Angstrom


@dataclass(frozen=True)
class KSParams:
    K: float = 1.75  # Wolfsberg-Helmholz constant
    eps: float = 4.0  # dielectric of the Coulomb term


REDUCED = KSParams(K=1.0, eps=1.0)

# pair feature -> (parameter, power): feature(params) = feature(REDUCED) * params.<p> ** power
SCALING = {
    "ks3": ("K", 2),
    "coupling_sq": ("K", 2),
    "gap_kernel": ("K", 0),
    "ks2": ("eps", -1),
    "ks1": ("K", 1),
    "inter": ("K", 0),
    "distance": ("K", 0),
}


def scale(params: KSParams, name: str) -> float:
    """Factor that turns a feature computed at `REDUCED` into its value at `params`."""
    param, power = SCALING[name]
    return float(getattr(params, param)) ** power


def shell_fock_ev(F: np.ndarray, ao_atom: np.ndarray, ao_l: np.ndarray, atom: int, ell: int):
    """Mean Fock diagonal (eV) of the AOs of shell `ell` on `atom`; NaN if there is none."""
    rows = (ao_atom == atom) & (ao_l == ell)
    if not rows.any():
        return float("nan")
    return float(np.diag(F)[rows].mean() * HARTREE_TO_EV)
