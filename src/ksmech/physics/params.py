"""The two empirical constants of the Klopman-Salem terms, and the resonance integral.

    beta_mu,nu = K S_mu,nu (F_mu,mu + F_nu,nu) / 2   (Wolfsberg-Helmholz, GFN2 Fock diagonal
                                                       averaged per shell, `oriented.ao_energies`;
                                                       `ks_terms.resonance_integral`)
    KS2        = q_k q_l / (eps R_kl)                (between fragments, R at the vdW contact;
                                                       within one, q q / eps (1/R_c - 1/R))

K and eps are shared learnable scalars initialized at their textbook values (1.75 and 4.0):
the prior is the functional form, not two numbers. The pair features are computed at K = eps =
1 (`REDUCED`) and every one depends on the parameters in a fixed way (`SCALING`): the log
features shift additively (log beta^2 and log|KS3| by 2 log K, log|KS1| by log K), KS2 scales
as 1 / eps and the cross-centre coupling as K. `to_params` applies a learned K and eps.
Standardization statistics must be frozen at the initial values: statistics recomputed at the
current K would cancel the learned shift again. `tau` (1 eV at initialization, learnable,
positive) is the frozen head's temperature, -(KS3 + KS2 + KS1) / tau; it enters no feature.
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
    tau: float = 1.0  # eV: temperature of the frozen head's logits -(KS3 + KS2 + KS1) / tau


REDUCED = KSParams(K=1.0, eps=1.0)

# pair feature -> (parameter, kind, power):
#   "log": feature(params) = feature(REDUCED) + power * log(params.<parameter>)
#   "mul": feature(params) = feature(REDUCED) * params.<parameter> ** power
SCALING = {
    "log_beta2_d2.0": ("K", "log", 2),
    "log_beta2_d2.3": ("K", "log", 2),
    "log_beta2_d3.0": ("K", "log", 2),
    "log_abs_g": ("K", "log", 0),
    "log_abs_ks3": ("K", "log", 2),
    "ks2": ("eps", "mul", -1),
    "log_abs_ks1": ("K", "log", 1),
    "inter": ("K", "mul", 0),
    "centroid_distance": ("K", "mul", 0),
    "cross_coupling": ("K", "mul", 1),
    "clash": ("K", "mul", 0),
}


def to_params(z_reduced: np.ndarray, params: KSParams, names) -> np.ndarray:
    """Features computed at `REDUCED` (last axis in the order of `names`) at `params`."""
    out = np.array(z_reduced, dtype=np.float64, copy=True)
    for k, name in enumerate(names):
        param, kind, power = SCALING[name]
        value = float(getattr(params, param))
        if kind == "log":
            out[..., k] += power * np.log(value)
        else:
            out[..., k] *= value**power
    return out


def shell_fock_ev(F: np.ndarray, ao_atom: np.ndarray, ao_l: np.ndarray, atom: int, ell: int):
    """Mean Fock diagonal (eV) of the AOs of shell `ell` on `atom`; NaN if there is none."""
    rows = (ao_atom == atom) & (ao_l == ell)
    if not rows.any():
        return float("nan")
    return float(np.diag(F)[rows].mean() * HARTREE_TO_EV)
