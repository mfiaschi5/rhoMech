"""Contact resonance integral between two tokens of any two molecules (spec section 5.4).

    beta_rs = K * a(r, s) * (F_r + F_s) / 2

F_r is the Fock diagonal (eV) of the dominant shell on r's main centre, the AO-level
Wolfsberg-Helmholz energy: an LMO's own Fock diagonal is negative for occupied and often
positive for virtual tokens, so their mean can cross zero and would erase the coupling. The
amplitude a(r, s) comes from the configured kernel:

- `table` (default): S[kappa] * w_m(r) * w_m(s), the contact-table overlap of the two main
  centres' element and shell, times each token's population on its main centre.
- `analytic`: the tblite two-atom probe of `ksmech.physics.contact` (|s| and p-vector norms of
  every centre pair at the covalent-radii distance, best pair), which already carries the
  orbital coefficients.
- `mlp` (an MLP over the two descriptor vectors, fitted to the same statistics) belongs to the
  ablation study and is not implemented.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ksmech.orbitals.tokens import MoleculeTokens, OrbitalKind, OrbitalToken
from ksmech.physics.contact import contact_coupling
from ksmech.physics.contact_table import ContactTable, contact_key
from ksmech.physics.params import KSParams, shell_fock_ev

KERNELS = ("table", "analytic", "mlp")


@dataclass(frozen=True)
class ContactConfig:
    kernel: str = "table"


def _z(t: MoleculeTokens) -> np.ndarray:
    return np.array([a.atomic_number for a in t.atoms])


def main_shell_fock(o: OrbitalToken, t: MoleculeTokens) -> float:
    """Fock diagonal (eV) of the dominant shell on the token's main centre."""
    d = o.descriptors
    return shell_fock_ev(t.F, t.ao_atom, t.ao_l, int(d.center_atoms[0]), contact_key(o, _z(t))[1])


class ContactKernel:
    def __init__(
        self, tables: dict[str, ContactTable] | None = None, config: ContactConfig | None = None
    ) -> None:
        self.config = config or ContactConfig()
        if self.config.kernel not in KERNELS:
            raise ValueError(f"unknown contact kernel {self.config.kernel!r}; one of {KERNELS}")
        if self.config.kernel == "mlp":
            raise NotImplementedError("the MLP contact kernel belongs to the ablation phase")
        if self.config.kernel == "table" and tables is None:
            raise ValueError("the table kernel needs contact tables (`load_tables`)")
        self.tables = tables

    def amplitude(
        self, r: OrbitalToken, t_r: MoleculeTokens, s: OrbitalToken, t_s: MoleculeTokens
    ) -> float:
        """Dimensionless contact amplitude of an occupied r with a token s of either block."""
        if self.config.kernel == "analytic":
            return contact_coupling(r.descriptors, _z(t_r), s.descriptors, _z(t_s))
        kind = "occ_occ" if s.kind == OrbitalKind.OCCUPIED else "occ_virt"
        overlap, _ = self.tables[kind].lookup((*contact_key(r, _z(t_r)), *contact_key(s, _z(t_s))))
        return overlap * r.descriptors.center_weights[0] * s.descriptors.center_weights[0]

    def beta(
        self,
        r: OrbitalToken,
        t_r: MoleculeTokens,
        s: OrbitalToken,
        t_s: MoleculeTokens,
        params: KSParams | None = None,
    ) -> float:
        """Contact resonance integral (eV)."""
        params = params or KSParams()
        f_mean = 0.5 * (main_shell_fock(r, t_r) + main_shell_fock(s, t_s))
        return params.K * self.amplitude(r, t_r, s, t_s) * f_mean
