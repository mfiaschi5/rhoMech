"""Contact resonance integral between two tokens of any two molecules (spec section 5.4).

Default kernel `oriented` (Phase 5b, brought forward 2026-10-20):

    beta_rs = K * sum_mu,nu l_mu,r l_nu,s S^c_mu,nu (F_mu,mu + F_nu,nu) / 2

the spec's AO-level Wolfsberg-Helmholz sum, with S^c the AO overlaps of the two LMOs placed with
their lobes antiparallel, the acceptor's site at distance d from the donor's site
(`ksmech.physics.contact.legacy_fixed23`, tblite's own basis, the acceptor on its local cap,
directions and rolls in local frames), and F_mu,mu each molecule's own AO Fock diagonal (eV).
The placement is the one with the largest |S_rs|, maximized over the donor's centres x the
acceptor's centres (the pair-track value; the argmax sites and the CLASH flag are in
`oriented_details`), and beta is taken in the phase that makes S_rs positive. The amplitude
a(r, s) is |S_rs|. (Until 2026-10-20 beta was K S_rs (E_r + E_s)/2 with the LMO energies: a
virtual LMO above 0 eV, such as sigma*(H-Br) at +8.8 eV, cancelled the occupied one's energy
and gave a near-zero beta.) Pass `same=True` for a pair of one molecule: the oriented block
then applies the intramolecular site rule (`legacy_fixed23.block_contacts`).

Ablation kernels, kept for comparison only:

    beta_rs = K * a(r, s) * (F_r + F_s) / 2,  F the Fock diagonal of the dominant shell on the
                                              main centre (AO-level Wolfsberg-Helmholz)

- `table`: a = S[kappa] w_m(r) w_m(s), the median overlap of intramolecular van der Waals
  contacts keyed by element and dominant shell of the two main centres. It fails as a
  reactivity amplitude: contacts at random orientation favour H and s shells
  (docs/decisions.md, Prompt 5).
- `analytic`: the tblite two-atom probe of `ksmech.physics.contact` (|s| and p-vector norms of
  every centre pair at the covalent-radii distance, best pair).

`amplitude_matrix` / `beta_matrix` evaluate every (row, column) token pair at once; rows are
occupied tokens, columns occupied or virtual (the occupied-occupied block serves the repulsion
term).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ksmech.orbitals.tokens import MoleculeTokens, OrbitalKind, OrbitalToken
from ksmech.physics.contact import contact_coupling
from ksmech.physics.contact.legacy_fixed23 import OrientedConfig, oriented_block
from ksmech.physics.contact_table import ContactTable, contact_key
from ksmech.physics.params import KSParams, shell_fock_ev

KERNELS = ("oriented", "table", "analytic")


@dataclass(frozen=True)
class ContactConfig:
    kernel: str = "oriented"
    oriented: OrientedConfig = OrientedConfig()


def _z(t: MoleculeTokens) -> np.ndarray:
    return np.array([a.atomic_number for a in t.atoms])


def main_shell_fock(o: OrbitalToken, t: MoleculeTokens) -> float:
    """Fock diagonal (eV) of the dominant shell on the token's main centre."""
    d = o.descriptors
    return shell_fock_ev(t.F, t.ao_atom, t.ao_l, int(d.center_atoms[0]), contact_key(o, _z(t))[1])


def _main_weights(tokens: list[OrbitalToken]) -> np.ndarray:
    return np.array([o.descriptors.center_weights[0] for o in tokens], dtype=float)


def _block_positions(t: MoleculeTokens, kind: OrbitalKind) -> dict[int, int]:
    """id(token) -> its position among the tokens of `kind` of `t` (block row / column)."""
    return {id(o): k for k, o in enumerate(o for o in t.orbitals if o.kind == kind)}


def _kinds(cols: list[OrbitalToken]) -> np.ndarray:
    return np.array(["occ_occ" if o.kind == OrbitalKind.OCCUPIED else "occ_virt" for o in cols])


class ContactKernel:
    def __init__(
        self,
        tables: dict[str, ContactTable] | None = None,
        config: ContactConfig | None = None,
    ) -> None:
        self.config = config or ContactConfig()
        kernel = self.config.kernel
        if kernel not in KERNELS:
            raise ValueError(f"unknown contact kernel {kernel!r}; one of {KERNELS}")
        if kernel == "table" and tables is None:
            raise ValueError("the table kernel needs contact tables (`load_tables`)")
        self.tables = tables

    def _table_overlap(self, rows, t_r, cols, t_s, kinds: np.ndarray) -> np.ndarray:
        """S[kappa] per pair, one lookup per distinct (key_r, key_s, kind)."""
        z_r, z_s = _z(t_r), _z(t_s)
        k_r = [contact_key(o, z_r) for o in rows]
        k_s = [contact_key(o, z_s) for o in cols]
        pairs: dict[tuple, list[tuple[int, int]]] = {}
        for i, a in enumerate(k_r):
            for j, b in enumerate(k_s):
                pairs.setdefault((kinds[j], *a, *b), []).append((i, j))
        out = np.empty((len(rows), len(cols)))
        for (kind, *kappa), where in pairs.items():
            value, _ = self.tables[kind].lookup(tuple(kappa), n=len(where))
            ii, jj = np.array(where).T
            out[ii, jj] = value
        return out

    def _oriented(self, rows, t_r, cols, t_s, field: str, same: bool = False) -> np.ndarray:
        """One field of the cached molecule-pair `Contacts` for every (row, col) pair."""
        occ_r = _block_positions(t_r, OrbitalKind.OCCUPIED)
        i = [occ_r[id(o)] for o in rows]
        out = None
        for kind, block in ((OrbitalKind.VIRTUAL, "occ_virt"), (OrbitalKind.OCCUPIED, "occ_occ")):
            where = [j for j, o in enumerate(cols) if o.kind == kind]
            if not where:
                continue
            pos = _block_positions(t_s, kind)
            intra = same and block == "occ_virt"
            values = getattr(oriented_block(t_r, t_s, block, self.config.oriented, intra), field)
            part = values[np.ix_(i, [pos[id(cols[j])] for j in where])]
            if out is None:
                out = np.zeros((len(rows), len(cols), *part.shape[2:]), dtype=part.dtype)
            out[:, where] = part
        return out

    def oriented_details(self, rows, t_r, cols, t_s, same: bool = False):
        """(sites (len(rows), len(cols), 2), CLASH flags) of the oriented kernel's pair-track
        values: the argmax donor and acceptor site atoms (indices within t_r and t_s)."""
        if not rows or not cols:
            return np.zeros((len(rows), len(cols), 2), int), np.zeros((len(rows), len(cols)), bool)
        return (
            self._oriented(rows, t_r, cols, t_s, "sites", same),
            self._oriented(rows, t_r, cols, t_s, "clash", same),
        )

    def amplitude_matrix(
        self,
        rows: list[OrbitalToken],
        t_r: MoleculeTokens,
        cols: list,
        t_s: MoleculeTokens,
        same: bool = False,
    ) -> np.ndarray:
        """(len(rows), len(cols)) dimensionless contact amplitudes; rows occupied."""
        if not rows or not cols:
            return np.zeros((len(rows), len(cols)))
        if self.config.kernel == "oriented":
            return self._oriented(rows, t_r, cols, t_s, "S", same)
        if self.config.kernel == "analytic":
            z_r, z_s = _z(t_r), _z(t_s)
            return np.array(
                [
                    [contact_coupling(r.descriptors, z_r, s.descriptors, z_s) for s in cols]
                    for r in rows
                ]
            )
        kinds = _kinds(cols)
        overlap = self._table_overlap(rows, t_r, cols, t_s, kinds)
        return overlap * _main_weights(rows)[:, None] * _main_weights(cols)[None, :]

    def amplitude(
        self,
        r: OrbitalToken,
        t_r: MoleculeTokens,
        s: OrbitalToken,
        t_s: MoleculeTokens,
        same: bool = False,
    ) -> float:
        """Dimensionless contact amplitude of an occupied r with a token s of either block."""
        return float(self.amplitude_matrix([r], t_r, [s], t_s, same)[0, 0])

    def beta_matrix(
        self, rows, t_r, cols, t_s, params: KSParams | None = None, same: bool = False
    ) -> np.ndarray:
        """(len(rows), len(cols)) contact resonance integrals (eV)."""
        params = params or KSParams()
        if not rows or not cols:
            return np.zeros((len(rows), len(cols)))
        if self.config.kernel == "oriented":
            return params.K * self._oriented(rows, t_r, cols, t_s, "beta", same)
        f_r = np.array([main_shell_fock(o, t_r) for o in rows])
        f_s = np.array([main_shell_fock(o, t_s) for o in cols])
        f_mean = 0.5 * (f_r[:, None] + f_s[None, :])
        return params.K * self.amplitude_matrix(rows, t_r, cols, t_s) * f_mean

    def beta(
        self,
        r: OrbitalToken,
        t_r: MoleculeTokens,
        s: OrbitalToken,
        t_s: MoleculeTokens,
        params: KSParams | None = None,
        same: bool = False,
    ) -> float:
        """Contact resonance integral (eV)."""
        return float(self.beta_matrix([r], t_r, [s], t_s, params, same)[0, 0])
