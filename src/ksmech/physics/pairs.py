"""The pair-track features z_rs (Prompt 6), for every (occupied r, virtual s) token pair.

    z_rs = [log beta^2 at d = 2.0, 2.3, 3.0 Angstrom, log|G|, log|KS3| at 2.3, KS2, log|KS1|,
            1[intermolecular], centroid distance, |cross-centre coupling|, CLASH]

- beta = K sum_mu,nu l_mu,r l_nu,s S^c_mu,nu (F_mu,mu + F_nu,nu) / 2, the AO-level
  Wolfsberg-Helmholz sum over the oriented contact placement at each distance (`contact_kernel`):
  three values whose decay with d encodes how diffuse the two lobes are.
- G: the spectral gap kernel (full spectra); KS3 = beta^2 G at the primary distance 2.3. Logs
  drop the sign of G (positive only for inverted levels, 0.02% of the PMechDB pairs); the sign
  is kept apart (`PairBlock.gap_sign`) so the untrained floor KS3 + KS2 stays exact.
- KS2: the Coulomb term between the two main centres, signed (eV): at the van der Waals contact
  across fragments; within a fragment in difference form q q / eps (1/R_contact - 1/R), the
  change on bringing the two centres to contact (`_ks2`); undefined when both sit on one atom.
- KS1: the closed-shell repulsion of r with the occupied tokens around s's main centre
  (Wolfsberg-Helmholz form within a fragment, its magnitude per orbital pair because the LMO
  phases are arbitrary; contact form with |beta| of the primary kernel across, so KS1 >= 0
  everywhere); undefined where it is 0
  (no other occupied token near s's main centre, or a single-atom fragment: no resonance
  integral within an atom) or not finite.
- Within a fragment only: the centroid distance and |l_r^T B l_s| over the AO blocks between the
  two tokens' centres (its sign is an arbitrary LMO phase).
- CLASH: the primary kernel's flag for the pair's contact amplitude (oriented kernel).

Logs use a floor of 1e-12; undefined entries are stored as 0. Values are computed at
K = eps = 1 (`params.REDUCED`) for storage; `params.to_params` applies other values (log
features shift additively, KS2 and the coupling scale). Each entry
is standardized with statistics of the defined entries of a training set (`fit_pair_stats`)
for the learned heads. The frozen head (`frozen_head`) is not standardized: its logits are
-(KS3 + KS2 + KS1) / tau in eV, so untrained it ranks like the floor -(KS3 + KS2) with the
repulsion subtracted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ksmech.orbitals.tokens import (
    CandidateConfig,
    MoleculeTokens,
    is_antibonding_partner,
    site_excluded_matrix,
)
from ksmech.physics.contact.legacy_fixed23 import ORIENTED_VERSION, OrientedConfig
from ksmech.physics.contact_kernel import ContactConfig, ContactKernel
from ksmech.physics.contact_table import vdw_sum
from ksmech.physics.coulomb import coulomb_matrix
from ksmech.physics.ks_terms import cross_center_coupling_matrix, gap_kernel_matrix, wh_matrix
from ksmech.physics.params import COULOMB_EV_ANG, KSParams
from ksmech.physics.repulsion import contact_repulsion_matrix, neighbourhood, repulsion_matrix

FEATURE_DISTANCES = (2.0, 2.3, 3.0)
PRIMARY_DISTANCE = 2.3
PAIR_FEATURES = (
    "log_beta2_d2.0",
    "log_beta2_d2.3",
    "log_beta2_d3.0",
    "log_abs_g",
    "log_abs_ks3",
    "ks2",
    "log_abs_ks1",
    "inter",
    "centroid_distance",
    "cross_coupling",
    "clash",
)
F = {name: k for k, name in enumerate(PAIR_FEATURES)}
LOG_FLOOR = 1e-12


@dataclass(frozen=True)
class PairKernels:
    """The contact kernel at each feature distance; the primary one gives KS3, KS1, CLASH."""

    by_distance: dict[float, ContactKernel]
    primary: float = PRIMARY_DISTANCE

    @property
    def main(self) -> ContactKernel:
        return self.by_distance[self.primary]

    def describe(self) -> dict:
        """The kernel configuration samples are stamped with (provenance)."""
        k = self.main.config
        return {
            "kernel": k.kernel,
            "distances": sorted(self.by_distance),
            "primary": self.primary,
            "oriented_version": ORIENTED_VERSION if k.kernel == "oriented" else None,
            "site_weight": k.oriented.site_weight,
        }


def oriented_kernels(cache_dir: Path | None = None) -> PairKernels:
    return PairKernels(
        {
            d: ContactKernel(
                config=ContactConfig(oriented=OrientedConfig(distance=d, cache_dir=cache_dir))
            )
            for d in FEATURE_DISTANCES
        }
    )


def single_kernel(kernel: ContactKernel) -> PairKernels:
    """An ablation kernel without a distance: the same beta at every feature distance."""
    return PairKernels(dict.fromkeys(FEATURE_DISTANCES, kernel))


def _pairwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)


def _log(x: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(np.abs(x), LOG_FLOOR))


def _main(tokens) -> np.ndarray:
    return np.array([int(o.descriptors.center_atoms[0]) for o in tokens], dtype=int)


KS2_INTRA_FORMS = ("difference", "geometric")


def _ks2(
    t_f, occ, t_g, virt, same: bool, params, form: str = "difference"
) -> tuple[np.ndarray, np.ndarray]:
    """KS2 between the main centres, and where it is defined. Across fragments: q q / (eps R)
    at the van der Waals contact R_c. Within a fragment, `form` "difference" (default):
    q q / eps (1 / R_c - 1 / R), the change on bringing the two centres from their distance in
    the reactant to contact, so a bond's own polarity (O(-) H(+) at 1 Angstrom) is not counted
    as an attraction; "geometric" (the ablation): q q / (eps R)."""
    if form not in KS2_INTRA_FORMS:
        raise ValueError(f"unknown KS2 form {form!r}; one of {KS2_INTRA_FORMS}")
    a, b = _main(occ), _main(virt)
    za, zb = [t_f.atoms[i].atomic_number for i in a], [t_g.atoms[i].atomic_number for i in b]
    R_c = np.array([[vdw_sum(x, y) for y in zb] for x in za], dtype=float).reshape(len(a), len(b))
    q_a = [t_f.atoms[i].mulliken_charge for i in a]
    qq = np.outer(q_a, [t_g.atoms[i].mulliken_charge for i in b])
    k = COULOMB_EV_ANG * qq / params.eps
    if not same:
        return k / R_c, np.ones(R_c.shape, dtype=bool)
    pos_f = np.array([x.position for x in t_f.atoms]).reshape(-1, 3)
    R = _pairwise(pos_f[a], pos_f[b])
    defined = R > 0  # not when both main centres are one atom
    inv = 1.0 / np.where(defined, R, 1.0)
    out = k * (1.0 / R_c - inv) if form == "difference" else k * inv
    return np.where(defined, out, 0.0), defined


def _ks1(t_f, occ, t_g, virt, same: bool, kernel, params) -> np.ndarray:
    """KS1 of each occupied r of f with the occupied tokens of g around each virtual s's main
    centre (r itself excluded within a fragment)."""
    N = neighbourhood(t_g.occupied(), t_g, _main(virt))  # (n_virt, n_occ_g)
    if same:
        return repulsion_matrix(t_f, params) @ N.T  # zero diagonal: r never repels itself
    C = contact_repulsion_matrix(occ, t_f, t_g.occupied(), t_g, kernel, params)
    return C @ N.T


def _terms(
    t_f, t_g, same: bool, kernels: PairKernels, params, ks2_form: str = "difference"
) -> dict[str, np.ndarray]:
    """Every raw term of the (occupied of f) x (virtual of g) block."""
    occ, virt = t_f.occupied(), t_g.virtual()
    out = {
        f"beta_{d}": kernels.by_distance[d].beta_matrix(occ, t_f, virt, t_g, params, same=same)
        for d in FEATURE_DISTANCES
    }
    out["G"] = gap_kernel_matrix(occ, virt)
    out["ks2"], out["ks2_defined"] = _ks2(t_f, occ, t_g, virt, same, params, ks2_form)
    out["ks1"] = _ks1(t_f, occ, t_g, virt, same, kernels.main, params)
    shape = out["G"].shape
    out["clash"] = np.zeros(shape, dtype=bool)
    out["sites"] = np.full((*shape, 2), -1, dtype=int)
    if kernels.main.config.kernel == "oriented":
        out["sites"], out["clash"] = kernels.main.oriented_details(occ, t_f, virt, t_g, same)
    if same:
        cen_o = np.array([o.descriptors.centroid for o in occ]).reshape(-1, 3)
        cen_v = np.array([o.descriptors.centroid for o in virt]).reshape(-1, 3)
        out["centroid_distance"] = _pairwise(cen_o, cen_v)
        B = wh_matrix(t_f, params)
        out["cross_coupling"] = np.abs(cross_center_coupling_matrix(occ, virt, t_f.ao_atom, B))
    return out


@dataclass
class PairBlock:
    z: np.ndarray  # (n_occ, n_virt, len(PAIR_FEATURES)), 0 where undefined
    defined: np.ndarray  # same shape, bool
    gap_sign: np.ndarray  # (n_occ, n_virt) int8: sign of G (bookkeeping, not a feature)
    sites: np.ndarray  # (n_occ, n_virt, 2) global atoms of the contact amplitude's argmax
    n_non_finite: int = 0


def _fill(block: PairBlock, cell, terms: dict, same: bool) -> None:
    z, ok = block.z[cell], block.defined[cell]
    for d in FEATURE_DISTANCES:
        z[..., F[f"log_beta2_d{d}"]] = _log(terms[f"beta_{d}"] ** 2)
    beta2 = terms[f"beta_{PRIMARY_DISTANCE}"] ** 2
    z[..., F["log_abs_g"]] = _log(terms["G"])
    z[..., F["log_abs_ks3"]] = _log(beta2 * terms["G"])
    z[..., F["ks2"]] = terms["ks2"]
    z[..., F["log_abs_ks1"]] = _log(terms["ks1"])
    z[..., F["inter"]] = 0.0 if same else 1.0
    z[..., F["clash"]] = terms["clash"].astype(float)
    ok[...] = True
    ok[..., F["ks2"]] = terms["ks2_defined"]
    ok[..., F["log_abs_ks1"]] = np.isfinite(terms["ks1"]) & (terms["ks1"] != 0)
    for name in ("centroid_distance", "cross_coupling"):
        ok[..., F[name]] = same
        if same:
            z[..., F[name]] = terms[name]
    block.z[cell], block.defined[cell] = z, ok
    block.gap_sign[cell] = np.where(terms["G"] > 0, 1, -1)


def pair_block(
    fragments: list[MoleculeTokens], kernels: PairKernels, params=None, ks2_form="difference"
) -> PairBlock:
    """z_rs of every (occupied, virtual) pair of the reactant set, in the sample's orbital
    order (fragment by fragment, each file's own order). `ks2_form`: the intramolecular KS2
    (`_ks2`; "geometric" only for the ablation)."""
    params = params or KSParams()
    n_occ = [len(t.occupied()) for t in fragments]
    n_virt = [len(t.virtual()) for t in fragments]
    shape = (sum(n_occ), sum(n_virt))
    block = PairBlock(
        np.zeros((*shape, len(PAIR_FEATURES))),
        np.zeros((*shape, len(PAIR_FEATURES)), dtype=bool),
        np.zeros(shape, dtype=np.int8),
        np.full((*shape, 2), -1, dtype=np.int64),
    )
    r0, atom0 = np.cumsum([0, *n_occ]), np.cumsum([0, *[len(t.atoms) for t in fragments]])
    c0 = np.cumsum([0, *n_virt])
    for f, t_f in enumerate(fragments):
        for g, t_g in enumerate(fragments):
            if not n_occ[f] or not n_virt[g]:
                continue
            cell = np.s_[r0[f] : r0[f + 1], c0[g] : c0[g + 1]]
            terms = _terms(t_f, t_g, f == g, kernels, params, ks2_form)
            _fill(block, cell, terms, f == g)
            if (terms["sites"] >= 0).all():
                block.sites[cell] = terms["sites"] + [atom0[f], atom0[g]]
    bad = ~np.isfinite(block.z)  # counted, never silent
    block.n_non_finite = int(bad.sum())
    block.defined[bad] = False
    block.z[~block.defined] = 0.0  # e.g. log|KS1| floored where KS1 == 0
    return block


def valid_pairs(fragments: list[MoleculeTokens], config: CandidateConfig | None = None):
    """(n_occ, n_virt) arrow-candidate filter: a doubly occupied donor, an empty acceptor that
    is not d-type (unless allowed), and, within one fragment, not the donor's own antibond and
    not a pair whose every site pair is a bond interacting with itself (sites identical or
    bonded, and the acceptor carries >= `config.site_weight` on the donor's site or the donor
    on the acceptor's: `tokens.site_excluded_matrix`). Every criterion is read from descriptors
    and the record geometry (`ksmech.orbitals.tokens`)."""
    config = config or CandidateConfig()
    donors = {id(o) for t in fragments for o in t.donor_candidates(config)}
    acceptors = {id(o) for t in fragments for o in t.acceptor_candidates(config)}
    occ = [o for t in fragments for o in t.occupied()]
    virt = [o for t in fragments for o in t.virtual()]
    ok_d = np.array([id(o) in donors for o in occ], dtype=bool)
    ok_a = np.array([id(o) in acceptors for o in virt], dtype=bool)
    out = ok_d[:, None] & ok_a[None, :]
    r0 = np.cumsum([0, *[len(t.occupied()) for t in fragments]])
    c0 = np.cumsum([0, *[len(t.virtual()) for t in fragments]])
    for f, t in enumerate(fragments):  # centres are atoms of one fragment
        o_f, v_f = t.occupied(), t.virtual()
        if not o_f or not v_f:
            continue
        own = np.array([[is_antibonding_partner(d, a) for a in v_f] for d in o_f])
        bad = own | site_excluded_matrix(t, o_f, v_f, config.site_weight)
        out[r0[f] : r0[f + 1], c0[f] : c0[f + 1]] &= ~bad
    return out


def atom_coulomb_matrix(fragments: list[MoleculeTokens], params: KSParams | None = None):
    """(A, A) symmetric KS2 between atoms, in the sample's atom order (fragment by fragment):
    the geometric distance within a fragment, the van der Waals contact across fragments,
    0 on the diagonal."""
    atoms = [a for t in fragments for a in t.atoms]
    frag = np.repeat(np.arange(len(fragments)), [len(t.atoms) for t in fragments])
    pos = np.array([a.position for a in atoms], dtype=float).reshape(-1, 3)
    z = [a.atomic_number for a in atoms]
    q = np.array([a.mulliken_charge for a in atoms], dtype=float)
    inter = frag[:, None] != frag[None, :]
    vdw = np.array([[vdw_sum(a, b) for b in z] for a in z]).reshape(inter.shape)
    distance = np.where(inter, vdw, _pairwise(pos, pos))
    return coulomb_matrix(q, distance, ~np.eye(len(atoms), dtype=bool), params)


# ---- standardization and the frozen head


@dataclass(frozen=True)
class PairStats:
    """Mean and standard deviation of each z entry over the defined entries of a training set,
    at the physical K and eps, then frozen."""

    mean: np.ndarray
    std: np.ndarray
    count: np.ndarray
    names: tuple[str, ...] = field(default=PAIR_FEATURES)

    def to_json(self) -> str:
        return json.dumps(
            {
                n: {"mean": m, "std": s, "count": int(c)}
                for n, m, s, c in zip(
                    self.names, self.mean.tolist(), self.std.tolist(), self.count, strict=True
                )
            },
            indent=1,
        )

    @classmethod
    def from_json(cls, text: str, names: tuple[str, ...] = PAIR_FEATURES) -> PairStats:
        d = json.loads(text)
        if tuple(d) != tuple(names):
            raise ValueError(f"pair statistics for {tuple(d)}, expected {tuple(names)}")
        values = (np.array([d[n][k] for n in names]) for k in ("mean", "std", "count"))
        return cls(*values, names=tuple(names))


def fit_pair_stats(blocks, names: tuple[str, ...] = PAIR_FEATURES) -> PairStats:
    """Streaming mean / std over the defined entries of many (z, defined) pairs whose last
    axis holds the features `names`."""
    k = len(names)
    n, s1, s2 = np.zeros(k), np.zeros(k), np.zeros(k)
    for z, defined in blocks:
        for j in range(k):
            v = z[..., j][defined[..., j]].astype(float)
            n[j], s1[j], s2[j] = n[j] + v.size, s1[j] + v.sum(), s2[j] + (v * v).sum()
    mean = np.divide(s1, n, out=np.zeros(k), where=n > 0)
    var = np.divide(s2, n, out=np.zeros(k), where=n > 0) - mean**2
    return PairStats(mean, np.sqrt(np.clip(var, 0.0, None)), n, tuple(names))


def standardize(z: np.ndarray, defined: np.ndarray, stats: PairStats) -> np.ndarray:
    """(z - mean) / std on defined entries, 0 elsewhere; a constant entry is only centred."""
    std = np.where(stats.std > 0, stats.std, 1.0)
    return np.where(defined, (z - stats.mean) / std, 0.0)


def ks_energies(z: np.ndarray, gap_sign: np.ndarray, defined: np.ndarray):
    """(KS3, KS2, KS1) in eV from stored features at the physical K, eps (`to_params` first):
    KS3 = sign(G) |KS3| at 2.3 Angstrom, KS2 signed, KS1 >= 0 the closed-shell repulsion; an
    undefined term is 0."""
    ks3 = gap_sign * np.exp(z[..., F["log_abs_ks3"]])
    ks2 = np.where(defined[..., F["ks2"]], z[..., F["ks2"]], 0.0)
    ks1 = np.where(defined[..., F["log_abs_ks1"]], np.exp(z[..., F["log_abs_ks1"]]), 0.0)
    return ks3, ks2, ks1


def frozen_head(
    z: np.ndarray, gap_sign: np.ndarray, defined: np.ndarray, params: KSParams | None = None
) -> np.ndarray:
    """Logits of the frozen KS head: -(KS3 + KS2 + KS1) / tau in eV, no standardization, from
    stored features at the physical K and eps (`ks_energies`). KS1 is the repulsion, >= 0, so
    it lowers the logit (the spec's term (1); the user's `-(KS3 + KS2 - KS1)` with KS1 written
    as the stabilization-signed sum). tau (`KSParams.tau`, a learnable positive scalar,
    1 eV at initialization) only scales: the ranking is the floor -(KS3 + KS2) minus KS1."""
    ks3, ks2, ks1 = ks_energies(z, gap_sign, defined)
    return -(ks3 + ks2 + ks1) / (params or KSParams()).tau
