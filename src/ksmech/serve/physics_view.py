"""Per-arrow physics: the inline brief of every network arrow and the full physics panel
(adapters A9-A12; design 7.3-7.5).

Every number is a cell of the pool's analysis (`analysis.PoolAnalysis`), computed on the
production call path in the worker; this module selects, indexes and converts. Two values are
re-derived and each is guarded against the upstream function: the dominant canonical MO pair
(k, m) of the gap kernel (its term matrix must sum to `ks_terms.token_gap_kernel`, else
CONSISTENCY_FAILED) and the floor rank (the `physics.ranking.label_rank` formula, mirrored so
the API process never imports `physics.ranking`, which loads torch; a test compares both on
every valid cell). The site-pair block (`t_site_pairs`, A9) calls `oriented.block_contacts`
for one site pair at a time; its result is cached as `site-pairs-<sha16>.json` next to the analysis
npz it was computed from (`site_pairs_path`), so a retried request submits no task.

Only numpy is imported at module top: the API process (the physics panel) and the worker (the
briefs at expansion and apply time) both import this module, and every upstream or serve
helper is imported inside the function that needs it.

Signs: KS3 < 0 is stabilizing, KS2 < 0 attractive, KS1 >= 0 repulsive, G < 0 the normal level
order, and a higher floor is better. Units: eV, 1/eV, eV^2, Angstrom.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from ksmech.arrows.reactants import ReactantTokens
    from ksmech.orbitals.tokens import MoleculeTokens, OrbitalToken
    from ksmech.physics.pairs import PairKernels
    from ksmech.serve.analysis import PoolAnalysis, PoolWork
    from ksmech.serve.contracts import Runtime

Occupations = Callable[[str], "np.ndarray | None"]

UNITS = {
    "ks1": "eV",
    "ks2": "eV",
    "ks3": "eV",
    "G": "1/eV",
    "beta": "eV",
    "beta2": "eV^2",
    "distance": "Angstrom",
    "energies": "eV",
}
DISTANCES = ("2.0", "2.3", "3.0")  # the beta2_d* members of the analysis
LEVEL_MIN_WEIGHT = 1e-3  # energy levels kept by weight (plus HOMO-2..HOMO, LUMO..LUMO+2)
G_RTOL, G_ATOL = 1e-5, 1e-9  # G (float32-derived) against G_exact
LOG_RTOL, LOG_ATOL = 1e-5, 1e-11  # stored log features against direct values (log floor)
SUM_RTOL, SUM_ATOL = 1e-10, 1e-12  # the dominant-pair term matrix against token_gap_kernel
SPECTRAL_TOL = 1e-9
FROZEN_SOURCE = "untrained_kernel"
FROZEN_NORMALIZATION = (
    "softmax over the donor row's valid acceptors; the spec's softmax_j runs over all tokens j "
    "with mask m_ij; no model exists"
)
SITE_PAIR_NOTE = (
    "the predictor scores the token pair at its contact sites; the values at your sites are "
    "diagnostics"
)
SITE_PAIR_TIMEOUT_S = 180


def _error(code: str, message: str, **kw: Any) -> Exception:
    from ksmech.serve.errors import ServeError

    return ServeError(code, message, **kw)


@cache
def primary_distance() -> float:
    """The oriented placement distance of the main kernel (`physics/pairs.py` PRIMARY_DISTANCE,
    = `kernels.main.config.oriented.distance`), read without importing the physics package."""
    from ksmech.serve.versions import source_constant

    return float(source_constant("physics/pairs.py", "PRIMARY_DISTANCE"))


def _tau() -> float:
    from ksmech.physics.params import KSParams

    return float(KSParams().tau)


def token_class(o: OrbitalToken, t: MoleculeTokens) -> str:
    """Diagnostic description of a token, display only: the classifier type and the elements
    of the two leading centres. Mirrors `physics.ranking.token_class` (that module loads
    torch); a test compares both."""
    centres = [a for a in o.descriptors.center_atoms[:2] if a >= 0]
    return f"{o.type}({'-'.join(t.symbol(int(a)) for a in centres)})"


# ---- canonical MO indices (A10)


@dataclass(frozen=True)
class BlockIndex:
    """Canonical MO index of every position of a species' occupied and virtual blocks."""

    occ: np.ndarray
    virt: np.ndarray
    source: str  # "qm_occupations" | "aufbau"

    @property
    def homo(self) -> int | None:
        return int(self.occ.max()) if len(self.occ) else None

    @property
    def lumo(self) -> int | None:
        return int(self.virt.min()) if len(self.virt) else None


def _block_len(tokens: Sequence[OrbitalToken]) -> int:
    return len(tokens[0].descriptors.full_energies) if tokens else 0


def block_indices(t: MoleculeTokens, occupations: np.ndarray | None) -> BlockIndex:
    """`full_energies` follows `flatnonzero(occ >= 0.5)` (occupied block) and
    `flatnonzero(occ < 0.5)` (virtual block) of the QM occupations (`orbitals/localize.py`).
    Without occupations, or when they do not match the token blocks, the aufbau prefix."""
    from ksmech.qm.fukui import OCC_THRESHOLD

    n_occ, n_virt = _block_len(t.occupied()), _block_len(t.virtual())
    if occupations is not None:
        occ = np.asarray(occupations, dtype=float)
        i_occ = np.flatnonzero(occ >= OCC_THRESHOLD)
        i_virt = np.flatnonzero(occ < OCC_THRESHOLD)
        if len(i_occ) == n_occ and len(i_virt) == n_virt:
            return BlockIndex(i_occ, i_virt, "qm_occupations")
    return BlockIndex(np.arange(n_occ), n_occ + np.arange(n_virt), "aufbau")


def spectral_consistent(o: OrbitalToken, idx: np.ndarray) -> bool:
    """Every retained spectral entry (weight > 0) names a canonical index of the block and
    the block energy at that position equals the entry's energy (1e-9 eV)."""
    d = o.descriptors
    full = np.asarray(d.full_energies, dtype=float)
    if len(idx) != len(full):
        return False
    pos = {int(c): p for p, c in enumerate(idx)}
    for e in np.flatnonzero(np.asarray(d.spectral_weights) > 0):
        c = int(d.spectral_indices[e])
        if c not in pos or abs(full[pos[c]] - float(d.spectral_energies[e])) > SPECTRAL_TOL:
            return False
    return True


def homo_label(canonical: int, homo: int | None) -> str:
    if homo is None or canonical == homo:
        return "HOMO"
    return f"HOMO-{homo - canonical}" if canonical < homo else f"HOMO+{canonical - homo}"


def lumo_label(canonical: int, lumo: int | None) -> str:
    if lumo is None or canonical == lumo:
        return "LUMO"
    return f"LUMO+{canonical - lumo}" if canonical > lumo else f"LUMO-{lumo - canonical}"


def _term_matrix(r: OrbitalToken, s: OrbitalToken) -> tuple[np.ndarray, np.ndarray]:
    """(T, clamped gap) with T[k, m] = 2 p_rk p_sm / (E_k - E_m): `ks_terms.gap_kernel`'s
    expression and clamp, keeping the block-local indices."""
    from ksmech.physics.ks_terms import MIN_GAP_EV

    er = np.asarray(r.descriptors.full_energies, dtype=np.float64)
    pr = np.asarray(r.descriptors.full_weights, dtype=np.float64)
    es = np.asarray(s.descriptors.full_energies, dtype=np.float64)
    ps = np.asarray(s.descriptors.full_weights, dtype=np.float64)
    gap = er[:, None] - es[None, :]
    gap = np.where(gap >= 0, 1.0, -1.0) * np.maximum(np.abs(gap), MIN_GAP_EV)
    return 2.0 * pr[:, None] * ps[None, :] / gap, gap


def dominant_gap_pair(
    r: OrbitalToken,
    s: OrbitalToken,
    idx_r: np.ndarray,
    idx_s: np.ndarray,
    homo_r: int,
    lumo_s: int,
    beta2: float,
) -> dict:
    """The canonical MO pair (k, m) with the largest |T[k, m]| (ties: lowest k, then m) of
    G = sum T. Raises ServeError(CONSISTENCY_FAILED) unless sum T equals
    `token_gap_kernel(r, s)` (rtol 1e-10, atol 1e-12) and the indices fit the blocks."""
    from ksmech.physics.ks_terms import token_gap_kernel

    T, gap = _term_matrix(r, s)
    total, exact = float(T.sum()), float(token_gap_kernel(r, s))
    if not math.isclose(total, exact, rel_tol=SUM_RTOL, abs_tol=SUM_ATOL):
        msg = f"dominant-pair term matrix sums to {total!r}, token_gap_kernel gives {exact!r}"
        raise _error("CONSISTENCY_FAILED", msg)
    if T.shape != (len(idx_r), len(idx_s)):
        msg = f"term matrix {T.shape} does not fit the blocks ({len(idx_r)}, {len(idx_s)})"
        raise _error("CONSISTENCY_FAILED", msg)
    k, m = (int(x) for x in np.unravel_index(int(np.argmax(np.abs(T))), T.shape))
    k_can, m_can = int(idx_r[k]), int(idx_s[m])
    term = float(T[k, m])
    return {
        "k": k,
        "m": m,
        "k_canonical": k_can,
        "m_canonical": m_can,
        "k_label": homo_label(k_can, homo_r),
        "m_label": lumo_label(m_can, lumo_s),
        "E_k": float(r.descriptors.full_energies[k]),
        "E_m": float(s.descriptors.full_energies[m]),
        "gap": float(gap[k, m]),
        "p_rk": float(r.descriptors.full_weights[k]),
        "p_sm": float(s.descriptors.full_weights[m]),
        "term": term,
        "share": term / total if total else float("nan"),
        "ks3_part": float(beta2) * term,
    }


def _levels(o: OrbitalToken, idx: np.ndarray, keep_pos: int, frontier: set[int], label) -> list:
    d = o.descriptors
    energies, weights = np.asarray(d.full_energies), np.asarray(d.full_weights)
    keep = {p for p in range(len(energies)) if weights[p] >= LEVEL_MIN_WEIGHT}
    keep |= {p for p, c in enumerate(idx) if int(c) in frontier} | {keep_pos}
    return [
        {
            "canonical": int(idx[p]),
            "label": label(int(idx[p])),
            "energy": float(energies[p]),
            "weight": float(weights[p]),
        }
        for p in sorted(keep, key=lambda p: int(idx[p]))
    ]


def energy_levels(
    r: OrbitalToken,
    s: OrbitalToken,
    ix_r: BlockIndex,
    ix_s: BlockIndex,
    k: int,
    m: int,
    donor_species: str,
    acceptor_species: str,
) -> dict:
    """`EnergyLevelsOut` fields: the donor species' occupied levels (weights p_rk) and the
    acceptor species' virtual levels (p_sm), each kept when p >= 1e-3, plus HOMO-2..HOMO and
    LUMO..LUMO+2 and always k and m."""
    homo, lumo = ix_r.homo, ix_s.lumo
    near_homo = set() if homo is None else {homo - 2, homo - 1, homo}
    near_lumo = set() if lumo is None else {lumo, lumo + 1, lumo + 2}
    er, es = r.descriptors.full_energies, s.descriptors.full_energies
    e_homo = float(er[int(np.argmax(ix_r.occ))]) if len(ix_r.occ) else float("nan")
    e_lumo = float(es[int(np.argmin(ix_s.virt))]) if len(ix_s.virt) else float("nan")
    return {
        "donor_species": donor_species,
        "acceptor_species": acceptor_species,
        "donor_levels": _levels(r, ix_r.occ, k, near_homo, lambda c: homo_label(c, homo)),
        "acceptor_levels": _levels(s, ix_s.virt, m, near_lumo, lambda c: lumo_label(c, lumo)),
        "donor_lmo_energy": float(r.energy_ev),
        "acceptor_lmo_energy": float(s.energy_ev),
        "homo_lumo_gap": e_lumo - e_homo,
    }


# ---- ranks and weights (A11, A12)


def label_rank_mirror(scores: np.ndarray, valid: np.ndarray, positive: np.ndarray) -> int | None:
    """`physics.ranking.label_rank`, line for line: the number of valid non-labeled
    candidates scoring at least as well as the best valid labeled pair; None when no labeled
    pair is valid."""
    pos = positive & valid
    if not pos.any():
        return None
    best = scores[pos].max()
    return int(((scores >= best) & valid & ~pos).sum())


def floor_rank(an: PoolAnalysis, i: int, j: int) -> int | None:
    """1-based rank of cell (i, j) under the floor among the valid pairs (ties count against
    it); None when the pair is invalid. Always over `pair_valid`, never scope-masked."""
    valid = np.asarray(an.valid, dtype=bool)
    if not valid[i, j]:
        return None
    positive = np.zeros_like(valid)
    positive[i, j] = True
    rank = label_rank_mirror(np.where(valid, an.floor, -np.inf), valid, positive)
    return None if rank is None else rank + 1


def frozen_head_weight(an: PoolAnalysis, i: int, j: int) -> dict:
    """`FrozenHeadOut` fields: the untrained frozen-head logit -(KS3 + KS2 + KS1)/tau and its
    softmax weight over the donor row's valid acceptors (None for an invalid pair)."""
    logits = np.asarray(an.frozen[i], dtype=np.float64)
    row_valid = np.asarray(an.valid[i], dtype=bool)
    weight = None
    if row_valid[j]:
        peak = logits[row_valid].max()
        weight = float(np.exp(logits[j] - peak) / np.exp(logits[row_valid] - peak).sum())
    return {
        "logit": float(logits[j]),
        "weight": weight,
        "tau": _tau(),
        "source": FROZEN_SOURCE,
        "normalization": FROZEN_NORMALIZATION,
    }


def cell_rank(scores: np.ndarray, valid: np.ndarray, i: int, j: int) -> int | None:
    """1-based position of cell (i, j) in the `Predictor.propose` order (score desc, ties by
    (row, col) asc) among the valid cells; None when (i, j) is not valid."""
    valid = np.asarray(valid, dtype=bool)
    if not valid[i, j]:
        return None
    s = np.asarray(scores, dtype=np.float64)
    flat = np.arange(s.size).reshape(s.shape)
    before = valid & ((s > s[i, j]) | ((s == s[i, j]) & (flat < flat[i, j])))
    return int(before.sum()) + 1


def predictor_cell(
    scores: np.ndarray, valid: np.ndarray, temperature: float, i: int, j: int
) -> tuple[float | None, float | None, int | None]:
    """(score, probability, rank) of cell (i, j): the predictor's value, its softmax over the
    valid cells at T (design 4.3) and its 1-based rank; None where the cell is not valid."""
    from ksmech.serve.contracts import log_softmax_valid

    valid = np.asarray(valid, dtype=bool)
    masked = np.where(valid, np.asarray(scores, dtype=np.float64), -np.inf)
    if not valid[i, j]:
        return None, None, None
    logp = log_softmax_valid(masked, valid, temperature)
    score = float(masked[i, j])
    return (
        (score if math.isfinite(score) else None),
        float(np.exp(logp[i, j])),
        cell_rank(masked, valid, i, j),
    )


# ---- one pair of the analysis


@dataclass(frozen=True)
class _Pair:
    """Cell (i, j) resolved to its tokens, fragments and species."""

    i: int
    j: int
    donor: int
    acceptor: int
    f: int
    g: int
    t_f: MoleculeTokens
    t_g: MoleculeTokens
    r: OrbitalToken
    s: OrbitalToken
    key_f: str
    key_g: str

    @property
    def species(self) -> tuple[str, str]:
        from ksmech.serve.keys import species_key_of

        return species_key_of(self.key_f)[0], species_key_of(self.key_g)[0]


def _pair(an: PoolAnalysis, rt: ReactantTokens, i: int, j: int) -> _Pair:
    donor, acceptor = int(an.rows[i]), int(an.cols[j])
    f = int(rt.orbital_fragment[donor - rt.n_atoms])
    g = int(rt.orbital_fragment[acceptor - rt.n_atoms])
    return _Pair(
        i,
        j,
        donor,
        acceptor,
        f,
        g,
        rt.fragments[f].tokens,
        rt.fragments[g].tokens,
        rt.orbital(donor),
        rt.orbital(acceptor),
        rt.fragments[f].key,
        rt.fragments[g].key,
    )


def _occupations(occupations: Occupations | None, key: str) -> np.ndarray | None:
    if occupations is None:
        return None
    try:
        return occupations(key)
    except (OSError, ValueError, KeyError):
        return None


@dataclass
class _Dominant:
    pair: dict | None  # dominant_gap_pair(...) plus index_source, species, same_molecule
    ix_f: BlockIndex
    ix_g: BlockIndex
    g_exact: float
    flags: list[str]


def _dominant(an: PoolAnalysis, p: _Pair, occupations: Occupations | None) -> _Dominant:
    from ksmech.physics.ks_terms import token_gap_kernel
    from ksmech.serve.errors import ServeError

    ix_f = block_indices(p.t_f, _occupations(occupations, p.key_f))
    ix_g = block_indices(p.t_g, _occupations(occupations, p.key_g))
    both_qm = ix_f.source == ix_g.source == "qm_occupations"
    source = "qm_occupations" if both_qm else "aufbau"
    flags = []
    if not (spectral_consistent(p.r, ix_f.occ) and spectral_consistent(p.s, ix_g.virt)):
        source += "_unchecked"
        flags.append("index_unchecked")
    beta2 = float(an.beta[p.i, p.j]) ** 2
    try:
        pair = dominant_gap_pair(p.r, p.s, ix_f.occ, ix_g.virt, ix_f.homo, ix_g.lumo, beta2)
    except ServeError:
        pair = None
        flags.append("CONSISTENCY_FAILED")
    if pair is not None:
        donor_species, acceptor_species = p.species
        pair |= {
            "index_source": source,
            "donor_species": donor_species,
            "acceptor_species": acceptor_species,
            "same_molecule": p.f == p.g,
        }
    return _Dominant(pair, ix_f, ix_g, float(token_gap_kernel(p.r, p.s)), flags)


def _warning_flags(t: MoleculeTokens) -> list[str]:
    out = [w for w in re.split(r"[,;\s]+", t.qm_warnings or "") if w]
    if str(t.method).upper().startswith("SYNTHETIC"):
        out.append("synthetic")
    return out


def _flags(an: PoolAnalysis, p: _Pair, dom: _Dominant) -> list[str]:
    """fermi_smeared, the QM warnings of both species, clash, synthetic, pair_invalid:<reason>,
    index_unchecked, G_PRECISION, BETA2_MISMATCH, CONSISTENCY_FAILED."""
    from ksmech.serve.candidates import reason_name

    i, j = p.i, p.j
    flags = ["fermi_smeared"] if an.fermi_smeared else []
    flags += _warning_flags(p.t_f) + _warning_flags(p.t_g)
    if bool(an.clash[i, j]):
        flags.append("clash")
    if not bool(an.valid[i, j]):
        flags.append(f"pair_invalid:{reason_name(int(an.reasons[i, j]))}")
    flags += dom.flags
    g = float(an.gap_kernel[i, j])
    if not abs(g - dom.g_exact) <= G_RTOL * abs(dom.g_exact) + G_ATOL:
        flags.append("G_PRECISION")
    stored = float(an.beta2("2.3")[i, j])
    beta2 = float(an.beta[i, j]) ** 2
    if not math.isnan(stored) and not math.isclose(
        beta2, stored, rel_tol=LOG_RTOL, abs_tol=LOG_ATOL
    ):
        flags.append("BETA2_MISMATCH")
    return list(dict.fromkeys(flags))


def ks2_form(an: PoolAnalysis, i: int, j: int) -> str:
    """ "undefined" where KS2 is not defined (both main centres on one atom), "contact" across
    fragments (q q / (eps R_c)), "difference" within one (q q / eps (1/R_c - 1/R))."""
    if not bool(an.ks2_ok[i, j]):
        return "undefined"
    return "contact" if bool(an.inter[i, j]) else "difference"


def _reason(an: PoolAnalysis, i: int, j: int) -> str | None:
    from ksmech.serve.candidates import reason_name

    return None if bool(an.valid[i, j]) else reason_name(int(an.reasons[i, j]))


def _opt(value: float, ok: bool) -> float | None:
    return float(value) if ok else None


DOMINANT_BRIEF = (
    "k_canonical",
    "m_canonical",
    "k_label",
    "m_label",
    "E_k",
    "E_m",
    "term",
    "share",
    "index_source",
    "same_molecule",
)


def brief_physics(
    an: PoolAnalysis, rt: ReactantTokens, i: int, j: int, occupations: Occupations | None
) -> dict:
    """`ArrowPhysicsBrief` fields of cell (i, j): pure numpy on the analysis and the tokens in
    memory, plus the species' QM occupations for the canonical MO indices."""
    p = _pair(an, rt, i, j)
    dom = _dominant(an, p, occupations)
    head = frozen_head_weight(an, i, j)
    return {
        "pair_valid": bool(an.valid[i, j]),
        "reason": _reason(an, i, j),
        "ks1": _opt(an.ks1[i, j], bool(an.ks1_ok[i, j])),
        "ks2": _opt(an.ks2[i, j], bool(an.ks2_ok[i, j])),
        "ks2_form": ks2_form(an, i, j),
        "ks3": float(an.ks3[i, j]),
        "G": float(an.gap_kernel[i, j]),
        "beta2": float(an.beta[i, j]) ** 2,
        "amplitude_S": float(an.amp[i, j]),
        "dominant": None if dom.pair is None else {k: dom.pair[k] for k in DOMINANT_BRIEF},
        "contact_distance_ang": primary_distance(),
        "contact_sites": [int(x) for x in an.contact_sites[i, j]],
        "clash": bool(an.clash[i, j]),
        "floor": float(an.floor[i, j]),
        "floor_rank": floor_rank(an, i, j),
        "n_valid": an.n_valid,
        "frozen_logit": head["logit"],
        "frozen_weight": head["weight"],
        "tau": head["tau"],
        "frozen_source": head["source"],
        "flags": _flags(an, p, dom),
    }


# ---- the full panel


def ks2_detail(an: PoolAnalysis, rt: ReactantTokens, i: int, j: int) -> dict:
    """`Ks2DetailOut` fields: the main centres (`center_atoms[0]`) as global sites, their
    Mulliken charges, the van der Waals contact R_c and, within a fragment, the record
    distance R."""
    from ksmech.physics.contact_table import vdw_sum
    from ksmech.physics.params import KSParams

    p = _pair(an, rt, i, j)
    a_loc = int(p.r.descriptors.center_atoms[0])
    b_loc = int(p.s.descriptors.center_atoms[0])
    atom_a, atom_b = p.t_f.atoms[a_loc], p.t_g.atoms[b_loc]
    r_geo = None
    if p.f == p.g:
        r_geo = float(np.linalg.norm(np.asarray(atom_a.position) - np.asarray(atom_b.position)))
    return {
        "form": ks2_form(an, i, j),
        "main_centres": [int(rt.atom_offsets[p.f]) + a_loc, int(rt.atom_offsets[p.g]) + b_loc],
        "q_donor": float(atom_a.mulliken_charge),
        "q_acceptor": float(atom_b.mulliken_charge),
        "R_contact": float(vdw_sum(atom_a.atomic_number, atom_b.atomic_number)),
        "R_geometric": r_geo,
        "eps": float(KSParams().eps),
    }


def _panel_values(an: PoolAnalysis, i: int, j: int) -> dict:
    return {
        "pair_valid": bool(an.valid[i, j]),
        "reason": _reason(an, i, j),
        "ks1": float(an.ks1[i, j]),
        "ks1_defined": bool(an.ks1_ok[i, j]),
        "ks2": float(an.ks2[i, j]),
        "ks2_defined": bool(an.ks2_ok[i, j]),
        "ks3": float(an.ks3[i, j]),
        "G": float(an.gap_kernel[i, j]),
        "beta": float(an.beta[i, j]),
        "beta2": float(an.beta[i, j]) ** 2,
        "beta2_by_distance": {d: float(an.beta2(d)[i, j]) for d in DISTANCES},
        "amplitude_S": float(an.amp[i, j]),
        "contact_sites": [int(x) for x in an.contact_sites[i, j]],
        "contact_distance_ang": primary_distance(),
        "clash": bool(an.clash[i, j]),
        "floor": float(an.floor[i, j]),
        "floor_rank": floor_rank(an, i, j),
        "n_valid": an.n_valid,
    }


def arrow_physics(
    an: PoolAnalysis,
    rt: ReactantTokens,
    i: int,
    j: int,
    donor_site: int,
    acceptor_site: int,
    scores: np.ndarray | None,
    temperature: float,
    occupations: Occupations | None,
    site_pair: dict | None,
) -> dict:
    """`PhysicsOut` fields minus `predictor` and `provenance` (the service adds them). Raises
    ServeError(CONSISTENCY_FAILED) when the dominant pair fails its check: the panel never
    shows a number that failed its identity."""
    p = _pair(an, rt, i, j)
    dom = _dominant(an, p, occupations)
    if dom.pair is None:
        raise _error("CONSISTENCY_FAILED", "the dominant MO pair failed its sum check", status=500)
    score, prob, rank = (None, None, None)
    if scores is not None:
        score, prob, rank = predictor_cell(scores, an.valid, temperature, i, j)
    contact = tuple(int(x) for x in an.contact_sites[i, j])
    donor_species, acceptor_species = p.species
    out = {"state_id": an.state_id, "donor": p.donor, "donor_site": int(donor_site)}
    out |= {"acceptor": p.acceptor, "acceptor_site": int(acceptor_site), "row": i, "col": j}
    out |= {"units": dict(UNITS), **_panel_values(an, i, j)}
    return out | {
        "ks2_detail": ks2_detail(an, rt, i, j),
        "G_exact": dom.g_exact,
        "requested_sites_are_contact": (int(donor_site), int(acceptor_site)) == contact,
        "site_pair": site_pair,
        "score": score,
        "probability": prob,
        "rank": rank,
        "frozen_head": frozen_head_weight(an, i, j),
        "dominant_pair": dom.pair,
        "energy_levels": energy_levels(
            p.r,
            p.s,
            dom.ix_f,
            dom.ix_g,
            dom.pair["k"],
            dom.pair["m"],
            donor_species,
            acceptor_species,
        ),
        "flags": _flags(an, p, dom),
    }


# ---- site pairs (A9)


def _site_map(rt: ReactantTokens, site: int) -> int:
    idx = rt.mol_atom.get(int(site))
    return 0 if idx is None else int(rt.mol.GetAtomWithIdx(idx).GetAtomMapNum())


def _site_pair_row(rt, p: _Pair, a: int, b: int, kernels, g_exact: float, contact) -> dict:
    from ksmech.orbitals.tokens import site_pairs_excluded
    from ksmech.physics.oriented import block_contacts
    from ksmech.physics.params import KSParams

    cfg = kernels.main.config.oriented
    off_f, off_g = int(rt.atom_offsets[p.f]), int(rt.atom_offsets[p.g])
    r_loc = p.donor - rt.n_atoms - int(rt.orbital_offsets[p.f])  # position in t_f.orbitals
    s_loc = p.acceptor - rt.n_atoms - int(rt.orbital_offsets[p.g])
    a_loc, b_loc = a - off_f, b - off_g
    bc = block_contacts(p.t_f, [r_loc], p.t_g, [s_loc], cfg, sites_F=[[a_loc]], sites_G=[[b_loc]])
    beta = float(KSParams().K) * float(bc.beta[0, 0])
    excluded = False
    if p.f == p.g:
        w = float(rt.candidate.site_weight)
        excluded = bool(site_pairs_excluded(p.t_f, [p.r], [a_loc], [p.s], [b_loc], w)[0])
    return {
        "donor_site": int(a),
        "acceptor_site": int(b),
        "donor_site_map": _site_map(rt, a),
        "acceptor_site_map": _site_map(rt, b),
        "S": float(bc.S[0, 0]),
        "beta": beta,
        "clash": bool(bc.clash[0, 0]),
        "ks3_site": beta**2 * g_exact,
        "site_rule_excluded": excluded,
        "is_contact": (int(a), int(b)) == contact,
    }


def _check_centres(rt: ReactantTokens, token: int, sites: Sequence[int], role: str) -> None:
    centres = rt.centres(token)
    bad = [int(x) for x in sites if int(x) not in centres]
    if bad:
        msg = f"{role} site(s) {bad} are not centres of token {token} (centres {sorted(centres)})"
        raise _error("SITE_NOT_A_CENTRE", msg, status=422, detail={"token": token, "sites": bad})


def site_pairs(
    work: PoolWork,
    kernels: PairKernels,
    donor: int,
    acceptor: int,
    donor_site: int | None = None,
    acceptor_site: int | None = None,
) -> dict:
    """`SitePairsOut` fields: the oriented contact of (donor, acceptor) at each requested site
    pair (`block_contacts` restricted to one site pair: S, K beta, CLASH), KS3 at those sites
    (beta^2 G_exact) and the intramolecular site rule. A side without a site lists every
    centre of its token (centres by weight), so no site at all gives every centre pair."""
    from ksmech.physics.ks_terms import token_gap_kernel

    an, rt = work.an, work.rt
    i, j = an.cell(int(donor), int(acceptor))
    p = _pair(an, rt, i, j)
    d_sites = [int(donor_site)] if donor_site is not None else list(rt.centres(p.donor))
    a_sites = [int(acceptor_site)] if acceptor_site is not None else list(rt.centres(p.acceptor))
    _check_centres(rt, p.donor, d_sites, "donor")
    _check_centres(rt, p.acceptor, a_sites, "acceptor")
    g_exact = float(token_gap_kernel(p.r, p.s))
    contact = tuple(int(x) for x in an.contact_sites[i, j])
    rows = [
        _site_pair_row(rt, p, a, b, kernels, g_exact, contact) for a in d_sites for b in a_sites
    ]
    return {
        "state_id": work.state.state_id,
        "donor": p.donor,
        "acceptor": p.acceptor,
        "G": g_exact,
        "site_pairs": rows,
        "note": SITE_PAIR_NOTE,
    }


def _site_key(site: int | None) -> int | None:
    return None if site is None else int(site)


def site_pairs_path(
    npz: Path, donor: int, acceptor: int, donor_site: int | None, acceptor_site: int | None
) -> Path:
    """The cached site-pair block of (donor, acceptor) at the requested sites (None: every
    centre of that side), next to the analysis npz it was computed from. The npz name holds
    the physics fingerprint and the token digest, so a new analysis never reads an old block."""
    import hashlib
    from pathlib import Path

    npz = Path(npz)
    d, a = int(donor), int(acceptor)
    key = f"{npz.name}|{d}|{a}|{_site_key(donor_site)}|{_site_key(acceptor_site)}"
    return npz.parent / f"site-pairs-{hashlib.sha256(key.encode()).hexdigest()[:16]}.json"


def t_site_pairs(rt: Runtime, payload: dict) -> dict:
    """{"state", "donor", "acceptor", "donor_site"?, "acceptor_site"?} -> `SitePairsOut`
    fields (design 7.4), also written to `site_pairs_path` of the analysis it read."""
    import json

    from ksmech.serve.analysis import pool_work
    from ksmech.serve.contracts import PoolState
    from ksmech.serve.pools import atomic_write_text

    state = PoolState.from_dict(payload["state"])
    work = pool_work(rt, state)
    donor, acceptor = int(payload["donor"]), int(payload["acceptor"])
    d_site, a_site = payload.get("donor_site"), payload.get("acceptor_site")
    block = site_pairs(work, rt.kernels, donor, acceptor, d_site, a_site)
    path = site_pairs_path(work.npz, donor, acceptor, d_site, a_site)
    atomic_write_text(path, json.dumps(block, sort_keys=True) + "\n", rt.settings)
    return block
