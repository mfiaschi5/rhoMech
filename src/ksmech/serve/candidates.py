"""Why a (donor, acceptor) token pair is or is not an arrow candidate (adapter A8).

Upstream (`orbitals.tokens` masks, `physics.pairs.valid_pairs`) returns booleans only.
`pair_reasons` gives every cell of the pair block a `PairReason`, the first matching code in
enum order, read from the same descriptors and the same calls; the analysis asserts
`(pair_reasons == VALID) == valid_pairs` cell for cell (`CONSISTENCY_FAILED` otherwise).
"""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np

from ksmech.serve.errors import Code, ServeError

if TYPE_CHECKING:
    from ksmech.arrows.reactants import ReactantTokens
    from ksmech.orbitals.tokens import CandidateConfig, MoleculeTokens

SCOPES = ("all", "inter")
# tuple-level reasons: never in the pair matrix
SELF_SITE_PAIR = "SELF_SITE_PAIR"  # one intramolecular site pair excluded by site_pairs_excluded
SCOPE_EXCLUDED = "SCOPE_EXCLUDED"  # outside candidate_scope


class PairReason(IntEnum):
    VALID = 0
    DONOR_FRACTIONAL = 1  # config.is_fractional(n_r)
    DONOR_NOT_DOUBLY_OCCUPIED = 2  # n_r <= config.full
    ACCEPTOR_FRACTIONAL = 3
    ACCEPTOR_NOT_EMPTY = 4  # n_s >= config.empty
    ACCEPTOR_D_TYPE = 5  # t.is_d_type(s) and not allow_d_acceptors
    OWN_ANTIBOND = 6  # same fragment, is_antibonding_partner
    SELF_SITE = 7  # same fragment, site_excluded_matrix(t, o_f, v_f, config.site_weight)


def _donor_reason(n: float, config: CandidateConfig) -> int:
    if config.is_fractional(n):
        return PairReason.DONOR_FRACTIONAL
    if n <= config.full:
        return PairReason.DONOR_NOT_DOUBLY_OCCUPIED
    return PairReason.VALID


def _acceptor_reason(t: MoleculeTokens, o, config: CandidateConfig) -> int:
    n = o.descriptors.occupation
    if config.is_fractional(n):
        return PairReason.ACCEPTOR_FRACTIONAL
    if n >= config.empty:
        return PairReason.ACCEPTOR_NOT_EMPTY
    if not config.allow_d_acceptors and t.is_d_type(o):
        return PairReason.ACCEPTOR_D_TYPE
    return PairReason.VALID


def _pair_block(t: MoleculeTokens, config: CandidateConfig) -> np.ndarray:
    """OWN_ANTIBOND / SELF_SITE / VALID for the (occupied x virtual) block of one fragment."""
    from ksmech.orbitals.tokens import is_antibonding_partner, site_excluded_matrix

    o_f, v_f = t.occupied(), t.virtual()
    out = np.zeros((len(o_f), len(v_f)), dtype=np.int8)
    if not o_f or not v_f:
        return out
    own = np.array([[is_antibonding_partner(d, a) for a in v_f] for d in o_f], dtype=bool)
    self_site = site_excluded_matrix(t, o_f, v_f, config.site_weight)
    out[self_site] = PairReason.SELF_SITE
    out[own] = PairReason.OWN_ANTIBOND
    return out


def pair_reasons(fragments: list[MoleculeTokens], config: CandidateConfig) -> np.ndarray:
    """(n_occ, n_virt) int8 `PairReason` of every pair, in the pair-block order (fragment by
    fragment, each file's own order); the donor codes win over the acceptor codes, which win
    over the pair codes."""
    occ = [(f, o) for f, t in enumerate(fragments) for o in t.occupied()]
    virt = [(f, t, o) for f, t in enumerate(fragments) for o in t.virtual()]
    donor = np.array([_donor_reason(o.descriptors.occupation, config) for _, o in occ], np.int8)
    acc = np.array([_acceptor_reason(t, o, config) for _, t, o in virt], dtype=np.int8)
    out = np.zeros((len(occ), len(virt)), dtype=np.int8)
    r0 = np.cumsum([0, *[len(t.occupied()) for t in fragments]])
    c0 = np.cumsum([0, *[len(t.virtual()) for t in fragments]])
    for f, t in enumerate(fragments):
        out[r0[f] : r0[f + 1], c0[f] : c0[f + 1]] = _pair_block(t, config)
    out = np.where(acc[None, :] > 0, acc[None, :], out)
    out = np.where(donor[:, None] > 0, donor[:, None], out)
    return out.astype(np.int8)


def reason_name(code: int) -> str:
    return PairReason(int(code)).name


def scope_mask(rt: ReactantTokens, scope: str) -> np.ndarray:
    """(n_occ, n_virt) bool: every cell for "all"; cells whose donor and acceptor lie in
    different fragments for "inter". Raises ServeError(REQUEST_INVALID) for other scopes."""
    rows = np.flatnonzero(rt.occupied)
    cols = np.flatnonzero(~rt.occupied)
    if scope == "all":
        return np.ones((len(rows), len(cols)), dtype=bool)
    if scope == "inter":
        frag = np.asarray(rt.orbital_fragment)
        return frag[rows][:, None] != frag[cols][None, :]
    raise ServeError(
        Code.REQUEST_INVALID, f"candidate scope must be one of {SCOPES}, got {scope!r}"
    )
