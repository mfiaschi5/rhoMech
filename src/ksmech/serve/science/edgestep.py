"""One step of a network as data: a reaction matrix applied to a pool state, with its arrow
reading and the orbital readings of its forming bonds (spec v9, "Inference": every returned
pathway lists, per step, the intermediate SMILES, the step probability, the pooled gap factor
of the anchor, the orbital pair that dominates it and the dominant canonical MO pair).

`step_record(state, R, ...)` is shared by the worker (`serve.predict.expansion`, every predicted
edge), the API's live check (`POST /api/step/check`) and the user's override. It returns the
`StepCheckOut` fields plus the edge fields: matrix, anchor, product state, product SMILES and
sigma keys (product fragment order), formal charges, the arrows (`serve.science.reading.decompose`
with the directed pooled gap) and one `OrbitalReadingOut` per forming bond that has a valid
channel. The readings need the pool's analysis (`serve.science.gatom.PooledGap`) and tokens; without
them the arrows are read without the gap (first matching) and `readings` is empty.

Atoms are pool atoms throughout; tokens are global token ids (`GET /api/pool/{id}/tokens`).
Diagnostic orbital labels (`donor_diag`, `acceptor_diag`: the classifier type and the elements
of the two leading centres) are display only, never inputs.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from ksmech.serve.science import reading, stepapply

if TYPE_CHECKING:
    from ksmech.arrows.delta_be import ReactionMatrix
    from ksmech.arrows.reactants import ReactantTokens
    from ksmech.serve.core.contracts import PoolState
    from ksmech.serve.science.gatom import PooledGap

Occupations = Callable[[str], object]  # token key -> QM occupations or None


def _directed(pg: PooledGap | None, rt: ReactantTokens | None):
    if pg is None or rt is None:
        return None
    g = rt.global_atom
    return lambda a, b: float(pg.directed[g[a], g[b]])


def diag_label(rt: ReactantTokens, token: int) -> str:
    """Display only: the classifier type and the elements of the two leading centres."""
    o = rt.orbital(token)
    f = int(rt.orbital_fragment[token - rt.n_atoms])
    t = rt.fragments[f].tokens
    centres = [a for a in o.descriptors.center_atoms[:2] if a >= 0]
    kind = getattr(o.type, "value", o.type)
    return f"{kind}({'-'.join(t.symbol(int(a)) for a in centres)})"


def _canonical(rt: ReactantTokens, r: int, s: int, occupations: Occupations | None) -> dict | None:
    from ksmech.serve.science.gatom import dominant_pair

    fr = rt.fragments[int(rt.orbital_fragment[r - rt.n_atoms])]
    fs = rt.fragments[int(rt.orbital_fragment[s - rt.n_atoms])]
    occ_r = occupations(fr.key) if occupations else None
    occ_s = occupations(fs.key) if occupations else None
    return dominant_pair(rt.orbital(r), rt.orbital(s), fr.tokens, fs.tokens, occ_r, occ_s)


def orbital_reading(pg, rt, i: int, j: int, occupations: Occupations | None) -> dict | None:
    """`OrbitalReadingOut` of the bond i-j (pool atoms; i the donor end of its arrow when the
    reading has one): the channel (r, s) with the largest term of G_atom[i, j]."""
    gi, gj = rt.global_atom[i], rt.global_atom[j]
    best = pg.best_channel(gi, gj)
    if best is None:
        return None
    row, col, term = best
    r, s = int(pg.rows[row]), int(pg.cols[col])
    return {
        "i": int(i),
        "j": int(j),
        "g_atom": float(pg.g_atom[gi, gj]),
        "donor_token": r,
        "acceptor_token": s,
        "g_rs": float(pg.G[row, col]),
        "term": float(term),
        "donor_diag": diag_label(rt, r),
        "acceptor_diag": diag_label(rt, s),
        "canonical": _canonical(rt, r, s, occupations),
    }


def _forming(R: ReactionMatrix, moves) -> list[tuple[int, int]]:
    """(donor end, other end) of every forming bond: from the reading, else sorted pairs."""
    if moves is not None:
        seen, out = set(), []
        for m in moves:
            f = m.forms
            if f is not None and tuple(sorted(f)) not in seen:
                seen.add(tuple(sorted(f)))
                out.append(f)
        return out
    return [(i, j) for i, j, d in R.bonds if d > 0]


def choose_anchor(R: ReactionMatrix, pg, rt) -> tuple[int, int, str] | None:
    """The admissible anchor of R with the largest |G_atom| (ties: the first)."""
    anchors = list(R.anchors())
    if not anchors:
        return None
    if pg is None or rt is None:
        return tuple(anchors[0])
    g = rt.global_atom
    return tuple(max(anchors, key=lambda a: abs(float(pg.g_atom[g[a[0]], g[a[1]]]))))


def _map_of(state: PoolState):
    from ksmech.data import chem

    mol = chem.parse_mapped(state.mapped_smiles)
    return lambda a: int(mol.GetAtomWithIdx(int(a)).GetAtomMapNum())


def _anchor_out(anchor) -> dict | None:
    if anchor is None:
        return None
    return {"i": int(anchor[0]), "j": int(anchor[1]), "sign": str(anchor[2])}


def step_record(
    state: PoolState,
    R: ReactionMatrix,
    *,
    pg: PooledGap | None = None,
    rt: ReactantTokens | None = None,
    occupations: Occupations | None = None,
    anchor: tuple[int, int, str] | None = None,
) -> dict:
    """The check and edge fields of R on `state` (module docstring). `ok` is False with the
    issues when R is no valid step; the product fields are filled whenever a product exists."""
    applied = stepapply.apply_matrix(state, R)
    issues = [x.to_dict() for x in applied.issues]
    out = {
        "ok": False,
        "issues": issues,
        "matrix": stepapply.matrix_io(R),
        "anchor": _anchor_out(anchor or choose_anchor(R, pg, rt)),
        "product_state": applied.state,
        "product_smiles": [],
        "sigma_keys": [],
        "formal_charges": [],
        "arrows": [],
        "readings": [],
        "touched_maps": [],
        "pmechdb_code": None,
        "readings_status": "ready" if pg is not None and rt is not None else "unavailable",
    }
    if any(x["code"] == stepapply.MATRIX_MALFORMED for x in issues) or R.size == 0:
        return out
    out["touched_maps"] = stepapply.touched_maps(state, R)
    if applied.product is not None:
        out["formal_charges"] = stepapply.formal_charges(applied.product, R)
    if applied.state is not None:
        out["product_smiles"], out["sigma_keys"] = stepapply.product_keys(applied.state)
    moves = reading.decompose(R, _directed(pg, rt))
    if moves is None:
        msg = "R has no decomposition into electron-pair moves (curved arrows)"
        issues.append(
            {"code": stepapply.NO_ARROW_READING, "message": msg, "atoms": stepapply.matrix_atoms(R)}
        )
    out["arrows"] = [m.to_dict() for m in moves or []]
    if moves:
        out["pmechdb_code"] = reading.pmech_code(moves, _map_of(state))
    if pg is not None and rt is not None:
        rows = [orbital_reading(pg, rt, a, b, occupations) for a, b in _forming(R, moves)]
        out["readings"] = [x for x in rows if x is not None]
    out["ok"] = not issues
    return out
