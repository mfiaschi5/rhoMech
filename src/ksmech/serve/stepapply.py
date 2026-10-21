"""Check, apply and encode the arrows of one elementary step (adapters A17-A20).

- `check_arrows` (A17): `apply_predicted_arrows` silently accepts an atom-token id (negative
  indexing) and raises IndexError on an id that is too large, so every user arrow is checked
  first. Always errors: TOKEN_OUT_OF_RANGE, NOT_AN_ORBITAL_TOKEN, SITE_OUT_OF_RANGE,
  SITE_NOT_DRAWN, SAME_SITE. Errors unless `allow_invalid`: DONOR_NOT_DONOR,
  ACCEPTOR_NOT_ACCEPTOR, PAIR_EXCLUDED (detail.reason = the PairReason name),
  SITE_NOT_A_CENTRE. A warning only: SITE_PAIR_EXCLUDED (an intramolecular site pair the
  site rule excludes, `orbitals.tokens.site_pairs_excluded`).
- `apply_step` (A18): `apply_predicted_arrows` in the given order (order matters: the
  map_arrows gotcha), then the product state (`pools.state_from_mol`).
- `edit_diff` (A19): the bond-order, nonbonding-electron and charge changes between two
  index-aligned Kekule molecules (the product keeps the parent's atom indices and maps).
- `encode_dataset_arrows` (A20): the step as dataset arrows on map numbers (PMechDB style),
  built from the edit over every Kekule base of the parent (the predicted-arrow path may switch
  Kekule structure silently) and verified with `data.validation.build_step`.
- `arrow_out`: the `ArrowOut` dict of one arrow.

`edit_diff` runs on both sides; the rest runs in the worker. Upstream arrow modules are
imported inside the functions.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from rdkit import Chem, rdBase

from ksmech.serve import keys
from ksmech.serve.contracts import ArrowSpec, PoolState
from ksmech.serve.errors import Code, ServeError, error_body
from ksmech.serve.pools import state_from_mol

if TYPE_CHECKING:
    from ksmech.arrows.reactants import ReactantTokens
    from ksmech.data.schema import Arrow

log = logging.getLogger(__name__)

ALWAYS_ERRORS = frozenset(
    {
        Code.TOKEN_OUT_OF_RANGE,
        Code.NOT_AN_ORBITAL_TOKEN,
        Code.SITE_OUT_OF_RANGE,
        Code.SITE_NOT_DRAWN,
        Code.SAME_SITE,
    }
)
ERRORS_UNLESS_INVALID = frozenset(
    {Code.DONOR_NOT_DONOR, Code.ACCEPTOR_NOT_ACCEPTOR, Code.PAIR_EXCLUDED, Code.SITE_NOT_A_CENTRE}
)
WARNING_CODES = frozenset({Code.SITE_PAIR_EXCLUDED})
DONOR_REASONS = (1, 2)  # PairReason DONOR_FRACTIONAL, DONOR_NOT_DOUBLY_OCCUPIED
ACCEPTOR_REASONS = (3, 4, 5)  # ACCEPTOR_FRACTIONAL, ACCEPTOR_NOT_EMPTY, ACCEPTOR_D_TYPE
ENCODE_STEP_ID = "encode-check"


# ---- check (A17)


@dataclass(frozen=True)
class Issue:
    """One finding of `check_arrows` on the user's arrow number `arrow` (0-based)."""

    code: str
    message: str
    arrow: int
    blocking: bool
    detail: dict = field(default_factory=dict)

    def to_body(self) -> dict:
        """`ErrorBody` fields (request_id empty)."""
        severity = "error" if self.blocking else "warning"
        detail = {"arrow": self.arrow, "severity": severity, **self.detail}
        return error_body(self.code, self.message, detail=detail)

    def error(self) -> ServeError:
        detail = {"arrow": self.arrow, **self.detail}
        return ServeError(
            self.code, f"arrow {self.arrow}: {self.message}", status=422, detail=detail
        )


def _rows_cols(rt: ReactantTokens) -> tuple[np.ndarray, np.ndarray]:
    occ = np.asarray(rt.occupied, dtype=bool)
    return rt.n_atoms + np.flatnonzero(occ), rt.n_atoms + np.flatnonzero(~occ)


def _position(ids: np.ndarray, token: int) -> int | None:
    k = int(np.searchsorted(ids, token))
    return k if k < len(ids) and int(ids[k]) == int(token) else None


def _token_issues(rt, k: int, a: ArrowSpec, rows, cols) -> list[Issue]:
    out = []
    for role, token, ids, kind in (
        ("donor", a.donor, rows, "occupied"),
        ("acceptor", a.acceptor, cols, "virtual"),
    ):
        if not 0 <= token < rt.n_tokens:
            msg = f"{role} token {token} outside 0..{rt.n_tokens - 1}"
            out.append(Issue(Code.TOKEN_OUT_OF_RANGE, msg, k, True, {"token": token}))
        elif token < rt.n_atoms:
            msg = f"{role} token {token} is an atom token, not an orbital"
            out.append(Issue(Code.NOT_AN_ORBITAL_TOKEN, msg, k, True, {"token": token}))
        elif _position(ids, token) is None:
            msg = f"{role} token {token} is not a {kind} orbital token"
            out.append(Issue(Code.NOT_AN_ORBITAL_TOKEN, msg, k, True, {"token": token}))
    return out


def _site_issues(rt, k: int, a: ArrowSpec) -> list[Issue]:
    out = []
    for role, site in (("donor", a.donor_site), ("acceptor", a.acceptor_site)):
        if not 0 <= site < rt.n_atoms:
            msg = f"{role} site {site} outside 0..{rt.n_atoms - 1}"
            out.append(Issue(Code.SITE_OUT_OF_RANGE, msg, k, True, {"site": site}))
        elif site not in rt.mol_atom:
            msg = f"{role} site {site} is not a drawn atom (an implicit hydrogen)"
            out.append(Issue(Code.SITE_NOT_DRAWN, msg, k, True, {"site": site}))
    if a.donor_site == a.acceptor_site:
        msg = f"donor and acceptor site are both atom {a.donor_site}"
        out.append(Issue(Code.SAME_SITE, msg, k, True, {"site": a.donor_site}))
    return out


def _pair_issues(rt, k, a, reasons, rows, cols, allow_invalid: bool) -> list[Issue]:
    from ksmech.serve.candidates import reason_name

    i, j = _position(rows, a.donor), _position(cols, a.acceptor)
    code = int(reasons[i, j])
    if code == 0:
        return []
    name = reason_name(code)
    if code in DONOR_REASONS:
        issue, msg = Code.DONOR_NOT_DONOR, f"token {a.donor} is not a donor ({name})"
    elif code in ACCEPTOR_REASONS:
        issue, msg = Code.ACCEPTOR_NOT_ACCEPTOR, f"token {a.acceptor} is not an acceptor ({name})"
    else:
        issue, msg = Code.PAIR_EXCLUDED, f"pair ({a.donor}, {a.acceptor}) is excluded ({name})"
    return [Issue(issue, msg, k, not allow_invalid, {"reason": name, "row": i, "col": j})]


def _centre_issues(rt, k: int, a: ArrowSpec, allow_invalid: bool) -> list[Issue]:
    out = []
    for role, token, site in (
        ("donor", a.donor, a.donor_site),
        ("acceptor", a.acceptor, a.acceptor_site),
    ):
        centres = rt.centres(token)
        if site not in centres:
            msg = f"{role} site {site} is not a centre of token {token} ({sorted(centres)})"
            detail = {"token": token, "site": site, "centres": sorted(centres)}
            out.append(Issue(Code.SITE_NOT_A_CENTRE, msg, k, not allow_invalid, detail))
    return out


def _site_pair_issue(rt, k: int, a: ArrowSpec) -> list[Issue]:
    """The intramolecular site rule on this one site pair: a warning only."""
    from ksmech.orbitals.tokens import site_pairs_excluded

    n = rt.n_atoms
    f = int(rt.orbital_fragment[a.donor - n])
    if f != int(rt.orbital_fragment[a.acceptor - n]):
        return []
    if int(rt.atom_fragment[a.donor_site]) != f or int(rt.atom_fragment[a.acceptor_site]) != f:
        return []
    off = int(rt.atom_offsets[f])
    t = rt.fragments[f].tokens
    r, s = rt.orbital(a.donor), rt.orbital(a.acceptor)
    weight = float(rt.candidate.site_weight)
    hit = site_pairs_excluded(t, [r], [a.donor_site - off], [s], [a.acceptor_site - off], weight)
    if not bool(hit[0]):
        return []
    msg = (
        f"site pair ({a.donor_site}, {a.acceptor_site}) is a bond interacting with itself "
        "(intramolecular site rule)"
    )
    return [Issue(Code.SITE_PAIR_EXCLUDED, msg, k, False, {"reason": "SELF_SITE_PAIR"})]


def check_arrows(
    rt: ReactantTokens,
    arrows: Sequence[ArrowSpec],
    reasons: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    allow_invalid: bool,
) -> list[Issue]:
    """Every issue of every arrow, in arrow order. `reasons`, `rows`, `cols`: the analysis
    members of the state. An arrow with a structural error (always an error) is not checked
    further."""
    out: list[Issue] = []
    for k, a in enumerate(arrows):
        structural = _token_issues(rt, k, a, rows, cols) + _site_issues(rt, k, a)
        out += structural
        if structural:
            continue
        out += _pair_issues(rt, k, a, reasons, rows, cols, allow_invalid)
        out += _centre_issues(rt, k, a, allow_invalid)
        out += _site_pair_issue(rt, k, a)
    return out


# ---- apply (A18)


@dataclass
class ApplyOutcome:
    """The product of a step, or why there is none (APPLY_FAILED with the upstream
    ApplyReason, or SPECIES_KEY_FAILED when a product fragment cannot be keyed)."""

    ok: bool
    code: str | None
    upstream: str | None
    detail: str
    product: Chem.Mol | None = None
    state: PoolState | None = None

    @property
    def failure(self) -> str | None:
        """The scan's failure key: "APPLY_FAILED:<reason>" or "SPECIES_KEY_FAILED"."""
        if self.ok:
            return None
        if self.code == Code.APPLY_FAILED:
            return f"{Code.APPLY_FAILED}:{self.upstream}"
        return str(self.code)


def apply_step(rt: ReactantTokens, arrows: Sequence[ArrowSpec]) -> ApplyOutcome:
    """`apply_predicted_arrows` on the state's drawn molecule, arrows in the given order, then
    the product state. Arrows that would reach upstream with an atom-token id, an id out of
    range or an undrawn site fail as APPLY_FAILED with that check's code as upstream."""
    from ksmech.arrows.apply_predicted import PredictedArrow, apply_predicted_arrows

    rows, cols = _rows_cols(rt)
    for k, a in enumerate(arrows):
        issues = _token_issues(rt, k, a, rows, cols) + _site_issues(rt, k, a)
        if issues:
            return ApplyOutcome(False, Code.APPLY_FAILED, str(issues[0].code), issues[0].message)
    predicted = [PredictedArrow(a.donor, a.donor_site, a.acceptor, a.acceptor_site) for a in arrows]
    try:
        res = apply_predicted_arrows(rt.mol, predicted, rt)
    except (IndexError, KeyError, ValueError, NotImplementedError) as exc:
        return ApplyOutcome(
            False, Code.APPLY_FAILED, "UNKNOWN_ARROW", f"{type(exc).__name__}: {exc}"
        )
    if not res.ok:
        return ApplyOutcome(False, Code.APPLY_FAILED, str(res.reason), res.detail)
    try:
        state = state_from_mol(res.mol)
    except ServeError as exc:
        return ApplyOutcome(False, Code.SPECIES_KEY_FAILED, str(exc.code), exc.message, res.mol)
    return ApplyOutcome(True, None, None, "", res.mol, state)


# ---- edit diff (A19)


def _pair(i: int, j: int) -> tuple[int, int]:
    return (i, j) if i < j else (j, i)


@dataclass(frozen=True)
class EditDiff:
    """after - before, keyed by atom index; only nonzero entries."""

    bond_delta: dict[tuple[int, int], int]
    lone_delta: dict[int, int]
    charge_delta: dict[int, int]
    touched: set[int]
    touched_maps: list[int]  # sorted map numbers of the touched atoms


def _orders(mol: Chem.Mol) -> dict[tuple[int, int], int]:
    from ksmech.arrows.apply import TYPE_TO_ORDER

    return {
        _pair(b.GetBeginAtomIdx(), b.GetEndAtomIdx()): TYPE_TO_ORDER.get(b.GetBondType(), 1)
        for b in mol.GetBonds()
    }


def edit_diff(before: Chem.Mol, after: Chem.Mol) -> EditDiff:
    """Bond-order, nonbonding-electron and formal-charge changes over the union of bonds of
    two index-aligned Kekule molecules (`arrows.apply.TYPE_TO_ORDER`, `nonbonding_electrons`).
    Raises ValueError when the atom counts differ."""
    from ksmech.arrows.apply import nonbonding_electrons

    if before.GetNumAtoms() != after.GetNumAtoms():
        raise ValueError(f"{before.GetNumAtoms()} != {after.GetNumAtoms()} atoms")
    ob, oa = _orders(before), _orders(after)
    bond = {k: oa.get(k, 0) - ob.get(k, 0) for k in sorted(set(ob) | set(oa))}
    bond = {k: v for k, v in bond.items() if v}
    lone, charge = {}, {}
    for x, y in zip(before.GetAtoms(), after.GetAtoms(), strict=True):
        dn = nonbonding_electrons(y) - nonbonding_electrons(x)
        dq = y.GetFormalCharge() - x.GetFormalCharge()
        if dn:
            lone[x.GetIdx()] = dn
        if dq:
            charge[x.GetIdx()] = dq
    touched = set(lone) | set(charge) | {i for pair in bond for i in pair}
    maps = sorted(m for m in (after.GetAtomWithIdx(i).GetAtomMapNum() for i in touched) if m)
    return EditDiff(bond, lone, charge, touched, maps)


# ---- dataset-arrow encoding (A20)


class EncodeError(Exception):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = str(code)
        self.detail = detail


@dataclass
class EncodedStep:
    """The dataset arrows of a step on map numbers, over the Kekule base that explains it."""

    base_smiles: str
    products_smiles: str
    arrows: list[Arrow]
    code: str  # PMechDB arrow code (`pmechdb.format_arrow_code`)
    per_arrow: list[dict]  # per user arrow: source_kind, source/sink partner (global, map)
    base: Chem.Mol | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict:
        return {
            "base_smiles": self.base_smiles,
            "products_smiles": self.products_smiles,
            "arrows": [a.to_dict() for a in self.arrows],
            "code": self.code,
            "source_kinds": [x["source_kind"] for x in self.per_arrow],
            "per_arrow": [dict(x) for x in self.per_arrow],
        }

    def arrow_info(self, k: int) -> dict:
        return dict(self.per_arrow[k])


def kekule_alternatives(mol: Chem.Mol) -> list[Chem.Mol]:
    """Other Kekule structures of `mol` with identical formal charges (the public
    `ResonanceMolSupplier(mol, KEKULE_ALL, 64)` call `apply_predicted` uses), atom indices
    kept, deduplicated by canonical mapped SMILES; `mol` itself excluded."""
    from rdkit.Chem import rdchem

    from ksmech.arrows.apply_predicted import MAX_KEKULE
    from ksmech.data import chem

    flags = int(rdchem.ResonanceFlags.KEKULE_ALL)
    with rdBase.BlockLogs():
        supplier = rdchem.ResonanceMolSupplier(mol, flags, MAX_KEKULE)
        structures = [supplier[k] for k in range(len(supplier))]
    charges = [a.GetFormalCharge() for a in mol.GetAtoms()]
    seen, out = {chem.canonical_mapped_smiles(mol)}, []
    for s in structures:
        if s is None or [a.GetFormalCharge() for a in s.GetAtoms()] != charges:
            continue
        m = Chem.Mol(s)
        try:
            with rdBase.BlockLogs():
                m.UpdatePropertyCache(strict=False)
                Chem.Kekulize(m, clearAromaticFlags=True)
        except Chem.rdchem.MolSanitizeException:
            continue
        smi = chem.canonical_mapped_smiles(m)
        if smi not in seen:
            seen.add(smi)
            out.append(m)
    return out


@dataclass
class _Budget:
    """The edit still to explain while arrows are emitted (design 9.5 `construct`)."""

    dB: Counter
    dN: Counter
    maps: dict[int, int]  # mol index -> map number

    def m(self, x: int) -> int:
        return self.maps[x]

    def losing(self, a: int, exclude: int) -> list[int]:
        """Partners x != exclude of a whose bond to a lost order (dB < 0)."""
        return [
            (y if x == a else x)
            for (x, y), v in self.dB.items()
            if v < 0 and a in (x, y) and exclude not in (x, y)
        ]


def _source(bud: _Budget, a: int, b: int, out: list) -> tuple[str, int | None]:
    from ksmech.data.schema import Arrow, SinkKind, SourceKind

    if bud.dN[a] <= -2:  # a lost a pair net (or regains one later in a relay)
        out.append(Arrow(SourceKind.LONE_PAIR, (bud.m(a),), SinkKind.ATOM, (bud.m(b),)))
        bud.dN[a] += 2
        return "lone_pair", None
    partners = sorted(bud.losing(a, b), key=lambda c: (bud.dB[_pair(a, c)], bud.m(c)))
    if not partners:
        raise EncodeError(Code.ARROW_ENCODING_FAILED, f"SOURCE_UNDERIVABLE at map {bud.m(a)}")
    c = partners[0]
    sink = (bud.m(a), bud.m(b))
    out.append(Arrow(SourceKind.BOND, (bud.m(a), bud.m(c)), SinkKind.BOND, sink))
    bud.dB[_pair(a, c)] += 1
    return "bond", c


def _sink(bud: _Budget, a: int, b: int, centres: set[int], out: list) -> int | None:
    from ksmech.data.schema import Arrow, SinkKind, SourceKind

    cands = bud.losing(b, a)
    if not cands:
        return None
    d = min(cands, key=lambda x: (x not in centres, bud.dB[_pair(b, x)], bud.m(x)))
    out.append(Arrow(SourceKind.BOND, (bud.m(b), bud.m(d)), SinkKind.ATOM, (bud.m(d),)))
    bud.dB[_pair(b, d)] += 1
    bud.dN[d] -= 2
    return d


def _construct(d: EditDiff, rt, base: Chem.Mol, arrows: Sequence[ArrowSpec]):
    """(dataset arrows, per-arrow info) explaining the edit `d`, or EncodeError."""
    bud = _Budget(Counter(d.bond_delta), Counter(d.lone_delta), {})
    bud.maps = {a.GetIdx(): a.GetAtomMapNum() for a in base.GetAtoms()}
    out: list = []
    info = []
    try:
        for arrow in arrows:
            a, b = rt.mol_atom[arrow.donor_site], rt.mol_atom[arrow.acceptor_site]
            kind, c = _source(bud, a, b, out)
            bud.dB[_pair(a, b)] -= 1  # the formed order is explained
            centres = {rt.mol_atom[g] for g in rt.centres(arrow.acceptor) if g in rt.mol_atom}
            sink = _sink(bud, a, b, centres, out)
            info.append(_info(rt, bud, kind, c, sink))
    except (KeyError, ValueError) as exc:
        raise EncodeError(Code.ARROW_ENCODING_FAILED, f"{type(exc).__name__}: {exc}") from exc
    left = {k: v for k, v in bud.dB.items() if v} | {k: v for k, v in bud.dN.items() if v}
    if left:
        raise EncodeError(Code.ARROW_ENCODING_FAILED, f"LEFTOVER edit {left}")
    return out, info


def _info(rt, bud: _Budget, kind: str, c: int | None, d: int | None) -> dict:
    def glob(x: int | None) -> int | None:
        return None if x is None else int(rt.global_atom[x])

    return {
        "source_kind": kind,
        "source_partner": glob(c),
        "source_partner_map": None if c is None else bud.m(c),
        "sink_partner": glob(d),
        "sink_partner_map": None if d is None else bud.m(d),
    }


def encode_dataset_arrows(
    rt: ReactantTokens, product: Chem.Mol, arrows: Sequence[ArrowSpec]
) -> EncodedStep:
    """The step's dataset arrows (design 9.5): for each Kekule base (the drawn state first,
    then its alternatives), the edit to the product is explained arrow by arrow and the
    result must pass `build_step` (source user_curated). Raises
    EncodeError(ARROW_ENCODING_FAILED, <last failure>)."""
    from ksmech.data import chem, pmechdb
    from ksmech.data.schema import Source
    from ksmech.data.validation import StepRejected, build_step

    p_smi = chem.canonical_mapped_smiles(product)
    last = "no Kekule base"
    for base in [rt.mol, *kekule_alternatives(rt.mol)]:
        try:
            out, info = _construct(edit_diff(base, product), rt, base, arrows)
        except EncodeError as exc:
            last = exc.detail
            continue
        b_smi = chem.canonical_mapped_smiles(base)
        try:
            rxn = f"{b_smi}>>{p_smi}"
            build_step(
                step_id=ENCODE_STEP_ID, source=Source.USER_CURATED, rxn_smiles=rxn, arrows=out
            )
        except StepRejected as exc:
            last = f"{exc.reason}: {exc.detail}"
            continue
        code = pmechdb.format_arrow_code(out)
        return EncodedStep(b_smi, p_smi, out, code, info, base)
    raise EncodeError(Code.ARROW_ENCODING_FAILED, last)


# ---- ArrowOut


def _site_map(rt: ReactantTokens, site: int) -> int:
    idx = rt.mol_atom.get(int(site))
    return 0 if idx is None else int(rt.mol.GetAtomWithIdx(idx).GetAtomMapNum())


def _element(rt: ReactantTokens, site: int) -> str:
    f = int(rt.atom_fragment[int(site)])
    t = rt.fragments[f].tokens
    return t.symbol(int(site) - int(rt.atom_offsets[f]))


def _species(rt: ReactantTokens, token: int) -> str:
    f = int(rt.orbital_fragment[int(token) - rt.n_atoms])
    return keys.species_key_of(rt.fragments[f].key)[0]


def centres_map(rt: ReactantTokens, token: int) -> dict[str, float]:
    """Map number (as a string) -> centre weight of an orbital token."""
    return {str(_site_map(rt, a)): float(w) for a, w in rt.centres(int(token)).items()}


def _token_class(rt: ReactantTokens, token: int) -> str:
    from ksmech.serve.physics_view import token_class

    f = int(rt.orbital_fragment[int(token) - rt.n_atoms])
    return token_class(rt.orbital(int(token)), rt.fragments[f].tokens)


UNKNOWN_SOURCE = {
    "source_kind": "unknown",
    "source_partner": None,
    "source_partner_map": None,
    "sink_partner": None,
    "sink_partner_map": None,
}


def arrow_out(
    rt: ReactantTokens,
    arrow: ArrowSpec,
    site_source: str,
    enc_info: dict | None,
    brief: dict | None,
) -> dict[str, Any]:
    """`ArrowOut` fields of one arrow. `enc_info`: `EncodedStep.arrow_info(k)` (None when the
    encoding failed: source kind "unknown"); `brief`: `physics_view.brief_physics`."""
    rows, cols = _rows_cols(rt)
    i, j = _position(rows, arrow.donor), _position(cols, arrow.acceptor)
    info = UNKNOWN_SOURCE | (enc_info or {})
    return {
        "donor": int(arrow.donor),
        "donor_site": int(arrow.donor_site),
        "acceptor": int(arrow.acceptor),
        "acceptor_site": int(arrow.acceptor_site),
        "donor_site_map": _site_map(rt, arrow.donor_site),
        "acceptor_site_map": _site_map(rt, arrow.acceptor_site),
        "donor_element": _element(rt, arrow.donor_site),
        "acceptor_element": _element(rt, arrow.acceptor_site),
        "donor_species": _species(rt, arrow.donor),
        "acceptor_species": _species(rt, arrow.acceptor),
        "electrons": 2,
        **{k: info[k] for k in UNKNOWN_SOURCE},
        "site_source": site_source,
        "row": -1 if i is None else i,
        "col": -1 if j is None else j,
        "donor_centres_map": centres_map(rt, arrow.donor),
        "acceptor_centres_map": centres_map(rt, arrow.acceptor),
        "diagnostic": {
            "donor_class": _token_class(rt, arrow.donor),
            "acceptor_class": _token_class(rt, arrow.acceptor),
        },
        "physics": brief,
    }
