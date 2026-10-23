"""Align dataset arrows to orbital tokens by atom, never by orbital type.

A dataset arrow that forms a bond has a source atom `a` (a lone pair on a, or a bond (a, c)
whose pair forms the new bond at a) and a sink atom `b` (where the new bond forms). Its label
is the 4-tuple of the spec, with sets in place of single orbitals:

- donor site a; donor set = every donor-candidate token with weight >= tau_atom on a (a token
  "on" an atom has that atom among its centres). For a bond source only the tokens that also
  sit on c are kept; if none does, all tokens on a are kept and the label is flagged
  SOURCE_RELAXED.
- acceptor site b; acceptor set = every acceptor-candidate token on b (d-type polarization
  functions excluded unless configured). When another arrow of the step moves the pair of a
  bond (b, x) away from b (a leaving arrow: bond b-x -> x, or the pair of b-x relayed on from
  x), that bond breaks at the sink and only the antibonding tokens (negative overlap
  population) also on x are kept. With none (the record often holds such a bond as nearly
  ionic: no token spans it), the tokens also on x are kept, and with none of those either,
  every token on b (SINK_RELAXED); `apply_predicted` then finds the leaving bond by its
  Wiberg index. For a bond source (a, c), an antibonding orbital of that bond
  (leading centres a and c, negative overlap population) is never an acceptor; a lone pair
  has no bond of its own, so nothing is excluded for it.
- a lone pair drawn on an atom that carries no one-centre occupied token (an enolate drawn as
  its carbanion) aligns to the tokens where the drawing has no room for the pair: those shared
  with a partner x by more tokens than the drawn bond order a-x, and those with three or more
  centres (RESONANCE_FORM).
- an empty set falls back to the nearest candidate tokens of the site's fragment by centroid
  distance (ALIGN_FALLBACK).

A heterolysis arrow (bond (p, q) -> lone pair on q) forms no bond. When another arrow of the
step forms a bond at p, it is that arrow's sink bond and is absorbed into its acceptor set;
otherwise it cannot be expressed as a donor -> acceptor pair and is reported as unaligned.

"One-centre" means `n_centers == 1`, "antibonding" a negative overlap population, "d-type" a
mostly-d main centre on an element without a valence d shell: all continuous descriptors.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from ksmech.arrows.reactants import ReactantTokens
from ksmech.data.schema import Arrow, ElementaryStep, SourceKind

FALLBACK_TOLERANCE = 0.25  # Angstrom beyond the nearest centroid still counted as nearest


class AlignFlag(StrEnum):
    SOURCE_RELAXED = "SOURCE_RELAXED"  # bond source: no token on both a and c
    SINK_RELAXED = "SINK_RELAXED"  # sink bond: no token on both b and d
    RESONANCE_FORM = "RESONANCE_FORM"  # lone pair drawn where no one-centre token sits
    ALIGN_FALLBACK = "ALIGN_FALLBACK"  # an empty set filled by centroid distance
    SINK_ONLY_D = "SINK_ONLY_D"  # every virtual token on the sink atom is d-type
    LABEL_FILTERED = "LABEL_FILTERED"  # set members with no valid partner (pair_valid) removed


class UnalignedReason(StrEnum):
    HETEROLYSIS_ONLY = "HETEROLYSIS_ONLY"  # bond -> lone pair that no forming arrow absorbs
    NO_DONOR = "NO_DONOR"  # no donor candidate in the site's fragment
    NO_ACCEPTOR = "NO_ACCEPTOR"  # no acceptor candidate in the site's fragment


@dataclass(frozen=True)
class ArrowLabel:
    """One aligned arrow. Token ids and sites are global indices of the reactant side
    (`ReactantTokens`): sites are atom tokens, so a site id is also its atom index."""

    donor_ids: frozenset[int]
    donor_site: int
    acceptor_ids: frozenset[int]
    acceptor_site: int
    flags: frozenset[AlignFlag] = frozenset()
    arrow_index: int = -1  # position of the dataset arrow in `step.arrows`
    source_kind: str = ""  # lone_pair | bond
    donor_partner: int = -1  # c of a bond source, -1 for a lone pair
    sink_partners: tuple[int, ...] = ()  # d of every bond breaking at the sink
    absorbed: tuple[int, ...] = ()  # heterolysis arrows folded into this acceptor set
    teacher_donor: int = -1  # r*: set by the sample builder, a valid pair with teacher_acceptor
    teacher_acceptor: int = -1  # s*
    breaks_bd: int | None = None  # 1 / 0: the bond b-d to s*'s partner d breaks in this step
    breaks_bd_forced: bool = False  # True: fixed by physics (`apply_predicted.break_labels`)

    def to_dict(self) -> dict:
        return {
            "donor_ids": sorted(self.donor_ids),
            "donor_site": self.donor_site,
            "acceptor_ids": sorted(self.acceptor_ids),
            "acceptor_site": self.acceptor_site,
            "flags": sorted(str(f) for f in self.flags),
            "arrow_index": self.arrow_index,
            "source_kind": self.source_kind,
            "donor_partner": self.donor_partner,
            "sink_partners": list(self.sink_partners),
            "absorbed": list(self.absorbed),
            "teacher_donor": self.teacher_donor,
            "teacher_acceptor": self.teacher_acceptor,
            "breaks_bd": self.breaks_bd,
            "breaks_bd_forced": self.breaks_bd_forced,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ArrowLabel:
        return cls(
            frozenset(d["donor_ids"]),
            d["donor_site"],
            frozenset(d["acceptor_ids"]),
            d["acceptor_site"],
            frozenset(AlignFlag(f) for f in d["flags"]),
            d["arrow_index"],
            d["source_kind"],
            d["donor_partner"],
            tuple(d["sink_partners"]),
            tuple(d["absorbed"]),
            d.get("teacher_donor", -1),
            d.get("teacher_acceptor", -1),
            d.get("breaks_bd"),
            d.get("breaks_bd_forced", False),
        )


@dataclass(frozen=True)
class Unaligned:
    arrow_index: int
    reason: UnalignedReason
    group: str  # "<source kind>><sink kind> <source element>-<sink element>"
    flags: frozenset[AlignFlag] = frozenset()  # what the attempt found (e.g. SINK_ONLY_D)


@dataclass
class StepAlignment:
    step_id: str
    n_arrows: int
    labels: list[ArrowLabel] = field(default_factory=list)
    absorbed: list[int] = field(default_factory=list)
    unaligned: list[Unaligned] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.unaligned


@dataclass(frozen=True)
class Role:
    """What one dataset arrow does, in map numbers."""

    index: int
    forms: bool  # a new bond (or a higher order) forms between donor and sink
    donor: int  # a: where the pair leaves from (heterolysis: the atom it leaves)
    partner: int | None  # c of a bond source
    sink: int  # b: where the new bond forms (heterolysis: the atom that keeps the pair)


def arrow_role(index: int, arrow: Arrow) -> Role:
    kind, dst = arrow.destination()
    src = arrow.source_atoms
    if arrow.source_kind is SourceKind.LONE_PAIR:
        (a,) = src
        return Role(index, True, a, None, next(x for x in dst if x != a))
    if arrow.source_kind is SourceKind.RADICAL:
        raise NotImplementedError("radical arrows: RMechDB extension")
    if kind == "lone_pair":
        (q,) = dst
        return Role(index, False, next(x for x in src if x != q), None, q)
    (a,) = set(src) & set(dst)
    c, b = (next(x for x in src if x != a), next(x for x in dst if x != a))
    return Role(index, True, a, c, b)


def leaving_pairs(roles: list[Role]) -> dict[int, list[tuple[int, int]]]:
    """Map atom p -> [(q, arrow index)] for every bond (p, q) whose pair moves away from p."""
    out: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r in roles:
        if not r.forms:
            out[r.donor].append((r.sink, r.index))
        elif r.partner is not None:
            out[r.partner].append((r.donor, r.index))
    return out


def antibonding_on(rt: ReactantTokens, token: int, bond: tuple[int, int] | None) -> bool:
    """`token` is an antibonding orbital of `bond` (global atoms): its two leading centres are
    the bond's atoms and its overlap population is negative. A lone-pair source has no bond
    (`None`), so nothing is its antibonding partner."""
    if bond is None:
        return False
    lead = list(rt.centres(token))[:2]
    overlap = rt.orbital(token).descriptors.overlap_population
    return len(lead) == 2 and set(lead) == set(bond) and overlap < 0


def nearest_tokens(rt: ReactantTokens, site: int, mask: np.ndarray) -> list[int]:
    """Candidate tokens (`mask` over orbitals) of the site's fragment whose centroid lies
    within FALLBACK_TOLERANCE of the nearest one."""
    frag = rt.atom_fragment[site]
    idx = np.flatnonzero(mask & (rt.orbital_fragment == frag))
    if idx.size == 0:
        return []
    dist = np.linalg.norm(rt.centroids[idx] - rt.atom_positions[site], axis=1)
    keep = idx[dist <= dist.min() + FALLBACK_TOLERANCE]
    return [rt.n_atoms + int(j) for j in keep]


def _on(rt: ReactantTokens, atom: int, mask: np.ndarray) -> list[int]:
    return [t for t in rt.tokens_on_atom.get(atom, []) if mask[t - rt.n_atoms]]


def excess_tokens(rt: ReactantTokens, a: int, on_a: list[int]) -> list[int]:
    """Tokens on `a` that the drawing has no bond for: shared with a partner x by more tokens
    than the drawn order a-x, or spread over three or more centres."""
    by_partner: dict[int, list[int]] = defaultdict(list)
    out = set()
    for t in on_a:
        centres = rt.centres(t)
        if len(centres) >= 3:
            out.add(t)
        for x in centres:
            if x != a:
                by_partner[x].append(t)
    for x, ts in by_partner.items():
        if len(ts) > rt.drawn_order(a, x):
            out.update(ts)
    return sorted(out)


def donor_set(rt: ReactantTokens, a: int, c: int | None) -> tuple[list[int], set[AlignFlag]]:
    on_a = _on(rt, a, rt.donor_ok)
    if c is not None:
        both = [t for t in on_a if c in rt.centres(t)]
        return (both, set()) if both else (on_a, {AlignFlag.SOURCE_RELAXED})
    if any(rt.orbital(t).descriptors.n_centers == 1 for t in on_a):
        return on_a, set()
    return excess_tokens(rt, a, on_a) or on_a, {AlignFlag.RESONANCE_FORM}


def acceptor_set(
    rt: ReactantTokens, b: int, partners: list[int], own: tuple[int, int] | None
) -> tuple[list[int], set[AlignFlag]]:
    on_b, flags = _on(rt, b, rt.acceptor_ok), set()
    if not on_b and _on(rt, b, ~rt.occupied):
        flags.add(AlignFlag.SINK_ONLY_D)
    if partners:
        both = [t for t in on_b if any(x in rt.centres(t) for x in partners)]
        anti = [t for t in both if rt.orbital(t).descriptors.overlap_population < 0]
        anti = [t for t in anti if not antibonding_on(rt, t, own)]
        if anti:
            on_b = anti
        elif both:
            on_b = both
        else:
            flags.add(AlignFlag.SINK_RELAXED)
    return [s for s in on_b if not antibonding_on(rt, s, own)], flags


def label_for(
    rt: ReactantTokens, role: Role, roles: list[Role], leaving: dict, group: str
) -> ArrowLabel | Unaligned:
    """The label of one bond-forming arrow, or why it has none."""
    a, b = rt.site_of_map(role.donor), rt.site_of_map(role.sink)
    c = None if role.partner is None else rt.site_of_map(role.partner)
    breaking = [(q, i) for q, i in leaving.get(role.sink, []) if i != role.index]
    partners = sorted({rt.site_of_map(q) for q, _ in breaking})
    donors, flags = donor_set(rt, a, c)
    if not donors:
        donors, flags = nearest_tokens(rt, a, rt.donor_ok), flags | {AlignFlag.ALIGN_FALLBACK}
    own = None if c is None else (a, c)
    acceptors, sink_flags = acceptor_set(rt, b, partners, own)
    flags |= sink_flags
    if not acceptors:
        allowed = rt.acceptor_ok.copy()
        allowed[
            [t - rt.n_atoms for t in rt.tokens_on_atom.get(a, []) if antibonding_on(rt, t, own)]
        ] = False
        acceptors, flags = nearest_tokens(rt, b, allowed), flags | {AlignFlag.ALIGN_FALLBACK}
    for ok, reason in (
        (donors, UnalignedReason.NO_DONOR),
        (acceptors, UnalignedReason.NO_ACCEPTOR),
    ):
        if not ok:
            return Unaligned(role.index, reason, group, frozenset(flags))
    return ArrowLabel(
        frozenset(donors),
        a,
        frozenset(acceptors),
        b,
        frozenset(flags),
        role.index,
        "lone_pair" if c is None else "bond",
        -1 if c is None else c,
        tuple(partners),
        tuple(sorted(i for _, i in breaking if not roles[i].forms)),
    )


def _group(rt: ReactantTokens, arrow: Arrow, role: Role) -> str:
    """Unaligned-report group: arrow shape and the elements of the donor and sink atoms."""
    sym = {a.GetAtomMapNum(): a.GetSymbol() for a in rt.mol.GetAtoms()}
    return f"{arrow.source_kind}>{arrow.sink_kind} {sym[role.donor]}-{sym[role.sink]}"


def align_step(step: ElementaryStep, rt: ReactantTokens) -> StepAlignment:
    """Labels, absorbed heterolysis arrows and unaligned arrows of one step."""
    roles = [arrow_role(i, a) for i, a in enumerate(step.arrows)]
    leaving = leaving_pairs(roles)
    out = StepAlignment(step.step_id, len(roles))
    sinks = {r.sink for r in roles if r.forms}
    for role, arrow in zip(roles, step.arrows, strict=True):
        group = _group(rt, arrow, role)
        if not role.forms:
            if role.donor in sinks:
                out.absorbed.append(role.index)
            else:
                out.unaligned.append(Unaligned(role.index, UnalignedReason.HETEROLYSIS_ONLY, group))
            continue
        item = label_for(rt, role, roles, leaving, group)
        (out.labels if isinstance(item, ArrowLabel) else out.unaligned).append(item)
    return out


def align_arrows(step: ElementaryStep, reactant_tokens: ReactantTokens) -> list[ArrowLabel]:
    """The ArrowLabels of `step` (see `align_step` for absorbed and unaligned arrows)."""
    return align_step(step, reactant_tokens).labels


def representatives(
    rt: ReactantTokens, label: ArrowLabel, valid: Callable[[int, int], bool] | None = None
) -> tuple[int, int]:
    """Teacher representatives (r*, s*): the donor and the acceptor with the largest weight on
    their site; ties go to the higher donor and the lower acceptor energy. `valid(r, s)`, when
    given, restricts the choice to valid pairs (s* among r*'s valid partners)."""
    ok = valid or (lambda r, s: True)
    donors = [t for t in label.donor_ids if any(ok(t, s) for s in label.acceptor_ids)]
    r = max(
        donors or label.donor_ids,
        key=lambda t: (rt.centres(t).get(label.donor_site, 0.0), rt.orbital(t).energy_ev, -t),
    )
    acceptors = [s for s in label.acceptor_ids if ok(r, s)]
    s = max(
        acceptors or label.acceptor_ids,
        key=lambda t: (rt.centres(t).get(label.acceptor_site, 0.0), -rt.orbital(t).energy_ev, -t),
    )
    return r, s


def report(alignments: list[StepAlignment], skipped: Counter) -> dict:
    """Alignment statistics over the steps that could be aligned; `skipped` counts the steps
    that could not be tried (reason -> count)."""
    labels = [lab for al in alignments for lab in al.labels]
    unaligned = [u for al in alignments for u in al.unaligned]
    n_arrows = sum(al.n_arrows for al in alignments)
    n_absorbed = sum(len(al.absorbed) for al in alignments)
    n_label_arrows = len(labels) + sum(
        u.reason != UnalignedReason.HETEROLYSIS_ONLY for u in unaligned
    )
    flags = Counter(f for lab in labels for f in lab.flags)
    groups = Counter((str(u.reason), u.group) for u in unaligned)
    only_d = sum(AlignFlag.SINK_ONLY_D in x.flags for x in [*labels, *unaligned])
    return {
        "n_steps_aligned_or_tried": len(alignments),
        "n_steps_complete": sum(al.complete for al in alignments),
        "steps_skipped": dict(skipped.most_common()),
        "n_arrows": n_arrows,
        "n_labels": len(labels),
        "n_absorbed_heterolysis": n_absorbed,
        "n_unaligned": len(unaligned),
        "arrows_aligned_fraction": (len(labels) + n_absorbed) / n_arrows if n_arrows else 0.0,
        "labels_aligned_fraction": len(labels) / n_label_arrows if n_label_arrows else 0.0,
        "mean_donor_set": float(np.mean([len(x.donor_ids) for x in labels])) if labels else 0,
        "mean_acceptor_set": float(np.mean([len(x.acceptor_ids) for x in labels])) if labels else 0,
        "flag_fraction": {str(f): flags[f] / len(labels) if labels else 0.0 for f in AlignFlag},
        "flag_count": {str(f): flags[f] for f in AlignFlag},
        "n_sinks_on_only_d_atoms": only_d,  # aligned by fallback or unaligned
        "unaligned_groups": [
            {"reason": r, "group": g, "count": n} for (r, g), n in groups.most_common()
        ],
    }
