"""Predicted (token, site) arrows applied from their sites (`ksmech.arrows.apply_predicted`)."""

from collections import Counter

import numpy as np

from ksmech.arrows.align import align_step
from ksmech.arrows.apply import ApplyReason
from ksmech.arrows.apply_predicted import (
    PredictedArrow,
    apply_predicted_arrows,
    labeled_arrows,
)
from ksmech.arrows.reactants import ReactantTokens
from ksmech.arrows.roundtrip import member_agreement, predicted_round_trip_step
from ksmech.data import chem
from ksmech.data.resonance import canonical_resonance_key


def _rt(smiles: str, loader) -> ReactantTokens:
    return ReactantTokens.from_smiles(smiles, loader)


def _token(
    rt: ReactantTokens,
    occupied: bool,
    on: set[int],
    n_centres: int | None = None,
    antibonding: bool | None = None,
) -> int:
    """First orbital token whose centres are exactly `on` (global atoms), by descriptors."""
    for j, o in enumerate(rt.orbitals):
        t = rt.n_atoms + j
        d = o.descriptors
        if bool(rt.occupied[j]) != occupied or set(rt.centres(t)) != on:
            continue
        if n_centres is not None and d.n_centers != n_centres:
            continue
        if antibonding is not None and (d.overlap_population < 0) != antibonding:
            continue
        return t
    raise LookupError(on)


def test_mini_fixture_predicted_round_trip(mini_steps, token_loader):
    for label, step in mini_steps.items():
        rt = _rt(step.reactants_smiles, token_loader)
        al = align_step(step, rt)
        assert predicted_round_trip_step(step, rt, al).status == "exact", label


def test_sn2_from_sites_breaks_the_antibonding_partner(token_loader):
    rt = _rt("[OH-:10].[CH3:20][Cl:21]", token_loader)
    o, c, cl = (rt.site_of_map(m) for m in (10, 20, 21))
    lone_pair = _token(rt, True, {o}, n_centres=1)
    sigma_star = _token(rt, False, {c, cl}, antibonding=True)
    out = apply_predicted_arrows(rt.mol, [PredictedArrow(lone_pair, o, sigma_star, c)], rt)
    assert out.ok and out.smiles == chem.canonicalize_mapped("[OH:10][CH3:20].[Cl-:21]")


def test_open_shell_sink_keeps_its_bonds(token_loader):
    """A carbocation takes the new bond without losing one, whatever acceptor is chosen."""
    rt = _rt("[OH2:10].[CH3:21][CH2+:20]", token_loader)
    o, c = rt.site_of_map(10), rt.site_of_map(20)
    lone_pair = _token(rt, True, {o}, n_centres=1)
    want = chem.canonicalize_mapped("[OH2+:10][CH2:20][CH3:21]")
    for s in rt.tokens_on_atom[c]:
        if rt.acceptor_ok[s - rt.n_atoms]:
            out = apply_predicted_arrows(rt.mol, [PredictedArrow(lone_pair, o, s, c)], rt)
            assert out.smiles == want


def test_e2_relay_works_in_either_arrow_order(mini_steps, token_loader):
    step = mini_steps["e2"]
    rt = _rt(step.reactants_smiles, token_loader)
    arrows = labeled_arrows(rt, align_step(step, rt).labels)
    for order in (arrows, arrows[::-1]):
        out = apply_predicted_arrows(rt.mol, order, rt)
        assert out.smiles == step.products_smiles


def test_pi_donor_without_a_lone_pair_uses_its_multiple_bond(token_loader):
    """Propene + H+: the C=C pi donates at the terminal carbon, the cation sits on C2."""
    rt = _rt("[CH2:10]=[CH:11][CH3:12].[H+:20]", token_loader)
    c1, c2, h = (rt.site_of_map(m) for m in (10, 11, 20))
    donors = [
        t for t in rt.tokens_on_atom[c1] if rt.donor_ok[t - rt.n_atoms] and c2 in rt.centres(t)
    ]
    acceptor = rt.tokens_on_atom[h][0]
    for r in donors:  # sigma or pi: the site rule decrements the multiple bond either way
        out = apply_predicted_arrows(rt.mol, [PredictedArrow(r, c1, acceptor, h)], rt)
        assert out.smiles == chem.canonicalize_mapped("[CH2:10]([H:20])[CH+:11][CH3:12]")


def _forcing_pair(rt: ReactantTokens) -> tuple[int, int, int]:
    """(map number, global site, ring pi token) where the token's centres exclude the site's
    drawn double-bond partner, so only another Kekulé form gives it a multiple bond there."""
    for m in range(1, 7):
        site = rt.site_of_map(m)
        a = rt.mol_atom[site]
        partner = next(
            rt.global_atom[b.GetOtherAtomIdx(a)]
            for b in rt.mol.GetAtomWithIdx(a).GetBonds()
            if b.GetBondTypeAsDouble() == 2
        )
        for t in rt.tokens_on_atom[site]:
            centres = rt.centres(t)
            if rt.donor_ok[t - rt.n_atoms] and len(centres) >= 3 and partner not in centres:
                return m, site, t
    raise LookupError("no ring pi token forces a Kekulé switch")


def test_kekule_switch_finds_the_multiple_bond(token_loader):
    """The donor is a ring pi token whose centres exclude the site's drawn double-bond
    partner: the site gets a multiple bond to a centre of the token by switching Kekulé form,
    and the product is the arenium cation."""
    rt = _rt("[cH:1]1[cH:2][cH:3][cH:4][cH:5][cH:6]1.[CH3+:20]", token_loader)
    m, site, ring_pi = _forcing_pair(rt)
    me = rt.site_of_map(20)
    acceptor = next(t for t in rt.tokens_on_atom[me] if rt.acceptor_ok[t - rt.n_atoms])
    out = apply_predicted_arrows(rt.mol, [PredictedArrow(ring_pi, site, acceptor, me)], rt)
    ring = [(m + k - 1) % 6 + 1 for k in range(6)]  # the ring numbered from the site
    arenium = chem.parse_mapped(
        "[CH3:20][CH:{}]1[CH+:{}][CH:{}]=[CH:{}][CH:{}]=[CH:{}]1".format(*ring)
    )
    assert out.ok and out.key.key == canonical_resonance_key(arenium).key


def test_sites_must_be_drawn_atoms(token_loader):
    rt = _rt("[OH-:10].[CH4:20]", token_loader)
    implicit_h = next(g for g in range(rt.n_atoms) if g not in rt.mol_atom)
    lp = _token(rt, True, {rt.site_of_map(10)}, n_centres=1)
    out = apply_predicted_arrows(
        rt.mol, [PredictedArrow(lp, rt.site_of_map(10), lp, implicit_h)], rt
    )
    assert out.reason == ApplyReason.UNKNOWN_ARROW


def test_set_members_agree_on_the_mini_fixture(mini_steps, token_loader):
    total = Counter()
    for step in mini_steps.values():
        rt = _rt(step.reactants_smiles, token_loader)
        total.update(member_agreement(step, rt, align_step(step, rt)))
    assert total[("donor", False)] == 0 and total[("acceptor", False)] == 0
    assert total[("donor", True)] > 0


def test_property_heavy_atoms_and_charge_are_conserved(token_loader):
    rng = np.random.default_rng(1)
    sides = [
        "[OH-:1].[CH3:2][Cl:3]",
        "[CH3:1][C:2](=[O:3])[CH3:4].[NH3:5]",
        "[CH2:1]=[CH:2][CH:3]=[O:4].[C-:5]#[N:6]",
        "[H:1][Br:2].[CH2:3]=[CH2:4]",
    ]
    n_ok = 0
    for _ in range(600):  # random arrows: few give a valid product
        rt = _rt(sides[rng.integers(len(sides))], token_loader)
        donors = np.flatnonzero(rt.donor_ok) + rt.n_atoms
        acceptors = np.flatnonzero(rt.acceptor_ok) + rt.n_atoms
        arrows = []
        for _ in range(rng.integers(1, 3)):
            r, s = int(rng.choice(donors)), int(rng.choice(acceptors))
            a = int(rng.choice(list(rt.centres(r))))
            b = int(rng.choice(list(rt.centres(s))))
            arrows.append(PredictedArrow(r, a, s, b))
        out = apply_predicted_arrows(rt.mol, arrows, rt)
        if not out.ok:
            continue
        n_ok += 1
        assert chem.heavy_atom_count(out.mol) == chem.heavy_atom_count(rt.mol)
        assert chem.formal_charge(out.mol) == chem.formal_charge(rt.mol)
        assert chem.hydrogen_count(out.mol) == chem.hydrogen_count(rt.mol)
    assert n_ok >= 30


def test_an_empty_orbital_of_an_expanding_atom_takes_the_pair_without_losing_a_bond(token_loader):
    """TiCl4 + a carbonyl O: the acceptor is Ti's own empty orbital (one centre); Ti (period 4)
    may exceed the octet, so no Ti-Cl bond breaks (the old rule expelled a chloride)."""
    rt = _rt("C=[O:10].[Cl][Ti:20]([Cl])([Cl])[Cl]", token_loader)
    o, ti = rt.site_of_map(10), rt.site_of_map(20)
    lone_pair = next(
        t for t in rt.tokens_on_atom[o] if rt.occupied[t - rt.n_atoms] and len(rt.centres(t)) == 1
    )
    empty = next(
        t
        for t in rt.tokens_on_atom[ti]
        if not rt.occupied[t - rt.n_atoms] and set(rt.centres(t)) == {ti}
    )
    out = apply_predicted_arrows(rt.mol, [PredictedArrow(lone_pair, o, empty, ti)], rt)
    want = chem.canonicalize_mapped("C=[O+:10][Ti-:20]([Cl])([Cl])([Cl])[Cl]")
    assert out.ok and out.smiles == want


def test_a_hydride_gives_the_pair_of_its_ionic_bond(token_loader):
    """NaH + an alcohol O-H: the hydride's pair sits on H (no occupied token spans Na-H), so the
    drawn H-Na bond gives it; the O-H sigma* breaks at O."""
    step = "CC[O:21][H:20].[H:10][Na:11]"
    rt = _rt(step, token_loader)
    h, hp, o = rt.site_of_map(10), rt.site_of_map(20), rt.site_of_map(21)
    occupied_on_h = [t for t in rt.tokens_on_atom[h] if rt.occupied[t - rt.n_atoms]]
    hydride = max(occupied_on_h, key=lambda t: rt.centres(t)[h])
    sigma_star = _token(rt, False, {hp, o}, antibonding=True)
    out = apply_predicted_arrows(rt.mol, [PredictedArrow(hydride, h, sigma_star, hp)], rt)
    want = chem.canonicalize_mapped("CC[O-:21].[H:10][H:20].[Na+:11]")
    assert out.ok and out.smiles == want


def _one_arrow_flag(rt, label, s):
    from ksmech.arrows.apply_predicted import break_labels

    (r,) = label.donor_ids
    arrow = PredictedArrow(r, label.donor_site, s, label.acceptor_site)
    (out,) = break_labels(rt, [arrow], [label])
    return out


def test_bond_break_labels_are_learned_only_where_the_tuple_cannot_decide(token_loader):
    from ksmech.arrows.align import ArrowLabel

    rt = _rt("C[O-:10].C[Si:20](C)(C)[Cl:21].[CH3:30][Cl:31]", token_loader)
    o, si, cl = rt.site_of_map(10), rt.site_of_map(20), rt.site_of_map(21)
    c, cl2 = rt.site_of_map(30), rt.site_of_map(31)
    donor = next(
        t for t in rt.tokens_on_atom[o] if rt.occupied[t - rt.n_atoms] and len(rt.centres(t)) == 1
    )
    si_star = _token(rt, False, {si, cl}, antibonding=True)
    c_star = _token(rt, False, {c, cl2}, antibonding=True)

    def label(site, s, partners=()):
        return ArrowLabel(frozenset({donor}), o, frozenset({s}), site, sink_partners=partners)

    # Si can expand its octet: the dataset decides, learned
    assert _one_arrow_flag(rt, label(si, si_star), si_star) == (0, False)  # pentacoordinate
    assert _one_arrow_flag(rt, label(si, si_star, (cl,)), si_star) == (1, False)  # SN2 at Si
    # a full-shell carbon must lose a bond: forced, whatever the label says
    assert _one_arrow_flag(rt, label(c, c_star), c_star) == (1, True)


def test_without_a_partner_in_s_the_lowest_wiberg_bond_at_b_leaves(token_loader):
    from dataclasses import replace

    from ksmech.arrows.align import ArrowLabel
    from ksmech.arrows.apply_predicted import LEAVING_AMBIGUOUS

    rt = _rt("C[O:10][Si:20](C)(C)[Cl:21]", token_loader)
    o, si, cl = rt.site_of_map(10), rt.site_of_map(20), rt.site_of_map(21)
    donor = next(
        t for t in rt.tokens_on_atom[o] if rt.occupied[t - rt.n_atoms] and len(rt.centres(t)) == 1
    )
    si_o_star = _token(rt, False, {si, o}, antibonding=True)  # its partner is the donor atom
    tokens = rt.fragments[0].tokens
    wbi = dict(tokens.bond_index)
    tokens.bond_index = {k: (0.40 if set(k) == {si, cl} else max(v, 0.9)) for k, v in wbi.items()}

    def label(partners=()):
        return ArrowLabel(frozenset({donor}), o, frozenset({si_o_star}), si, sink_partners=partners)

    # the flag is about the bond the rule names: Si-Cl, the lowest Wiberg index at Si
    assert _one_arrow_flag(rt, label(), si_o_star) == (0, False)
    assert _one_arrow_flag(rt, label((cl,)), si_o_star) == (1, False)
    arrow = PredictedArrow(donor, o, si_o_star, si, breaks_bd=1)
    out = apply_predicted_arrows(rt.mol, [arrow], rt)
    assert out.ok and "[Cl-:21]" in out.smiles and LEAVING_AMBIGUOUS not in out.flags
    tied = {k: (0.40 if si in k and o not in k else v) for k, v in tokens.bond_index.items()}
    tokens.bond_index = tied  # Si-C and Si-Cl both at 0.40: the choice is flagged
    out = apply_predicted_arrows(rt.mol, [replace(arrow)], rt)
    assert LEAVING_AMBIGUOUS in out.flags
    tokens.bond_index = wbi


def test_forced_flags_read_on_the_edited_state_agree_with_apply(mini_steps, token_loader):
    for step in mini_steps.values():
        rt = ReactantTokens.from_smiles(step.reactants_smiles, token_loader)
        labels = align_step(step, rt).labels
        learned = labeled_arrows(rt, labels)
        both = labeled_arrows(rt, labels, with_forced=True)
        assert any(a.breaks_bd is not None for a in both) or not labels
        one = apply_predicted_arrows(rt.mol, learned, rt)
        two = apply_predicted_arrows(rt.mol, both, rt)
        assert (one.ok, one.smiles) == (two.ok, two.smiles), step.step_id
