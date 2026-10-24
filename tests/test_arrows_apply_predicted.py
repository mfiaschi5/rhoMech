"""Predicted (token, site) arrows applied from their sites (`ksmech.arrows.apply_predicted`)."""

from collections import Counter

import numpy as np
import pytest

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
from ksmech.data.resonance import sigma_key


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
    out = apply_predicted_arrows(rt.mol, [PredictedArrow(lone_pair, o, sigma_star, c, cl)], rt)
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
    """The leaving bonds are labeled in the order the arrows are read: base -> H first breaks
    C-H at H; read the other way, H has already lost C-H and takes the base with NONE."""
    step = mini_steps["e2"]
    rt = _rt(step.reactants_smiles, token_loader)
    labels = align_step(step, rt).labels
    for order in (labels, labels[::-1]):
        out = apply_predicted_arrows(rt.mol, labeled_arrows(rt, order), rt)
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
    assert out.ok and out.key == sigma_key(arenium)


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
    from ksmech.arrows.align import NONE_LEAVING

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
            m = rt.mol_atom.get(b)
            bonds = [] if m is None else rt.mol.GetAtomWithIdx(m).GetNeighbors()
            choices = [NONE_LEAVING] + [rt.global_atom[n.GetIdx()] for n in bonds]
            arrows.append(PredictedArrow(r, a, s, b, int(rng.choice(choices))))
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
    out = apply_predicted_arrows(rt.mol, [PredictedArrow(hydride, h, sigma_star, hp, o)], rt)
    want = chem.canonicalize_mapped("CC[O-:21].[H:10][H:20].[Na+:11]")
    assert out.ok and out.smiles == want


def _one_arrow(rt, label, s):
    from ksmech.arrows.apply_predicted import leaving_labels

    (r,) = label.donor_ids
    arrow = PredictedArrow(r, label.donor_site, s, label.acceptor_site)
    (out,) = leaving_labels(rt, [arrow], [label])
    return out


def _lone_pair(rt, atom):
    occupied = [t for t in rt.tokens_on_atom[atom] if rt.occupied[t - rt.n_atoms]]
    return next(t for t in occupied if len(rt.centres(t)) == 1)


def test_the_leaving_bond_mask_follows_the_sink(token_loader):
    from ksmech.arrows.align import NONE_LEAVING, ArrowLabel

    rt = _rt("C[O-:10].C[Si:20](C)(C)[Cl:21].[CH3:30][Cl:31]", token_loader)
    o, si, cl = rt.site_of_map(10), rt.site_of_map(20), rt.site_of_map(21)
    c, cl2 = rt.site_of_map(30), rt.site_of_map(31)
    donor = _lone_pair(rt, o)
    si_star = _token(rt, False, {si, cl}, antibonding=True)
    c_star = _token(rt, False, {c, cl2}, antibonding=True)

    def label(site, s, partners=()):
        return ArrowLabel(frozenset({donor}), o, frozenset({s}), site, sink_partners=partners)

    # Si can expand its octet: NONE and every Si bond are choices, the dataset decides
    kept = _one_arrow(rt, label(si, si_star), si_star)
    assert kept.leaving == NONE_LEAVING and kept.none_allowed and kept.in_mask
    si_mol = rt.mol_atom[si]
    neighbours = {rt.global_atom[n.GetIdx()] for n in rt.mol.GetAtomWithIdx(si_mol).GetNeighbors()}
    assert {x.atom for x in kept.candidates} == neighbours
    assert {x.element for x in kept.candidates} == {6, 17}
    (partner,) = [x for x in kept.candidates if x.is_partner]
    assert partner.atom == cl and kept.breaks_bd == 0
    assert partner.wbi == pytest.approx(rt.bond_index(si, cl)) and partner.order == 1
    sn2 = _one_arrow(rt, label(si, si_star, (cl,)), si_star)
    assert sn2.leaving == cl and sn2.in_mask and sn2.breaks_bd == 1
    # a full-shell carbon must lose a bond: NONE is not a choice
    full = _one_arrow(rt, label(c, c_star, (cl2,)), c_star)
    assert not full.none_allowed and full.leaving == cl2 and full.in_mask
    assert not _one_arrow(rt, label(c, c_star), c_star).in_mask  # NONE at a full octet


def test_an_open_shell_sink_keeps_every_bond(token_loader):
    from ksmech.arrows.align import NONE_LEAVING, ArrowLabel

    rt = _rt("C[CH+:20]C.[OH-:10]", token_loader)
    o, c = rt.site_of_map(10), rt.site_of_map(20)
    empty = max(
        (t for t in rt.tokens_on_atom[c] if not rt.occupied[t - rt.n_atoms]),
        key=lambda t: rt.centres(t)[c],
    )
    lab = ArrowLabel(frozenset({_lone_pair(rt, o)}), o, frozenset({empty}), c)
    out = _one_arrow(rt, lab, empty)
    assert out.leaving == NONE_LEAVING and out.candidates == () and out.in_mask
    bad = ArrowLabel(lab.donor_ids, o, lab.acceptor_ids, c, sink_partners=(c - 1,))
    assert not _one_arrow(rt, bad, empty).in_mask


def test_apply_breaks_exactly_the_chosen_leaving_bond(token_loader):
    rt = _rt("C[O-:10].C[Si:20](C)(C)[Cl:21]", token_loader)
    o, si, cl = rt.site_of_map(10), rt.site_of_map(20), rt.site_of_map(21)
    donor, si_star = _lone_pair(rt, o), _token(rt, False, {si, cl}, antibonding=True)
    out = apply_predicted_arrows(rt.mol, [PredictedArrow(donor, o, si_star, si, cl)], rt)
    assert out.ok and "[Cl-:21]" in out.smiles
    methyl = next(g for g in rt.global_atom.values() if g not in (o, si, cl) and g != o - 1)
    out = apply_predicted_arrows(rt.mol, [PredictedArrow(donor, o, si_star, si, methyl)], rt)
    assert out.ok and "[CH3-]" in out.smiles and "[Cl:21]" in out.smiles  # the chosen bond
    out = apply_predicted_arrows(rt.mol, [PredictedArrow(donor, o, si_star, si, o - 1)], rt)
    assert out.reason == ApplyReason.UNKNOWN_ARROW  # not a bond at the sink


def test_labeled_leaving_bonds_reproduce_the_fixture_steps(mini_steps, token_loader):
    from ksmech.arrows.apply_predicted import leaving_labels

    for step in mini_steps.values():
        rt = ReactantTokens.from_smiles(step.reactants_smiles, token_loader)
        labels = align_step(step, rt).labels
        bare = labeled_arrows(rt, labels, with_leaving=False)
        assert all(lab is not None and lab.in_mask for lab in leaving_labels(rt, bare, labels))
        out = apply_predicted_arrows(rt.mol, labeled_arrows(rt, labels), rt)
        key = sigma_key(chem.parse_mapped(step.products_smiles))
        assert out.ok and out.key == key, step.step_id


def test_a_relayed_pi_bond_is_not_the_leaving_bond_of_the_later_arrow(token_loader):
    """Enolate alkylation: the C10=C11 pair moves to C10-C20 first, so when O- adds to C11
    the bond the dataset lists as leaving at C11 is already gone: C11 has an open shell and
    its leaving bond is NONE, inside the mask."""
    from ksmech.arrows.align import NONE_LEAVING
    from ksmech.arrows.apply_predicted import leaving_labels
    from ksmech.data.schema import Arrow, ElementaryStep

    step = ElementaryStep(
        "enolate",
        "pmechdb",
        None,
        None,
        "C[CH:10]=[C:11](C)[O-:12].C[CH:20](C)[Br:21]",
        "C[CH:10]([C:11](C)=[O:12])[CH:20](C)C.[Br-:21]",
        [
            Arrow("bond", (10, 11), "bond", (10, 20)),
            Arrow("bond", (20, 21), "atom", (21,)),
            Arrow("lone_pair", (12,), "atom", (11,)),
        ],
    )
    rt = _rt(step.reactants_smiles, token_loader)
    labels = align_step(step, rt).labels
    labs = leaving_labels(rt, labeled_arrows(rt, labels, with_leaving=False), labels)
    assert all(lab is not None and lab.in_mask for lab in labs)
    assert labs[0].leaving == rt.site_of_map(21) and labs[1].leaving == NONE_LEAVING
    out = apply_predicted_arrows(rt.mol, labeled_arrows(rt, labels), rt)
    key = sigma_key(chem.parse_mapped(step.products_smiles))
    assert out.ok and out.key == key
