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
    for _ in range(300):
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
