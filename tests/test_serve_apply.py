"""stepapply.py (synthetic chemistry): the arrow checks, the SN2 step on MeO- + MeCl, the edit
diff, the dataset-arrow encoding (lone pair, pi donor, a Kekule alternative) verified by
build_step, explicit H as a leaving atom, and the ArrowOut dict."""

import re

import numpy as np
import pytest

from ksmech.serve import keys
from ksmech.serve import stepapply as sa
from ksmech.serve.contracts import ArrowSpec
from ksmech.serve.errors import Code
from tests.test_serve_fixtures import (  # noqa: F401 (fixtures)
    no_xtb,
    sn2_state,
    synthetic_loader,
    synthetic_root,
    synthetic_runtime,
    synthetic_settings,
)

pytestmark = pytest.mark.serve

PI_TYPES = ("pi", "delocalized_pi")
SN2 = ArrowSpec(16, 1, 28, 5)  # n(O) -> sigma*(C-Cl) at sites (O, C)
CH_ACCEPTOR = ArrowSpec(16, 1, 34, 5)  # n(O) -> sigma*(C-H9) at site C


def _rt(smiles, loader, candidate=None):
    from ksmech.arrows.reactants import ReactantTokens

    return ReactantTokens.from_smiles(smiles, loader, candidate)


def _check(rt, arrows, allow_invalid=False):
    from ksmech.serve.candidates import pair_reasons

    reasons = pair_reasons([f.tokens for f in rt.fragments], rt.candidate)
    rows, cols = sa._rows_cols(rt)
    return sa.check_arrows(rt, arrows, reasons, rows, cols, allow_invalid)


def _codes(issues):
    return [(str(x.code), x.blocking) for x in issues]


def _state(smiles_list, settings):
    from ksmech.serve.pools import SpeciesInput, initial_state

    return initial_state([SpeciesInput(s, 1) for s in smiles_list], settings.limits)


@pytest.fixture(scope="module")
def rt(sn2_state, synthetic_loader):  # noqa: F811
    return _rt(sn2_state.mapped_smiles, synthetic_loader)


def test_check_accepts_the_sn2_arrow(rt):
    assert _check(rt, [SN2]) == []


@pytest.mark.parametrize(
    ("arrow", "code"),
    [
        (ArrowSpec(1, 1, 28, 5), Code.NOT_AN_ORBITAL_TOKEN),  # an atom token as donor
        (ArrowSpec(16, 1, 99, 5), Code.TOKEN_OUT_OF_RANGE),
        (ArrowSpec(28, 5, 16, 1), Code.NOT_AN_ORBITAL_TOKEN),  # virtual donor, occupied acceptor
        (ArrowSpec(16, 1, 28, 42), Code.SITE_OUT_OF_RANGE),
        (ArrowSpec(16, 1, 28, 1), Code.SAME_SITE),
    ],
)
def test_check_always_errors(rt, arrow, code):
    issues = _check(rt, [arrow], allow_invalid=True)
    assert issues and issues[0].code == code and issues[0].blocking
    err = issues[0].error()
    assert err.status == 422 and err.detail["arrow"] == 0


def test_check_site_not_drawn(sn2_state, synthetic_loader):  # noqa: F811
    implicit = _rt("[CH3:1][O-:2].[CH3:3][Cl:4]", synthetic_loader)
    assert implicit.n_atoms == 10 and len(implicit.mol_atom) == 4
    issues = _check(implicit, [ArrowSpec(12, 2, 28, 5)], allow_invalid=True)
    assert _codes(issues) == [(Code.SITE_NOT_DRAWN, True)]


def test_check_non_donor(sn2_state, synthetic_loader):  # noqa: F811
    from ksmech.orbitals.tokens import CandidateConfig

    strict = _rt(sn2_state.mapped_smiles, synthetic_loader, CandidateConfig(full=2.5))
    issues = _check(strict, [SN2])
    assert _codes(issues) == [(Code.DONOR_NOT_DONOR, True)]
    assert issues[0].detail["reason"] == "DONOR_NOT_DOUBLY_OCCUPIED"
    assert _codes(_check(strict, [SN2], allow_invalid=True)) == [(Code.DONOR_NOT_DONOR, False)]


def test_check_non_acceptor(rt):
    issues = _check(rt, [ArrowSpec(16, 1, 29, 6)])  # an empty d function of Cl
    assert _codes(issues) == [(Code.ACCEPTOR_NOT_ACCEPTOR, True)]
    assert issues[0].detail["reason"] == "ACCEPTOR_D_TYPE"


def test_check_excluded_pair_with_and_without_allow_invalid(rt):
    own = ArrowSpec(25, 6, 28, 5)  # sigma(C-Cl) -> its own sigma*(C-Cl)
    issues = _check(rt, [own])  # its sites (Cl, C) are also the bond itself: a warning
    assert _codes(issues) == [(Code.PAIR_EXCLUDED, True), (Code.SITE_PAIR_EXCLUDED, False)]
    assert issues[0].detail["reason"] == "OWN_ANTIBOND"
    relaxed = _check(rt, [own], allow_invalid=True)
    assert _codes(relaxed) == [(Code.PAIR_EXCLUDED, False), (Code.SITE_PAIR_EXCLUDED, False)]
    body = relaxed[0].to_body()
    assert body["code"] == "PAIR_EXCLUDED" and body["detail"]["severity"] == "warning"


def test_check_site_not_a_centre(rt):
    issues = _check(rt, [ArrowSpec(16, 0, 28, 5)])  # the O lone pair at the C atom
    assert _codes(issues) == [(Code.SITE_NOT_A_CENTRE, True)]
    assert _codes(_check(rt, [ArrowSpec(16, 0, 28, 5)], True)) == [(Code.SITE_NOT_A_CENTRE, False)]


def test_check_site_pair_excluded_is_a_warning(rt):
    # sigma(C0-H2) -> sigma*(C0-O1) is a valid pair, but the site pair (C0, C0) is the C-O
    # bond interacting with itself
    issues = _check(rt, [ArrowSpec(12, 0, 20, 0)])
    assert (Code.SAME_SITE, True) in _codes(issues)
    issues = _check(rt, [ArrowSpec(12, 2, 20, 0)])
    assert _codes(issues) == [(Code.SITE_PAIR_EXCLUDED, False)]


def test_sn2_apply_keys_and_maps(rt, synthetic_loader):  # noqa: F811
    from ksmech.data import chem

    out = sa.apply_step(rt, [SN2])
    assert out.ok and out.failure is None
    assert [s.species_key for s in out.state.species] == ["COC", "[Cl-]"]
    assert out.state.pool_key == "COC.[Cl-]"
    before = sorted(a.GetAtomMapNum() for a in rt.mol.GetAtoms())
    assert sorted(a.GetAtomMapNum() for a in out.product.GetAtoms()) == before
    assert chem.canonical_mapped_smiles(out.product) == out.state.mapped_smiles
    nxt = _rt(out.state.mapped_smiles, synthetic_loader)  # the next state rebuilds
    assert nxt.n_atoms == rt.n_atoms and len(nxt.mol_atom) == nxt.n_atoms
    o_map = rt.mol.GetAtomWithIdx(rt.mol_atom[1]).GetAtomMapNum()
    assert nxt.site_of_map(o_map) in nxt.global_atom.values()


def test_apply_failures_are_reason_coded(rt):
    out = sa.apply_step(rt, [ArrowSpec(12, 2, 20, 0)])  # H2 has no pair beyond its C0 bond
    assert not out.ok and out.code == Code.APPLY_FAILED and out.upstream == "VALENCE"
    assert out.failure == "APPLY_FAILED:VALENCE"
    guarded = sa.apply_step(rt, [ArrowSpec(1, 1, 28, 5)])  # an atom token never reaches upstream
    assert guarded.failure == f"APPLY_FAILED:{Code.NOT_AN_ORBITAL_TOKEN}"


def test_edit_diff_touched_maps(rt):
    out = sa.apply_step(rt, [SN2])
    d = sa.edit_diff(rt.mol, out.product)
    o, c, cl = rt.mol_atom[1], rt.mol_atom[5], rt.mol_atom[6]
    assert d.bond_delta == {tuple(sorted((o, c))): 1, tuple(sorted((c, cl))): -1}
    assert d.lone_delta == {o: -2, cl: 2}
    assert d.charge_delta == {o: 1, cl: -1}
    assert d.touched == {o, c, cl} and d.touched_maps == [2, 6, 7]
    with pytest.raises(ValueError):
        sa.edit_diff(rt.mol, _first_atoms(rt.mol))


def _first_atoms(mol):
    from rdkit import Chem

    rw = Chem.RWMol(mol)
    rw.RemoveAtom(rw.GetNumAtoms() - 1)
    return rw.GetMol()


def test_encode_sn2(rt):
    from ksmech.data import chem
    from ksmech.data.schema import Source
    from ksmech.data.validation import build_step

    out = sa.apply_step(rt, [SN2])
    enc = sa.encode_dataset_arrows(rt, out.product, [SN2])
    assert enc.code == "2=6;6,7=7"  # a=b;b,d=d
    assert enc.base_smiles == chem.canonical_mapped_smiles(rt.mol)
    step = build_step(
        step_id="t",
        source=Source.USER_CURATED,
        rxn_smiles=f"{enc.base_smiles}>>{enc.products_smiles}",
        arrows=enc.arrows,
    )
    assert step.products_smiles == chem.canonicalize_mapped(out.state.mapped_smiles)
    info = enc.arrow_info(0)
    assert info["source_kind"] == "lone_pair" and info["source_partner"] is None
    assert info["sink_partner"] == 6 and info["sink_partner_map"] == 7
    d = enc.to_dict()
    assert d["arrows"][0] == {
        "source_kind": "lone_pair",
        "source_atoms": [2],
        "sink_kind": "atom",
        "sink_atoms": [6],
    }
    assert d["code"] == enc.code and d["source_kinds"] == ["lone_pair"]


def _symbol(rt, site):
    f = int(rt.atom_fragment[site])
    return rt.fragments[f].tokens.symbol(site - int(rt.atom_offsets[f]))


def _cl_acceptor(rt):
    """sigma*(C-Cl) and its C site."""
    for t in range(rt.n_atoms, rt.n_tokens):
        cen = rt.centres(t)
        virtual = not rt.occupied[t - rt.n_atoms]
        antibonding = rt.orbital(t).descriptors.overlap_population < 0
        if virtual and antibonding and {_symbol(rt, a) for a in cen} == {"C", "Cl"}:
            return t, next(a for a in cen if _symbol(rt, a) == "C")
    raise AssertionError("no sigma*(C-Cl)")


def test_encode_pi_donor(synthetic_settings, synthetic_loader):  # noqa: F811
    from ksmech.data.schema import SinkKind, SourceKind

    st = _state(["C=CC=O", "CCl"], synthetic_settings)
    rt = _rt(st.mapped_smiles, synthetic_loader)
    acc, c_site = _cl_acceptor(rt)
    pi = [  # the C=C pi bond: two carbon centres joined by a drawn double bond
        t
        for t in range(rt.n_atoms, rt.n_atoms + rt.n_orbitals)
        if rt.occupied[t - rt.n_atoms]
        and str(rt.orbital(t).type) == "pi"
        and {_symbol(rt, a) for a in rt.centres(t)} == {"C"}
    ]
    assert len(pi) == 1
    cen = rt.centres(pi[0])
    site = max(cen, key=cen.get)
    arrow = ArrowSpec(pi[0], site, acc, c_site)
    out = sa.apply_step(rt, [arrow])
    assert out.ok, out.detail
    enc = sa.encode_dataset_arrows(rt, out.product, [arrow])
    first = enc.arrows[0]
    assert first.source_kind == SourceKind.BOND and first.sink_kind == SinkKind.BOND  # a,c=a,b
    a_map = rt.mol.GetAtomWithIdx(rt.mol_atom[site]).GetAtomMapNum()
    assert set(first.source_atoms) & set(first.sink_atoms) == {a_map}
    assert re.fullmatch(r"\d+,\d+=\d+,\d+;\d+,\d+=\d+", enc.code)
    assert enc.arrow_info(0)["source_kind"] == "bond"
    assert enc.arrow_info(0)["source_partner_map"] in first.source_atoms


def test_encode_needs_a_kekule_alternative(synthetic_settings, synthetic_loader):  # noqa: F811
    from ksmech.data import chem

    st = _state(["c1ccccc1", "CCl"], synthetic_settings)
    rt = _rt(st.mapped_smiles, synthetic_loader)
    drawn = chem.canonical_mapped_smiles(rt.mol)
    assert len(sa.kekule_alternatives(rt.mol)) == 1
    acc, c_site = _cl_acceptor(rt)
    switched = []
    for t in range(rt.n_atoms, rt.n_tokens):
        if not rt.occupied[t - rt.n_atoms] or str(rt.orbital(t).type) not in PI_TYPES:
            continue
        for a in rt.centres(t):
            if _symbol(rt, a) != "C" or rt.atom_fragment[a] == rt.atom_fragment[c_site]:
                continue
            arrow = ArrowSpec(t, a, acc, c_site)
            out = sa.apply_step(rt, [arrow])
            assert out.ok, out.detail
            enc = sa.encode_dataset_arrows(rt, out.product, [arrow])
            if enc.base_smiles != drawn:
                switched.append((arrow, enc))
    assert switched, "no aromatic donor needed the other Kekule structure"
    arrow, enc = switched[0]
    assert enc.base_smiles == chem.canonical_mapped_smiles(sa.kekule_alternatives(rt.mol)[0])
    d = sa.edit_diff(enc.base, sa.apply_step(rt, [arrow]).product)
    assert len(d.touched_maps) == 4  # a, c, b, d: only the arrow atoms move


def test_encode_failure_is_reported(rt):
    out = sa.apply_step(rt, [SN2])
    with pytest.raises(sa.EncodeError) as e:
        sa.encode_dataset_arrows(rt, out.product, [ArrowSpec(16, 1, 36, 7)])  # O -> H7: leftover
    assert e.value.code == Code.ARROW_ENCODING_FAILED and e.value.detail


def test_explicit_h_is_the_leaving_atom(rt, synthetic_loader):  # noqa: F811
    out = sa.apply_step(rt, [CH_ACCEPTOR])
    assert out.ok and out.state.pool_key == "COCCl.[H-]"
    enc = sa.encode_dataset_arrows(rt, out.product, [CH_ACCEPTOR])
    assert enc.code == "2=6;6,10=10" and enc.arrow_info(0)["sink_partner"] == 9
    implicit = _rt("[CH3:1][O-:2].[CH3:3][Cl:4]", synthetic_loader)
    hidden = sa.apply_step(implicit, [CH_ACCEPTOR])  # H9 is not drawn: the C-Cl bond breaks
    assert hidden.code == Code.SPECIES_KEY_FAILED and hidden.upstream == Code.IMPLICIT_H_IN_POOL
    frags = keys.fragment_keys(hidden.product)
    assert sorted(f.species_key for f in frags) == ["COC", "[Cl-]"]


def test_arrow_out_fields(rt, synthetic_runtime):  # noqa: F811
    from ksmech.serve.schemas import ArrowOut

    out = sa.apply_step(rt, [SN2])
    enc = sa.encode_dataset_arrows(rt, out.product, [SN2])
    d = sa.arrow_out(rt, SN2, "rule", enc.arrow_info(0), None)
    # site_pair (optional, default None) is added by expansion only for non-contact sites
    assert set(d) <= set(ArrowOut.model_fields)
    assert set(ArrowOut.model_fields) - set(d) <= {"site_pair"}
    m = ArrowOut.model_validate(d)
    assert (m.donor_site_map, m.acceptor_site_map) == (2, 6)
    assert (m.donor_element, m.acceptor_element) == ("O", "C")
    assert (m.donor_species, m.acceptor_species) == ("C[O-]", "CCl")
    assert m.source_kind == "lone_pair" and m.sink_partner == 6 and m.sink_partner_map == 7
    assert m.site_source == "rule" and m.electrons == 2
    rows = rt.n_atoms + np.flatnonzero(rt.occupied)
    cols = rt.n_atoms + np.flatnonzero(~rt.occupied)
    assert rows[m.row] == SN2.donor and cols[m.col] == SN2.acceptor
    assert set(m.donor_centres_map) == {"2"}
    assert set(m.acceptor_centres_map) == {"6", "7"}
    assert m.acceptor_centres_map["6"] == pytest.approx(rt.centres(28)[5])
    assert m.diagnostic == {"donor_class": "lone_pair(O)", "acceptor_class": "sigma_star(C-Cl)"}
    unknown = sa.arrow_out(rt, SN2, "contact", None, None)
    assert unknown["source_kind"] == "unknown" and unknown["sink_partner"] is None
