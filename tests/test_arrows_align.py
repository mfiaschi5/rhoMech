"""Arrow-to-orbital alignment by atom (`ksmech.arrows.align`, `ksmech.arrows.reactants`)."""

import dataclasses
from collections import Counter

import pytest

from ksmech.arrows.align import (
    AlignFlag,
    UnalignedReason,
    align_arrows,
    align_step,
    report,
)
from ksmech.arrows.reactants import (
    MissingTokens,
    ReactantTokens,
    TokenCache,
    record_correspondence,
)
from ksmech.arrows.roundtrip import evaluate_alignment
from ksmech.data import chem
from ksmech.data.io import read_training_steps, steps_path
from ksmech.data.pmechdb import parse_arrow_code
from ksmech.data.schema import Source
from ksmech.data.validation import build_step
from ksmech.qm.embed import prepare_mol


def make_step(rxn_code: str):
    rxn, code = rxn_code.split()
    return build_step(
        step_id="t", source=Source.PMECHDB, rxn_smiles=rxn, arrows=parse_arrow_code(code)
    )


def local(rt: ReactantTokens, token_id: int) -> int:
    """Orbital index within its own token file: comparable across reactant sides that share
    the token object."""
    j = token_id - rt.n_atoms
    return j - int(rt.orbital_offsets[int(rt.orbital_fragment[j])])


def local_atom(rt: ReactantTokens, g: int) -> int:
    return g - int(rt.atom_offsets[int(rt.atom_fragment[g])])


def test_correspondence_maps_heavy_atoms_and_explicit_hydrogens():
    frag = chem.parse_mapped("[H:20][CH2:21][CH:22](C)[Br:23]")
    key = chem.unmapped_canonical(frag)
    record = prepare_mol(key)
    mapping = record_correspondence(frag, record, key)
    assert len(mapping) == frag.GetNumAtoms() and len(set(mapping.values())) == len(mapping)
    for i, j in mapping.items():
        a, b = frag.GetAtomWithIdx(i), record.GetAtomWithIdx(j)
        assert a.GetAtomicNum() == b.GetAtomicNum()
    for bond in frag.GetBonds():  # every drawn bond is a record bond
        i, j = mapping[bond.GetBeginAtomIdx()], mapping[bond.GetEndAtomIdx()]
        assert record.GetBondBetweenAtoms(i, j) is not None


def test_correspondence_falls_back_to_the_skeleton_for_another_resonance_form():
    frag = chem.parse_mapped("[CH2-:10][CH:11]=[O:12]")
    record = prepare_mol("C=C[O-]")
    mapping = record_correspondence(frag, record, "C=C[O-]")
    symbols = {frag.GetAtomWithIdx(i).GetSymbol(): j for i, j in mapping.items()}
    assert record.GetAtomWithIdx(symbols["O"]).GetFormalCharge() == -1


def test_missing_tokens_are_reported_by_key():
    with pytest.raises(MissingTokens) as exc:
        ReactantTokens.from_smiles("[OH-:1].[CH3:2][Cl:3]", lambda key: None)
    assert sorted(exc.value.keys) == ["CCl", "[OH-]"]


def test_mini_fixture_aligns_fully(mini_steps, token_loader):
    for label, step in mini_steps.items():
        rt = ReactantTokens.from_smiles(step.reactants_smiles, token_loader)
        al = align_step(step, rt)
        assert al.complete, (label, al.unaligned)
        assert len(al.labels) + len(al.absorbed) == len(step.arrows), label
        for lab in al.labels:
            assert AlignFlag.ALIGN_FALLBACK not in lab.flags, label
            assert all(rt.occupied[t - rt.n_atoms] for t in lab.donor_ids)
            assert not any(rt.occupied[t - rt.n_atoms] for t in lab.acceptor_ids)
            assert all(lab.donor_site in rt.centres(t) for t in lab.donor_ids)
            assert all(lab.acceptor_site in rt.centres(t) for t in lab.acceptor_ids)


def test_sn2_acceptor_is_the_c_cl_antibonding_orbital(mini_steps, token_loader):
    step = mini_steps["sn2"]
    rt = ReactantTokens.from_smiles(step.reactants_smiles, token_loader)
    (label,) = align_arrows(step, rt)
    c, cl = rt.site_of_map(20), rt.site_of_map(21)
    assert label.donor_site == rt.site_of_map(10) and label.acceptor_site == c
    assert label.sink_partners == (cl,) and label.absorbed == (1,)
    for s in label.acceptor_ids:  # on both C and Cl, antibonding: descriptors only
        assert cl in rt.centres(s) and rt.orbital(s).descriptors.overlap_population < 0


def test_e2_chains_lone_pair_to_sigma_ch_to_sigma_star_c_br(mini_steps, token_loader):
    step = mini_steps["e2"]
    rt = ReactantTokens.from_smiles(step.reactants_smiles, token_loader)
    first, second = align_arrows(step, rt)
    h, c21, c22, br = (rt.site_of_map(m) for m in (20, 21, 22, 23))
    assert first.acceptor_site == h and first.sink_partners == (c21,)
    assert second.donor_site == c21 and second.donor_partner == h
    assert all(h in rt.centres(t) for t in second.donor_ids)
    assert second.sink_partners == (br,) and second.absorbed == (2,)


def test_acetaldehyde_enolate_drawn_both_ways_gives_the_same_label(token_loader):
    enolate = token_loader("C=C[O-]")

    def loader(key):  # one wavefunction for both drawings: resonance forms share the tokens
        return enolate if key in ("C=C[O-]", "[CH2-]C=O") else token_loader(key)

    carbanion = make_step(
        "[CH2-:10][CH:11]=[O:12].[CH3:20][Br:21]>>"
        "[CH2:10]([CH3:20])[CH:11]=[O:12].[Br-:21] 10=20;20,21=21"
    )
    enol = make_step(
        "[CH2:10]=[CH:11][O-:12].[CH3:20][Br:21]>>"
        "[CH2:10]([CH3:20])[CH:11]=[O:12].[Br-:21] 12=11;10,11=10,20;20,21=21"
    )
    labels = {}
    for name, step in (("carbanion", carbanion), ("enolate", enol)):
        rt = ReactantTokens.from_smiles(step.reactants_smiles, loader)
        al = align_step(step, rt)
        assert al.complete, name
        (lab,) = [x for x in al.labels if x.acceptor_site == rt.site_of_map(20)]
        assert any(f.tokens is enolate for f in rt.fragments)
        labels[name] = (rt, lab)
    (rt_a, a), (rt_b, b) = labels["carbanion"], labels["enolate"]
    assert AlignFlag.RESONANCE_FORM in a.flags
    assert local_atom(rt_a, a.donor_site) == local_atom(rt_b, b.donor_site)
    assert {local(rt_a, t) for t in a.donor_ids} == {local(rt_b, t) for t in b.donor_ids}
    assert len(a.donor_ids) == 2  # the C-C sigma and the C=C pi, not the C-H bonds
    assert {local(rt_a, t) for t in a.acceptor_ids} == {local(rt_b, t) for t in b.acceptor_ids}


def test_alignment_never_reads_the_classifier_type(mini_steps, token_loader):
    def relabelled(key):
        t = token_loader(key)
        orbitals = [dataclasses.replace(o, type="unknown") for o in t.orbitals]
        return dataclasses.replace(t, orbitals=orbitals)

    for step in mini_steps.values():
        a = align_arrows(step, ReactantTokens.from_smiles(step.reactants_smiles, token_loader))
        b = align_arrows(step, ReactantTokens.from_smiles(step.reactants_smiles, relabelled))
        assert a == b


def test_lone_heterolysis_is_unaligned_and_grouped(token_loader):
    step = make_step("[CH3:10][Br:11]>>[CH3+:10].[Br-:11] 10,11=11")
    al = align_step(step, ReactantTokens.from_smiles(step.reactants_smiles, token_loader))
    (u,) = al.unaligned
    assert not al.labels and u.reason == UnalignedReason.HETEROLYSIS_ONLY
    assert u.group == "bond>atom C-Br"


def test_sink_on_an_atom_with_only_d_virtuals_is_reported(token_loader):
    step = make_step("[OH-:10].[Cl-:20]>>[OH:10][Cl-2:20] 10=20")
    rt = ReactantTokens.from_smiles(step.reactants_smiles, token_loader)
    chloride = rt.site_of_map(20)
    assert all(
        rt.d_type[t - rt.n_atoms]
        for t in rt.tokens_on_atom[chloride]
        if not rt.occupied[t - rt.n_atoms]
    )
    al = align_step(step, rt)
    (u,) = al.unaligned  # the chloride fragment has no acceptor candidate at all
    assert u.reason == UnalignedReason.NO_ACCEPTOR and AlignFlag.SINK_ONLY_D in u.flags
    assert report([al], Counter())["n_sinks_on_only_d_atoms"] == 1


def test_label_dict_round_trip(mini_steps, token_loader):
    step = mini_steps["e2"]
    for lab in align_arrows(step, ReactantTokens.from_smiles(step.reactants_smiles, token_loader)):
        assert type(lab).from_dict(lab.to_dict()) == lab


@pytest.mark.realdata
def test_pmechdb_alignment_and_predicted_round_trip():
    path = steps_path("pmechdb")
    cache = TokenCache()
    if not path.exists() or not any(cache.tokens_dir.glob("*.npz")):
        pytest.skip("PMechDB steps or token cache missing")
    report = evaluate_alignment(read_training_steps(path), cache, member_check=False)
    print({k: v for k, v in report.items() if k != "unaligned_groups"})
    assert report["n_steps_aligned_or_tried"] > 0
    assert report["arrows_aligned_fraction"] >= 0.90
    assert report["flag_fraction"]["ALIGN_FALLBACK"] <= 0.10
    assert report["predicted"]["match_rate"] >= 0.95
