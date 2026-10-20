"""Training samples, split files and the collate function (`ksmech.arrows.steps`)."""

import dataclasses
import json

import pytest
import torch

from ksmech.arrows.cli import main as arrows_main
from ksmech.arrows.steps import (
    SAMPLE_VERSION,
    MechanismSample,
    SkipReason,
    build_all_samples,
    build_samples,
    collate_samples,
    load_samples,
    make_loader,
    sample_for_step,
    samples_path,
)
from ksmech.data.io import steps_path, write_steps
from ksmech.data.schema import Arrow, ElementaryStep


@pytest.fixture
def mini_processed(tmp_path, mini_result):
    """The mini fixture's accepted steps as data/processed/pmechdb/steps.parquet, with mechanism
    classes so leave-one-class-out can run; plus a step whose tokens do not exist."""
    classes = ["substitution", "proton_transfer", "addition", "elimination", "resonance"]
    steps = [
        dataclasses.replace(s, mechanism_class=c)
        for s, c in zip(mini_result.steps, classes, strict=True)
    ]
    steps.append(
        ElementaryStep(
            step_id="no-tokens",
            source="pmechdb",
            pathway_id=None,
            step_index=None,
            reactants_smiles="[CH3:10][I:11]",
            products_smiles="[CH3+:10].[I-:11]",
            arrows=[Arrow("bond", (10, 11), "atom", (11,))],
            mechanism_class="substitution",
        )
    )
    write_steps(steps, steps_path("pmechdb", tmp_path))
    return tmp_path


def missing_iodide(token_loader):
    return lambda key: None if key == "CI" else token_loader(key)


def test_sample_holds_tokens_labels_and_product_key(mini_steps, token_loader, contact_kernel):
    step = mini_steps["sn2"]
    s = sample_for_step(step, token_loader, contact_kernel)
    assert isinstance(s, MechanismSample)
    assert len(s.fragment_keys) == 2 and set(s.atom_fragment) == {0, 1}
    assert s.atom_features.shape[0] == s.n_atoms == len(s.atom_map)
    assert s.orbital_spectra.shape == (s.n_orbitals, 8, 2)
    assert (s.atom_map > 0).sum() == 3  # the three mapped atoms
    assert s.product_key and s.product_key_flag == "OK"
    (label,) = s.labels
    assert s.atom_map[label.donor_site] == 10 and s.atom_map[label.acceptor_site] == 20
    j, g, w = s.incidence.T
    assert (w >= 0.15 - 1e-6).all() and j.max() < s.n_orbitals and g.max() < s.n_atoms


def test_samples_round_trip_through_torch_save(mini_steps, token_loader, contact_kernel, tmp_path):
    s = sample_for_step(mini_steps["e2"], token_loader, contact_kernel)
    torch.save({"version": SAMPLE_VERSION, "samples": [s.to_dict()]}, tmp_path / "x.pt")
    (back,) = load_samples(tmp_path / "x.pt")
    assert back.labels == s.labels and back.product_key == s.product_key
    assert (back.orbital_features == s.orbital_features).all()


def test_skip_reasons(token_loader, contact_kernel):
    lone = ElementaryStep(
        "x",
        "pmechdb",
        None,
        None,
        "[CH3:10][Br:11]",
        "[CH3+:10].[Br-:11]",
        [Arrow("bond", (10, 11), "atom", (11,))],
    )
    assert sample_for_step(lone, token_loader, contact_kernel) == SkipReason.UNALIGNED_ARROW
    assert sample_for_step(lone, lambda key: None, contact_kernel) == SkipReason.MISSING_TOKENS


def test_samples_build_for_every_split_strategy(mini_processed, token_loader, contact_kernel):
    report = build_all_samples(
        "pmechdb",
        contact_kernel,
        processed_dir=mini_processed,
        loader=missing_iodide(token_loader),
    )
    assert not report["strategies_not_applicable"]
    assert report["skipped"] == {"MISSING_TOKENS": 1}
    names = set(report["splits"])
    assert {"random_train", "scaffold_test"} <= names
    assert any(n.startswith("loco-substitution_") for n in names)
    for split, r in report["splits"].items():
        path = samples_path("pmechdb", split, mini_processed)
        assert path.exists() and len(load_samples(path)) == r["n_samples"]
        assert r["n_samples"] + sum(r["skipped"].values()) == r["n_steps"]
    held_out = report["splits"]["loco-substitution_test"]
    assert held_out["n_steps"] == 2 and held_out["skipped"] == {"MISSING_TOKENS": 1}


def test_build_samples_writes_one_split(mini_processed, token_loader, contact_kernel):
    path = build_samples("pmechdb", "random_train", contact_kernel, mini_processed, token_loader)
    assert path.name == "samples_random_train.pt"
    report = json.loads(path.with_suffix(".json").read_text())
    assert report["split"] == "random_train" and report["n_samples"] >= 1
    with pytest.raises(ValueError):
        build_samples("pmechdb", "random_bogus", contact_kernel, mini_processed, token_loader)


def test_collate_pads_and_builds_masks(mini_steps, token_loader, contact_kernel):
    batch = [
        sample_for_step(mini_steps[k], token_loader, contact_kernel)
        for k in ("sn2", "e2", "resonance")
    ]
    b = collate_samples(batch)
    n_atoms = max(s.n_atoms for s in batch)
    n = n_atoms + max(s.n_orbitals for s in batch)
    assert b["token_mask"].shape == (3, n) and b["incidence"].shape == (3, n, n_atoms)
    for i, s in enumerate(batch):
        assert b["token_mask"][i].sum() == s.n_atoms + s.n_orbitals
        assert b["is_atom"][i, : s.n_atoms].all() and not b["is_atom"][i, s.n_atoms :].any()
        assert (b["occupied"][i] | b["virtual"][i]).sum() == s.n_orbitals
        assert not (b["donor_mask"][i] & ~b["occupied"][i]).any()
        assert not (b["acceptor_mask"][i] & ~b["virtual"][i]).any()
        assert b["incidence"][i, : s.n_atoms, : s.n_atoms].equal(torch.eye(s.n_atoms))
        frag = b["fragment"][i]
        same = b["same_fragment"][i]
        real = b["token_mask"][i]
        assert same.equal((frag[:, None] == frag[None, :]) & real[:, None] & real[None, :])
        for k, lab in enumerate(s.labels):
            donors = b["donor_sets"][i, k].nonzero().flatten()
            assert len(donors) == len(lab.donor_ids)
            assert (b["incidence"][i, donors, b["donor_site"][i, k]] > 0).all()
            assert b["occupied"][i, donors].all()
        assert b["label_mask"][i].sum() == len(s.labels)
    assert (b["donor_site"][~b["label_mask"]] == -1).all()


def test_loader_and_cli(mini_processed, token_loader, contact_tables, monkeypatch, capsys):
    import ksmech.arrows.cli as cli
    from ksmech.physics.contact_table import save_tables

    monkeypatch.setattr(cli, "TokenCache", lambda *a, **k: missing_iodide(token_loader))
    args = ["--source", "pmechdb", "--processed-dir", str(mini_processed)]
    table = ["--contact-table", str(save_tables(contact_tables, mini_processed / "contact.json"))]
    missing = ["--kernel", "table", "--contact-table", str(mini_processed / "none.json")]
    assert arrows_main(["samples", *args, *missing]) == 2  # no contact table: refused, not guessed
    assert arrows_main(["samples", *args, "--kernel", "table", *table, "--strategy", "random"]) == 0
    oriented = ["--oriented-cache", str(mini_processed / "oriented")]  # the default kernel
    assert arrows_main(["samples", *args, *oriented, "--strategy", "random", "scaffold"]) == 0
    assert arrows_main(["roundtrip", *args]) == 0
    assert "100.00%" in capsys.readouterr().out
    loader = make_loader(samples_path("pmechdb", "random_train", mini_processed), batch_size=2)
    batch = next(iter(loader))
    assert batch["donor_sets"].dtype == torch.bool and len(batch["step_id"]) <= 2


def test_sample_stores_the_physics_pair_block(mini_steps, token_loader, contact_kernel):
    import numpy as np

    from ksmech.physics.pairs import (
        PAIR_FEATURES,
        atom_coulomb_matrix,
        donor_acceptor_block,
        pair_features,
        valid_pairs,
    )
    from ksmech.physics.params import REDUCED

    step = mini_steps["sn2"]
    s = sample_for_step(step, token_loader, contact_kernel)
    n_occ, n_virt = int(s.orbital_occupied.sum()), int((~s.orbital_occupied).sum())
    assert s.pair_z.shape == s.pair_defined.shape == (n_occ, n_virt, len(PAIR_FEATURES))
    assert s.pair_z.dtype == np.float32 and s.pair_valid.shape == (n_occ, n_virt)
    tokens = s.tokens(token_loader)
    z, defined = donor_acceptor_block(pair_features(tokens, contact_kernel, REDUCED))
    assert np.allclose(s.pair_z, z, rtol=1e-6, atol=0) and np.array_equal(s.pair_defined, defined)
    assert np.array_equal(s.pair_valid, valid_pairs(tokens))
    donors = s.donor_mask[s.orbital_occupied]
    acceptors = s.acceptor_mask[~s.orbital_occupied]
    assert not (s.pair_valid & ~(donors[:, None] & acceptors[None, :])).any()
    assert s.atom_coulomb.shape == (s.n_atoms, s.n_atoms)
    assert np.allclose(s.atom_coulomb, s.atom_coulomb.T)
    assert np.allclose(s.atom_coulomb, atom_coulomb_matrix(tokens, REDUCED), rtol=1e-6)


def test_old_sample_versions_are_refused(mini_steps, token_loader, contact_kernel, tmp_path):
    s = sample_for_step(mini_steps["e2"], token_loader, contact_kernel)
    torch.save({"version": SAMPLE_VERSION - 1, "samples": [s.to_dict()]}, tmp_path / "old.pt")
    with pytest.raises(ValueError, match="sample version"):
        load_samples(tmp_path / "old.pt")


def test_sample_stores_the_contact_sites_and_clash_flags(mini_steps, token_loader, contact_kernel):
    import numpy as np

    s = sample_for_step(mini_steps["sn2"], token_loader, contact_kernel)
    n_occ, n_virt = s.pair_valid.shape
    assert s.pair_contact_sites.shape == (n_occ, n_virt, 2) and s.pair_clash.shape == (
        n_occ,
        n_virt,
    )
    occ = s.n_atoms + np.flatnonzero(s.orbital_occupied)
    virt = s.n_atoms + np.flatnonzero(~s.orbital_occupied)
    centres = {}
    for j, g, _ in s.incidence:
        centres.setdefault(s.n_atoms + int(j), set()).add(int(g))
    for i in range(n_occ):  # the argmax sites are centres of the two tokens
        for k in range(n_virt):
            a, b = s.pair_contact_sites[i, k]
            assert a in centres[occ[i]] and b in centres[virt[k]]
