"""Training samples, split files and the collate function (`ksmech.arrows.steps`)."""

import dataclasses
import json
import os

import numpy as np
import pytest
import torch

from ksmech.arrows.cli import main as arrows_main
from ksmech.arrows.steps import (
    PARTS,
    SAMPLE_VERSION,
    MechanismSample,
    SkipReason,
    build_all_samples,
    collate_samples,
    load_samples,
    make_loader,
    sample_for_step,
    samples_path,
    write_split_file,
)
from ksmech.data import provenance as prov
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


def test_sample_holds_tokens_labels_and_product_key(mini_steps, token_loader, pair_kernels):
    step = mini_steps["sn2"]
    s = sample_for_step(step, token_loader, pair_kernels)
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


def _stored(tmp_path, samples: dict, version=SAMPLE_VERSION):
    """Write `samples` to a store in tmp_path and one view over all of them."""
    from ksmech.arrows import store

    parquet = tmp_path / "steps.parquet"
    parquet.write_bytes(b"stand-in parquet")
    split = write_split_file(tmp_path / "splits_x.json", "x", 0, {"x": sorted(samples)}, parquet)
    index = store.write_store(tmp_path, samples, {}, parquet, version)
    config = {"split": "x", "sample_version": version}
    return store.write_view(tmp_path / "samples_x.json", sorted(samples), {}, index, split, config)


def test_samples_round_trip_through_the_store(mini_steps, token_loader, pair_kernels, tmp_path):
    s = sample_for_step(mini_steps["e2"], token_loader, pair_kernels)
    (back,) = load_samples(_stored(tmp_path, {s.step_id: s}))
    assert back.labels == s.labels and back.product_key == s.product_key
    assert (back.orbital_features == s.orbital_features).all()
    valid = s.pair_valid
    assert np.array_equal(back.pair_valid, valid)
    assert np.array_equal(back.pair_defined[valid], s.pair_defined[valid])
    assert np.allclose(back.pair_z[valid], s.pair_z[valid], rtol=1e-3, atol=1e-6)  # float16
    assert np.array_equal(back.pair_contact_sites[valid], s.pair_contact_sites[valid])
    assert not back.pair_defined[~valid].any() and (back.pair_contact_sites[~valid] == -1).all()
    assert back.pair_z.dtype == np.float32 and back.atom_coulomb.dtype == np.float32


def test_skip_reasons(token_loader, pair_kernels):
    lone = ElementaryStep(
        "x",
        "pmechdb",
        None,
        None,
        "[CH3:10][Br:11]",
        "[CH3+:10].[Br-:11]",
        [Arrow("bond", (10, 11), "atom", (11,))],
    )
    assert sample_for_step(lone, token_loader, pair_kernels) == SkipReason.UNALIGNED_ARROW
    assert sample_for_step(lone, lambda key: None, pair_kernels) == SkipReason.MISSING_TOKENS


def test_samples_build_for_every_split_strategy(mini_processed, token_loader, pair_kernels):
    report = build_all_samples(
        "pmechdb",
        processed_dir=mini_processed,
        loader=missing_iodide(token_loader),
    )
    assert not report["strategies_not_applicable"]
    assert report["skipped"] == {"MISSING_TOKENS": 1}
    names = set(report["splits"])
    assert {"molecule_train", "scaffold_test"} <= names
    assert any(n.startswith("loco-substitution_") for n in names)
    for split, r in report["splits"].items():
        path = samples_path("pmechdb", split, mini_processed)
        assert path.exists() and len(load_samples(path)) == r["n_samples"]
        assert r["n_samples"] + sum(r["skipped"].values()) == r["n_steps"]
    held_out = report["splits"]["loco-substitution_test"]
    assert held_out["n_steps"] == 2 and held_out["skipped"] == {"MISSING_TOKENS": 1}


def test_collate_pads_and_builds_masks(mini_steps, token_loader, pair_kernels):
    batch = [
        sample_for_step(mini_steps[k], token_loader, pair_kernels)
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
    assert arrows_main(["contacts", *args]) == 2  # no split file yet: run `samples` first
    assert arrows_main(["samples", *args, "--strategy", "random_leaky"]) == 0
    leaky = ["--split", "random_leaky"]
    assert arrows_main(["contacts", *args, *leaky, *missing]) == 2  # no table: refused, not guessed
    assert arrows_main(["contacts", *args, *leaky, "--kernel", "table", *table]) == 0
    assert arrows_main(["samples", *args, "--strategy", "molecule", "scaffold"]) == 0
    oriented = ["--oriented-cache", str(mini_processed / "oriented")]  # the default kernel
    assert arrows_main(["contacts", *args, *oriented]) == 0
    assert (mini_processed / "pmechdb" / "contacts_report_molecule.json").exists()
    assert arrows_main(["roundtrip", *args]) == 0
    assert "100.00%" in capsys.readouterr().out
    loader = make_loader(samples_path("pmechdb", "molecule_train", mini_processed), batch_size=2)
    batch = next(iter(loader))
    assert batch["donor_sets"].dtype == torch.bool and len(batch["step_id"]) <= 2


def test_sample_stores_the_physics_pair_block(mini_steps, token_loader, pair_kernels):
    import numpy as np

    from ksmech.physics.pairs import PAIR_FEATURES, atom_coulomb_matrix, pair_block, valid_pairs
    from ksmech.physics.params import REDUCED

    step = mini_steps["sn2"]
    s = sample_for_step(step, token_loader, pair_kernels)
    n_occ, n_virt = int(s.orbital_occupied.sum()), int((~s.orbital_occupied).sum())
    assert s.pair_z.shape == s.pair_defined.shape == (n_occ, n_virt, len(PAIR_FEATURES))
    assert s.pair_z.dtype == np.float32 and s.pair_valid.shape == (n_occ, n_virt)
    tokens = s.tokens(token_loader)
    block = pair_block(tokens, pair_kernels, REDUCED)
    assert np.allclose(s.pair_z, block.z, rtol=1e-6, atol=1e-6)
    assert np.array_equal(s.pair_defined, block.defined)
    assert np.array_equal(s.pair_gap_sign, block.gap_sign)
    assert np.array_equal(s.pair_valid, valid_pairs(tokens))
    donors = s.donor_mask[s.orbital_occupied]
    acceptors = s.acceptor_mask[~s.orbital_occupied]
    assert not (s.pair_valid & ~(donors[:, None] & acceptors[None, :])).any()
    assert s.atom_coulomb.shape == (s.n_atoms, s.n_atoms)
    assert np.allclose(s.atom_coulomb, s.atom_coulomb.T)
    assert np.allclose(s.atom_coulomb, atom_coulomb_matrix(tokens, REDUCED), rtol=1e-6)


def test_old_sample_versions_are_refused(mini_steps, token_loader, pair_kernels, tmp_path):
    s = sample_for_step(mini_steps["e2"], token_loader, pair_kernels)
    view = _stored(tmp_path, {s.step_id: s}, version=SAMPLE_VERSION - 1)
    with pytest.raises(prov.StaleArtifact, match="schema_version"):
        load_samples(view)
    bare = tmp_path / "samples_bare.json"
    bare.write_text(json.dumps({"ids": [s.step_id], "report": {}}))
    with pytest.raises(prov.StaleArtifact, match="no provenance header"):
        load_samples(bare)  # written without a header: refused, never guessed


def test_touching_the_ingest_parquet_makes_the_samples_refuse_to_load(
    mini_processed, token_loader, monkeypatch, tmp_path
):
    import ksmech.arrows.cli as cli
    from ksmech.data.io import read_steps

    monkeypatch.setattr(cli, "TokenCache", lambda *a, **k: missing_iodide(token_loader))
    args = ["--source", "pmechdb", "--processed-dir", str(mini_processed)]
    assert arrows_main(["samples", *args, "--strategy", "molecule"]) == 0
    path = samples_path("pmechdb", "molecule_train", mini_processed)
    assert load_samples(path)
    parquet = steps_path("pmechdb", mini_processed)
    shard = prov.read_header(next((mini_processed / "pmechdb" / "samples").glob("shard_*.npz")))
    assert prov.file_input_name(parquet) in shard.inputs
    cache = ["--oriented-cache", str(tmp_path / "oriented")]
    assert arrows_main(["contacts", *args, *cache]) == 0
    side = mini_processed / "pmechdb" / "samples" / "contacts_molecule"
    side_shard = prov.read_header(next(side.glob("shard_*.npz")))
    assert any("oriented" in name for name in side_shard.inputs)  # the contact blocks it used
    os.utime(parquet)  # a timestamp alone is not a change of content
    assert load_samples(path)
    steps = read_steps(parquet)
    write_steps(steps[:-1], parquet)  # re-ingested: one step fewer
    with pytest.raises(prov.StaleArtifact, match=r"input changed: .*steps\.parquet"):
        load_samples(path)
    assert load_samples(path, allow_stale=True)  # diagnostics may still read it


def test_sample_stores_the_contact_sites_and_clash_flags(mini_steps, token_loader, pair_kernels):
    import numpy as np

    s = sample_for_step(mini_steps["sn2"], token_loader, pair_kernels)
    n_occ, n_virt = s.pair_valid.shape
    assert s.pair_contact_sites.shape == (n_occ, n_virt, 2)
    occ = s.n_atoms + np.flatnonzero(s.orbital_occupied)
    virt = s.n_atoms + np.flatnonzero(~s.orbital_occupied)
    centres = {}
    for j, g, _ in s.incidence:
        centres.setdefault(s.n_atoms + int(j), set()).add(int(g))
    for i in range(n_occ):  # the argmax sites are centres of the two tokens
        for k in range(n_virt):
            a, b = s.pair_contact_sites[i, k]
            assert a in centres[occ[i]] and b in centres[virt[k]]


def test_the_per_step_split_is_random_leaky_and_never_a_default(mini_processed):
    from ksmech.arrows.steps import DEFAULT_STRATEGIES, split_ids
    from ksmech.data.io import read_training_steps

    steps = read_training_steps(steps_path("pmechdb", mini_processed))
    assert "random_leaky" not in DEFAULT_STRATEGIES and DEFAULT_STRATEGIES[0] == "molecule"
    with pytest.raises(ValueError, match="random_leaky"):
        split_ids(steps, "random")
    assert set(split_ids(steps, "random_leaky")) == {f"random_leaky_{p}" for p in PARTS}


@pytest.mark.realdata
def test_no_reactant_molecule_is_in_two_parts_of_the_real_molecule_split():
    """The PMechDB molecule split: no reactant (maps stripped, canonical, > 3 heavy atoms) in
    two of train / val / test. Products are checked too, as `split_by_molecule` promises."""
    from ksmech.arrows.steps import split_ids
    from ksmech.data.io import read_training_steps

    path = steps_path("pmechdb")
    if not path.exists():
        pytest.skip("PMechDB steps missing")
    steps = {s.step_id: s for s in read_training_steps(path)}
    ids = split_ids(list(steps.values()), "molecule")
    for side in ("reactants_smiles", "products_smiles"):
        seen = {
            p: {m for i in ids[f"molecule_{p}"] for m in _species(steps[i], side)} for p in PARTS
        }
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            assert not seen[a] & seen[b], (side, a, b, sorted(seen[a] & seen[b])[:5])
    assert sum(len(v) for v in ids.values()) == len(steps)


def _species(step, side: str) -> list[str]:
    from ksmech.data.chem import nontrivial_species

    return nontrivial_species(getattr(step, side))


def test_filtered_label_cells_are_counted_flagged_and_never_the_teacher(
    mini_steps, token_loader, pair_kernels
):
    from dataclasses import replace

    from ksmech.arrows.align import AlignFlag
    from ksmech.arrows.reactants import ReactantTokens
    from ksmech.arrows.steps import _pair_index, filter_labels

    step = mini_steps["e2"]
    s = sample_for_step(step, token_loader, pair_kernels)
    rt = ReactantTokens.from_smiles(step.reactants_smiles, token_loader)
    lab = max(s.labels, key=lambda x: len(x.donor_ids) * len(x.acceptor_ids))
    row, col = _pair_index(s)
    fresh = replace(s, labels=[lab], label_filter={}, pair_valid=s.pair_valid.copy())
    # every pair of the teacher donor rejected: it must leave the set, the label is flagged
    r0 = lab.teacher_donor
    for a in lab.acceptor_ids:
        fresh.pair_valid[row[r0], col[a]] = False
    if len(lab.donor_ids) > 1:
        (out,) = filter_labels(fresh, rt)
        assert r0 not in out.donor_ids and AlignFlag.LABEL_FILTERED in out.flags
        assert out.teacher_donor != r0
        assert fresh.pair_valid[row[out.teacher_donor], col[out.teacher_acceptor]]
        assert sum(fresh.label_filter.values()) == len(lab.acceptor_ids)
    # no valid cell left at all: the step is skipped, not kept with a dead label
    for r in lab.donor_ids:
        for a in lab.acceptor_ids:
            fresh.pair_valid[row[r], col[a]] = False
    assert filter_labels(replace(fresh, label_filter={}), rt) is None


def test_every_label_carries_a_valid_teacher_pair(mini_steps, token_loader, pair_kernels):
    from ksmech.arrows.steps import _pair_index

    for step in mini_steps.values():
        s = sample_for_step(step, token_loader, pair_kernels)
        if not isinstance(s, MechanismSample):
            continue
        row, col = _pair_index(s)
        for lab in s.labels:
            assert s.pair_valid[row[lab.teacher_donor], col[lab.teacher_acceptor]]


def test_stage_a_samples_round_trip_through_the_store(mini_steps, token_loader, tmp_path):
    from ksmech.physics.stage_a import STAGE_A_FEATURES, stage_a_block

    s = sample_for_step(mini_steps["e2"], token_loader)  # no kernels: stage A
    assert s.pair_features == STAGE_A_FEATURES
    assert s.pair_z.shape[-1] == len(STAGE_A_FEATURES) and (s.pair_contact_sites == -1).all()
    block = stage_a_block(s.tokens(token_loader))
    assert np.allclose(s.pair_z, block.z, rtol=1e-6, atol=1e-6)
    (back,) = load_samples(_stored(tmp_path, {s.step_id: s}))
    assert back.pair_features == STAGE_A_FEATURES
    valid = s.pair_valid
    assert np.array_equal(back.pair_defined[valid], s.pair_defined[valid])
    assert np.allclose(back.pair_z[valid], s.pair_z[valid], rtol=1e-3, atol=1e-3)  # float16
    k = STAGE_A_FEATURES.index("bonded")
    assert np.array_equal(back.pair_z[valid][:, k], s.pair_z[valid][:, k])  # bits, exact


def test_the_parallel_build_matches_the_sequential_one(mini_steps, token_loader, tmp_path):
    """Two spawned workers give the same samples as one process (tokens read from files, so
    the loader can be sent to the workers)."""
    from ksmech.arrows.reactants import TokenCache, _fragment_keys
    from ksmech.arrows.steps import build_step_samples, build_step_samples_parallel
    from ksmech.data import chem
    from ksmech.orbitals.build import tokens_path
    from ksmech.orbitals.tokens import save_tokens

    steps = list(mini_steps.values())
    cache = TokenCache(tmp_path / "tokens")
    for step in steps:
        for _, _, key in _fragment_keys(chem.parse_mapped(step.reactants_smiles)):
            if key and key != "CI":
                t = token_loader(key)
                save_tokens(t, tokens_path(cache.tokens_dir, key, 0, cache.method))
    one = build_step_samples(steps, cache)
    two = build_step_samples_parallel(steps, TokenCache(cache.tokens_dir), n_workers=2)
    assert one[1] == two[1] and sorted(one[0]) == sorted(two[0])
    for sid, s in one[0].items():
        other = two[0][sid]
        assert other.labels == s.labels and np.array_equal(other.pair_valid, s.pair_valid)
        assert np.allclose(other.pair_z, s.pair_z, rtol=1e-12, atol=1e-12)
