"""Training samples, split files and the collate function (`ksmech.arrows.steps`)."""

import dataclasses
import json
import os

import numpy as np
import pytest
import torch

from ksmech.arrows.batches import ANCHOR_SIDE, collate_samples, make_loader
from ksmech.arrows.cli import main as arrows_main
from ksmech.arrows.steps import (
    PARTS,
    READING_ALIGNED,
    SAMPLE_VERSION,
    MechanismSample,
    SkipReason,
    build_all_samples,
    load_pair_stats,
    load_samples,
    pair_stats,
    pair_stats_path,
    sample_for_step,
    samples_path,
    write_split_file,
)
from ksmech.data import chem
from ksmech.data import provenance as prov
from ksmech.data.io import read_training_steps, steps_path, write_steps
from ksmech.data.resonance import sigma_key
from ksmech.data.schema import Arrow, ElementaryStep
from ksmech.physics.stage_a import STAGE_A_FEATURES


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


def test_sample_holds_tokens_labels_and_product_key(mini_steps, token_loader):
    step = mini_steps["sn2"]
    s = sample_for_step(step, token_loader)
    assert isinstance(s, MechanismSample)
    assert len(s.fragment_keys) == 2 and set(s.atom_fragment) == {0, 1}
    assert s.atom_features.shape[0] == s.n_atoms == len(s.atom_map)
    assert s.orbital_spectra.shape == (s.n_orbitals, 8, 2)
    assert (s.atom_map > 0).sum() == 3  # the three mapped atoms
    assert s.product_key == sigma_key(chem.parse_mapped(step.products_smiles))
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


def test_samples_round_trip_through_the_store(mini_steps, token_loader, tmp_path):
    s = sample_for_step(mini_steps["e2"], token_loader)
    (back,) = load_samples(_stored(tmp_path, {s.step_id: s}))
    assert back.labels == s.labels and back.product_key == s.product_key
    assert back.edit_label == s.edit_label and back.orbital_reading == s.orbital_reading
    for name in ("bonds", "lone_pairs", "wiberg"):
        a, b = getattr(back, name), getattr(s, name)
        assert a.dtype == b.dtype and np.array_equal(a, b), name
    assert (back.orbital_features == s.orbital_features).all()
    valid = s.pair_valid
    assert np.array_equal(back.pair_valid, valid)
    assert np.array_equal(back.pair_defined[valid], s.pair_defined[valid])
    assert np.allclose(back.pair_z[valid], s.pair_z[valid], rtol=1e-3, atol=1e-6)  # float16
    assert np.array_equal(back.pair_contact_sites[valid], s.pair_contact_sites[valid])
    assert not back.pair_defined[~valid].any() and (back.pair_contact_sites[~valid] == -1).all()
    assert back.pair_z.dtype == np.float32 and back.atom_coulomb.dtype == np.float32


def test_skip_reasons(token_loader):
    lone = ElementaryStep(
        "x",
        "pmechdb",
        None,
        None,
        "[CH3:10][Br:11]",
        "[CH3+:10].[Br-:11]",
        [Arrow("bond", (10, 11), "atom", (11,))],
    )
    assert sample_for_step(lone, lambda key: None) == SkipReason.MISSING_TOKENS

    class Stale:  # a loader that serves no file and says why (`TokenCache.missing_reason`)
        def __call__(self, key):
            return None

        def missing_reason(self, key):
            return prov.STALE_ARTIFACT

    assert sample_for_step(lone, Stale()) == SkipReason.STALE_ARTIFACT
    s = sample_for_step(lone, token_loader)  # a lone heterolysis is a sample
    assert isinstance(s, MechanismSample)
    assert s.labels == [] and s.orbital_reading == "HETEROLYSIS_ONLY"
    c, br = (int(np.flatnonzero(s.atom_map == m)[0]) for m in (10, 11))
    (r,) = s.edit_label.variants
    assert r.bonds == ((min(c, br), max(c, br), -1),) and r.lone == ((br, 2),)
    assert s.edit_label.anchors == (((min(c, br), max(c, br), "break"),),)


def test_a_resonance_step_has_no_sample(mini_steps, token_loader):
    """Its edit label is RESONANCE_STEP: it is not part of the training target."""
    out = sample_for_step(mini_steps["resonance"], token_loader)
    assert out == SkipReason.RESONANCE_STEP


def _reproduces(s: MechanismSample, prod, r) -> bool:
    """On the mapped token atoms of `s`, its bond orders + R are the bond orders of `prod`, and
    its lone pairs + R the non-bonding electrons of `prod`."""
    from ksmech.arrows.apply import nonbonding_electrons

    by_map = {a.GetAtomMapNum(): a for a in prod.GetAtoms() if a.GetAtomMapNum()}
    mapped = {int(m): g for g, m in enumerate(s.atom_map) if m}
    after = {(int(i), int(j)): int(o) for i, j, o in s.bonds}
    for i, j, d in r.bonds:
        after[(i, j)] = after.get((i, j), 0) + d
    lone = {i: 2 * int(s.lone_pairs[i]) + d for i, d in r.lone}
    for m, g in mapped.items():
        if lone.get(g, 2 * int(s.lone_pairs[g])) != nonbonding_electrons(by_map[m]):
            return False
    for m, g in mapped.items():
        for n, h in mapped.items():
            if m < n:
                bond = prod.GetBondBetweenAtoms(by_map[m].GetIdx(), by_map[n].GetIdx())
                want = 0 if bond is None else int(bond.GetBondTypeAsDouble())
                if after.get((min(g, h), max(g, h)), 0) != want:
                    return False
    return True


def test_the_edit_label_takes_the_reactant_to_the_drawn_product(mini_steps, token_loader):
    """On the mapped token atoms, reactant bond orders + R give the drawn product's orders,
    and reactant lone pairs + R its lone pairs, for a variant of every training step."""
    for name in ("sn2", "proton_transfer", "carbonyl_addition", "e2"):
        step = mini_steps[name]
        s = sample_for_step(step, token_loader)
        prod = chem.parse_mapped(step.products_smiles)
        assert any(_reproduces(s, prod, r) for r in s.edit_label.variants), name


def test_samples_build_for_every_split_strategy(mini_processed, token_loader):
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


def test_pair_stats_are_those_of_the_stored_valid_cells(mini_processed, token_loader):
    """The statistics file equals a refit on the samples as loaded (valid cells, float16
    values), counts only valid defined entries, and is refused once its view changes."""
    build_all_samples("pmechdb", ["molecule"], mini_processed, token_loader)
    path = pair_stats_path("pmechdb", "molecule", mini_processed)
    view = samples_path("pmechdb", "molecule_train", mini_processed)
    stats, train = load_pair_stats(path), load_samples(view)
    ids = {s.step_id for s in train}
    steps = read_training_steps(steps_path("pmechdb", mini_processed))
    built = [sample_for_step(s, token_loader) for s in steps if s.step_id in ids]
    assert stats.names == STAGE_A_FEATURES
    for refit in (pair_stats(train), pair_stats(built)):  # as loaded, and as just built
        for a in ("mean", "std", "count"):
            assert np.array_equal(getattr(stats, a), getattr(refit, a)), a
    valid = sum((s.pair_defined & s.pair_valid[..., None]).sum((0, 1)) for s in built)
    assert np.array_equal(stats.count, valid)
    assert (sum(s.pair_defined.sum((0, 1)) for s in built) > valid).any()  # invalid cells exist
    header = prov.read_header(path)
    assert header.kind == "pair_stats" and header.config["cells"] == "valid"
    from ksmech.arrows import store
    from ksmech.arrows.steps import splits_path

    ids = json.loads(view.read_text())["ids"][:1]  # the training view is rebuilt, changed
    split_file = splits_path("pmechdb", "molecule", mini_processed)
    index = store.store_dir(view.parent) / "index.json"
    store.write_view(view, ids, {}, index, split_file, prov.read_header(view).config)
    with pytest.raises(prov.StaleArtifact, match="input changed"):
        load_pair_stats(path)


def test_collate_pads_and_builds_masks(mini_steps, token_loader):
    lone = ElementaryStep(
        "x",
        "pmechdb",
        None,
        None,
        "[CH3:10][Br:11]",
        "[CH3+:10].[Br-:11]",
        [Arrow("bond", (10, 11), "atom", (11,))],
    )
    batch = [sample_for_step(mini_steps[k], token_loader) for k in ("sn2", "e2")]
    batch.append(sample_for_step(lone, token_loader))
    b = collate_samples(batch)
    _check_edit_tensors(b, batch)
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
    _check_pair_tensors(b, batch)


def _check_pair_tensors(b: dict, batch: list) -> None:
    """Each sample's pair arrays sit unchanged in the top-left of the padded block, their rows
    and columns point at its occupied and virtual token positions, and padding is empty."""
    assert b["pair_features"] == batch[0].pair_features
    assert "atom_coulomb" not in b  # stored, no model input
    for i, s in enumerate(batch):
        r, c = s.pair_valid.shape
        assert np.array_equal(b["pair_z"][i, :r, :c].numpy(), s.pair_z)
        assert np.array_equal(b["pair_defined"][i, :r, :c].numpy(), s.pair_defined)
        assert np.array_equal(b["pair_valid"][i, :r, :c].numpy(), s.pair_valid)
        assert np.array_equal(b["pair_gap_sign"][i, :r, :c].numpy(), s.pair_gap_sign)
        assert not b["pair_valid"][i, r:].any() and not b["pair_defined"][i, :, c:].any()
        rows, cols = b["pair_row_pos"][i], b["pair_col_pos"][i]
        assert b["occupied"][i, rows[:r]].all() and b["virtual"][i, cols[:c]].all()
        assert (rows[r:] == -1).all() and (cols[c:] == -1).all()
        assert int(b["occupied"][i].sum()) == r and int(b["virtual"][i].sum()) == c


def _check_edit_tensors(b: dict, batch: list) -> None:
    """The R targets are the labels' entries (symmetric), every anchor is one the masks
    allow, the reactant tensors hold the stored structure, and padding is empty."""
    a = max(s.n_atoms for s in batch)
    assert b["r_pairs"].shape[2:] == (a, a) and b["bond_order"].shape == (len(batch), a, a)
    for i, s in enumerate(batch):
        assert b["variant_mask"][i].sum() == len(s.edit_label.variants)
        slots = zip(s.edit_label.variants, s.edit_label.anchors, strict=True)
        for x, (r, anchors) in enumerate(slots):
            for p, q, d in r.bonds:
                assert b["r_pairs"][i, x, p, q] == b["r_pairs"][i, x, q, p] == d
            assert int(b["r_pairs"][i, x].ne(0).sum()) == 2 * len(r.bonds)
            assert int(b["r_atoms"][i, x].ne(0).sum()) == len(r.lone)
            for p, q, kind in anchors:
                assert b["r_anchors"][i, x, p, q, ANCHOR_SIDE[kind]]
                mask = b["can_form"] if kind == "form" else b["can_break"]
                assert mask[i, p, q] and mask[i, q, p]
        for p, q, o in s.bonds:
            assert b["bond_order"][i, p, q] == b["bond_order"][i, q, p] == o > 0
        n = s.n_atoms
        assert (b["graph_distance"][i, :n, :n].diagonal() == 0).all()
        assert (b["graph_distance"][i, n:] == -1).all() and not b["can_form"][i, n:].any()
    assert b["orbital_reading"] == [READING_ALIGNED, READING_ALIGNED, "HETEROLYSIS_ONLY"]
    assert not b["r_anchors"][2, :, :, :, ANCHOR_SIDE["form"]].any()  # a heterolysis breaks


def test_loader_and_cli(mini_processed, token_loader, monkeypatch, capsys):
    import ksmech.arrows.cli as cli

    monkeypatch.setattr(cli, "TokenCache", lambda *a, **k: missing_iodide(token_loader))
    args = ["--source", "pmechdb", "--processed-dir", str(mini_processed)]
    assert arrows_main(["samples", *args, "--strategy", "random_leaky"]) == 0
    assert arrows_main(["samples", *args, "--strategy", "molecule", "scaffold"]) == 0
    assert arrows_main(["roundtrip", *args]) == 0
    assert "100.00%" in capsys.readouterr().out
    loader = make_loader(samples_path("pmechdb", "molecule_train", mini_processed), batch_size=2)
    batch = next(iter(loader))
    assert batch["donor_sets"].dtype == torch.bool and len(batch["step_id"]) <= 2


def test_sample_stores_the_pair_block(mini_steps, token_loader):
    import numpy as np

    from ksmech.physics.pairs import atom_coulomb_matrix, valid_pairs
    from ksmech.physics.params import REDUCED
    from ksmech.physics.stage_a import STAGE_A_FEATURES, stage_a_block

    step = mini_steps["sn2"]
    s = sample_for_step(step, token_loader)
    n_occ, n_virt = int(s.orbital_occupied.sum()), int((~s.orbital_occupied).sum())
    assert s.pair_z.shape == s.pair_defined.shape == (n_occ, n_virt, len(STAGE_A_FEATURES))
    assert s.pair_z.dtype == np.float32 and s.pair_valid.shape == (n_occ, n_virt)
    tokens = s.tokens(token_loader)
    block = stage_a_block(tokens, REDUCED)
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


def test_old_sample_versions_are_refused(mini_steps, token_loader, tmp_path):
    s = sample_for_step(mini_steps["e2"], token_loader)
    view = _stored(tmp_path, {s.step_id: s}, version=SAMPLE_VERSION - 1)
    with pytest.raises(prov.StaleArtifact, match="schema_version"):
        load_samples(view)
    bare = tmp_path / "samples_bare.json"
    bare.write_text(json.dumps({"ids": [s.step_id], "report": {}}))
    with pytest.raises(prov.StaleArtifact, match="no provenance header"):
        load_samples(bare)  # written without a header: refused, never guessed


def test_touching_the_ingest_parquet_makes_the_samples_refuse_to_load(
    mini_processed, token_loader, monkeypatch
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
    os.utime(parquet)  # a timestamp alone is not a change of content
    assert load_samples(path)
    steps = read_steps(parquet)
    write_steps(steps[:-1], parquet)  # re-ingested: one step fewer
    with pytest.raises(prov.StaleArtifact, match=r"input changed: .*steps\.parquet"):
        load_samples(path)
    assert load_samples(path, allow_stale=True)  # diagnostics may still read it


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


def test_filtered_label_cells_are_counted_flagged_and_never_the_teacher(mini_steps, token_loader):
    from dataclasses import replace

    from ksmech.arrows.align import AlignFlag
    from ksmech.arrows.reactants import ReactantTokens
    from ksmech.arrows.steps import _pair_index, filter_labels

    step = mini_steps["e2"]
    s = sample_for_step(step, token_loader)
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


def test_every_label_carries_a_valid_teacher_pair(mini_steps, token_loader):
    from ksmech.arrows.steps import _pair_index

    for step in mini_steps.values():
        s = sample_for_step(step, token_loader)
        if not isinstance(s, MechanismSample):
            continue
        row, col = _pair_index(s)
        for lab in s.labels:
            assert s.pair_valid[row[lab.teacher_donor], col[lab.teacher_acceptor]]


def test_stage_a_samples_round_trip_through_the_store(mini_steps, token_loader, tmp_path):
    from ksmech.physics.stage_a import STAGE_A_FEATURES, stage_a_block

    s = sample_for_step(mini_steps["e2"], token_loader)
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


def test_the_parallel_build_matches_the_sequential_one(mini_steps, mini_token_cache):
    """Two spawned workers give the same samples as one process (tokens read from files, so
    the loader can be sent to the workers; its copies start empty)."""
    from ksmech.arrows.steps import build_step_samples, build_step_samples_parallel

    steps = list(mini_steps.values())
    one = build_step_samples(steps, mini_token_cache)
    two = build_step_samples_parallel(steps, mini_token_cache, n_workers=2)
    assert one[0], one[1]  # samples were built, not every step skipped
    assert one[1] == two[1] and sorted(one[0]) == sorted(two[0])
    for sid, s in one[0].items():
        other = two[0][sid]
        assert other.labels == s.labels and np.array_equal(other.pair_valid, s.pair_valid)
        assert np.allclose(other.pair_z, s.pair_z, rtol=1e-12, atol=1e-12)
        assert other.edit_label == s.edit_label and np.array_equal(other.bonds, s.bonds)
        assert np.array_equal(other.wiberg, s.wiberg)
