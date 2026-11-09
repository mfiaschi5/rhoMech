"""The previous step (`arrows.backward`, reaction-network plan item H): the backward labels,
the backward store and its views, training and decoding on it, and the forward path as it
was."""

import dataclasses
import io
import json
from collections import Counter

import pandas as pd
import pytest
import torch
from rich.console import Console

from ksmech.arrows.apply import ArrowApplyError, apply_edit, sanitize_product
from ksmech.arrows.backward import (
    BACKWARD,
    FORWARD,
    backward_agreement,
    backward_step,
    direction_config,
    direction_dir,
    label_agreement,
    molecule_leakage,
    oriented,
    pilot_steps,
    qm_status,
    require_direction,
    species_coverage,
    view_direction,
)
from ksmech.arrows.batches import ANCHOR_SIDE, collate_samples
from ksmech.arrows.cli import main as arrows_main
from ksmech.arrows.delta_be import mol_label, to_global
from ksmech.arrows.reactants import ReactantTokens
from ksmech.arrows.steps import (
    MechanismSample,
    build_all_samples,
    load_samples,
    load_split_file,
    pair_stats,
    pair_stats_path,
    sample_for_step,
    samples_path,
    splits_path,
    write_views_from_store,
)
from ksmech.data import chem
from ksmech.data import provenance as prov
from ksmech.data.io import read_training_steps, steps_path, write_steps
from ksmech.data.resonance import identity_key
from ksmech.data.schema import Arrow, ElementaryStep, SinkKind, SourceKind
from ksmech.eval.decode import hit_rank, lewis_apply

LP, BOND, ATOM, SINK_BOND = (
    SourceKind.LONE_PAIR,
    SourceKind.BOND,
    SinkKind.ATOM,
    SinkKind.BOND,
)
# a pure addition: one bond forms, none breaks; going back it is a heterolysis
ADDITION = ElementaryStep(
    "addition", "pmechdb", None, None, "[CH3+:10].[Br-:11]", "[CH3:10][Br:11]",
    [Arrow(LP, (11,), ATOM, (10,))],
)  # fmt: skip
# PMechDB's conjugate addition (pmechdb-aab2c8a5010071a7): the forward class reaches the
# product's carbanion form, the backward class starts from the drawn enolate
CONJUGATE = ElementaryStep(
    "conjugate", "pmechdb", None, None,
    "C[C:22]([CH:21]=[CH2:20])=[O:23].N#[C-:10]",
    "C[C:22](=[CH:21][CH2:20][C:10]#N)[O-:23]",
    [Arrow(LP, (10,), ATOM, (20,)), Arrow(BOND, (20, 21), SINK_BOND, (21, 22)),
     Arrow(BOND, (22, 23), ATOM, (23,))],
)  # fmt: skip
KINDS = ("sn2", "proton_transfer", "carbonyl_addition", "e2", "addition")


def _step(mini_steps, kind: str) -> ElementaryStep:
    return ADDITION if kind == "addition" else mini_steps[kind]


@pytest.fixture(scope="module")
def backward_samples(mini_steps, token_loader) -> dict[str, MechanismSample]:
    out = {k: sample_for_step(backward_step(_step(mini_steps, k)), token_loader) for k in KINDS}
    bad = {k: s for k, s in out.items() if not isinstance(s, MechanismSample)}
    assert not bad, bad
    return out


def test_the_backward_step_swaps_the_sides_and_drops_the_arrows(mini_steps):
    step = mini_steps["sn2"]
    back = backward_step(step)
    assert (back.reactants_smiles, back.products_smiles) == (
        step.products_smiles,
        step.reactants_smiles,
    )
    assert back.arrows == [] and back.step_id == step.step_id and back.source == step.source
    assert backward_step(back).reactants_smiles == step.reactants_smiles
    assert oriented([step], FORWARD) == [step] and oriented([step], BACKWARD) == [back]
    with pytest.raises(ValueError, match="direction"):
        oriented([step], "sideways")


@pytest.mark.parametrize("kind", KINDS)
def test_minus_r_on_the_child_gives_the_parent_back(kind, mini_steps, backward_samples):
    """Every member of the backward class takes the child, as drawn, to the parent: the
    identity key v2 with the atom maps kept; and on the sample's own Lewis structure
    (`eval.decode.lewis_apply`, as decode checks it) to the parent's map-free key."""
    step = _step(mini_steps, kind)
    child, label = mol_label(backward_step(step))
    parent = identity_key(chem.parse_mapped(step.reactants_smiles), keep_maps=True)
    for v in label.variants:
        prod = sanitize_product(apply_edit(chem.parse_mapped(child.reactants_smiles), v.to_edit()))
        assert identity_key(prod, keep_maps=True) == parent
    sample = backward_samples[kind]
    assert sample.product_key == identity_key(chem.parse_mapped(step.reactants_smiles))
    apply = lewis_apply(sample)
    for v in sample.edit_label.variants:
        assert apply(v.bonds, v.lone) == (sample.product_key, None)


@pytest.mark.parametrize("kind", KINDS)
def test_the_backward_class_is_the_negated_forward_class(
    kind, mini_steps, backward_samples, token_loader
):
    """On these steps the class of the swapped step is the forward class negated (through the
    atom maps); on the sample it is that class on the child's token atoms, closed under the
    child pool's symmetry (the extra members are its images)."""
    step = _step(mini_steps, kind)
    child, label = mol_label(backward_step(step))
    assert label_agreement(mol_label(step), (child, label)) == "equal"
    rt = ReactantTokens.from_smiles(child.reactants_smiles, token_loader)
    on_tokens = set(to_global(label, rt.global_atom).variants)
    sample = backward_samples[kind].edit_label
    assert on_tokens <= set(sample.variants)
    assert len(sample.variants) == len(on_tokens) + sample.n_symmetric
    assert sample.data_parsimonious and not any(lab.n_rejected for lab in (label, sample))


def test_up_to_resonance_the_backward_class_starts_from_the_drawn_child():
    """A conjugate addition: the forward class reaches the carbanion form of the product, so
    its negation cannot even be applied to the enolate the dataset draws; the backward class
    takes that enolate to the parent (both classes are -R of each other modulo resonance)."""
    fwd, bwd = mol_label(CONJUGATE), mol_label(backward_step(CONJUGATE))
    assert label_agreement(fwd, bwd) == "disjoint"
    assert backward_agreement([CONJUGATE])["agreement"] == {"disjoint": 1}
    child = chem.parse_mapped(CONJUGATE.products_smiles)
    parent = identity_key(chem.parse_mapped(CONJUGATE.reactants_smiles), keep_maps=True)
    for v in bwd[1].variants:
        prod = sanitize_product(apply_edit(child, v.to_edit()))
        assert identity_key(prod, keep_maps=True) == parent
    index = {a.GetAtomMapNum(): a.GetIdx() for a in child.GetAtoms() if a.GetAtomMapNum()}
    fmaps = {
        a.GetIdx(): a.GetAtomMapNum()
        for a in chem.parse_mapped(CONJUGATE.reactants_smiles).GetAtoms()
    }
    for v in fwd[1].variants:
        negated = dataclasses.replace(
            v,
            bonds=tuple((*sorted((index[fmaps[i]], index[fmaps[j]])), -d) for i, j, d in v.bonds),
            lone=tuple((index[fmaps[i]], -d) for i, d in v.lone),
        )
        with pytest.raises(ArrowApplyError):
            sanitize_product(apply_edit(child, negated.to_edit()))


def _tiny_model(samples):
    from ksmech.model.config import ModelConfig
    from ksmech.model.inputs import fit_feature_stats
    from ksmech.model.net import KSMech

    torch.manual_seed(0)
    cfg = ModelConfig(
        d=32, d_pair=16, layers=2, heads=4, decoder_hidden=16, physics_hidden=8, checkpoint=False
    )
    return KSMech(cfg, fit_feature_stats(samples), pair_stats(samples)).eval()


def test_a_pure_addition_goes_back_by_its_breaking_bonds(backward_samples):
    """The forward addition breaks no bond, so -R forms none: its anchors are its -1 bond
    entries. The batch marks them on the break side, the step loss is finite, and the beam
    decoder completes the heterolysis from a breaking anchor (lewis_apply accepts it)."""
    from ksmech.model.decode import DecodeConfig, decode_pool
    from ksmech.model.loss import anchor_ranks, step_loss

    _, forward = mol_label(ADDITION)
    assert not any(d < 0 for v in forward.variants for *_, d in v.bonds)
    sample = backward_samples["addition"]
    for v, anchors in zip(sample.edit_label.variants, sample.edit_label.anchors, strict=True):
        assert not any(d > 0 for *_, d in v.bonds)
        assert set(anchors) == {(i, j, "break") for i, j, d in v.bonds if d < 0} != set()
    batch = collate_samples([sample])
    assert batch["r_anchors"][..., ANCHOR_SIDE["break"]].any()
    assert not batch["r_anchors"][..., ANCHOR_SIDE["form"]].any()
    model = _tiny_model(list(backward_samples.values()))
    with torch.no_grad():
        h, p, y, _ = model.encode(batch)
        loss, parts = step_loss(model.decoder, h, p, y, batch, 0.0)
    assert torch.isfinite(loss) and torch.isfinite(parts["anchor_nll"]).all()
    assert (anchor_ranks(parts["lp_anchor"], batch) >= 0).all()
    # the map-free key of CH3+ + Br- is that of CH3- + Br+ (charges zeroed), so the decoder's
    # merge by product key keeps whichever an untrained model scores higher: keyed by the
    # matrix too, every completion lewis_apply accepts stays
    check = lewis_apply(sample)

    def apply(bonds, lone):
        key, why = check(bonds, lone)
        return (None, why) if why else ((key, bonds, lone), None)

    cfg = DecodeConfig(anchors=64, per_anchor=10)
    found = decode_pool(model, sample, cfg, apply, "all", Counter())
    rank = hit_rank(found, sample.edit_label)
    assert rank is not None and found[rank].anchor[2] == 1  # model.decode.BREAK


# ---- the backward store


@pytest.fixture
def mini_processed(tmp_path, mini_result):
    """The mini fixture's accepted steps and the pure addition as a steps parquet."""
    classes = ["substitution", "proton_transfer", "addition", "elimination", "resonance"]
    steps = [
        dataclasses.replace(s, mechanism_class=c)
        for s, c in zip(mini_result.steps, classes, strict=True)
    ]
    write_steps([*steps, dataclasses.replace(ADDITION, mechanism_class="addition")],
                steps_path("pmechdb", tmp_path))  # fmt: skip
    return tmp_path


def _files(root) -> dict:
    return {p: p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def test_the_backward_store_never_touches_the_forward_one(mini_processed, token_loader):
    forward = build_all_samples("pmechdb", ["molecule", "scaffold"], mini_processed, token_loader)
    root = mini_processed / "pmechdb"
    before = _files(root)
    report = build_all_samples(
        "pmechdb", ["molecule", "scaffold"], mini_processed, token_loader, direction=BACKWARD
    )
    after = _files(root)
    assert {p: b for p, b in after.items() if p in before} == before  # not one byte changed
    new = set(after) - set(before)
    assert new and all(p.is_relative_to(root / "backward") for p in new)
    assert report["direction"] == BACKWARD and report["n_samples"] == forward["n_samples"]
    assert report["anchor_kind"] == {"form": report["n_samples"] - 1, "break": 1}
    for strategy in ("molecule", "scaffold"):  # the same parts, step by step
        assert load_split_file(splits_path("pmechdb", strategy, mini_processed, BACKWARD)) == (
            load_split_file(splits_path("pmechdb", strategy, mini_processed))
        )
    leaks = report["leakage"]["molecule"]
    assert set(leaks) == {"molecule_val", "molecule_test"}
    assert all(r["n_leaking"] == 0 for r in leaks.values())
    for split, counts in report["splits"].items():
        view = samples_path("pmechdb", split, mini_processed, BACKWARD)
        assert view.parent == direction_dir("pmechdb", mini_processed, BACKWARD)
        assert view_direction(view) == BACKWARD
        assert prov.read_header(view).config["direction"] == BACKWARD
        assert len(load_samples(require_direction(view, BACKWARD))) == counts["n_samples"]
        with pytest.raises(ValueError, match="backward samples"):
            require_direction(view, FORWARD)
    stats = pair_stats_path("pmechdb", "molecule", mini_processed, "valid", BACKWARD)
    assert view_direction(stats) == BACKWARD
    forward_view = samples_path("pmechdb", "molecule_train", mini_processed)
    assert view_direction(forward_view) == FORWARD


def test_the_forward_artifacts_carry_no_direction(mini_processed, token_loader):
    build_all_samples("pmechdb", ["molecule"], mini_processed, token_loader)
    root = mini_processed / "pmechdb"
    assert samples_path("pmechdb", "molecule_train", mini_processed) == (
        root / "samples_molecule_train.json"
    )
    assert pair_stats_path("pmechdb", "molecule", mini_processed) == (
        root / "pair_feature_stats_molecule.json"
    )
    assert direction_config(FORWARD, 5) == {} and not (root / "backward").exists()
    paths = [*root.glob("samples_*.json"), *root.glob("pair_feature_stats_*.json")]
    paths += [*(root / "samples").glob("*")]
    for path in paths:
        assert "direction" not in prov.read_header(path).config, path


def test_a_pilot_builds_a_random_subset_and_keeps_its_views_to_it(mini_processed, token_loader):
    with pytest.raises(ValueError, match="backward only"):
        build_all_samples("pmechdb", ["molecule"], mini_processed, token_loader, limit=3)
    report = build_all_samples(
        "pmechdb", ["molecule"], mini_processed, token_loader, seed=1, direction=BACKWARD, limit=3
    )
    assert report["n_built"] == 3 and report["limit"] == 3
    assert sum(r["n_steps"] for r in report["splits"].values()) == 3
    every = read_training_steps(steps_path("pmechdb", mini_processed))
    drawn = {s.step_id for s in pilot_steps(every, 3)}  # whatever the split's seed
    views = [samples_path("pmechdb", split, mini_processed, BACKWARD) for split in report["splits"]]
    assert {i for v in views for i in json.loads(v.read_text())["ids"]} == drawn
    index = direction_dir("pmechdb", mini_processed, BACKWARD) / "samples" / "index.json"
    assert prov.read_header(index).config == {
        "sample_version": prov.read_header(index).schema_version,
        "shard_size": 256,
        "direction": BACKWARD,
        "limit": 3,
    }
    kept = {v: v.read_bytes() for v in views}  # views of another strategy over the pilot store
    more = write_views_from_store("pmechdb", ["scaffold"], mini_processed, 1, BACKWARD)
    assert {v: v.read_bytes() for v in views} == kept
    added = [samples_path("pmechdb", s, mini_processed, BACKWARD) for s in more["splits"]]
    assert {i for v in added for i in json.loads(v.read_text())["ids"]} == drawn
    assert all(prov.read_header(v).config["limit"] == 3 for v in added)
    steps = list(range(10))
    assert pilot_steps(steps, 4, 0) == pilot_steps(steps, 4, 0) != pilot_steps(steps, 4, 1)
    assert pilot_steps(steps, 0) == steps and pilot_steps(steps, 20) == steps


def test_the_arrows_cli_takes_the_direction(mini_processed, token_loader, monkeypatch, capsys):
    import ksmech.arrows.cli as cli

    monkeypatch.setattr(cli, "TokenCache", lambda *a, **k: token_loader)
    args = ["--source", "pmechdb", "--processed-dir", str(mini_processed)]
    assert arrows_main(["samples", *args, "--strategy", "molecule", "--limit", "2"]) == 2
    back = [*args, "--direction", "backward"]
    assert arrows_main(["samples", *back, "--strategy", "molecule"]) == 0
    assert "leakage of the backward inputs, molecule" in capsys.readouterr().out
    assert arrows_main(["edit-labels", *back]) == 0
    out = direction_dir("pmechdb", mini_processed, BACKWARD)
    report = json.loads((out / "edit_labels_report.json").read_text())
    assert report["direction"] == BACKWARD and report["labeled_fraction"] == 1.0
    assert sum(report["agreement"].values()) == report["n_steps"]
    assert (out / "samples_report.json").exists()
    assert not (mini_processed / "pmechdb" / "samples_report.json").exists()


# ---- training and decoding


def test_a_backward_run_reads_the_backward_views_and_records_its_direction(
    mini_processed, token_loader
):
    from ksmech.train.loop import TrainConfig, load_source_part, run_notes, train_config

    build_all_samples("pmechdb", ["molecule"], mini_processed, token_loader)
    build_all_samples("pmechdb", ["molecule"], mini_processed, token_loader, direction=BACKWARD)
    forward, backward = TrainConfig(), TrainConfig(direction=BACKWARD)
    assert "direction" not in train_config(forward)  # a forward header as before
    assert train_config(backward)["direction"] == BACKWARD
    assert "backward samples" in run_notes(backward, None)["selection"]
    assert "backward" not in run_notes(forward, None)["selection"]
    samples, info = load_source_part(backward, "pmechdb", "train", mini_processed)
    assert info["path"] == str(samples_path("pmechdb", "molecule_train", mini_processed, BACKWARD))
    keyed = {s.step_id: s for s in samples}
    fwd, _ = load_source_part(forward, "pmechdb", "train", mini_processed)
    assert set(keyed) == {s.step_id for s in fwd}  # the same steps, seen from the child
    for s in fwd:
        assert keyed[s.step_id].product_key != s.product_key


def test_the_decode_command_takes_the_direction():
    from ksmech.eval.cli import _directions, build_parser

    args = build_parser().parse_args(["decode", "--run", "x", "--direction", "backward"])
    assert args.direction == BACKWARD
    assert build_parser().parse_args(["decode", "--run", "x"]).direction == FORWARD
    backward = {"provenance": {"config": {"train": {"direction": BACKWARD}}}}
    forward = {"provenance": {"config": {"train": {"split": "molecule"}}}}
    out = io.StringIO()
    assert _directions(FORWARD, forward, Console(file=out)) == {}
    assert _directions(BACKWARD, backward, Console(file=out)) == {
        "direction": BACKWARD,
        "checkpoint_direction": BACKWARD,
    }
    assert out.getvalue() == ""
    got = _directions(BACKWARD, forward, Console(file=out))
    assert got["checkpoint_direction"] == FORWARD and "decoding backward" in out.getvalue()


def test_hits_of_the_two_directions_are_never_paired(tmp_path):
    from ksmech.eval import paired
    from ksmech.eval.cli import DECODE_VERSION

    def hits(path, **config):
        rows = [{"step_id": f"s{k}", "rank": 0, "n_found": 1} for k in range(3)]
        header = prov.stamp("decode_hits", DECODE_VERSION, {}, {"split": "molecule"} | config)
        return prov.write_json(path, {"per_sample": rows}, header)

    forward, backward = hits(tmp_path / "f.json"), hits(tmp_path / "b.json", direction=BACKWARD)
    with pytest.raises(ValueError, match="direction"):
        paired.compare_arms([forward], [backward])
    assert paired.compare_arms([backward], [hits(tmp_path / "b2.json", direction=BACKWARD)])


# ---- coverage and leakage


def test_species_coverage_counts_each_side_and_the_product_only_species(mini_steps, tmp_path):
    steps = [mini_steps[k] for k in ("sn2", "proton_transfer")]
    loader = lambda key: None if key == "CO" else object()  # noqa: E731 - methanol missing
    manifest = tmp_path / "qm_manifest.parquet"
    pd.DataFrame(
        {"smiles": ["CO", "CO", "[Cl-]"], "status": ["failed", "ok", "failed"],
         "reason": ["OPT_NOT_CONVERGED", "", "SCF_NOT_CONVERGED"]}
    ).to_parquet(manifest)  # fmt: skip
    assert qm_status(["CO", "[Cl-]", "O"], manifest) == {
        "CO": "ok",
        "[Cl-]": "failed:SCF_NOT_CONVERGED",
        "O": "absent",
    }
    report = species_coverage(steps, loader, manifest)
    assert report["reactants"]["missing_tokens"] == {}
    assert report["reactants"]["n_steps_served"] == 2
    products = report["products"]
    assert products["missing_tokens"] == {"MISSING_TOKENS": 1} and products["n_steps_served"] == 1
    assert products["examples_missing"] == ["CO"]
    assert report["products_only"]["missing_tokens"] == {"MISSING_TOKENS": 1}
    assert report["products"]["without_qm"]["failed:SCF_NOT_CONVERGED"] == 1


def _toy(step_id: str, reactants: str, products: str) -> ElementaryStep:
    return ElementaryStep(step_id, "pmechdb", None, None, reactants, products)


def test_molecule_leakage_reads_the_backward_inputs_against_both_sides_of_training():
    steps = [
        _toy("a", "CCCCO", "CCCC[O-]"),
        _toy("b", "CCCCCl", "CCCCCCN"),  # its product is a training reactant below
        _toy("c", "CCCCCCN", "CCCCCCN"),
        _toy("d", "CCCCBr", "O"),  # only a trivial product
    ]
    ids = {"m_train": ["a", "c"], "m_val": ["b"], "m_test": ["d"]}
    got = molecule_leakage(steps, ids, BACKWARD)
    assert got["m_val"]["n_leaking"] == 1 and got["m_val"]["examples"] == ["b"]
    assert got["m_test"] == {"n_steps": 1, "n_leaking": 0, "examples": []}
    assert molecule_leakage(steps, ids, FORWARD)["m_val"]["n_leaking"] == 0
