"""`eval.decode`: B + R on a stored sample's Lewis structure, the exact-R hit rank and the
top-k report of the engine's beam decoder."""

from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
from rdkit import Chem

from ksmech.arrows.delta_be import EditLabel, ReactionMatrix, label_counts
from ksmech.arrows.steps import SAMPLE_VERSION, samples_path
from ksmech.data import provenance as prov
from ksmech.data.provenance import StaleArtifact
from ksmech.data.resonance import identity_key, sigma_key
from ksmech.eval.decode import caps_from_report, decode_caps, decode_report, hit_rank, lewis_apply
from ksmech.model.decode import FORM, Completion, DecodeConfig, EntryTable, complete_anchor


def _sample(z, q, bonds, lone_pairs, label=None, nonbonding=None):
    lone = np.asarray(lone_pairs, dtype=np.int8)
    return SimpleNamespace(
        atom_categorical=np.stack([np.asarray(z), np.asarray(q)], axis=1).astype(np.int64),
        bonds=np.asarray(bonds, dtype=np.int16).reshape(-1, 3),
        lone_pairs=lone,
        nonbonding=2 * lone if nonbonding is None else np.asarray(nonbonding, dtype=np.int8),
        edit_label=label,
    )


# chloromethane + hydroxide: C0 Cl1 H2 H3 H4 O5 H6
MECL_OH = _sample(
    [6, 17, 1, 1, 1, 8, 1],
    [0, 0, 0, 0, 0, -1, 0],
    [(0, 1, 1), (0, 2, 1), (0, 3, 1), (0, 4, 1), (5, 6, 1)],
    [0, 3, 0, 0, 0, 3, 0],
)
SN2 = (((0, 5, 1), (0, 1, -1)), ((1, 2), (5, -2)))


def test_sn2_gives_methanol_and_chloride():
    key, why = lewis_apply(MECL_OH)(*SN2)
    assert why is None
    assert key == sigma_key(Chem.MolFromSmiles("CO.[Cl-]"))


def test_a_missing_bond_cannot_break():
    assert lewis_apply(MECL_OH)(((1, 5, -1), (0, 5, 1)), ()) == (None, "BOND_ORDER")


def test_a_fifth_bond_on_carbon_exceeds_the_octet():
    assert lewis_apply(MECL_OH)(((0, 5, 1),), ((5, -2),)) == (None, "OCTET_EXCEEDED")


def test_a_new_dianion_on_a_nonmetal_is_refused():
    assert lewis_apply(MECL_OH)(((5, 6, -1),), ((5, 2),)) == (None, "FORMAL_CHARGE_RANGE")


def test_a_homolysis_moves_one_electron_to_each_atom():
    """A homolysis' +1 on each atom is a one-electron entry (open-shell plan P6): N is tracked,
    so the product is two radicals, keyed apart from the heterolysis of the same bond."""
    key, why = lewis_apply(MECL_OH)(((0, 1, -1),), ((0, 1), (1, 1)))
    assert why is None
    assert key == identity_key(Chem.MolFromSmiles("[CH3].[Cl].[OH-]"))
    hetero, _ = lewis_apply(MECL_OH)(((0, 1, -1),), ((1, 2),))
    assert hetero == identity_key(Chem.MolFromSmiles("[CH3+].[Cl-].[OH-]")) != key


def test_a_radical_recombines_through_its_odd_electron():
    """Two methyl radicals (N = 1 on each C) form ethane: the -1 entries pass the N check, and
    a -2 on an atom with one electron is LONE_PAIRS."""
    z, q = [6, 1, 1, 1, 6, 1, 1, 1], [0] * 8
    bonds = [(0, 1, 1), (0, 2, 1), (0, 3, 1), (4, 5, 1), (4, 6, 1), (4, 7, 1)]
    methyls = _sample(z, q, bonds, [0] * 8, nonbonding=[1, 0, 0, 0, 1, 0, 0, 0])
    key, why = lewis_apply(methyls)(((0, 4, 1),), ((0, -1), (4, -1)))
    assert why is None and key == sigma_key(Chem.MolFromSmiles("CC"))
    assert lewis_apply(methyls)(((0, 4, 1),), ((0, -2),)) == (None, "LONE_PAIRS")


def test_a_pi_homolysis_is_a_step_not_resonance():
    """C=C -> .C-C. keeps every sigma bond, but the identity key v2 counts the unpaired
    electrons: a real step, not RESONANCE_ONLY_STEP."""
    ethene = _sample([6, 6, 1, 1, 1, 1], [0] * 6, [(0, 1, 2), (0, 2, 1), (0, 3, 1), (1, 4, 1),
                     (1, 5, 1)], [0] * 6)  # fmt: skip
    key, why = lewis_apply(ethene)(((0, 1, -1),), ((0, 1), (1, 1)))
    assert why is None and key == identity_key(Chem.MolFromSmiles("[CH2][CH2]"))


def test_moving_only_the_pi_bond_is_resonance():
    formaldehyde = _sample(
        [6, 8, 1, 1], [0, 0, 0, 0], [(0, 1, 2), (0, 2, 1), (0, 3, 1)], [0, 2, 0, 0]
    )
    assert lewis_apply(formaldehyde)(((0, 1, -1),), ((1, 2),)) == (None, "RESONANCE_ONLY_STEP")


def _completion(bonds, lone, logp=0.0):
    return Completion(logp, tuple(bonds), tuple(lone), (0, 5, 0))


def _label(*variants):
    return SimpleNamespace(variants=[ReactionMatrix(b, lo) for b, lo in variants])


def test_hit_rank_finds_any_variant_of_the_class():
    wrong = _completion([(0, 5, 1), (0, 2, -1)], [(5, -2)])
    right = _completion([(0, 1, -1), (0, 5, 1)], [(5, -2), (1, 2)])
    label = _label(([(0, 5, 1), (0, 1, -1)], [(1, 2), (5, -2)]))
    assert hit_rank([wrong, right], label) == 1
    assert hit_rank([wrong], label) is None


def test_report_counts_hits_misses_and_unlabelled_samples():
    label = _label(SN2)
    right = _completion(*SN2)
    wrong = _completion([(0, 5, 1), (0, 2, -1)], [(5, -2)])
    found = {"a": [right], "b": [wrong, right], "c": [wrong], "d": []}
    samples = [SimpleNamespace(step_id=k, edit_label=label) for k in found]
    samples.append(SimpleNamespace(step_id="e", edit_label=None))

    def decode(sample, skips: Counter):
        return found[sample.step_id]

    r = decode_report(samples, decode)
    assert (r["n"], r["top1"], r["top3"], r["not_found"]) == (4, 0.25, 0.5, 2)
    assert r["skips"] == {"DECODE_NOTHING_FOUND": 1, "NO_EDIT_LABEL": 1}
    rows = {row["step_id"]: (row["rank"], row["n_found"]) for row in r["per_sample"]}
    assert rows == {"a": (0, 1), "b": (1, 2), "c": (None, 1), "d": (None, 0)}
    for k in (1, 3):  # the per-sample ranks reproduce the report
        assert sum(x is not None and x < k for x, _ in rows.values()) / r["n"] == r[f"top{k}"]


def test_decode_takes_the_run_source_apart_from_the_decoded_samples():
    from ksmech.eval.cli import build_parser

    args = build_parser().parse_args(
        ["decode", "--run", "sparse_resonance", "--run-source", "pmechdb+reactmech"]
    )
    assert (args.source, args.run_source) == ("pmechdb", "pmechdb+reactmech")
    assert build_parser().parse_args(["decode", "--run", "x"]).run_source is None


def test_a_sample_version_mismatch_warns_and_is_recorded(tmp_path):
    import io

    from rich.console import Console

    from ksmech.data import provenance as prov
    from ksmech.eval.cli import _sample_versions

    view = tmp_path / "samples_molecule_val.json"
    prov.write_json(view, {"ids": []}, prov.stamp("samples", 17, {}, {}))
    out = io.StringIO()
    payload = {"provenance": {"config": {"sample_version": 15}}}
    got = _sample_versions(view, payload, Console(file=out))
    assert got == {"checkpoint_sample_version": 15, "view_sample_version": 17}
    assert "v15" in out.getvalue() and "v17" in out.getvalue()
    same = {"provenance": {"config": {"sample_version": 17}}}
    assert _sample_versions(view, same, Console(file=io.StringIO()))["view_sample_version"] == 17


# ---- the caps (spec v9.1: legacy, or the label maxima of the source's training part)

LEGACY = {"max_bonds": 8, "max_atoms": 2, "max_candidates": 48}


def _view(path, report):
    """A split view holding only its report (`arrows.steps.split_report`)."""
    prov.write_json(
        path, {"ids": [], "report": report}, prov.stamp("samples", SAMPLE_VERSION, {}, {})
    )
    return path


def test_caps_are_read_from_the_training_part_report(tmp_path):
    view = _view(
        tmp_path / "samples_molecule_train.json", {"label_maxima": {"bonds": 6, "atoms": 4}}
    )
    assert decode_caps("labels", view, SAMPLE_VERSION, {}) == LEGACY | {
        "max_bonds": 6,
        "max_atoms": 4,
    }
    assert decode_caps("legacy", tmp_path / "absent.json", SAMPLE_VERSION, {}) == LEGACY
    over = {"max_bonds": None, "max_atoms": 3, "max_candidates": 64}
    want = {"max_bonds": 6, "max_atoms": 3, "max_candidates": 64}
    assert decode_caps("labels", view, SAMPLE_VERSION, over) == want
    with pytest.raises(ValueError, match="no label maxima"):
        caps_from_report({"class_size": {"1": 3}})  # a report written before the maxima
    with pytest.raises(StaleArtifact):
        decode_caps("labels", view, SAMPLE_VERSION + 1, {})


def test_the_decode_cli_reads_the_training_part_of_the_decoded_source(tmp_path):
    """--caps labels reads `samples_<split>_train.json` of --source, whatever --part and
    --run-source are, and names it as an input; legacy reads nothing."""
    from ksmech.eval.cli import _decode_caps, build_parser

    train = samples_path("reactmech", "molecule_train", tmp_path)
    _view(train, {"label_maxima": {"bonds": 8, "atoms": 4}})
    test = samples_path("reactmech", "molecule_test", tmp_path)
    _view(test, {"label_maxima": {"bonds": 6, "atoms": 2}})
    argv = ["decode", "--run", "x", "--source", "reactmech", "--run-source", "pmechdb+reactmech"]
    argv += ["--part", "test", "--processed-dir", str(tmp_path)]
    labels = build_parser().parse_args([*argv, "--caps", "labels", "--max-candidates", "64"])
    caps, inputs = _decode_caps(labels)
    assert caps == {"max_bonds": 8, "max_atoms": 4, "max_candidates": 64}
    assert list(inputs) == [prov.file_input_name(train)]
    assert _decode_caps(build_parser().parse_args(argv)) == (LEGACY, {})


def _table(r: ReactionMatrix, n: int = 6) -> EntryTable:
    """Entry log-probabilities with 0.9 on each entry of `r` but its forming anchor (0, 1),
    0.98 on 0 everywhere else."""
    pair = np.log(np.tile([0.01, 0.98, 0.01], (n, n, 1)))
    atom = np.log(np.tile([0.01, 0.98, 0.01], (n, 1)))
    for i, j, d in r.bonds:
        pair[i, j] = np.log([0.9 if d == -1 else 0.05, 0.05, 0.9 if d == 1 else 0.05])
    for i, d in r.lone:
        atom[i] = np.log([0.9 if d == -2 else 0.05, 0.05, 0.9 if d == 2 else 0.05])
    return EntryTable((0, 1, FORM), -0.1, pair, atom)


def test_a_label_with_four_atom_entries_is_reachable_with_label_caps_only(tmp_path):
    """Four lone-pair changes, as 11 ReactMech training labels have: the legacy caps (2 atom
    changes) can never write it, the caps read from its training report can."""
    r = ReactionMatrix(((0, 1, 1), (1, 2, -1)), ((0, -2), (3, 2), (4, -2), (5, 2)))
    label = EditLabel((r,), (r.anchors(),), True, 1, False)
    view = _view(tmp_path / "samples_molecule_train.json", label_counts([label]))
    apply = lambda bonds, lone: (repr((bonds, lone)), None)  # noqa: E731
    found = {}
    for mode in ("legacy", "labels"):
        cfg = DecodeConfig(**decode_caps(mode, view, SAMPLE_VERSION, {}))
        found[mode] = list(complete_anchor(_table(r), apply, cfg, Counter()).values())
    assert (cfg.max_bonds, cfg.max_atoms) == (2, 4)
    assert hit_rank(found["labels"], label) == 0
    assert found["legacy"] and hit_rank(found["legacy"], label) is None
    assert all(len(c.lone) <= 2 for c in found["legacy"])


def test_decode_parses_the_caps_and_their_overrides():
    from ksmech.eval.cli import build_parser

    args = build_parser().parse_args(
        ["decode", "--run", "x", "--caps", "labels", "--max-atoms", "4"]
    )
    assert (args.caps, args.max_bonds, args.max_atoms, args.max_candidates) == (
        "labels",
        None,
        4,
        None,
    )
    assert build_parser().parse_args(["decode", "--run", "x"]).caps == "legacy"
