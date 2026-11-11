"""--drop-domain-class (open-shell plan P4 step 7): the training samples of the named domain
classes are left out of training, never out of evaluation, and counted."""

from dataclasses import replace
from types import SimpleNamespace

from ksmech.data.schema import ElementaryStep, Source
from ksmech.train import loop
from ksmech.train.loop import DROPPED_DOMAIN_CLASS, TrainConfig, load_source_part

BASE = ElementaryStep("a", Source.PMECHDB, None, None, "[CH4:1]", "[CH4:1]")
STEPS = [
    BASE,
    replace(BASE, step_id="b", domain_class="OSM_CU2"),
    replace(BASE, step_id="c", domain_class="REPAIRED_V1+OSM_FE3"),
    replace(BASE, step_id="d", domain_class="REPAIRED_V1"),
]


def _samples(path, ids=None, compact=False, workers=1):  # load_samples' signature
    return [SimpleNamespace(step_id=s.step_id, n_atoms=3, n_orbitals=4, drawings=[]) for s in STEPS]


def _patched(monkeypatch):
    monkeypatch.setattr(loop, "load_samples", _samples)
    monkeypatch.setattr(loop, "read_steps", lambda path: STEPS)


def test_the_named_classes_leave_training_and_are_counted(monkeypatch, tmp_path):
    _patched(monkeypatch)
    cfg = TrainConfig(drop_domain_class=("OSM_CU2", "OSM_FE3"))
    keep, info = load_source_part(cfg, "pmechdb", "train", tmp_path)
    assert [s.step_id for s in keep] == ["a", "d"]
    assert info["n"] == 4 and info["used"] == 2
    assert info["skipped"] == {DROPPED_DOMAIN_CLASS: 2}


def test_evaluation_parts_keep_every_sample(monkeypatch, tmp_path):
    _patched(monkeypatch)
    cfg = TrainConfig(drop_domain_class=("OSM_CU2", "OSM_FE3"))
    for part in ("val", "test"):
        keep, info = load_source_part(cfg, "pmechdb", part, tmp_path)
        assert len(keep) == 4 and DROPPED_DOMAIN_CLASS not in info["skipped"]


def test_without_the_option_nothing_is_dropped(monkeypatch, tmp_path):
    _patched(monkeypatch)
    keep, info = load_source_part(TrainConfig(), "pmechdb", "train", tmp_path)
    assert len(keep) == 4 and info["skipped"] == {}


def test_the_option_is_on_the_command_line():
    from ksmech.train.cli import build_parser

    args = build_parser().parse_args(["fit", "--run", "x", "--drop-domain-class", "OSM_CU2"])
    assert args.drop_domain_class == ["OSM_CU2"]
