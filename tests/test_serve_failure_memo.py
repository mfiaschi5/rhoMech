"""Serve's featurizer consults the failure memo (`qm.failures`) of the shared cache and of its
own before it runs xTB, and a failure it computes lands in its own memo."""

from ksmech.qm import batch, failures
from ksmech.qm.batch import process_molecule as real_process_molecule
from ksmech.qm.xtb_calc import QMConfig
from ksmech.serve.chem.featurize import FeaturizeStatus, QMFeaturizer
from tests.test_serve_fixtures import no_xtb, serve_settings  # noqa: F401


def _serve_config(s, cache_dir):
    return QMConfig(cache_dir=cache_dir, timeout_s=int(s.qm_timeout_s))


def test_a_shared_failure_is_not_recomputed(serve_settings):  # noqa: F811
    s = serve_settings(chemistry="real")
    failures.remember(_serve_config(s, s.shared_qm), "CCO", 0, "SCF_NOT_CONVERGED", "250", 40.0)
    feat = QMFeaturizer(s, allow_compute=True)
    res = feat.ensure("CCO")  # batch.process_molecule raises under the xTB guard
    assert res.status == FeaturizeStatus.FAILED and res.code == "FEATURIZATION_FAILED"
    assert res.upstream == "SCF_NOT_CONVERGED"
    assert res.detail.startswith(failures.KNOWN_PREFIX)
    assert "SCF_NOT_CONVERGED" in s.failures_log.read_text()


def test_a_larger_serve_budget_runs_again(serve_settings, monkeypatch):  # noqa: F811
    s = serve_settings(chemistry="real")
    small = QMConfig(cache_dir=s.shared_qm, timeout_s=int(s.qm_timeout_s) - 1)
    failures.remember(small, "CCO", 0, "TIMEOUT", "> 899s", 899.0)
    calls = []
    monkeypatch.setattr(batch, "process_molecule", lambda task: calls.append(task) or [{}])
    feat = QMFeaturizer(s, allow_compute=True)
    assert feat._known_failure("CCO") is None
    feat._run_qm("CCO")
    assert len(calls) == 1


def test_a_computed_failure_is_remembered_in_serve(serve_settings, monkeypatch):  # noqa: F811
    s = serve_settings(chemistry="real")
    calls = []

    def scf_fails(task):
        calls.append(task.smiles)
        return [batch._row(task.smiles, 0, reason="SCF_NOT_CONVERGED", detail="250", seconds=3.0)]

    monkeypatch.setattr(batch, "process_molecule", real_process_molecule)
    monkeypatch.setattr(batch, "_run_molecule", scf_fails)
    feat = QMFeaturizer(s, allow_compute=True)
    assert feat.ensure("[CH2-]O").upstream == "SCF_NOT_CONVERGED"
    config = _serve_config(s, s.serve_qm)
    assert failures.tombstone_path(s.serve_qm, "[CH2-]O", 0, config.method).is_file()
    second = feat.ensure("[CH2-]O")
    assert second.detail.startswith(failures.KNOWN_PREFIX) and calls == ["[CH2-]O"]
