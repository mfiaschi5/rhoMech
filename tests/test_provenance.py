"""Provenance headers: stamp, verify, the QM record -> token chain, backfill and the CLI."""

import dataclasses

import numpy as np
import pytest

from ksmech.data import provenance as prov
from ksmech.data.cli import main as data_main

REAL_CODE_STATE = prov.code_state  # conftest replaces it with a clean stub in every test


def _stamped(path, config=None, inputs=None, schema=1):
    arrays = {"x": np.arange(4)}
    header = prov.stamp("demo", schema, inputs or {}, config or {"tau": 0.15}, "")
    header = dataclasses.replace(header, content_sha256=prov.array_digest(arrays))
    return prov.save_npz(path, arrays, header)


def test_verify_names_every_change(tmp_path):
    src = tmp_path / "input.txt"
    src.write_text("a")
    path = _stamped(tmp_path / "a.npz", inputs=prov.file_input(src))
    assert prov.verify(path, "demo", 1, {"tau": 0.15}).kind == "demo"
    src.write_text("b")
    with pytest.raises(prov.StaleArtifact) as exc:
        prov.verify(path, "demo", 2, {"tau": 0.2})
    changes = exc.value.changes
    assert "schema_version 1 -> 2" in changes
    assert "config tau: 0.15 -> 0.2" in changes
    assert any(c.startswith("input changed:") and "input.txt" in c for c in changes)
    assert exc.value.reason == prov.STALE_ARTIFACT
    src.unlink()
    with pytest.raises(prov.StaleArtifact, match="input missing"):
        prov.verify(path)


def test_named_inputs_and_missing_or_corrupt_headers(tmp_path):
    path = _stamped(tmp_path / "a.npz", inputs={"smiles": prov.sha256_text("CCO")})
    prov.verify(path, expect={"smiles": prov.sha256_text("CCO")})
    with pytest.raises(prov.StaleArtifact, match="input changed: smiles"):
        prov.verify(path, expect={"smiles": prov.sha256_text("CCN")})
    np.savez(tmp_path / "bare.npz", x=np.arange(3))
    with pytest.raises(prov.StaleArtifact, match="no provenance header"):
        prov.verify(tmp_path / "bare.npz")
    (tmp_path / "junk.npz").write_bytes(b"not an npz")
    with pytest.raises(prov.CorruptArtifact):
        prov.verify(tmp_path / "junk.npz")


def test_hash_comes_from_the_header_and_bytes_only_without_one(tmp_path):
    path = _stamped(tmp_path / "a.npz")
    assert prov.current_hash(path) == prov.array_digest({"x": np.arange(4)})
    plain = tmp_path / "plain.bin"
    plain.write_bytes(b"xyz")
    assert prov.current_hash(plain) == prov.sha256_bytes(b"xyz")


def test_framed_and_json_headers_round_trip(tmp_path):
    header = prov.stamp("samples", 3, {"a": "1"}, {"k": [1, 2]}, prov.sha256_bytes(b"data"))
    prov.write_framed(tmp_path / "f.pt", b"data", header)
    back, payload = prov.read_framed(tmp_path / "f.pt")
    assert back == header and payload == b"data"
    prov.write_json(tmp_path / "s.json", {"splits": {}}, header)
    assert prov.read_header(tmp_path / "s.json") == header


def test_collecting_reports_inputs_to_every_open_block():
    with prov.collecting() as outer:
        prov.note_input("file:a", "1")
        with prov.collecting() as inner:
            prov.note_input("file:b", "2")
    prov.note_input("file:c", "3")  # outside any block: dropped
    assert outer == {"file:a": "1", "file:b": "2"} and inner == {"file:b": "2"}


def test_qm_record_header_checks_molecule_method_and_conformer(sp_record, tmp_path):
    from ksmech.qm.records import load_record, save_record

    rec = sp_record("CCO")
    path = save_record(rec, tmp_path / "rec.npz")
    back = load_record(path, (rec.smiles, rec.conformer_id, rec.method))
    assert np.array_equal(back.C, rec.C)
    with pytest.raises(prov.StaleArtifact, match="config method"):
        load_record(path, (rec.smiles, rec.conformer_id, "GFN2-xTB/gas/other-optimizer"))
    with pytest.raises(prov.StaleArtifact, match="input changed: smiles"):
        load_record(path, ("CCN", rec.conformer_id, rec.method))


def test_tokens_go_stale_when_their_qm_record_changes(sp_record, tmp_path):
    from ksmech.orbitals.tokens import build_molecule_tokens, load_tokens, save_tokens
    from ksmech.qm.records import save_record

    rec = sp_record("CC=O")
    qm_path = save_record(rec, tmp_path / "qm.npz")
    tokens = build_molecule_tokens(rec)
    tok_path = save_tokens(tokens, tmp_path / "tok.npz", prov.file_input(qm_path))
    assert load_tokens(tok_path).digest == prov.current_hash(tok_path)
    save_record(dataclasses.replace(rec, charges=rec.charges + 0.01), qm_path)
    with pytest.raises(prov.StaleArtifact, match="input changed: .*qm.npz"):
        load_tokens(tok_path)
    with pytest.raises(prov.StaleArtifact, match="config tau_atom"):
        load_tokens(tok_path, {"tau_atom": 0.2})
    assert load_tokens(tok_path, verify=False).smiles == rec.smiles


def test_backfill_stamps_old_qm_records_without_touching_the_arrays(sp_record, tmp_path):
    from dataclasses import fields

    from ksmech.qm.records import backfill_headers, load_record

    rec = sp_record("CC#N")
    arrays = {f.name: np.asarray(getattr(rec, f.name)) for f in fields(rec)}
    np.savez_compressed(tmp_path / "old.npz", **arrays)  # written before headers existed
    with pytest.raises(prov.StaleArtifact, match="no provenance header"):
        load_record(tmp_path / "old.npz")
    assert data_main(["provenance", "--backfill", "qm", "--qm-cache-dir", str(tmp_path)]) == 0
    assert dict(backfill_headers(tmp_path)) == {"already": 1}
    back = load_record(tmp_path / "old.npz", (rec.smiles, rec.conformer_id, rec.method))
    assert np.array_equal(back.C, rec.C) and np.array_equal(back.fock, rec.fock)
    assert prov.read_header(tmp_path / "old.npz").extra == {"backfilled": True}


def test_provenance_cli_prints_the_chain(tmp_path, capsys):
    src = tmp_path / "input.txt"
    src.write_text("a")
    path = _stamped(tmp_path / "a.npz", inputs=prov.file_input(src))
    assert data_main(["provenance", str(path)]) == 0
    out = capsys.readouterr().out
    assert "demo schema 1" in out and "ok" in out and "input.txt" in out
    src.write_text("b")
    assert data_main(["provenance", str(path)]) == 1
    assert "STALE" in capsys.readouterr().out
    assert data_main(["provenance", str(tmp_path / "none.npz")]) == 2


def test_dirty_builds_are_flagged_and_refused_for_training(monkeypatch, tmp_path):
    monkeypatch.setattr(prov, "code_state", lambda: ("a" * 40, True, "d" * 64))
    path = _stamped(tmp_path / "a.npz")
    header = prov.read_header(path)
    assert header.code_dirty and header.code_diff_sha256 == "d" * 64
    with pytest.raises(prov.StaleArtifact, match="dirty tree"):
        prov.require_clean(header, path)
    legacy = prov.Provenance.from_dict(
        {**dataclasses.asdict(header), "code_git_hash": "b" * 40 + "+dirty", "code_dirty": False}
    )
    assert legacy.code_dirty and legacy.code_git_hash == "b" * 40  # pre-flag headers


def _git(repo, *args):
    """git in a scratch repository, without the GIT_* variables a commit hook exports
    (`prov.git_env`): with `git commit --only`, `git add` here overwrote the index being
    committed."""
    import subprocess

    subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, env=prov.git_env()
    )


def test_code_state_sees_tracked_edits_and_untracked_sources(monkeypatch, tmp_path):
    def git(*args):
        _git(tmp_path, *args)

    git("init", "-q")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    git("add", ".")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "a")
    monkeypatch.setattr(prov, "REPO_ROOT", tmp_path)
    state = REAL_CODE_STATE.__wrapped__
    head, dirty, diff = state()
    assert len(head) == 40 and not dirty and diff == ""
    (tmp_path / "src" / "b.py").write_text("y = 2\n")  # an untracked module a build may import
    _, dirty, diff_b = state()
    assert dirty and len(diff_b) == 64
    (tmp_path / "src" / "a.py").write_text("x = 3\n")
    _, dirty, diff_ab = state()
    assert dirty and diff_ab != diff_b


def test_a_dirty_sample_store_is_refused_unless_allow_stale(
    mini_steps, token_loader, pair_kernels, monkeypatch, tmp_path
):
    from ksmech.arrows import store
    from ksmech.arrows.steps import SAMPLE_VERSION, load_samples, sample_for_step, write_split_file

    s = sample_for_step(mini_steps["e2"], token_loader, pair_kernels)
    monkeypatch.setattr(prov, "code_state", lambda: ("a" * 40, True, "d" * 64))
    parquet = tmp_path / "steps.parquet"
    parquet.write_bytes(b"x")
    split = write_split_file(tmp_path / "splits_x.json", "x", 0, {"x": [s.step_id]}, parquet)
    index = store.write_store(tmp_path, {s.step_id: s}, {}, parquet, SAMPLE_VERSION)
    config = {"split": "x", "sample_version": SAMPLE_VERSION}
    view = store.write_view(tmp_path / "samples_x.json", [s.step_id], {}, index, split, config)
    with pytest.raises(prov.StaleArtifact, match="dirty tree"):
        load_samples(view)
    assert len(load_samples(view, allow_stale=True)) == 1


def test_git_never_touches_an_inherited_index(monkeypatch, tmp_path):
    """A hook of `git commit --only` exports GIT_INDEX_FILE, the absolute path of git's
    temporary commit index. Neither the tests' git nor `code_state` may read or write it: once,
    a copy of the index went from 154 files to 1."""
    commit_index = tmp_path / "next-index.lock"
    commit_index.write_bytes(b"the index being committed")
    monkeypatch.setenv("GIT_INDEX_FILE", str(commit_index))
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "a.py").write_text("x = 1\n")
    _git(repo, "add", ".")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "a")
    monkeypatch.setattr(prov, "REPO_ROOT", repo)
    head, dirty, _ = REAL_CODE_STATE.__wrapped__()
    assert len(head) == 40 and not dirty
    assert commit_index.read_bytes() == b"the index being committed"


def _raise_stale(_task) -> None:
    raise prov.StaleArtifact("shard_00000.npz", ["input changed: tokens.npz"])


def test_every_skip_exception_survives_pickling():
    """Exceptions cross from spawned workers to the parent by pickle; one whose __init__ takes
    other arguments than its message used to fail there (`StaleArtifact` hung the pool)."""
    import pickle

    from ksmech.arrows.apply import ApplyReason, ArrowApplyError
    from ksmech.arrows.reactants import MissingTokens
    from ksmech.data.validation import RejectReason, StepRejected
    from ksmech.qm.records import QMError, ReasonCode

    for exc, fields in (
        (prov.StaleArtifact("a.npz", ["x", "y"]), ("path", "changes")),
        (ArrowApplyError(ApplyReason.VALENCE, "atom 3"), ("reason", "detail")),
        (StepRejected(RejectReason.ARROWS_MISMATCH, "bond"), ("reason", "detail")),
        (QMError(ReasonCode.TIMEOUT, "600 s"), ("reason", "detail")),
        (MissingTokens(["CC", "O"], {"O": prov.STALE_ARTIFACT}), ("keys", "reasons", "reason")),
    ):
        back = pickle.loads(pickle.dumps(exc))
        assert type(back) is type(exc) and str(back) == str(exc)
        for f in fields:
            assert getattr(back, f) == getattr(exc, f), f


def test_a_worker_stale_artifact_reaches_the_parent():
    from multiprocessing import get_context

    with get_context("spawn").Pool(1) as pool, pytest.raises(prov.StaleArtifact) as exc:
        pool.apply_async(_raise_stale, (0,)).get(timeout=120)
    assert exc.value.changes == ["input changed: tokens.npz"]
