"""Provenance headers: stamp, verify, the QM record -> token chain, backfill and the CLI."""

import dataclasses

import numpy as np
import pytest

from ksmech.data import provenance as prov
from ksmech.data.cli import main as data_main


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
