"""Species lists for the QM batch and the token build (`qm.species`): the file shapes, map-free
canonical fragments, the spin census on listed species, a manifest that never meets a source's,
and the options `--species-from` refuses beside it."""

import json

import pandas as pd
import pytest
from rich.console import Console

from ksmech.data.schema import Source
from ksmech.orbitals import cli as orbitals_cli
from ksmech.qm import batch, species
from ksmech.qm import cli as qm_cli
from ksmech.qm.records import ReasonCode, cache_path
from ksmech.qm.xtb_calc import QMConfig
from tests.fixtures.molecules import MOLECULES

LISTED = [MOLECULES["ethane"], "[CH3:1][OH:2].[Cl-]", MOLECULES["chloromethane"]]
REFUSED = "[CH2]CCC[CH2]"  # RDKit u = 2 without a count: the census refuses it


def test_read_species_file_takes_three_shapes(tmp_path):
    path = tmp_path / "s.json"
    path.write_text(json.dumps(LISTED))
    assert species.read_species_file(path) == (LISTED, {})
    path.write_text(json.dumps({"provenance": {}, "species": LISTED}))
    assert species.read_species_file(path) == (LISTED, {})
    coverage = {"qm_missing": ["CC", "CO"], "tokens_old_schema": ["CCO"], "qm_tombstone": ["[Na]"]}
    path.write_text(json.dumps({"need_qm_batch": 2, "species": coverage}))
    assert species.read_species_file(path) == (["CC", "CCO", "CO"], {"qm_tombstone": 1})
    for bad in ({"ids": ["a"]}, {"species": [1]}, "CC", {"species": {"qm_missing": "CC"}}):
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match="neither a list"):
            species.read_species_file(path)


def test_species_run_folders_never_meet_a_source_folder(tmp_path):
    assert species.SPECIES_DIR not in {str(s) for s in Source}
    sources = {batch.manifest_path(str(s), tmp_path) for s in Source}
    for name in [*(str(s) for s in Source), "human_benchmark", "pilot.v2"]:
        path = species.manifest_path(name, tmp_path)
        assert path not in sources and path.parent.parent == tmp_path / species.SPECIES_DIR
    for bad in ("", "..", "a/b", "../pmechdb", ".hidden"):
        with pytest.raises(ValueError, match="species run name"):
            species.species_dir(bad, tmp_path)
    assert species.run_name(tmp_path / "qm_species.json", None) == "qm_species"


def test_collect_species_canonicalizes_and_applies_the_census():
    smiles, rows = species.collect_species([*LISTED, REFUSED, "C(C", "CC"])
    assert smiles == ["CC", "CCl", "CO", "[Cl-]"]
    reasons = {r["smiles"]: (r["reason"], r["conformer_id"]) for r in rows}
    assert reasons == {
        "C(C": (str(ReasonCode.CANONICALIZE_FAILED), -1),
        "[CH2]CCC[CH2]": (str(ReasonCode.SPIN_CENSUS_REFUSED), -1),
    }


def test_featurize_species_writes_its_own_manifest_and_resumes(tmp_path, monkeypatch):
    processed, cfg = tmp_path / "processed", QMConfig(cache_dir=tmp_path / "cache", timeout_s=120)
    source_manifest = batch.manifest_path("pmechdb", processed)
    source_manifest.parent.mkdir(parents=True)
    sentinel = pd.DataFrame([batch._row("CCO", 0, status="ok", method="m")])
    sentinel.to_parquet(source_manifest, index=False)
    view = species.featurize_species(
        [*LISTED, REFUSED], "bench", processed, config=cfg, console=Console(quiet=True)
    )
    assert list(view.columns) == batch.MANIFEST_COLUMNS
    ok = view[view["status"] == "ok"]
    assert sorted(ok["smiles"]) == ["CC", "CCl", "CO", "[Cl-]"]
    assert all(cache_path(cfg.cache_dir, s, 0, cfg.method).is_file() for s in ok["smiles"])
    refused = view[view["status"] == "failed"].iloc[0]
    assert refused["smiles"] == REFUSED and refused["reason"] == "SPIN_CENSUS_REFUSED"
    on_disk = pd.read_parquet(species.manifest_path("bench", processed))
    assert len(on_disk) == 5
    pd.testing.assert_frame_equal(pd.read_parquet(source_manifest), sentinel)
    calls = []
    monkeypatch.setattr(batch, "process_molecule", lambda t: calls.append(t.smiles) or [])
    species.featurize_species(LISTED, "bench", processed, config=cfg, console=Console(quiet=True))
    assert calls == []  # every listed species is cached


def test_species_from_with_source_is_refused(tmp_path, capsys):
    path = tmp_path / "s.json"
    path.write_text(json.dumps(LISTED))
    for main, command in ((qm_cli.main, "run"), (orbitals_cli.main, "build")):
        with pytest.raises(SystemExit) as exc:
            main([command, "--species-from", str(path), "--source", "pmechdb"])
        assert exc.value.code == 2
        assert "not allowed with argument" in capsys.readouterr().err
        with pytest.raises(SystemExit) as exc:
            main([command])  # one of the two is required
        assert exc.value.code == 2
        assert "one of the arguments --source --species-from" in capsys.readouterr().err


def test_species_options_are_refused_where_they_do_not_apply(tmp_path, capsys):
    path = tmp_path / "s.json"
    path.write_text(json.dumps(LISTED))
    dirs = ["--processed-dir", str(tmp_path / "p"), "--cache-dir", str(tmp_path / "c")]
    run = ["run", "--species-from", str(path), *dirs]
    assert qm_cli.main([*run, "--steps-from", str(path)]) == 2
    assert qm_cli.main([*run, "--name", "../pmechdb"]) == 2
    assert qm_cli.main(["run", "--source", "pmechdb", "--name", "x", *dirs]) == 2
    assert qm_cli.main(["run", "--species-from", str(tmp_path / "none.json"), *dirs]) == 2
    build = ["build", "--processed-dir", str(tmp_path / "p")]
    assert orbitals_cli.main([*build, "--source", "pmechdb", "--name", "x"]) == 2
    assert orbitals_cli.main([*build, "--species-from", str(path), "--name", "a/b"]) == 2
    assert not (tmp_path / "p").exists() and not (tmp_path / "c").exists()
    capsys.readouterr()


def test_cli_species_run_then_token_build(tmp_path):
    """`ksmech-qm run --species-from` then `ksmech-orbitals build --species-from` on the same
    file: records and tokens of the listed species, both summaries under species/<name>/, the
    token cache's own summary.json untouched."""
    path = tmp_path / "listed.json"
    path.write_text(json.dumps({"species": [*LISTED, REFUSED]}))
    processed, cache, tokens = tmp_path / "processed", tmp_path / "qm", tmp_path / "tokens"
    common = ["--species-from", str(path), "--processed-dir", str(processed)]
    qm = ["run", *common, "--cache-dir", str(cache), "--timeout", "120", "--min-success", "0.5"]
    assert qm_cli.main(qm) == 0  # 4 of 5 rows ok: the census refusal is a failure row
    run_dir = processed / species.SPECIES_DIR / "listed"
    assert (run_dir / batch.MANIFEST_NAME).is_file()
    build = ["build", *common, "--qm-cache-dir", str(cache), "--out-dir", str(tokens)]
    assert orbitals_cli.main(build) == 0
    summary = json.loads((run_dir / "tokens_summary.json").read_text())
    assert summary["n_molecules"] == 4 and summary["n_built"] == 4
    assert summary["source"] == "species/listed"
    assert summary["not_collected"] == {REFUSED: "SPIN_CENSUS_REFUSED"}
    assert len(list(tokens.glob("*.npz"))) == 4 and not (tokens / "summary.json").exists()
