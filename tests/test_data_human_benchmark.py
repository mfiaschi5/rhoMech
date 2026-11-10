"""The PMechRP human benchmark (`data.human_benchmark`, `data.benchmark_checks`): a
hand-written CSV in the released layout, the overlap with hand-made PMechDB parts, the cache
lookups, and the guard that keeps it out of training; the real-file tests read the released
CSV."""

import csv
import json
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ksmech.data import benchmark_checks as checks
from ksmech.data import human_benchmark as hb
from ksmech.data import provenance as prov
from ksmech.data.cli import LOADERS, main
from ksmech.data.schema import Source

C = hb.BenchmarkCode
# row 1: one step at room temperature; row 2: three steps at -80 °C, the second target blank,
# a cell with a trailing space; row 3: two steps, the first target a whole PMechDB record
ROWS = [
    ["n/a", "CC(C)Cl.[I-].[Na+]", "CC(C)I", "C3H7I", "1", "p 1, Clayden", "CC(C)I"]
    + [""] * 6,
    ["- 80 °C", "CC(=O)Cl.CO ", "COC(C)=O", "C3H6O2", "3", "p 2", "CC([O-])(Cl)[OH+]C", "",
     "COC(C)=O", "", "", "", ""],
    ["100 °C", "CC(=O)OC.[OH-]", "CC(=O)[O-]", "C2H3O2-", "2", "p 3",
     "[CH3:1][C:2](=[O:3])[O:4][CH3:5].[OH-:6]>>[CH3:1][C:2]([O-:3])([OH:6])[O:4][CH3:5] 6=2",
     "CC(=O)[O-]", "", "", "", "", ""],
]  # fmt: skip


def write_csv(path, rows=ROWS, header=hb.HEADER):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return path


def test_temperatures():
    counts = Counter()
    assert hb.parse_temperature("n/a", counts) is None
    assert hb.parse_temperature("-78 °C", counts) == -78.0
    assert hb.parse_temperature("- 80 °C", counts) == -80.0
    assert hb.parse_temperature("100 °C", counts) == 100.0
    assert counts[C.TEMPERATURE_UNREAD] == 0
    assert hb.parse_temperature("reflux", counts) is None
    assert counts[C.TEMPERATURE_UNREAD] == 1


def test_read_benchmark_normalizations(tmp_path):
    pathways, counts = hb.read_benchmark(write_csv(tmp_path / "b.csv"))
    one, three, two = pathways
    assert (one.n_steps, one.targets, one.temperature_c, one.row) == (1, ("CC(C)I",), None, 1)
    assert three.targets == ("CC([O-])(Cl)[OH+]C", None, "COC(C)=O")
    assert three.reactants == "CC(=O)Cl.CO" and three.temperature_c == -80.0
    assert two.targets[0] == "[CH3:1][C:2]([O-:3])([OH:6])[O:4][CH3:5]"
    assert counts[C.WHITESPACE_STRIPPED] == 1 and counts[C.TARGET_WAS_REACTION] == 1
    assert counts[C.STEP_TARGET_MISSING] == 1 and counts[C.PATHWAY_WITH_MISSING_TARGET] == 1
    assert counts[C.STEP_COUNT_MISMATCH] == 0
    assert one.pathway_id == hb.benchmark_id("CC(C)Cl.[I-].[Na+]", "CC(C)I")


def test_read_benchmark_refuses_another_header(tmp_path):
    with pytest.raises(ValueError, match="header"):
        hb.read_benchmark(write_csv(tmp_path / "b.csv", header=hb.HEADER[:-1] + ("x",)))


def test_step_count_mismatch_extends_the_steps():
    counts = Counter()
    targets = hb.step_targets(["A", "", "C", "", "", "", ""], 2, counts)
    assert targets == ("A", None, "C") and counts[C.STEP_COUNT_MISMATCH] == 1


def test_unreadable_species_have_reason_codes():
    failed = {}
    assert hb.try_species("CC#[N+:10][CH@:20]1CCCC1", failed) is None
    assert hb.try_species("CCCC[Br-]C1=CC=CC=C1", failed) is None
    assert set(failed.values()) == {"SMILES_PARSE", "SANITIZE_FAILED"}
    assert hb.try_species("[CH3:1][OH:2].[Na+]", failed) == ["CO", "[Na+]"]


def test_write_and_load_the_table(tmp_path):
    table, report_path, report, pathways = hb.write_benchmark(
        write_csv(tmp_path / "b.csv"), tmp_path
    )
    assert report["pathways"] == 3 and report["steps"] == 6 and report["with_temperature"] == 2
    assert report["evaluation_only"] is True
    header = prov.read_header(report_path)
    assert header.config == {"evaluation_only": True}
    assert header.content_sha256 == prov.file_sha256(table)
    rows = hb.load_benchmark(tmp_path).to_pylist()
    assert all(r["evaluation_only"] for r in rows)
    assert rows[1]["step_target_smiles"] == ["C[OH+]C(C)([O-])Cl", None, "COC(C)=O"]
    assert rows[2]["step_target_smiles"][0] == "COC(C)([O-])O"
    assert [p.pathway_id for p in hb.load_pathways(tmp_path)] == [p.pathway_id for p in pathways]


def test_last_target_stereo_only():
    counts = Counter()
    p = hb.BenchmarkPathway("x", 1, "C", "CC(O)CC", "", 1, "", None, "n/a", ("C[C@H](O)CC",))
    hb.last_target_check(p, "CC[C@@H](C)O", counts)
    assert counts[C.LAST_TARGET_STEREO_ONLY] == 1 and counts[C.LAST_TARGET_NOT_FINAL] == 0


# ---- the guard


def test_the_benchmark_is_no_source_and_training_refuses_it():
    assert hb.NAME not in {str(s) for s in Source}
    assert hb.NAME not in {str(s) for s in LOADERS}
    with pytest.raises(hb.EvaluationOnly):
        hb.refuse_training(["pmechdb", "human_benchmark"])
    hb.refuse_training(["pmechdb", "reactmech"])
    from ksmech.train.loop import sources

    assert sources(SimpleNamespace(source="pmechdb+reactmech")) == ("pmechdb", "reactmech")
    with pytest.raises(hb.EvaluationOnly):
        sources(SimpleNamespace(source="pmechdb+human_benchmark"))


# ---- overlap


def test_species_index_makes_a_target():
    idx = checks.SpeciesIndex()
    idx.add(["A", "B"], ["C", "D"])
    assert idx.makes({"C"}, {"A"}, any_of=False)
    assert not idx.makes({"C"}, {"A", "X"}, any_of=False)
    assert idx.makes({"C"}, {"A", "X"}, any_of=True)
    assert idx.makes({"C", "D"}, None, any_of=False)
    assert not idx.makes({"C", "E"}, None, any_of=False)


def test_steps_found_follows_the_previous_target():
    def frag(*names):
        return [(n, n) for n in names]

    idx = checks.SpeciesIndex()
    idx.add(["CCCCO", "[Na+]"], ["CCCCO[Na]"])
    idx.add(["CCCCO[Na]"], ["CCCCOC"])
    ids = checks.BenchmarkIds(
        frag("CCCCO", "[Na+]"), frag("CCCCOC"), [frag("CCCCO[Na]"), frag("CCCCOC")]
    )
    assert checks.steps_found(ids, idx, 0) == [1, 2]
    ids.targets[0] = None  # step 2 then checked on its target alone
    assert checks.steps_found(ids, idx, 0) == [2]
    other = checks.BenchmarkIds(frag("CCCCN"), frag("CCCCOC"), [frag("CCCCO[Na]")])
    assert checks.steps_found(other, idx, 0) == []


def pmechdb_tree(tmp_path):
    """PMechDB steps (sides only) and a molecule split file over them."""
    from ksmech.arrows.steps import splits_path, write_split_file

    processed = tmp_path / "processed"
    folder = processed / "pmechdb"
    folder.mkdir(parents=True)
    steps = {
        "pmechdb-a": ("[CH3:1][CH:2]([CH3:3])[Cl:4].[I-:5]", "[CH3:1][CH:2]([CH3:3])[I:5].[Cl-:4]"),
        "pmechdb-b": ("CC(=O)Cl.[OH:1][CH3:2]", "CC([O-])(Cl)[OH+:1][CH3:2]"),
        "pmechdb-c": ("c1ccccc1O", "c1ccccc1[O-].[H+]"),
    }
    table = pa.table(
        {
            "step_id": list(steps),
            "reactants_smiles": [r for r, _ in steps.values()],
            "products_smiles": [p for _, p in steps.values()],
        }
    )
    pq.write_table(table, folder / "steps.parquet")
    ids = {"molecule_train": ["pmechdb-a", "pmechdb-b"], "molecule_val": ["pmechdb-c"]}
    ids["molecule_test"] = []
    write_split_file(
        splits_path("pmechdb", "molecule", processed), "molecule", 0, ids, folder / "steps.parquet"
    )
    return processed


def test_overlap_against_hand_made_parts(tmp_path):
    processed = pmechdb_tree(tmp_path)
    out_dir = processed / hb.NAME
    *_, pathways = hb.write_benchmark(write_csv(tmp_path / "b.csv"), out_dir)
    path, body = checks.write_overlap(pathways, processed, out_dir)
    assert body["part_steps"] == {"train": 2, "val": 1, "test": 0, "any_part": 3}
    train = body["pathways"]["train"]["smiles"]
    assert train["pathways_with_a_step_found"] == 2  # rows 1 and 2 (its first step)
    assert train["pathways_with_every_step_found"] == 1 and train["steps_found"] == 2
    assert train["targets_made"] == 2
    detail = body["per_pathway"][pathways[1].pathway_id]["train"]
    assert detail["steps_found_smiles"] == [1] and detail["steps_found_identity_key"] == [1]
    species = body["species"]
    assert species["found"]["train"]["smiles"]["all"] >= 4  # CC(C)Cl, [I-], CC(C)I, ...
    header = prov.read_header(path)
    assert header.kind == checks.OVERLAP_KIND and any("splits_molecule" in n for n in header.inputs)


def test_identity_key_finds_a_resonance_form_smiles_misses():
    idx = {"train": {v: checks.SpeciesIndex() for v in checks.VIEWS}}
    form = checks.fragment_ids("C=CC=[OH+]")[0]
    other = checks.fragment_ids("[CH2+]C=CO")[0]
    idx["train"]["smiles"].add([form[0]], [])
    idx["train"]["identity_key"].add([form[1]], [])
    found = checks.species_overlap({other}, idx)["found"]["train"]
    assert found == {
        "smiles": {"all": 0, "nontrivial": 1 - 1},
        "identity_key": {"all": 1, "nontrivial": 1},
    }


# ---- coverage


def test_cache_status_reads_files_only(tmp_path):
    from ksmech.orbitals.build import tokens_path
    from ksmech.orbitals.tokens import SCHEMA_VERSION as TOKEN_SCHEMA
    from ksmech.qm.failures import tombstone_path
    from ksmech.qm.records import cache_path
    from ksmech.qm.spin import method_for
    from ksmech.qm.xtb_calc import QMConfig

    qm, tok = tmp_path / "qm", tmp_path / "tokens"
    qm.mkdir()
    tok.mkdir()
    base = QMConfig().method
    cache_path(qm, "CCO", 0, method_for("CCO", base)).write_bytes(b"x")
    header = prov.stamp("tokens", TOKEN_SCHEMA)
    prov.save_npz(tokens_path(tok, "CCO", 0, method_for("CCO", base)), {"a": np.zeros(1)}, header)
    tombstone_path(qm, "CC=O", 0, method_for("CC=O", base)).write_text("{}")
    assert checks.cache_status("CCO", qm, tok) == ("record", "current")
    assert checks.cache_status("CC=O", qm, tok) == ("tombstone", "missing")
    assert checks.cache_status("CC(C)=O", qm, tok) == ("missing", "missing")


def test_coverage_counts_by_role(tmp_path):
    *_, pathways = hb.write_benchmark(write_csv(tmp_path / "b.csv"), tmp_path / "out")
    body = checks.coverage(pathways, tmp_path / "qm", tmp_path / "tokens")
    assert body["qm"] == {"missing": body["distinct_species"]}
    assert body["need_qm_batch"] == body["distinct_species"] == len(body["species"]["qm_missing"])
    assert body["by_role"]["final_products"]["species"] == 3


def test_cli_human_benchmark(tmp_path, capsys):
    processed = pmechdb_tree(tmp_path)
    raw = write_csv(tmp_path / "b.csv")
    caches = ["--qm-cache-dir", str(tmp_path / "qm"), "--tokens-dir", str(tmp_path / "tok")]
    code = main(["human-benchmark", "--raw", str(raw), "--processed-dir", str(processed),
                 "--workers", "1", *caches])  # fmt: skip
    assert code == 0
    out = processed / hb.NAME
    for name in (
        hb.TABLE_NAME,
        hb.REPORT_NAME,
        checks.overlap_name("molecule"),
        checks.COVERAGE_NAME,
    ):
        assert (out / name).is_file()
    assert (
        json.loads((out / checks.COVERAGE_NAME).read_text())["provenance"]["kind"]
        == checks.COVERAGE_KIND
    )
    assert "evaluation only" in capsys.readouterr().out


# ---- real file


@pytest.mark.realdata
@pytest.mark.skipif(not hb.RAW_FILE.is_file(), reason="PMechRP release not in data/raw/pmechrp")
def test_real_benchmark_file():
    pathways, counts = hb.read_benchmark()
    assert len(pathways) == 350 and sum(p.n_steps for p in pathways) == 1034
    assert Counter(p.n_steps for p in pathways) == {
        1: 37,
        2: 113,
        3: 108,
        4: 35,
        5: 38,
        6: 16,
        7: 3,
    }
    assert sum(p.temperature_c is not None for p in pathways) == 168
    assert counts[C.STEP_TARGET_MISSING] == 25 and counts[C.PATHWAY_WITH_MISSING_TARGET] == 19
    assert counts[C.TARGET_WAS_REACTION] == 1 and counts[C.TEMPERATURE_UNREAD] == 0
    assert counts[C.TARGET_MAPPED] == 320 and counts[C.STEP_COUNT_MISMATCH] == 0
    _, failed = hb.table_rows(pathways, counts)
    assert sorted(failed.values()) == ["SANITIZE_FAILED", "SANITIZE_FAILED", "SMILES_PARSE"]
    assert counts[C.LAST_TARGET_STEREO_ONLY] == 2 and counts[C.LAST_TARGET_NOT_FINAL] == 0
