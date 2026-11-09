"""ReactMech pathway chains (`data.pathways`): chains read from hand-written mechanism files,
matched to hand-written step lines, the atom maps compared at each boundary, the sidecar and
its provenance; the real-file tests read the released mechanism files and the written
sidecar."""

import csv
import json
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from rdkit import Chem

from ksmech.data import pathways as pw
from ksmech.data import provenance as prov
from ksmech.data import reactmech
from ksmech.data.cli import main
from ksmech.data.schema import Source
from ksmech.data.validation import make_step_id

# acetyl chloride + hydroxide -> tetrahedral intermediate -> acetic acid + chloride
A0 = "CC(=O)Cl.[OH-]"
A1 = "CC([O-])(O)Cl"
A2 = "CC(=O)O.[Cl-]"
L1 = "[CH3:1][C:2](=[O:3])[Cl:4].[OH-:5]>>[CH3:1][C:2]([O-:3])([OH:5])[Cl:4]"
L2 = "[CH3:6][C:7]([O-:8])([OH:9])[Cl:10]>>[CH3:6][C:7](=[O:8])[OH:9].[Cl-:10]"  # renumbered
# methylamine + acetyl chloride, one step only in the step files
B0 = "CC(=O)Cl.CN"
B1 = "C[NH2+]C(C)([O-])Cl"
L3 = "[CH3:1][C:2](=[O:3])[Cl:4].[CH3:5][NH2:6]>>[CH3:1][C:2]([O-:3])([NH2+:6][CH3:5])[Cl:4]"


def sid(line: str) -> str:
    return make_step_id(Source.REACTMECH, line)


def write_mechanisms(path, rows, with_class=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["mechanistic_class", "mechanisms"] if with_class else ["mechanisms"])
        for cls, species in rows:
            w.writerow([cls, str(species)] if with_class else [str(species)])


@pytest.fixture
def tree(tmp_path):
    """A raw ReactMech tree (ID split with ID_CRM, OOD split with OOD_Set1) and a processed
    directory holding L1 in its steps parquet and L2 in its rejections."""
    raw = tmp_path / "raw" / "data"
    for split in reactmech.SPLITS:
        (raw / split).mkdir(parents=True)
        for name in reactmech.SPLIT_FILES:
            (raw / split / name).write_text("")
    (raw / "ID_Split42" / "train.txt").write_text(f"{L1}\n{L2}\n\n")
    (raw / "OOD_Split42" / "train.txt").write_text(f"{L1}\n{L3}\n")  # L1 again: read once
    chains = [(None, [A0, A1, A2]), (None, [A0, A1, A2]), (None, ["CCO", "CC=O"])]
    write_mechanisms(raw / "ID_Split42" / "ID_CRM_dataset" / "ID_CRM.csv", chains)
    ood = [("acid_halide", [A0, A1]), ("amide", [B0, B1, "CC(=O)NC.Cl"])]
    write_mechanisms(raw / "OOD_Split42" / "OOD_CRM_dataset" / "OOD_Set1.csv", ood, True)
    processed = tmp_path / "processed"
    folder = processed / "reactmech"
    folder.mkdir(parents=True)
    pq.write_table(pa.table({"step_id": [sid(L1)]}), folder / "steps.parquet")
    with open(folder / "rejections.csv", "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows([["step_id", "reason", "detail", "raw"], [sid(L2), "X", "", L2]])
    return tmp_path / "raw", processed


def test_pathway_id_is_stable_and_order_sensitive():
    assert pw.pathway_id([A0, A1]) == pw.pathway_id((A0, A1))
    assert pw.pathway_id([A0, A1]) != pw.pathway_id([A1, A0])
    assert pw.pathway_id([A0, A1]).startswith("reactmech-path-")


def test_read_chains_drops_repeats(tree):
    raw, _ = tree
    counts = Counter()
    chains = pw.read_chains(pw.chain_files(raw), counts)
    assert counts[pw.ChainCode.DUPLICATE_CHAIN] == 1
    assert [c.chain_file for c in chains] == ["ID_CRM.csv"] * 2 + ["OOD_Set1.csv"] * 2
    assert [c.n_steps for c in chains] == [2, 1, 1, 2]
    assert chains[2].mechanism_class == "acid_halide" and chains[0].mechanism_class is None
    assert chains[3].row == 2  # 1-based data row in its own file


def test_step_lines_read_like_the_ingest(tree):
    raw, _ = tree
    lines = pw.step_lines(raw)
    assert set(lines) == {sid(L1), sid(L2), sid(L3)}
    index = pw.line_index(lines)
    assert index[(reactmech.side_key(A0), reactmech.side_key(A1))] == [sid(L1)]


def test_boundary_maps_same_renumbered_inconsistent():
    p = "[CH3:1][C:2]([O-:3])([OH:5])[Cl:4]"
    assert pw.boundary_maps(p, "[Cl:4][C:2]([CH3:1])([O-:3])[OH:5]") == pw.MapAgreement.SAME
    renum = L2.split(">>")[0]
    assert pw.boundary_maps(p, renum) == pw.MapAgreement.RENUMBERED
    assert pw.boundary_maps(p, "[CH3:1][C:2](=[O:3])[OH:5].[Cl-:4]") == pw.MapAgreement.INCONSISTENT
    # a spectator unmapped in one step and mapped in the next: one drawing, a graph match
    partial = "[CH3:7][C:8]([O-:9])([OH:10])[Cl:11].[Na+]"
    assert pw.boundary_maps(p + ".[Na+:1]", partial) == pw.MapAgreement.PARTLY_MAPPED
    assert pw.boundary_maps(p + ".[Na+:1]", partial + ".[K+]") == pw.MapAgreement.INCONSISTENT
    # symmetric atoms swapped are one mapped drawing
    assert pw.boundary_maps("[CH3:1][C:2]([CH3:3])=[O:4]", "[CH3:3][C:2]([CH3:1])=[O:4]") == "same"


def test_renumbering_maps_each_atom():
    p = Chem.MolFromSmiles("[CH3:1][C:2]([O-:3])([OH:5])[Cl:4]")
    r = Chem.MolFromSmiles(L2.split(">>")[0])
    assert pw.renumbering(p, r) == {1: 6, 2: 7, 3: 8, 5: 9, 4: 10}
    assert pw.renumbering(p, Chem.MolFromSmiles("[CH3:1][C:2](=[O:3])[Cl:4]")) is None
    half = Chem.MolFromSmiles("[CH3:6][C:7]([O-:8])(O)Cl")
    assert pw.renumbering(p, half) == {1: 6, 2: 7, 3: 8}  # the atoms mapped on both sides


def test_build_rows_codes_and_maps(tree):
    raw, processed = tree
    rows, report = pw.build(raw, processed)
    by = {(r["pathway_id"], r["step_index"]): r for r in rows}
    chain = pw.pathway_id([A0, A1, A2])
    first, second = by[(chain, 0)], by[(chain, 1)]
    assert (first["step_id"], first["is_last"], first["n_steps"]) == (sid(L1), 0, 2)
    assert (second["step_id"], second["is_last"]) == (sid(L2), 1)
    assert first["maps_to_next"] == "renumbered" and second["maps_to_next"] is None
    assert (first["ingest_status"], second["ingest_status"]) == ("steps", "rejected")
    assert by[(pw.pathway_id([B0, B1, "CC(=O)NC.Cl"]), 0)]["ingest_status"] == "not_sampled"
    codes = report["codes"]
    assert codes["DUPLICATE_CHAIN"] == 1 and codes["NO_STEP_LINE"] == 2
    assert codes["CHAIN_UNMATCHED"] == 1 and codes["CHAIN_PARTIAL"] == 1
    assert codes["CHAIN_LAST_UNMATCHED"] == 2
    assert codes["STEP_IN_SEVERAL_CHAINS"] == 1 and codes["STOP_LABEL_CONFLICT"] == 1
    assert report["chain_files"]["ID_CRM.csv"] == {
        "chains": 2, "steps": 3, "matched_rows": 2, "matched_steps": 2,
        "whole": 1, "partial": 0, "unmatched": 1, "duplicate_rows": 1,
    }  # fmt: skip
    assert report["maps"]["boundaries"] == {"renumbered": 1}
    maps = report["maps"]
    assert maps["consistent_share"] == maps["renumbering_share"] == 1.0
    assert maps["same_number_share"] == 0
    assert report["stop_rows"] == 2 and report["ingest_status"]["steps"] == 2


def test_write_and_load_the_sidecar(tree):
    raw, processed = tree
    table, out, report = pw.write_pathways(raw, processed)
    header = prov.read_header(out)
    assert header.kind == pw.KIND and header.content_sha256 == prov.file_sha256(table)
    assert any(name.endswith("ID_CRM.csv") for name in header.inputs)
    assert any(name.endswith("steps.parquet") for name in header.inputs)
    loaded = pw.load_pathways(processed)
    assert loaded.num_rows == report["rows"] and loaded.schema == pw.TABLE_SCHEMA
    table.write_bytes(table.read_bytes() + b"\0")
    with pytest.raises(prov.StaleArtifact):
        pw.load_pathways(processed)


def test_load_refuses_when_the_steps_change(tree):
    raw, processed = tree
    pw.write_pathways(raw, processed)
    pq.write_table(pa.table({"step_id": [sid(L1), sid(L3)]}), processed / "reactmech/steps.parquet")
    with pytest.raises(prov.StaleArtifact, match="steps.parquet"):
        pw.load_pathways(processed)


def test_cli_pathways(tree, capsys):
    raw, processed = tree
    code = main(["pathways", "--source", "reactmech", "--raw", str(raw), "--processed-dir",
                 str(processed), "--workers", "1"])  # fmt: skip
    assert code == 0
    report = json.loads((processed / "reactmech" / pw.REPORT_NAME).read_text())
    assert report["chains"] == 4 and "wrote" in capsys.readouterr().out


# ---- real files


REAL_CHAINS = reactmech.RAW_DIR.is_dir()


@pytest.mark.realdata
@pytest.mark.skipif(not REAL_CHAINS, reason="ReactMech raw files not in data/raw/reactmech")
def test_real_chain_files():
    counts = Counter()
    chains = pw.read_chains(pw.chain_files(), counts)
    files = Counter(c.chain_file for c in chains)
    assert files == {"ID_CRM.csv": 2748, "OOD_Set1.csv": 3311, "OOD_Set2.csv": 398}
    assert counts[pw.ChainCode.DUPLICATE_CHAIN] == 72  # rows repeating a chain: 3 + 29 + 40
    assert counts["DUPLICATE_CHAIN:OOD_Set2.csv"] == 40
    lengths = Counter(c.n_steps for c in chains if c.chain_file == "ID_CRM.csv")
    assert min(lengths) == 1 and max(lengths) == 10
    assert {c.n_steps for c in chains if c.chain_file != "ID_CRM.csv"} == {3, 4}


SIDECAR = pw.source_dir(Source.REACTMECH, pw.DEFAULT_PROCESSED_DIR) / pw.REPORT_NAME


@pytest.mark.realdata
@pytest.mark.skipif(not SIDECAR.is_file(), reason="run `ksmech-data pathways --source reactmech`")
def test_real_sidecar_stop_labels():
    rows = pw.load_pathways().to_pylist()
    assert rows and all(0 <= r["step_index"] < r["n_steps"] for r in rows)
    assert all(r["is_last"] == (r["step_index"] == r["n_steps"] - 1) for r in rows)
    assert {r["ingest_status"] for r in rows} <= {"steps", "not_sampled", "rejected"}
    assert {r["maps_to_next"] for r in rows} <= {None, *map(str, pw.MapAgreement)}
