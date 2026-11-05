"""The rmechdb loader (open-shell plan P8 step 2) on tests/fixtures/mini_flower.txt: nine lines
written by hand in FlowER's format (never copied from FlowER): an O-O homolysis, a
recombination, a HAT as RS and again as RC with other maps (the collision), an addition, a
beta-scission beside N2 drawn [N]=[N], an allyl resonance line, a methyl recombination with an
[Ar] third body and its bath-free twin."""

from __future__ import annotations

from pathlib import Path

import pytest

from ksmech.data import domain, rmechdb
from ksmech.data.domain import DomainClass
from ksmech.data.ingest import write_outputs
from ksmech.data.io import read_steps, steps_path
from ksmech.data.schema import Source

MINI = Path(__file__).parent / "fixtures" / "mini_flower.txt"
LINES = MINI.read_text().splitlines()


def line(k: int) -> tuple[str, str]:
    rxn, _, tag = LINES[k].rpartition("|")
    return rxn, tag


@pytest.fixture(scope="module")
def result():
    return rmechdb.load(MINI)


def by_tag_rxn(result, k: int):
    rxn, tag = line(k)
    sid = rmechdb.read_line(None, tag, rxn).step_id
    return next(s for s in result.steps if s.step_id == sid)


def test_one_step_per_reaction_and_every_other_line_counted(result):
    assert result.n_records == 9 and len(result.steps) == 7
    assert sorted(r.detail for r in result.rejections) == [
        "an earlier line is kept",
        "the RC copy is kept",
    ]
    assert {str(r.reason) for r in result.rejections} == {"DUPLICATE_STEP"}
    assert all(s.step_id.startswith("rmechdb-") for s in result.steps)


def test_the_bath_gas_twins_share_one_id_and_ar_is_stripped_and_counted(result):
    ids = [rmechdb.read_line(None, tag, rxn).step_id for rxn, tag in map(line, (6, 7))]
    assert ids[0] == ids[1]
    assert result.notes["NOBLE_GAS_STRIPPED"] == 1
    step = by_tag_rxn(result, 6)
    assert "Ar" not in step.reactants_smiles and "Ar" not in step.products_smiles
    stripped, noble = rmechdb.strip_bath_gases(line(6)[0])
    assert noble and "Ar" not in stripped


def test_the_collision_keeps_the_rc_copy_in_any_file_order(result):
    step = by_tag_rxn(result, 2)  # the RS copy comes first in the file
    assert step.flower_suffix == "RC" and ":21]" in step.reactants_smiles
    assert result.notes["rc_wins_over_rs"] == 1


def test_n2_drawn_with_radicals_is_paired_before_the_id(result):
    rxn, tag = line(4)
    assert rmechdb.read_line(None, tag, rxn).step_id == (
        rmechdb.read_line(None, tag, rxn.replace("[N:9]=[N:10]", "[N:9]#[N:10]")).step_id
    )
    step = by_tag_rxn(result, 4)
    assert "[N:9]#[N:10]" in step.reactants_smiles
    assert DomainClass.PAIRED_DRAWING in domain.domain_classes(step.domain_class)
    assert domain.PAIRED_RULE in step.charge_repair


def test_every_admitted_step_carries_radical_step_and_the_resonance_line_is_flagged(result):
    assert all(s.domain_exclusion is None for s in result.steps)
    assert all(
        str(DomainClass.RADICAL_STEP) in domain.domain_classes(s.domain_class) for s in result.steps
    )
    assert [s.resonance_step for s in result.steps].count(True) == 1
    assert by_tag_rxn(result, 5).resonance_step


def test_only_rs_and_rc_lines_are_read(tmp_path):
    other = tmp_path / "train.txt"
    other.write_text(
        "[H:1][O:2][O:3][H:4]>>[H:1][O:2].[O:3][H:4]|PM\n"
        "[H:1][O:2][O:3][H:4]>>[H:1][O:2].[O:3][H:4]|12\n" + LINES[0] + "\n"
    )
    result = rmechdb.load(other)
    assert result.n_records == 1 and result.steps[0].flower_part == "train"


def test_a_reaction_in_both_parts_is_flagged(tmp_path):
    (tmp_path / "train.txt").write_text(LINES[0] + "\n" + LINES[1] + "\n")
    (tmp_path / "test.txt").write_text(LINES[0] + "\n")
    result = rmechdb.load(tmp_path)
    parts = {s.flower_part for s in result.steps}
    assert parts == {"train+test", "train"} and result.notes["in_train_and_test"] == 1


def test_the_flower_columns_round_trip_through_parquet(result, tmp_path):
    write_outputs(result, tmp_path)
    back = read_steps(steps_path(Source.RMECHDB, tmp_path))
    assert {(s.step_id, s.flower_suffix, s.flower_part) for s in back} == {
        (s.step_id, s.flower_suffix, s.flower_part) for s in result.steps
    }


def test_a_line_that_does_not_parse_is_rejected_with_its_reason(tmp_path):
    bad = tmp_path / "test.txt"
    bad.write_text("[C:1](>>[C:1]|RC\n")
    result = rmechdb.load(bad)
    assert not result.steps and len(result.rejections) == 1


@pytest.mark.realdata
def test_five_real_lines_load():
    raw = rmechdb.RAW_DIR / "test.txt"
    if not raw.is_file():
        pytest.skip("FlowER's files are not under data/raw/flower")
    picked = []
    for _part, tag, rxn in rmechdb.radical_lines([("test", raw)]):
        picked.append(f"{rxn}|{tag}")
        if len(picked) == 5:
            break
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.txt"
        path.write_text("\n".join(picked) + "\n")
        result = rmechdb.load(path)
    assert result.n_records == 5
    assert len(result.steps) + len(result.rejections) == 5
    assert all(s.flower_part == "test" and s.flower_suffix in ("RS", "RC") for s in result.steps)


def test_a_pool_of_single_atoms_packs_its_empty_drawings():
    """[O] + [O] -> O2: no bond row, no other drawing (the store stacked (-1, 0) before)."""
    from types import SimpleNamespace

    import numpy as np

    from ksmech.arrows.store import _pack_drawings

    s = SimpleNamespace(n_atoms=2, bonds=np.zeros((0, 3), np.int16), drawings=[])
    shapes = {k: v.shape for k, v in _pack_drawings(s).items()}
    assert shapes["drawing_orders"] == (0, 0) and shapes["drawing_charges"] == (0, 2)


def test_a_samples_tokens_line_up_with_its_orbitals(token_loader, mini_steps):
    """`MechanismSample.tokens` gives the tokens as the sample holds them (P7's roles and
    holes), so the GBDT's per-orbital energies index the sample's own orbitals (P8 refit)."""
    from ksmech.arrows.steps import sample_for_step

    s = sample_for_step(mini_steps["e2"], token_loader)
    tokens = s.tokens(token_loader)
    assert sum(len(t.orbitals) for t in tokens) == s.n_orbitals
