"""`eval.roundtrip`: the forward beam reranked by the backward model's log P of -R at each
child (reaction-network plan, items H and I5), on chloromethane + hydroxide with a stub
backward scorer."""

from collections import Counter
from types import SimpleNamespace

from rdkit import Chem

from ksmech.arrows.reactants import TokenCache
from ksmech.data.chem import parse_mapped, unmapped_canonical
from ksmech.eval.decode import lewis_product
from ksmech.eval.roundtrip import (
    CHILD_INVALID,
    Rescored,
    child_state,
    combined,
    missing_species,
    reorder,
    reverse_on_maps,
    roundtrip_report,
    roundtrip_rows,
)
from ksmech.eval.roundtrip_cli import add_parser
from ksmech.model.decode import FORM, Completion
from ksmech.network.pools import apply_on_mol, state_of_mol
from tests.test_eval_decode import MECL_OH, SN2

# the label: SN2 to methanol + chloride; an alternative: the C-H proton to hydroxide
SN2_C = Completion(-0.1, SN2[0], SN2[1], (0, 5, FORM))
PT_C = Completion(-3.0, ((0, 2, -1), (2, 5, 1)), ((0, 2), (5, -2)), (2, 5, FORM))
LABEL = SimpleNamespace(variants=[SimpleNamespace(bonds=SN2[0], lone=SN2[1])])


def _sample(step_id="s1"):
    return SimpleNamespace(**(vars(MECL_OH) | {"step_id": step_id, "edit_label": LABEL}))


class StubScorer:
    """-8 for any child with chloride (the SN2 child), -0.5 otherwise; records the calls."""

    def __init__(self):
        self.calls = []

    def step_logp(self, state, bonds, lone):
        self.calls.append((state, bonds, lone))
        chloride = any(unmapped_canonical(Chem.MolFromSmiles(k)) == "[Cl-]" for k in _keys(state))
        return SimpleNamespace(marginal=-8.0 if chloride else -0.5)


def _keys(state):
    return state.smiles.split(".")


def test_the_child_is_the_product_on_map_numbers_k_plus_1():
    child = child_state(lewis_product(MECL_OH), SN2_C)
    keys = sorted(unmapped_canonical(f) for f in Chem.GetMolFrags(parse_mapped(child.smiles),
                                                                  asMols=True))  # fmt: skip
    assert keys == ["CO", "[Cl-]"]
    assert {a.GetAtomMapNum() for a in parse_mapped(child.smiles).GetAtoms()} == set(range(1, 8))


def test_minus_r_on_the_child_gives_the_parent_back():
    """-R on the child's map numbers is the previous step: applied there, the parent's pool."""
    parent = state_of_mol(lewis_product(MECL_OH)((), ()))
    child = child_state(lewis_product(MECL_OH), SN2_C)
    bonds, lone = reverse_on_maps(SN2_C)
    assert bonds == [(1, 6, -1), (1, 2, 1)] and lone == [(2, -2), (6, 2)]
    mol = parse_mapped(child.smiles)
    idx = {a.GetAtomMapNum(): a.GetIdx() for a in mol.GetAtoms()}
    back = apply_on_mol(mol, [(idx[m], idx[n], d) for m, n, d in bonds],
                        [(idx[m], d) for m, d in lone])  # fmt: skip
    assert back.key == parent.key


def test_a_product_rdkit_cannot_read_is_child_invalid():
    hypervalent = Completion(-1.0, ((1, 5, 1),), (), (1, 5, FORM))  # Cl- and O2- bonded
    broken = Completion(-1.0, ((1, 5, -1),), (), (1, 5, FORM))  # no Cl-O bond to break
    assert child_state(lewis_product(MECL_OH), hypervalent) == CHILD_INVALID
    assert child_state(lewis_product(MECL_OH), broken) == CHILD_INVALID


def test_reorder_permutes_only_the_scored_places():
    c0, c1, c2 = (Completion(lp, (), (), (0, 1, FORM)) for lp in (-1.0, -2.0, -3.0))
    rescored = [Rescored(c0, -10.0), Rescored(c1, None, "MISSING_TOKENS"), Rescored(c2, 0.0)]
    assert reorder([c0, c1, c2], rescored, 0.0) == [c0, c1, c2]
    assert reorder([c0, c1, c2], rescored, 1.0) == [c2, c1, c0]
    extra = Completion(-9.0, (), (), (0, 1, FORM))  # past `top`: stays last
    assert reorder([c0, c1, c2, extra], rescored, 1.0)[-1] is extra


def test_a_backward_minus_inf_is_ignored_at_lambda_0_and_last_otherwise():
    c0, c1 = (Completion(lp, (), (), (0, 1, FORM)) for lp in (-1.0, -2.0))
    r = [Rescored(c0, float("-inf")), Rescored(c1, -5.0)]
    assert combined(r[0], 0.0) == -1.0
    assert reorder([c0, c1], r, 0.5) == [c1, c0]


def test_rows_and_report_count_the_top1_flips_against_lambda_0():
    scorer = StubScorer()
    rows = roundtrip_rows([_sample()], lambda s, k: [SN2_C, PT_C], scorer, [0.0, 0.25, 1.0], 5,
                          {}, Counter())  # fmt: skip
    assert len(scorer.calls) == 2 and rows[0]["full"]
    assert rows[0]["backward"] == [-8.0, -0.5]
    by = rows[0]["by_lambda"]
    assert by["0"]["rank"] == 0 and by["0.25"]["rank"] == 0  # -2.1 against -3.125
    assert by["1"]["rank"] == 1  # -8.1 against -3.5: the proton transfer first
    report = roundtrip_report(rows, [0.0, 0.25, 1.0])
    assert report["by_lambda"]["0"]["top1"] == 1.0
    assert report["by_lambda"]["1"]["lost_top1"] == 1 and report["by_lambda"]["1"]["top1"] == 0
    assert report["coverage"] == {"completions": 2, "scored": 2, "unscored": {}}
    assert set(by["1"]) >= {"step_id", "rank", "rank_any", "n_found"}  # a decode_hits row


def test_an_unscored_completion_is_counted_and_stays_in_place():
    class Missing(StubScorer):
        def step_logp(self, state, bonds, lone):
            return "MISSING_TOKENS"

    skips: Counter = Counter()
    rows = roundtrip_rows([_sample()], lambda s, k: [SN2_C, PT_C], Missing(), [0.0, 4.0], 5, {},
                          skips)  # fmt: skip
    assert not rows[0]["full"] and rows[0]["by_lambda"]["4"]["rank"] == 0
    assert skips["ROUNDTRIP_UNSCORED:MISSING_TOKENS"] == 2
    assert roundtrip_report(rows, [0.0, 4.0])["coverage"]["unscored"] == {"MISSING_TOKENS": 2}


def test_missing_species_lists_the_children_without_tokens(tmp_path):
    loader = TokenCache(tmp_path / "tokens", bond_index_dir=tmp_path / "bond_index")
    skips: Counter = Counter()
    found = missing_species([_sample()], lambda s, k: [SN2_C, PT_C], loader, 5, skips)
    assert found == {"CO": "MISSING_TOKENS", "[Cl-]": "MISSING_TOKENS", "O": "MISSING_TOKENS",
                     "[CH2-]Cl": "MISSING_TOKENS"}  # fmt: skip
    only_first = missing_species([_sample()], lambda s, k: [SN2_C, PT_C], loader, 1, skips)
    assert set(only_first) == {"CO", "[Cl-]"}


def test_the_command_parses_with_the_decode_options():
    import argparse

    sub = argparse.ArgumentParser().add_subparsers(dest="command")
    add_parser(sub)
    args = sub.choices["roundtrip"].parse_args(["--run", "a", "--backward-run", "b",
                                                "--cohort", "closed_base"])  # fmt: skip
    assert (args.direction, args.top, args.beam, args.caps) == ("forward", 5, 5, "legacy")
    assert args.lambdas == [0.25, 0.5, 1.0, 2.0] and args.species_file is None
