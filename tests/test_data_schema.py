import pytest

from ksmech.data.io import read_steps, write_steps
from ksmech.data.schema import Arrow, ElementaryStep, SinkKind, Source, SourceKind

LP, BOND = SourceKind.LONE_PAIR, SourceKind.BOND
ATOM, SBOND = SinkKind.ATOM, SinkKind.BOND


def test_lone_pair_to_atom_forms_bond():
    a = Arrow(LP, (10,), ATOM, (20,))
    assert a.destination() == ("bond", (10, 20))
    assert a.atoms == {10, 20}


def test_bond_to_own_atom_makes_lone_pair():
    assert Arrow(BOND, (21, 20), ATOM, (21,)).destination() == ("lone_pair", (21,))


def test_bond_to_bond_shifts_pair():
    a = Arrow(BOND, (21, 20), SBOND, (22, 21))
    assert a.source_atoms == (20, 21) and a.sink_atoms == (21, 22)  # bonds stored sorted
    assert a.destination() == ("bond", (21, 22))


def test_lone_pair_to_bond_containing_source():
    assert Arrow(LP, (9,), SBOND, (9, 1)).destination() == ("bond", (1, 9))


def test_string_kinds_are_coerced():
    a = Arrow("lone_pair", (10,), "atom", (20,))
    assert a.source_kind is LP and a.sink_kind is ATOM


@pytest.mark.parametrize(
    "args",
    [
        (BOND, (20, 21), ATOM, (22,)),  # bond pair cannot land on a foreign atom
        (BOND, (20, 21), SBOND, (22, 23)),  # bond -> disjoint bond
        (BOND, (20, 21), SBOND, (20, 21)),  # bond -> itself
        (LP, (10,), ATOM, (10,)),  # lone pair onto its own atom
        (LP, (10,), SBOND, (20, 21)),  # lone pair onto a bond it is not part of
        (LP, (10, 11), ATOM, (20,)),  # wrong arity
        (BOND, (10, 10), ATOM, (10,)),  # repeated atom
        (LP, (0,), ATOM, (20,)),  # map numbers start at 1
    ],
)
def test_invalid_arrows_raise(args):
    with pytest.raises(ValueError):
        Arrow(*args)


def _step(**kw) -> ElementaryStep:
    base = dict(
        step_id="pmechdb-x",
        source=Source.PMECHDB,
        pathway_id=None,
        step_index=None,
        reactants_smiles="[CH3:20][Cl:21].[OH-:10]",
        products_smiles="[Cl-:21].[OH:10][CH3:20]",
        arrows=[Arrow(LP, (10,), ATOM, (20,)), Arrow(BOND, (20, 21), ATOM, (21,))],
    )
    return ElementaryStep(**{**base, **kw})


def test_parquet_round_trip(tmp_path):
    steps = [
        _step(),
        _step(step_id="b", pathway_id="p1", step_index=2, mechanism_class="SN2", conditions="rt"),
    ]
    path = write_steps(steps, tmp_path / "pmechdb" / "steps.parquet")
    assert read_steps(path) == steps


def test_source_enum_values():
    assert {s.value for s in Source} == {
        "pmechdb",
        "rmechdb",
        "reactmech",
        "mech_uspto",
        "flower",
        "user_curated",
    }
