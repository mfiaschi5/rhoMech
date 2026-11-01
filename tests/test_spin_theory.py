"""The theory rules of open-shell plan P2 step 3 (B, `data.spin_theory`): restricted GFN2 single
points (ALPB water, the lowest spin the electron count allows) at fixed idealized geometries,
written by hand here, never copied from a dataset."""

import ast
import json
import math
import re
from functools import cache
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from ksmech.data import chem, domain
from ksmech.data import spin_theory as st
from ksmech.qm import point_group as pg
from ksmech.qm.embed import embed, prepare_mol
from ksmech.qm.geometry import BOHR_ANG, new_calculator
from ksmech.qm.xtb_calc import GFN2Calculator

TD = [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]
OH = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
SQUARE = OH[:4]
LINE = [[0, 0, 1], [0, 0, -1]]


def _record(smiles: str, xyz, solvent: str | None = "water"):
    """(mol, record) of a restricted single point at u0 = the electron count's parity."""
    can = chem.unmapped_canonical(Chem.MolFromSmiles(smiles))
    mol = prepare_mol(can)
    z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int32)
    charge = sum(a.GetFormalCharge() for a in mol.GetAtoms())
    u0 = (int(z.sum()) - charge) % 2
    xyz = np.asarray(xyz, dtype=float)
    calc = new_calculator(z, xyz / BOHR_ANG, charge, u0, solvent, save_integrals=True)
    return mol, GFN2Calculator()._assemble(calc.singlepoint(), z, xyz, can, 0, charge, u0)


def _complex(smiles: str, dist: float, dirs, h_dist: float = 0.97) -> np.ndarray:
    """The metal at the origin, its neighbours along `dirs`, a ligand's other neighbour beyond
    it on the same line."""
    mol = prepare_mol(chem.unmapped_canonical(Chem.MolFromSmiles(smiles)))
    xyz = np.zeros((mol.GetNumAtoms(), 3))
    metal = next(a for a in mol.GetAtoms() if a.GetAtomicNum() in domain.GROUP)
    ligands = [n.GetIdx() for n in metal.GetNeighbors()]
    units = np.asarray(dirs, dtype=float)
    units /= np.linalg.norm(units, axis=1, keepdims=True)
    for i, u in zip(ligands, units, strict=True):
        xyz[i] = dist * u
        for far in mol.GetAtomWithIdx(i).GetNeighbors():
            if far.GetIdx() != metal.GetIdx():
                xyz[far.GetIdx()] = (dist + h_dist) * u
    return xyz


@cache
def verdict(name: str) -> st.TheoryVerdict:
    smiles, xyz = SPECIES[name]()
    return st.theory_verdict(*_record(smiles, xyz))


def _bent(r: float, angle_deg: float, centre_first: bool) -> list:
    a = math.radians(angle_deg) / 2
    ends = [[r * math.sin(a), 0, r * math.cos(a)], [-r * math.sin(a), 0, r * math.cos(a)]]
    return [[0, 0, 0], *ends] if centre_first else [ends[0], [0, 0, 0], ends[1]]


def _vinylidene() -> tuple:
    # prepare_mol('[C]=C'): the carbene C, the CH2 carbon, two H
    s60, c60 = math.sin(math.radians(60)), math.cos(math.radians(60))
    h = [[1.09 * s60, 0, 1.30 + 1.09 * c60], [-1.09 * s60, 0, 1.30 + 1.09 * c60]]
    return "[C]=C", [[0, 0, 0], [0, 0, 1.30], *h]


def _tmm() -> tuple:
    """Planar D3h trimethylenemethane: its degenerate e'' pair holds two electrons."""
    mol = prepare_mol(chem.unmapped_canonical(Chem.MolFromSmiles("[CH2]C(=C)[CH2]")))
    xyz = np.zeros((mol.GetNumAtoms(), 3))
    centre = next(a for a in mol.GetAtoms() if all(n.GetAtomicNum() == 6 for n in a.GetNeighbors()))
    for k, c in enumerate(centre.GetNeighbors()):
        d = np.array([math.cos(2 * math.pi * k / 3), math.sin(2 * math.pi * k / 3), 0.0])
        perp = np.array([-d[1], d[0], 0.0])
        xyz[c.GetIdx()] = 1.40 * d
        hs = [h.GetIdx() for h in c.GetNeighbors() if h.GetAtomicNum() == 1]
        for h, sign in zip(hs, (1, -1), strict=True):
            xyz[h] = xyz[c.GetIdx()] + 1.08 * (0.5 * d + sign * math.sqrt(0.75) * perp)
    return Chem.MolToSmiles(Chem.RemoveHs(mol)), xyz


SPECIES = {
    "o2": lambda: ("[O][O]", [[0, 0, -0.605], [0, 0, 0.605]]),
    "dichlorocarbene": lambda: ("Cl[C]Cl", _bent(1.72, 109.0, centre_first=False)),
    "dibromocarbene": lambda: ("Br[C]Br", _bent(1.87, 112.0, centre_first=False)),
    "methylene_134": lambda: ("[CH2]", _bent(1.08, 134.0, centre_first=True)),
    "co_drawn": lambda: ("[C]=O", [[0, 0, 0], [0, 0, 1.128]]),
    "vinylidene": _vinylidene,
    "tmm": _tmm,
    "bromine_atom": lambda: ("[Br]", [[0, 0, 0]]),
    "oxygen_atom": lambda: ("[O]", [[0, 0, 0]]),
    "cucl2_linear": lambda: ("Cl[Cu]Cl", _complex("Cl[Cu]Cl", 2.05, LINE)),
    "cucl4_square": lambda: ("Cl[Cu-2](Cl)(Cl)Cl", _complex("Cl[Cu-2](Cl)(Cl)Cl", 2.25, SQUARE)),
    "cucl4_td": lambda: ("Cl[Cu-2](Cl)(Cl)Cl", _complex("Cl[Cu-2](Cl)(Cl)Cl", 2.25, TD)),
    "febr4_td": lambda: ("Br[Fe-](Br)(Br)Br", _complex("Br[Fe-](Br)(Br)Br", 2.33, TD)),
    "fe_oh6": lambda: ("O[Fe-3](O)(O)(O)(O)O", _complex("O[Fe-3](O)(O)(O)(O)O", 2.0, OH)),
    "fe_cl6": lambda: (
        "Cl[Fe-3](Cl)(Cl)(Cl)(Cl)Cl",
        _complex("Cl[Fe-3](Cl)(Cl)(Cl)(Cl)Cl", 2.4, OH),
    ),
    "fe_cn6": lambda: (
        "N#C[Fe-3](C#N)(C#N)(C#N)(C#N)C#N",
        _complex("N#C[Fe-3](C#N)(C#N)(C#N)(C#N)C#N", 1.95, OH, h_dist=1.16),
    ),
}


# ---- the AO operators


def test_ao_operator_follows_tblite_conventions():
    """A rotated molecule's overlap and density are the operator's images of the unrotated
    ones (s, p and d shells: S and Cl carry d)."""
    can = chem.unmapped_canonical(Chem.MolFromSmiles("CSC(=O)Cl"))
    mol = prepare_mol(can)
    embed(mol, n_conformers=1, seed=0)
    xyz = mol.GetConformer(0).GetPositions()
    R = np.linalg.qr(np.random.default_rng(3).normal(size=(3, 3)))[0]
    R *= np.sign(np.linalg.det(R))
    _, rec = _record(can, xyz, solvent=None)
    _, turned = _record(can, xyz @ R.T, solvent=None)
    T = st.ao_operator(rec.ao_atom, rec.ao_l, R, np.arange(rec.n_atoms))
    density = lambda r: r.C @ np.diag(r.occupations) @ r.C.T  # noqa: E731
    assert (rec.ao_l == 2).any()
    assert np.abs(T.T @ turned.S @ T - rec.S).max() < 1e-10
    assert np.abs(T @ density(rec) @ T.T - density(turned)).max() < 1e-5


def test_every_operation_of_an_octahedron_reads_characters():
    """FeCl6 3- (Oh, 48 operations; Fe's shells come d, s, p): the operators are valid and the
    frontier sets are irreducible, three t2g MOs below two e_g."""
    v = verdict("fe_cl6")
    assert v.point_group == "Oh" and not v.notes
    homo, lumo = v.frontier
    assert (homo["dim"], homo["label"], homo["irreducible"]) == (3, "Tg", True)
    assert (lumo["dim"], lumo["label"], lumo["irreducible"]) == (2, "Eg", True)


# ---- the rules


@pytest.mark.parametrize("name", ["cucl2_linear", "cucl4_square", "cucl4_td"])
def test_cu_ii_d9_has_one_unpaired_electron_in_any_geometry(name):
    v = verdict(name)
    assert v.unpaired == 1
    (cu,) = v.metals
    assert (cu["element"], cu["oxidation_state"], cu["d"]) == ("Cu", 2, 9)
    assert "by count" in cu["rule"]


def test_the_cu_polyhedra_are_read_from_the_geometry():
    assert verdict("cucl4_square").metals[0]["polyhedron"] == "square planar"
    assert verdict("cucl4_td").metals[0]["polyhedron"] == "tetrahedral"
    assert verdict("cucl4_square").metals[0]["jahn_teller"] is False  # not octahedral


def test_a_tetrahedral_d5_ferrate_is_high_spin():
    v = verdict("febr4_td")
    assert v.unpaired == 5 and v.point_group == "Td"
    assert any("Td high spin" in r for r in v.rule_chain)


def test_an_octahedral_d5_with_a_mid_series_ligand_is_undetermined():
    v = verdict("fe_oh6")
    assert v.unpaired is None
    assert v.metals[0]["field"] == "middle or mixed"
    assert v.rule_chain[-1] == "undetermined"


def test_the_octahedral_field_ends_decide_d5():
    assert verdict("fe_cl6").unpaired == 5  # halides: weak field, high spin
    assert verdict("fe_cn6").unpaired == 1  # C donors: strong field, low spin


def test_triplet_o2_by_hund():
    v = verdict("o2")
    assert v.unpaired == 2 and v.term == "3Sigma_g-"
    assert v.open_shell["label"] == "Pi_g" and v.open_shell["d"] == 2
    assert any("Hund" in r for r in v.rule_chain)


def test_hund_in_a_finite_group():
    """D3h trimethylenemethane: two electrons in e'' give the triplet, 3A2' (subscript not
    assigned)."""
    v = verdict("tmm")
    assert v.point_group == "D3h" and v.unpaired == 2
    assert v.open_shell["label"] == "E″" and v.term == "3A′"


@pytest.mark.parametrize("name,term", [("bromine_atom", "2P"), ("oxygen_atom", "3P")])
def test_atoms_follow_hunds_rules(name, term):
    v = verdict(name)
    assert v.term == term and v.unpaired == int(term[0]) - 1


@pytest.mark.parametrize("name", ["dichlorocarbene", "dibromocarbene"])
def test_dihalocarbenes_are_singlets_by_their_pi_donors(name):
    """Rule 4b (v2): the GFN2 gap is only a proxy, and dibromocarbene's reads a triplet."""
    v = verdict(name)
    assert v.unpaired == 0 and v.carbene is not None and v.carbene["centre"] == 1
    assert v.rule_chain == ["two pi-donor neighbours with lone pairs: singlet"]


def test_a_vinylidene_is_an_unsaturated_carbene():
    v = verdict("vinylidene")
    assert v.unpaired == 0 and "stang" in v.citations
    assert v.rule_chain[0].startswith("unsaturated carbene")


def test_gleiter_hoffmann_reaches_both_ends():
    """Methylene at its triplet angle has a small sigma-p_pi gap, CO a large one."""
    assert verdict("methylene_134").unpaired == 2
    assert verdict("co_drawn").unpaired == 0
    assert any("Gleiter-Hoffmann" in r for r in verdict("co_drawn").rule_chain)


@pytest.mark.parametrize(
    ("smiles", "line"),
    [
        ("Cl[C]Cl", "two pi-donor"),
        ("CN(C)[C]N(C)C", "two pi-donor"),  # a diaminocarbene
        ("[C]=C(C)C", "unsaturated"),
        ("[CH]Cl", None),  # one donor: Gleiter-Hoffmann decides
        ("C[N+](C)(C)[C][N+](C)(C)C", None),  # oniums hold no lone pair
        ("[C]=O", None),  # the neighbour is not a carbon
        ("C[C]C", None),
    ],
)
def test_the_structural_carbene_rules(smiles, line):
    mol = prepare_mol(chem.unmapped_canonical(Chem.MolFromSmiles(smiles)))
    centre = st.carbene_centres(mol)[0]
    got = st.structural_carbene(mol.GetAtomWithIdx(centre))
    assert (got is None) if line is None else got.startswith(line)


def test_every_verdict_has_a_rule_chain_and_its_citations():
    for name in SPECIES:
        v = verdict(name)
        assert v.rule_chain, name
        assert v.citations and set(v.citations) <= set(st.CITATIONS), name
        assert v.unpaired is not None or v.rule_chain[-1] == "undetermined", name
        json.dumps(v.as_dict())


# ---- the pure ligand-field rules and the polyhedra


@pytest.mark.parametrize(
    "z,d,poly,field,expected",
    [
        (26, 5, "octahedral", "weak", 5),
        (26, 5, "octahedral", "strong", 1),
        (26, 6, "octahedral", "strong", 0),
        (27, 7, "octahedral", "weak", 3),
        (26, 5, "octahedral", "middle or mixed", None),
        (44, 6, "octahedral", "middle or mixed", 0),  # Ru(II): 4d, low spin
        (28, 8, "octahedral", "middle or mixed", 2),
        (24, 3, "octahedral", "middle or mixed", 3),
        (28, 8, "square planar", "middle or mixed", 0),
        (27, 7, "square planar", "middle or mixed", None),
        (25, 5, "tetrahedral", "middle or mixed", 5),
        (28, 8, "tetrahedral", "strong", 2),
        (45, 6, "tetrahedral", "weak", None),  # Td beyond the first row
        (26, 5, "trigonal planar", "weak", None),
        (29, 9, "see-saw", "weak", 1),
        (30, 10, "CN2", "weak", 0),
        (22, 0, "octahedral", "weak", 0),
        (26, 11, "octahedral", "weak", None),
    ],
)
def test_centre_rule(z, d, poly, field, expected):
    unpaired, rule = st.centre_rule(z, d, poly, field)
    assert unpaired == expected and rule


def test_ideal_polyhedra_classify_as_themselves():
    class Rec:  # only coords_ang is read
        pass

    for cn, shapes in st.ideal_polyhedra().items():
        for name in shapes:
            vecs = _ideal_vectors(cn, name)
            rec = Rec()
            rec.coords_ang = np.r_[[[0.0, 0.0, 0.0]], 2.0 * vecs]
            assert st.polyhedron(rec, 0, list(range(1, cn + 1))) == (name, 0.0), name


def _ideal_vectors(cn: int, name: str) -> np.ndarray:
    """The angles alone define the polyhedra: rebuild each from its sorted angle list's source."""
    o = np.array(OH, dtype=float)
    t = np.array(TD, dtype=float) / math.sqrt(3)
    tri = np.array([[math.cos(a), math.sin(a), 0] for a in (0, 2 * math.pi / 3, 4 * math.pi / 3)])
    prism = np.r_[tri + [0, 0, math.sqrt(0.75)], tri - [0, 0, math.sqrt(0.75)]]
    prism /= np.linalg.norm(prism, axis=1, keepdims=True)
    return {
        "trigonal planar": tri,
        "trigonal pyramidal": t[:3],
        "T-shaped": o[[0, 1, 2]],
        "tetrahedral": t,
        "square planar": o[:4],
        "see-saw": np.r_[o[4:], tri[:2]],
        "trigonal bipyramidal": np.r_[o[4:], tri],
        "square pyramidal": o[:5],
        "octahedral": o,
        "trigonal prismatic": prism,
    }[name]


def test_ligand_field_bands():
    assert st.ligand_field([17, 17, 35]) == "weak"
    assert st.ligand_field([6, 15]) == "strong"
    assert st.ligand_field([17, 8]) == "middle or mixed"
    assert st.ligand_field([1]) == "middle or mixed"  # hydride: not in the cited series
    assert st.ligand_field([]) == "middle or mixed"


# ---- no species literal, the table, the workbench


def test_no_rule_names_a_species():
    """No SMILES or SMARTS literal in spin_theory.py, and no RDKit parser call: every rule reads
    an element, a count, a geometry or the MOs."""
    src = Path(st.__file__).read_text()
    assert not re.search(r"MolFrom|Smarts|SMARTS|smarts", src)
    pattern = re.compile(r"\[[A-Z][a-z]?[^\]\s]*\]|\b[A-Z][a-z]?[=#][A-Z\[]")
    literals = [
        n.value
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    ]
    assert [s for s in literals if pattern.search(s)] == []


def test_the_theory_table_carries_its_digest(tmp_path):
    entries = {verdict(n).smiles: verdict(n) for n in ("o2", "febr4_td")}
    path = st.write_theory_table(entries, tmp_path / "theory.json")
    table = st.load_theory_table(path)
    assert table["version"] == st.VERSION and set(table["entries"]) == set(entries)
    assert set(table["citations"]) == set(st.CITATIONS)
    table["entries"][verdict("o2").smiles]["unpaired"] = 0
    path.write_text(json.dumps(table))
    with pytest.raises(ValueError, match="digest"):
        st.load_theory_table(path)


def test_the_workbench_reads_the_engine_detector():
    from ksmech.serve.workbench import symmetry as sym

    assert sym.search is pg.search and sym.expected_order is pg.expected_order
    assert sym.DEFAULT_TOLERANCE == pg.DEFAULT_TOLERANCE == st.SYMMETRY_TOLERANCE_ANG
