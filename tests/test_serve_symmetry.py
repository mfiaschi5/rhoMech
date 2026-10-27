"""The workbench point-group finder (`ksmech.serve.workbench.symmetry`) and GET
/molecule/{id}/symmetry:
known molecules (RDKit ETKDG + MMFF, or hand-built ideal geometries) give their Schoenflies
group, order and elements; perturbed geometries show the tolerance (C1 below the noise, the
group above it, the looser-tolerance hint), the symmetrized structure is exactly symmetric; the
fixed molecule set; the endpoint with its cache, limits and reason codes."""

from __future__ import annotations

import itertools
import math
from collections import Counter

import numpy as np
import pytest

from ksmech.serve.workbench import symmetry as sym
from tests.fixtures.molecules import MOLECULES
from tests.test_serve_fixtures import no_xtb, synthetic_root  # noqa: F401 (fixtures)

pytestmark = pytest.mark.serve


# ---- geometries


def mmff(smiles: str, n_conf: int = 8, seed: int = 7) -> tuple[list[int], np.ndarray]:
    """The lowest-MMFF-energy conformer of `smiles` (explicit H)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    cids = list(AllChem.EmbedMultipleConfs(mol, n_conf, randomSeed=seed))
    res = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=5000)
    best = min(range(len(cids)), key=lambda k: res[k][1])
    z = [a.GetAtomicNum() for a in mol.GetAtoms()]
    return z, np.asarray(mol.GetConformer(cids[best]).GetPositions())


def ring(n: int, r: float, h: float, z: int, phase: float = 0.0) -> list[tuple[int, tuple]]:
    t = [2 * math.pi * k / n + phase for k in range(n)]
    return [(z, (r * math.cos(a), r * math.sin(a), h)) for a in t]


def split(atoms) -> tuple[list[int], np.ndarray]:
    return [a[0] for a in atoms], np.asarray([a[1] for a in atoms], dtype=float)


def ferrocene(staggered: bool):
    ph = math.pi / 5 if staggered else 0.0
    atoms = [(26, (0.0, 0.0, 0.0))]
    atoms += ring(5, 1.21, 1.66, 6) + ring(5, 2.28, 1.66, 1)
    atoms += ring(5, 1.21, -1.66, 6, ph) + ring(5, 2.28, -1.66, 1, ph)
    return split(atoms)


def ethane(staggered: bool):
    cc, ch = 0.765, 1.09
    up = ch * math.cos(math.radians(70.53))
    r = ch * math.sin(math.radians(70.53))
    atoms = [(6, (0.0, 0.0, cc)), (6, (0.0, 0.0, -cc))]
    atoms += ring(3, r, cc + up, 1) + ring(3, r, -cc - up, 1, math.pi / 3 if staggered else 0.0)
    return split(atoms)


def octahedral(centre: int, ligand: int, d: float):
    axes = np.eye(3)
    return split(
        [(centre, (0, 0, 0))]
        + [(ligand, tuple(s * d * axes[i])) for i in range(3) for s in (1, -1)]
    )


def pcl5():
    return split([(15, (0, 0, 0)), (17, (0, 0, 2.12)), (17, (0, 0, -2.12)), *ring(3, 2.02, 0, 17)])


def ptcl4():
    return split([(78, (0, 0, 0)), *ring(4, 2.31, 0, 17)])


def c60():
    """The truncated icosahedron: cyclic permutations of (0, ±1, ±3φ), (±1, ±(2+φ), ±2φ),
    (±φ, ±2, ±φ³), 0.7 Å per unit (C-C about 1.4 Å)."""
    phi = (1 + 5**0.5) / 2
    pts = set()
    for base in [(0, 1, 3 * phi), (1, 2 + phi, 2 * phi), (phi, 2, phi**3)]:
        for cyc in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
            for signs in itertools.product((1, -1), repeat=3):
                pts.add(tuple(round(base[c] * s, 9) for c, s in zip(cyc, signs, strict=True)))
    assert len(pts) == 60
    return [6] * 60, np.asarray(sorted(pts)) * 0.7


def meso_tartaric():
    """meso-tartaric acid with one half the inversion image of the other through the C2-C3
    midpoint (Ci by construction; an MMFF conformer is rarely exactly centrosymmetric)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdCIPLabeler

    mol = Chem.AddHs(Chem.MolFromSmiles("OC(=O)[C@@H](O)[C@@H](O)C(O)=O"))
    rdCIPLabeler.AssignCIPLabels(mol)
    centres = [a.GetIdx() for a in mol.GetAtoms() if a.HasProp("_CIPCode")]
    assert sorted(mol.GetAtomWithIdx(i).GetProp("_CIPCode") for i in centres) == ["R", "S"]
    AllChem.EmbedMolecule(mol, randomSeed=3)
    AllChem.MMFFOptimizeMolecule(mol)
    x = np.asarray(mol.GetConformer().GetPositions())
    c2, c3 = centres
    swap = next(
        m for m in mol.GetSubstructMatches(mol, uniquify=False, useChirality=False) if m[c2] == c3
    )
    dist = Chem.GetDistanceMatrix(mol)
    mid = (x[c2] + x[c3]) / 2
    for a in range(mol.GetNumAtoms()):
        if dist[a, c2] < dist[a, c3]:
            x[swap[a]] = 2 * mid - x[a]
    return [a.GetAtomicNum() for a in mol.GetAtoms()], x


# ---- known molecules

MMFF_CASES = [
    ("water", "O", "C2v", 4),
    ("ammonia", "N", "C3v", 6),
    ("methane", "C", "Td", 24),
    ("benzene", "c1ccccc1", "D6h", 24),
    ("ethylene", "C=C", "D2h", 8),
    ("allene", "C=C=C", "D2d", 8),
    ("CO2", "O=C=O", "D∞h", None),
    ("HCN", "C#N", "C∞v", None),
    ("CHFClBr", "FC(Cl)Br", "C1", 1),
    ("trans-1,2-dichloroethylene", "Cl/C=C/Cl", "C2h", 4),
    ("cis-1,2-dichloroethylene", "Cl/C=C\\Cl", "C2v", 4),
    ("cubane", "C12C3C4C1C5C2C3C45", "Oh", 48),
    ("adamantane", "C1C2CC3CC1CC(C2)C3", "Td", 24),
    ("cyclohexane chair", "C1CCCCC1", "D3d", 12),
    ("twistane", "C1CC2CC3CCC2CC13", "D2", 4),
    ("H2O2", "OO", "C2", 2),
]

IDEAL_CASES = [
    ("ethane staggered", lambda: ethane(True), "D3d", 12),
    ("ethane eclipsed", lambda: ethane(False), "D3h", 12),
    ("SF6", lambda: octahedral(16, 9, 1.56), "Oh", 48),
    ("ferrocene staggered", lambda: ferrocene(True), "D5d", 20),
    ("ferrocene eclipsed", lambda: ferrocene(False), "D5h", 20),
    ("C60", c60, "Ih", 120),
    ("PCl5", pcl5, "D3h", 12),
    ("[PtCl4]2-", ptcl4, "D4h", 16),
    ("meso-tartaric acid", meso_tartaric, "Ci", 2),
]


@pytest.mark.parametrize(
    ("name", "smiles", "group", "order"), MMFF_CASES, ids=[c[0] for c in MMFF_CASES]
)
def test_mmff_molecules(name, smiles, group, order):
    z, x = mmff(smiles)
    out = sym.point_group(z, x, 0.1)
    assert (out["point_group"], out["order"]) == (group, order), name
    assert out["max_deviation_ang"] < 0.1 and out["rms_ang"] <= out["max_deviation_ang"]
    assert len(out["symmetrized_xyz"]) == len(z) and out["looser"] is None


@pytest.mark.parametrize(
    ("name", "build", "group", "order"), IDEAL_CASES, ids=[c[0] for c in IDEAL_CASES]
)
def test_ideal_geometries(name, build, group, order):
    z, x = build()
    out = sym.point_group(z, x, 0.1)
    assert (out["point_group"], out["order"]) == (group, order), name
    assert out["max_deviation_ang"] < 1e-6


def labels(out: dict) -> Counter:
    return Counter(e["label"] for e in out["elements"])


def test_benzene_elements_and_classes():
    z, x = mmff("c1ccccc1")
    out = sym.point_group(z, x, 0.1)
    assert labels(out) == Counter(
        {"C6": 1, "C2′": 3, "C2″": 3, "S6": 1, "σh": 1, "σv": 3, "σd": 3, "i": 1}
    )
    c6 = next(e for e in out["elements"] if e["label"] == "C6")
    assert c6["principal"] and c6["kind"] == "C" and c6["order"] == 6 and c6["atoms"] == []
    sigma_h = next(e for e in out["elements"] if e["label"] == "σh")
    assert abs(abs(np.dot(sigma_h["vector"], c6["vector"])) - 1) < 1e-6
    assert len(sigma_h["atoms"]) == 12  # the molecular plane
    c2p = [e for e in out["elements"] if e["label"] == "C2′"]
    assert all(len(e["atoms"]) == 4 for e in c2p)  # through two C and two H
    assert sorted(len(c["atoms"]) for c in out["classes"]) == [6, 6]
    assert {c["element"] for c in out["classes"]} == {"C", "H"}
    assert out["top"] == "symmetric"


def test_water_and_methane_elements():
    z, x = mmff("O")
    water = sym.point_group(z, x, 0.1)
    assert labels(water) == Counter({"C2": 1, "σv": 1, "σv′": 1})
    plane = next(e for e in water["elements"] if e["label"] == "σv")
    assert sorted(plane["atoms"]) == [0, 1, 2]  # σv holds the molecule, σv′ only O
    assert sorted(len(c["atoms"]) for c in water["classes"]) == [1, 2]
    z, x = mmff("C")
    methane = sym.point_group(z, x, 0.1)
    assert labels(methane) == Counter({"C3": 4, "C2": 3, "S4": 3, "σd": 6})
    assert methane["top"] == "spherical"


def test_cubic_and_linear_elements():
    z, x = octahedral(16, 9, 1.56)
    sf6 = sym.point_group(z, x, 0.1)
    assert labels(sf6) == Counter(
        {"C4": 3, "C3": 4, "C2": 6, "S4": 3, "S6": 4, "σh": 3, "σd": 6, "i": 1}
    )
    z, x = mmff("O=C=O")
    co2 = sym.point_group(z, x, 0.1)
    assert co2["top"] == "linear" and labels(co2) == Counter(
        {"C∞": 1, "∞σv": 1, "∞C2′": 1, "σh": 1, "i": 1}
    )
    assert sorted(len(c["atoms"]) for c in co2["classes"]) == [1, 2]
    z, x = c60()
    ih = sym.point_group(z, x, 0.1, hint=False)
    want = {"C5": 6, "C3": 10, "C2": 15, "S10": 6, "S6": 10, "σ": 15, "i": 1}
    assert labels(ih) == Counter(want) and len(ih["classes"]) == 1


def test_single_atom():
    out = sym.point_group([18], [[1.0, 2.0, 3.0]], 0.1)
    assert out["point_group"] == "Kh" and out["order"] is None and out["top"] == "atom"
    assert out["centre"] == [1.0, 2.0, 3.0]


# ---- tolerance behaviour


def perturbed(z, x, sigma: float, seed: int = 0):
    return z, x + np.random.default_rng(seed).normal(0.0, sigma, x.shape)


def test_perturbed_methane_tolerance():
    z, x = perturbed(*mmff("C"), 0.02)
    tight = sym.point_group(z, x, 0.01)
    assert tight["point_group"] == "C1"
    assert tight["looser"] is not None and tight["looser"]["point_group"] == "Td"
    assert tight["looser"]["tolerance"] <= 0.15
    loose = sym.point_group(z, x, 0.15)
    assert loose["point_group"] == "Td" and loose["looser"] is None
    assert 0.005 < loose["rms_ang"] < 0.05 and loose["max_deviation_ang"] <= 0.15
    # the symmetrized structure is Td at any tolerance
    again = sym.point_group(z, np.asarray(loose["symmetrized_xyz"]), 0.01)
    assert again["point_group"] == "Td" and again["max_deviation_ang"] < 1e-6


def test_distorted_water_hint():
    """One O-H stretched by 0.15 Å: Cs at 0.1 Å, C2v at a looser tolerance."""
    z, x = mmff("O")
    h = 1 if z[1] == 1 else 2
    x = x.copy()
    x[h] += 0.15 * (x[h] - x[0]) / np.linalg.norm(x[h] - x[0])
    out = sym.point_group(z, x, 0.1)
    assert out["point_group"] == "Cs" and labels(out) == Counter({"σ": 1})
    assert out["looser"] == {"point_group": "C2v", "order": 4, "tolerance": pytest.approx(0.15)}
    assert sym.point_group(z, x, 0.3)["point_group"] == "C2v"


@pytest.mark.parametrize(
    ("build", "group"), [(lambda: ferrocene(True), "D5d"), (c60, "Ih"), (ptcl4, "D4h")]
)
def test_perturbed_ideal(build, group):
    """Noise of 0.01 Å per coordinate: an atom pair then differs by up to about 0.06 Å."""
    z, x = perturbed(*build(), 0.01, seed=1)
    assert sym.point_group(z, x, 0.1, hint=False)["point_group"] == group
    tight = sym.point_group(z, x, 0.01)  # a lower group (C1, or Cs: 5 atoms near a plane)
    assert tight["order"] < sym.expected_order(group) and tight["looser"]["point_group"] == group


def test_bent_almost_linear():
    """HCN bent by 0.05 Å at H: linear at 0.1 Å, Cs at 0.02 Å."""
    z, x = split([(1, (0, 0, -1.07)), (6, (0, 0, 0)), (7, (0, 0, 1.16))])
    x[0, 0] = 0.05
    assert sym.point_group(z, x, 0.1)["point_group"] == "C∞v"
    tight = sym.point_group(z, x, 0.02)
    assert tight["point_group"] == "Cs" and tight["looser"]["point_group"] == "C∞v"


def test_fixed_molecule_set():
    """The repo's fixed set with the app's own embedding (prepare_mol + ETKDG/MMFF)."""
    from ksmech.qm.embed import embed, prepare_mol

    want = {
        "ethane": {"D3d"},
        "ethanol": {"C1", "Cs"},  # gauche or anti, whichever conformer MMFF ranks first
        "acetone": {"C2v"},
        "acetaldehyde": {"Cs"},
        "methyl_acetate": {"Cs"},
        "chloromethane": {"C3v"},
        "methylamine": {"Cs"},
        "acetonitrile": {"C3v"},
        "benzene": {"D6h"},
        "acrolein": {"Cs"},
    }
    for name, smiles in MOLECULES.items():
        mol = prepare_mol(smiles)
        embed(mol, 1, seed=0)
        z = [a.GetAtomicNum() for a in mol.GetAtoms()]
        out = sym.point_group(z, mol.GetConformer(0).GetPositions(), 0.1, hint=False)
        assert out["point_group"] in want[name], (name, out["point_group"])


@pytest.mark.parametrize(
    ("z", "xyz", "tol", "code"),
    [
        ([6], [[0, 0, 0]], 0.001, "REQUEST_INVALID"),
        ([6], [[0, 0, 0]], 0.6, "REQUEST_INVALID"),
        ([], np.zeros((0, 3)), 0.1, "REQUEST_INVALID"),
        ([6, 1], [[0, 0, 0]], 0.1, "REQUEST_INVALID"),
        ([6], [[np.nan, 0, 0]], 0.1, "REQUEST_INVALID"),
        ([1] * (sym.MAX_ATOMS + 1), np.zeros((sym.MAX_ATOMS + 1, 3)), 0.1, "LIMIT_EXCEEDED"),
    ],
)
def test_input_errors(z, xyz, tol, code):
    with pytest.raises(sym.SymmetryError) as exc:
        sym.point_group(z, xyz, tol)
    assert exc.value.code == code


def test_expected_order_table():
    table = {"C1": 1, "Cs": 2, "Ci": 2, "C3v": 6, "C2h": 4, "S4": 4, "S6": 6, "D3": 6}
    table |= {"D6h": 24, "D5d": 20, "Td": 24, "Oh": 48, "Ih": 120}
    assert {k: sym.expected_order(k) for k in table} == table


# ---- the endpoint


@pytest.fixture(scope="module")
def app(tmp_path_factory, synthetic_root):  # noqa: F811
    from ksmech.serve.core.settings import load_settings
    from tests.test_serve_fixtures_api import api_client, copy_synthetic_root

    root = copy_synthetic_root(synthetic_root, tmp_path_factory.mktemp("sym") / "root")
    settings = load_settings({}, data_root=root, chemistry="synthetic")
    with api_client(settings) as client:
        yield client


def test_symmetry_endpoint(app):
    from ksmech.serve.core import schemas as S

    created = app.post("/api/molecule", json={"smiles": "C"})
    assert created.status_code == 200, created.text
    mid = created.json()["molecule_id"]
    r = app.get(f"/api/molecule/{mid}/symmetry")
    assert r.status_code == 200, r.text
    out = S.PointGroupOut.model_validate(r.json())
    assert out.point_group == "Td" and out.order == 24 and out.tolerance == 0.1
    assert out.molecule_id == mid and out.key == "C" and out.n_atoms == 5 and out.qm_digest
    assert len(out.symmetrized_xyz) == 5 and len(out.classes) == 2
    geometry = app.get(f"/api/molecule/{mid}/geometry").json()
    assert out.centre == pytest.approx([geometry["atoms"][0][k] for k in "xyz"], abs=1e-6)
    tight = app.get(f"/api/molecule/{mid}/symmetry", params={"tolerance": 0.01})
    assert tight.status_code == 200 and tight.json()["tolerance"] == 0.01
    assert app.get(f"/api/molecule/{mid}/symmetry").json() == r.json()  # memoized


def test_symmetry_endpoint_errors(app):
    from tests.test_serve_api import error_of

    mid = app.post("/api/molecule", json={"smiles": "O"}).json()["molecule_id"]
    url = f"/api/molecule/{mid}/symmetry"
    error_of(app.get(url, params={"tolerance": 0.001}), 422, "REQUEST_INVALID")
    error_of(app.get(url, params={"tolerance": 0.9}), 422, "REQUEST_INVALID")
    error_of(app.get(url, params={"tolerance": "x"}), 422, "REQUEST_INVALID")
    error_of(app.get(f"/api/molecule/m{'0' * 20}/symmetry"), 404, "MOLECULE_NOT_FOUND")


def test_symmetry_service_limit(app, monkeypatch):
    from tests.test_serve_api import error_of

    mid = app.post("/api/molecule", json={"smiles": "CC"}).json()["molecule_id"]
    monkeypatch.setattr(sym, "MAX_ATOMS", 4)
    body = error_of(
        app.get(f"/api/molecule/{mid}/symmetry", params={"tolerance": 0.2}), 422, "LIMIT_EXCEEDED"
    )
    assert body["detail"]["n_atoms"] == 8
