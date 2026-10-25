"""Metal-complex start geometries (serve.metal_geometry) and the metal path's wiring
(serve.metal_qm, serve.featurize). No xTB: every test runs inside `xtb_forbidden()`."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from ksmech.data.chem import unmapped_canonical
from ksmech.qm.embed import prepare_mol
from ksmech.qm.xtb_calc import QMConfig
from ksmech.serve import metal_geometry as mg
from tests.test_serve_fixtures import xtb_forbidden

STD = QMConfig().method


@pytest.fixture(autouse=True)
def no_xtb():
    with xtb_forbidden():
        yield


def _mol(smiles: str) -> Chem.Mol:
    return prepare_mol(unmapped_canonical(Chem.MolFromSmiles(smiles)))


def _built(smiles: str):
    mol = _mol(smiles)
    xyz, plans = mg.build_start_geometry(mol)
    return mol, xyz, plans[0]


def _trans_symbols(mol, plan, xyz) -> list[str]:
    vecs = mg.site_vectors(mol, plan.metal, plan.sites, xyz)
    sym = [mol.GetAtomWithIdx(s.atoms[0]).GetSymbol() for s in plan.sites]
    return sorted("-".join(sorted((sym[i], sym[j]))) for i, j in mg.trans_pairs(vecs))


# ---- ideal polyhedra and the builder, CN 2-6


@pytest.mark.parametrize(
    ("name", "angles"),
    [
        ("linear", {180}),
        ("trigonal", {120}),
        ("tetrahedral", {109}),
        ("square_planar", {90, 180}),
        ("tbp", {90, 120, 180}),
        ("square_pyramidal", {100, 88, 160}),
        ("octahedral", {90, 180}),
    ],
)
def test_ideal_polyhedra_angles(name, angles):
    v = mg.POLYHEDRA[name]
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0)
    got = {int(round(a)) for a in mg._angles(v)[np.triu_indices(len(v), 1)]}
    assert got == angles


CARBONYL = "<-[C-]#[O+]"


@pytest.mark.parametrize(
    ("smiles", "poly"),
    [
        ("N->[Cu+]<-N", "linear"),
        ("[Cu+](<-P(C)(C)C)(<-P(C)(C)C)<-P(C)(C)C", "trigonal"),
        (f"[Ni]({CARBONYL})({CARBONYL})({CARBONYL}){CARBONYL}", "tetrahedral"),
        ("Cl[Pt-2](Cl)(Cl)Cl", "square_planar"),
        (f"[Fe]({CARBONYL})({CARBONYL})({CARBONYL})({CARBONYL}){CARBONYL}", "tbp"),
        ("Cl[Ru](Cl)(=C)(<-P(C)(C)C)<-P(C)(C)C", "square_pyramidal"),
        (
            f"[Cr]({CARBONYL})({CARBONYL})({CARBONYL})({CARBONYL})({CARBONYL}){CARBONYL}",
            "octahedral",
        ),
        ("N->[Co+3](<-N)(<-N)(<-N)(<-N)<-N", "octahedral"),
    ],
)
def test_builder_places_the_polyhedron(smiles, poly):
    mol, xyz, plan = _built(smiles)
    assert plan.polyhedron == poly
    vecs = mg.site_vectors(mol, plan.metal, plan.sites, xyz)
    best, rms = mg.fit_polyhedra(vecs)[0]
    assert best == poly and rms < 1.0
    for site in plan.sites:
        d = np.linalg.norm(xyz[site.atoms[0]] - xyz[plan.metal])
        assert d == pytest.approx(mg.ml_distance(mol, plan.metal, site.atoms[0]), abs=1e-6)
    assert mg.closest_contact(mol, xyz) > 1.5
    # every ligand bond keeps a sane length (ETKDG around the fixed donors)
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if not mg.is_metal_bond(mol, i, j):
            assert 0.9 < np.linalg.norm(xyz[i] - xyz[j]) < 2.0


def test_monodentate_lone_pairs_point_at_the_metal():
    mol, xyz, plan = _built(
        f"[Cr]({CARBONYL})({CARBONYL})({CARBONYL})({CARBONYL})({CARBONYL}){CARBONYL}"
    )
    for site in plan.sites:
        c = site.atoms[0]
        o = next(n.GetIdx() for n in mol.GetAtomWithIdx(c).GetNeighbors() if n.GetSymbol() == "O")
        u, v = xyz[c] - xyz[plan.metal], xyz[o] - xyz[c]
        assert u @ v / np.linalg.norm(u) / np.linalg.norm(v) > 0.99  # M-C-O linear


@pytest.mark.parametrize(
    "smiles",
    [
        "CC1=CC(C)=[O]->[Pd+2]2(<-[O]=C(C)C=C(C)[O-]->2)<-[O-]1",  # Pd(acac)2
        "C1C[NH2]->[Co+3]23(<-[NH2]1)(<-[NH2]CC[NH2]->2)<-[NH2]CC[NH2]->3",  # [Co(en)3]3+
    ],
)
def test_chelates_build_without_clashes(smiles):
    mol, xyz, plan = _built(smiles)
    assert mg.contact_ratio(mol, xyz) >= mg._GOOD_CONTACT
    for b in mol.GetBonds():  # the chelate rings keep their ETKDG bond lengths
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if not mg.is_metal_bond(mol, i, j):
            assert 0.9 < np.linalg.norm(xyz[i] - xyz[j]) < 1.6


def test_ferrocene_sandwich():
    smiles = (
        "[cH]12->[Fe+2]3456789(<-[cH]1[cH]->3[cH-]->4[cH]->52)<-[cH]1[cH]->6[cH]->7[cH-]->8[cH]->91"
    )
    mol, xyz, plan = _built(smiles)
    assert [s.hapto for s in plan.sites] == [5, 5] and plan.polyhedron == "linear"
    assert plan.d_count == 6
    for site in plan.sites:  # ring order: consecutive atoms bonded
        ring = site.atoms + site.atoms[:1]
        assert all(mol.GetBondBetweenAtoms(a, b) for a, b in itertools.pairwise(ring))
        fe_c = [np.linalg.norm(xyz[a] - xyz[plan.metal]) for a in site.atoms]
        assert np.allclose(fe_c, fe_c[0], atol=1e-6) and 1.9 < fe_c[0] < 2.2
    h = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 1]
    assert min(np.linalg.norm(xyz[h] - xyz[plan.metal], axis=1)) > 2.6  # no C-H bent to Fe


def test_zeise_alkene_stands_across_the_square_plane():
    mol, xyz, plan = _built("[Cl-]->[Pt+2]1(<-[Cl-])(<-[Cl-])<-[CH2]=[CH2]->1")
    eta = next(s for s in plan.sites if s.hapto == 2)
    cl = [s.atoms[0] for s in plan.sites if s.hapto == 1]
    normal = np.cross(xyz[cl[0]] - xyz[plan.metal], xyz[cl[1]] - xyz[plan.metal])
    cc = xyz[eta.atoms[0]] - xyz[eta.atoms[1]]
    assert abs(normal @ cc) / np.linalg.norm(normal) / np.linalg.norm(cc) > 0.95


# ---- isomers


@pytest.mark.parametrize(
    ("smiles", "trans"),
    [
        ("Cl[Pt@SP1](Cl)(<-N)<-N", ["Cl-N", "Cl-N"]),  # cisplatin
        ("Cl[Pt@SP2](Cl)(<-N)<-N", ["Cl-Cl", "N-N"]),  # transplatin
        ("Cl[Pt@SP3](Cl)(<-N)<-N", ["Cl-N", "Cl-N"]),
        ("Cl[Co@OH1](Cl)(Cl)(<-N)(<-N)<-N", ["Cl-N", "Cl-N", "Cl-N"]),  # fac
        ("Cl[Co@OH25](Cl)(<-N)(Cl)(<-N)<-N", ["Cl-Cl", "Cl-N", "N-N"]),  # mer
        ("Cl[Co@OH1+](Cl)(<-N)(<-N)(<-N)<-N", ["Cl-N", "Cl-N", "N-N"]),  # cis
        ("Cl[Co@OH25+](Cl)(<-N)(<-N)(<-N)<-N", ["Cl-Cl", "N-N", "N-N"]),  # trans
    ],
)
def test_isomer_from_smiles_stereo(smiles, trans):
    mol, xyz, plan = _built(smiles)
    assert plan.isomer_source == "smiles-stereo"
    assert _trans_symbols(mol, plan, xyz) == trans


@pytest.mark.parametrize(
    ("smiles", "trans"),
    [
        ("Cl[Pt](Cl)(<-N)<-N", ["Cl-Cl", "N-N"]),  # trans-MA2B2
        ("Cl[Co](Cl)(Cl)(<-N)(<-N)<-N", ["Cl-Cl", "Cl-N", "N-N"]),  # mer-MA3B3
        ("Cl[Co+](Cl)(<-N)(<-N)(<-N)<-N", ["Cl-Cl", "N-N", "N-N"]),  # trans-MA4B2
        ("Cl[Ru](Cl)(=C)(<-P(C)(C)C)<-P(C)(C)C", ["Cl-Cl", "P-P"]),  # Grubbs-type, CH2 apical
    ],
)
def test_default_isomer_puts_like_ligands_trans(smiles, trans):
    mol, xyz, plan = _built(smiles)
    assert plan.isomer_source == "default"
    assert _trans_symbols(mol, plan, xyz) == trans


def test_default_isomer_keeps_chelates_cis():
    mol, xyz, plan = _built("C1C[NH2]->[Co+3]23(<-[NH2]1)(<-[NH2]CC[NH2]->2)<-[NH2]CC[NH2]->3")
    frag = mg._fragment_of(mol)
    vecs = mg.site_vectors(mol, plan.metal, plan.sites, xyz)
    for i, j in mg.trans_pairs(vecs):
        assert frag[plan.sites[i].atoms[0]] != frag[plan.sites[j].atoms[0]]


@pytest.mark.parametrize(
    ("tag", "table", "n"),
    [("SP", mg._SP_TABLE, 4), ("TB", mg._TB_TABLE, 5), ("OH", mg._OH_TABLE, 6)],
)
def test_stereo_tables_match_rdkit_embedding(tag, table, n):
    """Each table entry is the best vertex assignment (L-M-L angles) of RDKit's own embedding
    of a proxy complex with distinct ligands, so it carries RDKit's cis/trans meaning."""
    ligands = ["[F]", "[Cl]", "[Br]", "[I]", "[O]", "[S]"][:n]
    poly = {"SP": "square_planar", "TB": "tbp", "OH": "octahedral"}[tag]
    ideal = mg._angles(mg.POLYHEDRA[poly])
    for perm, slots in table.items():
        smi = f"[Co@{tag}{perm}]" + "".join(f"({x})" for x in ligands[:-1]) + ligands[-1]
        mol = Chem.MolFromSmiles(smi)
        params = AllChem.ETKDGv3()
        params.randomSeed = 0
        assert AllChem.EmbedMolecule(mol, params) == 0
        x = mol.GetConformer().GetPositions()
        got = mg._angles((x[1:] - x[0]) / np.linalg.norm(x[1:] - x[0], axis=1)[:, None])
        best = min(_angle_rms(got, ideal, p) for p in itertools.permutations(range(n)))
        assert _angle_rms(got, ideal, list(slots)) <= best + 1e-6, smi


def _angle_rms(got: np.ndarray, ideal: np.ndarray, perm) -> float:
    iu = np.triu_indices(len(got), 1)
    return float(np.sqrt(np.mean((got[iu] - ideal[np.ix_(perm, perm)][iu]) ** 2)))


# ---- donors, oxidation state, polyhedron rules, spin


def test_sites_on_dative_smiles():
    mol = _mol("N->[Co+3](<-N)(<-N)(<-N)(<-N)<-N")
    (m,) = mg.metal_atoms(mol)
    sites = mg.find_sites(mol, m)
    assert len(sites) == 6 and all(s.hapto == 1 for s in sites)
    assert {mol.GetAtomWithIdx(s.atoms[0]).GetSymbol() for s in sites} == {"N"}
    assert mg.oxidation_state(mol, m) == 3 and mg.d_count(mol, m) == 6


@pytest.mark.parametrize(
    ("smiles", "ox", "d"),
    [
        ("Cl[Pt-2](Cl)(Cl)Cl", 2, 8),  # covalent X ligands
        ("Cl[Pt-2](Cl)([NH3+])[NH3+]", 2, 8),  # charge-separated ammines
        ("Cl[Pt](Cl)(<-N)<-N", 2, 8),  # dative ammines
        ("[Fe+2](<-[C-]#N)(<-[C-]#N)(<-[C-]#N)(<-[C-]#N)(<-[C-]#N)<-[C-]#N", 2, 6),
        (f"[Ni]({CARBONYL})({CARBONYL})({CARBONYL}){CARBONYL}", 0, 10),
        ("Cl[Ru](Cl)(=C)(<-P(C)(C)C)<-P(C)(C)C", 2, 6),  # the carbene counts as L
        ("Cl[Ti](Cl)(Cl)Cl", 4, 0),
        ("C=C[CH2][Cu]([CH2]C=C)[Mg][Br]", 1, 10),  # a PMechDB cuprate: Cu-Mg counts -1
        ("CC(=[O+][Ti-]([Cl])([Cl])([Cl])[Cl])c1ccccc1", 4, 0),  # PMechDB TiCl4 adduct
    ],
)
def test_oxidation_state_and_d_count(smiles, ox, d):
    mol = _mol(smiles)
    (m,) = mg.metal_atoms(mol)
    assert (mg.oxidation_state(mol, m), mg.d_count(mol, m)) == (ox, d)


@pytest.mark.parametrize(
    ("smiles", "poly"),
    [
        ("Cl[Pd-2](Cl)(Cl)Cl", "square_planar"),
        ("[Ni+2](<-[C-]#N)(<-[C-]#N)(<-[C-]#N)<-[C-]#N", "square_planar"),
        ("Cl[Ni-2](Cl)(Cl)Cl", "tetrahedral"),  # weak field
        ("Cl[Zn-2](Cl)(Cl)Cl", "tetrahedral"),
        ("Cl[Rh](<-P(C)(C)C)(<-P(C)(C)C)<-P(C)(C)C", "square_planar"),
        ("Cl[Pt@SP1](Cl)(<-N)<-N", "square_planar"),
    ],
)
def test_four_coordinate_rules(smiles, poly):
    mol = _mol(smiles)
    (m,) = mg.metal_atoms(mol)
    assert mg.choose_polyhedron(mol, m, mg.find_sites(mol, m)) == poly


def test_polyhedron_override():
    mol = _mol("Cl[Zn-2](Cl)(Cl)Cl")
    (m,) = mg.metal_atoms(mol)
    xyz, plans = mg.build_start_geometry(mol, overrides={m: "square_planar"})
    assert plans[0].polyhedron == "square_planar"
    with pytest.raises(ValueError):
        mg.choose_polyhedron(mol, m, mg.find_sites(mol, m), "octahedral")


@pytest.mark.parametrize(
    ("smiles", "open_shell"),
    [
        ("[Fe+2](<-O)(<-O)(<-O)(<-O)(<-O)<-O", True),  # high-spin d6
        ("Cl[Ni-2](Cl)(Cl)Cl", True),  # tetrahedral d8
        ("[Ni+2](<-O)(<-O)(<-O)(<-O)(<-O)<-O", True),  # octahedral d8
        ("N->[Co+3](<-N)(<-N)(<-N)(<-N)<-N", False),  # low-spin Co(III)
        ("[Fe+2](<-[C-]#N)(<-[C-]#N)(<-[C-]#N)(<-[C-]#N)(<-[C-]#N)<-[C-]#N", False),
        ("Cl[Zn-2](Cl)(Cl)Cl", False),
        ("Cl[Pt-2](Cl)(Cl)Cl", False),
    ],
)
def test_open_shell_rule(smiles, open_shell):
    mol = _mol(smiles)
    assert bool(mg.preflight_problems(mol)) is open_shell


def test_no_template_cases():
    with pytest.raises(mg.MetalTemplateError):
        mg.plan_complex(_mol("[Fe+2].c1cc[cH-]c1.c1cc[cH-]c1"))  # three fragments
    with pytest.raises(mg.MetalTemplateError):
        mg.plan_complex(_mol("Cl[Pd]1(Cl)Cl[Pd](Cl)(Cl)Cl1"))  # dinuclear, bridging Cl
    assert mg.preflight_problems(_mol("CCO")) == []


# ---- method tag: metal keys only


def test_method_tag_only_on_metal_keys():
    from ksmech.serve.metal_qm import MetalQMConfig
    from ksmech.serve.synthetic import SYNTHETIC_METHOD

    metal = unmapped_canonical(Chem.MolFromSmiles("Cl[Pt-2](Cl)(Cl)Cl"))
    assert mg.method_for(metal, STD) == STD + mg.METAL_GEOM_TAG != STD
    assert mg.method_for(metal, mg.method_for(metal, STD)) == STD + mg.METAL_GEOM_TAG
    for key in ("CCO", "CC(=O)C", "[Na+]", "C[Mg]Cl"):  # main-group metals stay standard
        assert mg.method_for(key, STD) == STD
    assert mg.method_for(metal, SYNTHETIC_METHOD) == SYNTHETIC_METHOD
    cfg = MetalQMConfig()
    assert cfg.method == STD + mg.METAL_GEOM_TAG
    from ksmech.qm.records import cache_path, record_config

    assert record_config(cfg.method, 0)["solvation"] == record_config(STD, 0)["solvation"]
    assert cache_path("x", metal, 0, cfg.method) != cache_path("x", metal, 0, STD)


def _settings(tmp_path):
    from ksmech.serve.settings import load_settings

    return load_settings({}, data_root=tmp_path, chemistry="cached")


def test_featurizer_paths_follow_the_key(tmp_path):
    from ksmech.orbitals.build import tokens_path
    from ksmech.qm.records import cache_path
    from ksmech.serve.featurize import QMFeaturizer, qm_npz_path

    s = _settings(tmp_path)
    feat = QMFeaturizer(s, allow_compute=False)
    metal = unmapped_canonical(Chem.MolFromSmiles("Cl[Pt-2](Cl)(Cl)Cl"))
    tagged = STD + mg.METAL_GEOM_TAG
    assert feat.method == STD and feat.method_for(metal) == tagged
    assert feat.method_for("CCO") == STD
    s.serve_qm.mkdir(parents=True)
    cache_path(s.serve_qm, metal, 0, STD).write_bytes(b"x")  # a standard record: not used
    assert qm_npz_path(s, metal, STD) is None
    cache_path(s.serve_qm, metal, 0, tagged).write_bytes(b"x")
    assert qm_npz_path(s, metal, STD) == cache_path(s.serve_qm, metal, 0, tagged)
    cache_path(s.serve_qm, "CCO", 0, STD).write_bytes(b"x")
    assert qm_npz_path(s, "CCO", STD) == cache_path(s.serve_qm, "CCO", 0, STD)
    cands = feat.loader()._candidates(metal)
    assert cands == [tokens_path(d, metal, 0, tagged) for d in (s.shared_tokens, s.serve_tokens)]
    assert feat.loader()._candidates("CCO") == [
        tokens_path(d, "CCO", 0, STD) for d in (s.shared_tokens, s.serve_tokens)
    ]


def test_run_qm_dispatch(tmp_path, monkeypatch):
    from ksmech.qm import batch
    from ksmech.serve import metal_qm
    from ksmech.serve.featurize import QMFeaturizer

    calls = []
    monkeypatch.setattr(
        metal_qm, "process_metal_molecule", lambda key, cfg: calls.append(("metal", key, cfg)) or {}
    )
    monkeypatch.setattr(batch, "process_molecule", lambda task: calls.append(("std", task)) or [{}])
    feat = QMFeaturizer(_settings(tmp_path), allow_compute=True)
    metal = unmapped_canonical(Chem.MolFromSmiles("Cl[Pt-2](Cl)(Cl)Cl"))
    feat._run_qm(metal)
    feat._run_qm("CCO")
    (kind, key, cfg), (kind2, task) = calls
    assert (kind, key, kind2, task.smiles) == ("metal", metal, "std", "CCO")
    assert isinstance(cfg, metal_qm.MetalQMConfig) and cfg.cache_dir == feat.settings.serve_qm
    assert type(task.config) is QMConfig


def test_preflight_rejects_open_shell_metal(tmp_path):
    from ksmech.serve.featurize import QMFeaturizer

    feat = QMFeaturizer(_settings(tmp_path), allow_compute=False)
    hs = unmapped_canonical(Chem.MolFromSmiles("[Fe+2](<-O)(<-O)(<-O)(<-O)(<-O)<-O"))
    codes = [c for c, _ in feat.preflight(hs).errors]
    assert codes == ["RADICAL_NOT_SUPPORTED"]
    ok = unmapped_canonical(Chem.MolFromSmiles("N->[Co+3](<-N)(<-N)(<-N)(<-N)<-N"))
    assert feat.preflight(ok).errors == []
    assert feat.preflight("CCO").errors == []


def test_metal_path_is_guarded_and_never_raises(tmp_path):
    """Inside `xtb_forbidden` the inherited `GFN2Calculator.run` raises: the metal path
    reports it as a failed row instead of computing."""
    from ksmech.serve import metal_qm

    key = unmapped_canonical(Chem.MolFromSmiles("Cl[Pt-2](Cl)(Cl)Cl"))
    row = metal_qm.process_metal_molecule(key, metal_qm.MetalQMConfig(cache_dir=tmp_path))
    assert row["status"] == "failed" and row["reason"] == "UNEXPECTED"
    assert "forbidden" in row["detail"]
    assert not list(tmp_path.iterdir())


# ---- connectivity tolerances and walls


def _ptcl4_stretched(factor: float, which: int = 1):
    mol = _mol("Cl[Pt-2](Cl)(Cl)Cl")
    xyz, plans = mg.build_start_geometry(mol)
    m = plans[0].metal
    r = Chem.GetPeriodicTable().GetRcovalent
    d0 = r(78) + r(17)
    cl = [s.atoms[0] for s in plans[0].sites]
    for c in cl:
        u = (xyz[c] - xyz[m]) / np.linalg.norm(xyz[c] - xyz[m])
        xyz[c] = xyz[m] + u * d0
    u = (xyz[cl[which]] - xyz[m]) / np.linalg.norm(xyz[cl[which]] - xyz[m])
    xyz[cl[which]] = xyz[m] + u * d0 * factor
    return mol, xyz


def test_metal_connectivity_tolerances():
    from ksmech.qm.records import QMError
    from ksmech.serve.metal_qm import broken_bonds, check_metal_connectivity

    mol, xyz = _ptcl4_stretched(1.25)
    assert broken_bonds(mol, xyz, 1.3, mg.ML_COV_FACTOR) == []
    mol, xyz = _ptcl4_stretched(1.38)  # the Pt-Cl dissociation seen in the diagnosis
    assert len(broken_bonds(mol, xyz, 1.3, mg.ML_COV_FACTOR)) == 1
    with pytest.raises(QMError, match="M-L"):
        check_metal_connectivity(mol, xyz, 1.3, mg.ML_COV_FACTOR)
    assert broken_bonds(mol, xyz, 1.3, 1.5) == []  # the M-L limit is its own knob
    assert broken_bonds(mol, xyz, 1.5, 1.3) != []  # ... independent of the organic one


def test_walls_only_when_a_bond_breaks(monkeypatch):
    from ksmech.serve import metal_qm

    mol, ok = _ptcl4_stretched(1.0)
    _, broken = _ptcl4_stretched(1.5)
    z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()])
    bohr = metal_qm.BOHR_ANG
    walls_seen = []

    def fake_walls(*args, **kwargs):
        walls_seen.append(args[7])
        return ok / bohr

    monkeypatch.setattr(metal_qm, "optimize_tblite", fake_walls)
    calc = metal_qm.MetalGFN2Calculator(metal_qm.MetalQMConfig())
    monkeypatch.setattr(calc, "_optimize", lambda *a: ok / bohr)
    assert calc._relax(mol, z, ok / bohr, -2, 0)[1] is False and walls_seen == []
    monkeypatch.setattr(calc, "_optimize", lambda *a: broken / bohr)
    xyz, restrained = calc._relax(mol, z, ok / bohr, -2, 0)
    assert restrained is True and len(walls_seen) == 1
    walls = walls_seen[0]
    r = Chem.GetPeriodicTable().GetRcovalent
    assert np.allclose(walls.d0_bohr * bohr, mg.ML_WALL_FACTOR * (r(78) + r(17)))


def test_check_geometry_flags_polyhedron_and_isomer():
    mol, xyz, plan = _built("Cl[Pt@SP1](Cl)(<-N)<-N")
    assert mg.check_geometry(mol, plan, xyz) == []
    tet = mg.POLYHEDRA["tetrahedral"]
    moved = xyz.copy()
    for site, v in zip(plan.sites, tet, strict=True):
        moved[list(site.atoms)] += xyz[plan.metal] + 2.2 * v - xyz[site.atoms[0]]
    assert mg.check_geometry(mol, plan, moved) == ["METAL_POLYHEDRON_CHANGED"]
    other = mg.CentrePlan(**{**plan.__dict__, "slots": mg._SP_TABLE[2]})
    assert mg.check_geometry(mol, other, xyz) == ["METAL_ISOMER_CHANGED"]
