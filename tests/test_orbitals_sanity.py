"""Chemistry sanity checks of the localized-orbital tokens on the optimized fixture molecules.

Run with `-s` to see the per-molecule token tables (type, energy, parent atoms).
"""

from collections import Counter

import numpy as np
import pytest
from rich.console import Console

from ksmech.orbitals.classify import OrbitalType
from ksmech.orbitals.localize import localize_record, valence_electrons
from ksmech.orbitals.tokens import build_molecule_tokens, token_table

pytestmark = pytest.mark.slow
T = OrbitalType


@pytest.fixture(scope="module")
def tokens(fixture_records):
    return {name: build_molecule_tokens(rec) for name, rec in fixture_records.items()}


def show(tokens, name) -> None:
    Console(width=110).print(token_table(tokens[name], title=name))


def occ_types(t) -> Counter:
    return Counter(o.type for o in t.occupied())


def atoms_of(t, symbol: str) -> set[int]:
    return {a.atom_index for a in t.atoms if t.symbol(a.atom_index) == symbol}


def test_ethane_is_seven_sigma_bonds(tokens):
    show(tokens, "ethane")
    t = tokens["ethane"]
    assert occ_types(t) == {T.SIGMA: 7}


def test_acetone_lone_pairs_pi_and_lowest_pi_star(tokens):
    show(tokens, "acetone")
    t = tokens["acetone"]
    oxygen = atoms_of(t, "O")
    lone_pairs = [o for o in t.occupied() if o.type == T.LONE_PAIR]
    assert len(lone_pairs) == 2 and all(set(o.parent_atoms) == oxygen for o in lone_pairs)
    pis = [o for o in t.occupied() if o.type == T.PI]
    assert len(pis) == 1 and oxygen < set(pis[0].parent_atoms)
    lowest = t.virtual()[0]  # virtual tokens are sorted by energy
    assert lowest.type == T.PI_STAR and oxygen < set(lowest.parent_atoms)
    sigma_stars = [o.energy_ev for o in t.virtual() if o.type == T.SIGMA_STAR]
    assert sigma_stars and lowest.energy_ev < min(sigma_stars)


def test_methylamine_lone_pair_is_the_highest_occupied(tokens):
    show(tokens, "methylamine")
    t = tokens["methylamine"]
    lone_pairs = [o for o in t.occupied() if o.type == T.LONE_PAIR]
    assert len(lone_pairs) == 1 and set(lone_pairs[0].parent_atoms) == atoms_of(t, "N")
    sigmas = [o.energy_ev for o in t.occupied() if o.type == T.SIGMA]
    assert len(sigmas) == 6 and lone_pairs[0].energy_ev > max(sigmas)


def test_benzene_pi_system(tokens):
    """PM with Mulliken populations gives three 3-centre pi orbitals (0.50/0.24/0.21), the same
    functional value from random orthogonal starts, so they are `delocalized_pi`, not `pi`."""
    show(tokens, "benzene")
    counts = occ_types(tokens["benzene"])
    n_pi, n_deloc = counts[T.PI], counts[T.DELOCALIZED_PI]
    assert (n_pi, n_deloc) in {(3, 0), (0, 3)}, counts
    assert counts[T.SIGMA] == 12 and counts[T.UNKNOWN] == 0
    print(f"benzene pi system: {n_pi} pi, {n_deloc} delocalized_pi")


def test_acetonitrile_two_pi_and_two_pi_star_on_cn(tokens):
    show(tokens, "acetonitrile")
    t = tokens["acetonitrile"]
    nitrogen = atoms_of(t, "N")
    cn = {
        o.parent_atoms
        for o in t.orbitals
        if nitrogen < set(o.parent_atoms) and len(o.parent_atoms) == 2
    }
    pis = [o for o in t.occupied() if o.type == T.PI]
    pi_stars = [o for o in t.virtual() if o.type == T.PI_STAR]
    assert len(pis) == 2 and len(pi_stars) == 2
    assert all(o.parent_atoms in cn for o in pis + pi_stars)


def test_chloromethane_lowest_sigma_star_is_c_cl(tokens):
    show(tokens, "chloromethane")
    t = tokens["chloromethane"]
    sigma_stars = [o for o in t.virtual() if o.type == T.SIGMA_STAR]
    lowest = min(sigma_stars, key=lambda o: o.energy_ev)
    assert set(lowest.parent_atoms) == atoms_of(t, "C") | atoms_of(t, "Cl")


def test_every_fixture_is_resolved_and_converged(tokens):
    for name, t in tokens.items():
        assert t.pm_converged, name
        unknown = [o for o in t.orbitals if o.type == T.UNKNOWN]
        assert not unknown, (name, unknown)
        n_val = sum(valence_electrons(a.atomic_number) - a.formal_charge for a in t.atoms)
        assert len(t.occupied()) == n_val // 2, name
        assert len(t.orbitals) == t.S.shape[0], name  # one token per AO


def test_print_remaining_tables(tokens):
    for name in ("ethanol", "acetaldehyde", "methyl_acetate", "acrolein"):
        show(tokens, name)


# --- leaving-group physics -------------------------------------------------------------------

PROTONATION_SHIFT_EV = 3.0  # sigma*(C-O) must drop by at least this much on protonating methanol


def heavy_sigma_star(tokens, symbols: set[str]):
    """The sigma* token whose two parents are exactly the given element symbols."""
    found = [
        o
        for o in tokens.virtual()
        if o.type == T.SIGMA_STAR and {tokens.symbol(a) for a in o.parent_atoms} == symbols
    ]
    assert len(found) == 1, (tokens.smiles, found)
    return found[0]


@pytest.fixture(scope="module")
def gas_config(tmp_path_factory):
    from ksmech.qm.xtb_calc import QMConfig

    return QMConfig(cache_dir=tmp_path_factory.mktemp("qm_gas"), solvent=None)


def test_protonation_lowers_sigma_star_c_o_gas_phase(qm_record, gas_config):
    neutral = build_molecule_tokens(qm_record("CO", gas_config))
    protonated = build_molecule_tokens(qm_record("C[OH2+]", gas_config))
    shift = (
        heavy_sigma_star(neutral, {"C", "O"}).energy_ev
        - heavy_sigma_star(protonated, {"C", "O"}).energy_ev
    )
    print(f"gas phase: sigma*(C-O) drops by {shift:.2f} eV on protonation")
    assert shift >= PROTONATION_SHIFT_EV


def test_protonation_lowers_sigma_star_c_o_in_water(qm_record):
    """ALPB water screens the cation: the drop is 2.9 eV, just under the gas-phase criterion.
    Only the sign is asserted here; the measured value is printed and recorded in decisions."""
    neutral = build_molecule_tokens(qm_record("CO"))
    protonated = build_molecule_tokens(qm_record("C[OH2+]"))
    shift = (
        heavy_sigma_star(neutral, {"C", "O"}).energy_ev
        - heavy_sigma_star(protonated, {"C", "O"}).energy_ev
    )
    print(f"ALPB water: sigma*(C-O) drops by {shift:.2f} eV on protonation")
    assert shift > 0


def halide_energies(qm_record, config=None) -> dict[str, tuple[float, float]]:
    """{X: (sigma*(C-X) LMO energy, canonical LUMO)} for CH3Cl, CH3Br, CH3I."""
    out = {}
    for x in ("Cl", "Br", "I"):
        rec = qm_record(f"C{x}", config)
        tokens = build_molecule_tokens(rec)
        out[x] = (heavy_sigma_star(tokens, {"C", x}).energy_ev, rec.lumo_ev)
    print(
        "sigma*(C-X) / canonical LUMO (eV): "
        + ", ".join(f"{x} {e:.2f} / {lumo:.2f}" for x, (e, lumo) in out.items())
    )
    return out


@pytest.mark.xfail(
    strict=True,
    reason="GFN2 + PM sigma*(C-X) LMO energies are Cl -2.07, Br -0.65, I -1.63 eV in ALPB "
    "water (gas phase -1.94, -0.29, -1.45): the C-Cl sigma* is the lowest, not the highest. "
    "The canonical LUMOs do order Cl > Br > I (next test). Recorded in docs/decisions.md.",
)
def test_sigma_star_c_x_orders_cl_br_i(qm_record):
    e = {x: v[0] for x, v in halide_energies(qm_record).items()}
    assert e["Cl"] > e["Br"] > e["I"], e


def test_canonical_lumo_orders_cl_br_i(qm_record):
    lumo = {x: v[1] for x, v in halide_energies(qm_record).items()}
    assert lumo["Cl"] > lumo["Br"] > lumo["I"], lumo


def test_grignard_carbanion_is_a_metal_bound_lone_pair(qm_record):
    tokens = build_molecule_tokens(qm_record("C[Mg]Br"))
    show({"CH3MgBr": tokens}, "CH3MgBr")
    carbanion = [
        o
        for o in tokens.occupied()
        if o.type == T.LONE_PAIR and tokens.symbol(o.parent_atoms[0]) == "C"
    ]
    assert len(carbanion) == 1 and carbanion[0].metal_bound
    assert tokens.symbol(carbanion[0].metal_atom) == "Mg"
    assert carbanion[0] in tokens.donor_candidates()
    assert not any(o.type == T.UNKNOWN for o in tokens.occupied())


# --- continuous descriptors and the spectral KS term ------------------------------------------


def canonical_pi_mos(rec) -> list[int]:
    """Occupied canonical MOs of a planar ring that are pure p and perpendicular to the plane."""
    from ksmech.orbitals.classify import ao_shares, p_vector
    from ksmech.qm.fukui import OCC_THRESHOLD

    carbons = [i for i, z in enumerate(rec.atomic_numbers) if z == 6]
    xyz = rec.coords_ang[carbons] - rec.coords_ang[carbons].mean(axis=0)
    normal = np.linalg.svd(xyz)[2][-1]
    out = []
    for k in np.flatnonzero(rec.occupations >= OCC_THRESHOLD):
        c = rec.C[:, k]
        if ao_shares(c, rec.S, rec.ao_l)[0] > 0.02:
            continue
        vecs = [p_vector(c, rec.ao_atom, rec.ao_l, a) for a in carbons]
        aligned = [abs(v @ normal) / np.linalg.norm(v) for v in vecs if np.linalg.norm(v) > 0.05]
        if aligned and min(aligned) > 0.9:
            out.append(int(k))
    return out


def test_benzene_occupied_tokens_carry_the_three_pi_mos(fixture_records):
    rec = fixture_records["benzene"]
    pi = canonical_pi_mos(rec)
    assert len(pi) == 3, pi
    loc = localize_record(rec)
    full = (loc.U_occ[pi, :] ** 2).sum()  # weight of the pi MOs over every occupied LMO
    assert abs(full - 3.0) < 1e-8
    tokens = build_molecule_tokens(rec)
    stored = sum(
        float(w)
        for o in tokens.occupied()
        for i, w in zip(o.descriptors.spectral_indices, o.descriptors.spectral_weights, strict=True)
        if i in pi
    )
    print(f"benzene: stored spectral weight on the pi MOs = {stored:.3f} (full {full:.6f})")
    assert stored > 2.95


def test_localization_is_deterministic(fixture_records):
    rec = fixture_records["benzene"]
    first, second = build_molecule_tokens(rec), build_molecule_tokens(rec)
    assert np.array_equal(first.orbital_features(), second.orbital_features())
    assert np.array_equal(first.C_loc, second.C_loc)
    assert [o.type for o in first.orbitals] == [o.type for o in second.orbitals]
    assert [o.parent_atoms for o in first.orbitals] == [o.parent_atoms for o in second.orbitals]


def test_acetone_pi_star_descriptors(tokens):
    t = tokens["acetone"]
    pi_star = [o for o in t.virtual() if o.type == T.PI_STAR]
    assert len(pi_star) == 1
    d = pi_star[0].descriptors
    assert d.cos_character() < 0.3 and d.overlap_sign < 0 and d.overlap_magnitude > 0
    sigma_stars = [o.descriptors for o in t.virtual() if o.type == T.SIGMA_STAR]
    assert sigma_stars and all(s.cos_character() > 0.7 for s in sigma_stars)
    assert all(s.overlap_sign < 0 for s in sigma_stars)


def test_gap_kernel_prefers_the_c_cl_sigma_star(qm_record, tokens):
    from ksmech.physics.ks_terms import token_gap_kernel

    methanol = build_molecule_tokens(qm_record("CO"))
    lone = max(
        (
            o
            for o in methanol.occupied()
            if o.type == T.LONE_PAIR and methanol.symbol(o.parent_atoms[0]) == "O"
        ),
        key=lambda o: o.energy_ev,
    )
    t = tokens["chloromethane"]
    c_cl = heavy_sigma_star(t, {"C", "Cl"})
    c_h = next(o for o in t.virtual() if o.type == T.SIGMA_STAR and o is not c_cl)
    g_cl, g_h = token_gap_kernel(lone, c_cl), token_gap_kernel(lone, c_h)
    print(f"G(O lone pair, sigma* C-Cl) = {g_cl:.4f} 1/eV, G(O lone pair, sigma* C-H) = {g_h:.4f}")
    assert g_cl < 0 and g_cl < g_h


def ring_roles(mol) -> dict[str, list[int]]:
    """ipso / ortho / meta / para ring carbons of a monosubstituted benzene."""
    ring = next(r for r in mol.GetRingInfo().AtomRings() if len(r) == 6)
    ring_set = set(ring)
    ipso = next(
        a
        for a in ring
        if any(
            n.GetIdx() not in ring_set and n.GetAtomicNum() > 1
            for n in mol.GetAtomWithIdx(a).GetNeighbors()
        )
    )
    ortho = [n.GetIdx() for n in mol.GetAtomWithIdx(ipso).GetNeighbors() if n.GetIdx() in ring_set]
    meta = [
        n.GetIdx()
        for o in ortho
        for n in mol.GetAtomWithIdx(o).GetNeighbors()
        if n.GetIdx() in ring_set and n.GetIdx() != ipso
    ]
    para = [a for a in ring if a not in {ipso, *ortho, *meta}]
    return {"ipso": [ipso], "ortho": ortho, "meta": meta, "para": para}


def test_anisole_homo_weight_is_ortho_para(qm_record):
    """The ring pi tokens of anisole are centred on the ortho and para carbons, never on meta,
    and the para-centred token carries the largest HOMO weight. The per-centre sum
    sum_r p_r,HOMO w_r(c) is *not* the HOMO population (it drops the LMO cross terms: meta
    0.157 vs exact 0.048 in GFN2), so the regiochemistry lives in where PM centres the tokens
    and in their spectral weights, which is what is asserted here."""
    from ksmech.qm.embed import prepare_mol
    from ksmech.qm.fukui import homo_index

    rec = qm_record("COc1ccccc1")
    tokens = build_molecule_tokens(rec)
    homo = homo_index(rec.occupations)
    roles = ring_roles(prepare_mol(rec.atom_order_smiles))
    role_of = {a: role for role, atoms in roles.items() for a in atoms}
    ring_pi = []  # (dominant centre role, HOMO weight) of the occupied ring pi tokens
    for o in tokens.occupied():
        d = o.descriptors
        centre = int(d.center_atoms[0])
        if centre not in role_of or d.center_spd[0, 1] < 0.9 or d.n_centers < 2:
            continue
        hit = np.flatnonzero(d.spectral_indices == homo)
        ring_pi.append((role_of[centre], float(d.spectral_weights[hit[0]]) if len(hit) else 0.0))
    print("anisole ring pi tokens (dominant centre, HOMO weight): " + str(ring_pi))
    assert len(ring_pi) == 3
    assert {role for role, _ in ring_pi} <= {"ortho", "para"}  # none centred at meta or ipso
    best_role = max(ring_pi, key=lambda rw: rw[1])[0]
    assert best_role == "para"
    assert min(w for role, w in ring_pi if role in ("ortho", "para")) > 0.1


def test_anisole_atom_homo_weights_are_ortho_para(qm_record):
    from ksmech.qm.embed import prepare_mol

    rec = qm_record("COc1ccccc1")
    tokens = build_molecule_tokens(rec)
    roles = ring_roles(prepare_mol(rec.atom_order_smiles))
    homo = {a: tokens.atoms[a].fmo_weights[1] for atoms in roles.values() for a in atoms}
    print(
        "anisole atom HOMO weights: "
        + str({k: [round(homo[a], 3) for a in v] for k, v in roles.items()})
    )
    assert min(homo[a] for a in roles["ortho"] + roles["para"]) > max(
        homo[a] for a in roles["meta"]
    )


def test_nitrobenzene_atom_lumo_weights_are_ortho_para(qm_record):
    from ksmech.qm.embed import prepare_mol

    rec = qm_record("O=[N+]([O-])c1ccccc1")
    tokens = build_molecule_tokens(rec)
    roles = ring_roles(prepare_mol(rec.atom_order_smiles))
    lumo = {a: tokens.atoms[a].fmo_weights[2] for atoms in roles.values() for a in atoms}
    print(
        "nitrobenzene atom LUMO weights: "
        + str({k: [round(lumo[a], 3) for a in v] for k, v in roles.items()})
    )
    assert min(lumo[a] for a in roles["ortho"] + roles["para"]) > max(
        lumo[a] for a in roles["meta"]
    )
