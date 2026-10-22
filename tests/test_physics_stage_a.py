"""Stage-A pair features (`physics.stage_a`) on methanol + chloromethane and on ethane."""

import numpy as np
import pytest

from ksmech.orbitals.tokens import build_molecule_tokens
from ksmech.physics.ks_terms import gap_kernel_matrix
from ksmech.physics.stage_a import GRAPH_CAP, STAGE_A_FEATURES, stage_a_block
from tests.token_select import highest_lone_pair, leading, sigma_star, sigma_stars

A = {name: k for k, name in enumerate(STAGE_A_FEATURES)}


@pytest.fixture(scope="module")
def pair(sp_record):
    methanol = build_molecule_tokens(sp_record("CO"))
    chloromethane = build_molecule_tokens(sp_record("CCl"))
    return methanol, chloromethane, stage_a_block([methanol, chloromethane])


def _cell(methanol, chloromethane, donor, acceptor) -> tuple[int, int]:
    occ = [*methanol.occupied(), *chloromethane.occupied()]
    virt = [*methanol.virtual(), *chloromethane.virtual()]
    return occ.index(donor), virt.index(acceptor)


def test_layout_and_always_defined_entries(pair):
    methanol, chloromethane, block = pair
    n_occ = len(methanol.occupied()) + len(chloromethane.occupied())
    n_virt = len(methanol.virtual()) + len(chloromethane.virtual())
    assert block.z.shape == (n_occ, n_virt, len(STAGE_A_FEATURES))
    for name in ("g", "log_abs_g", "inter", "bonded", "bond_order", "graph_distance"):
        assert block.defined[..., A[name]].all(), name
    assert np.isfinite(block.z).all() and block.n_non_finite == 0
    assert (block.sites == -1).all()  # no contact placement in stage A
    assert (block.z[~block.defined] == 0).all()


def test_gap_factor_and_fragment_flag(pair):
    methanol, chloromethane, block = pair
    occ = [*methanol.occupied(), *chloromethane.occupied()]
    virt = [*methanol.virtual(), *chloromethane.virtual()]
    np.testing.assert_allclose(block.z[..., A["g"]], gap_kernel_matrix(occ, virt))
    n_o, n_v = len(methanol.occupied()), len(methanol.virtual())
    inter = block.z[..., A["inter"]]
    assert (inter[:n_o, :n_v] == 0).all() and (inter[n_o:, n_v:] == 0).all()
    assert (inter[:n_o, n_v:] == 1).all() and (inter[n_o:, :n_v] == 1).all()
    cross = block.z[:n_o, n_v:]
    assert (cross[..., A["graph_distance"]] == GRAPH_CAP).all()
    assert (cross[..., A["bonded"]] == 0).all()
    assert not block.defined[:n_o, n_v:, A["centroid_distance"]].any()


def test_graph_features_within_a_fragment(pair):
    methanol, chloromethane, block = pair
    lp_cl = highest_lone_pair(chloromethane, "Cl")
    cs = sigma_star(chloromethane, {"C", "Cl"})
    i, j = _cell(methanol, chloromethane, lp_cl, cs)
    assert block.z[i, j, A["graph_distance"]] == 0  # they share the Cl atom
    assert block.z[i, j, A["bonded"]] == 1 and block.z[i, j, A["bond_order"]] == 1
    lp_o = highest_lone_pair(methanol, "O")
    ch = sigma_stars(methanol, {"C", "H"})
    assert ch
    i, j = _cell(methanol, chloromethane, lp_o, ch[0])
    assert block.z[i, j, A["graph_distance"]] == 1  # O-C-H: one bond between the sets
    assert block.defined[i, j, A["centroid_distance"]]
    assert block.z[i, j, A["centroid_distance"]] > 0


def test_antiperiplanar_bonds_have_a_dihedral_cosine_near_minus_one(sp_record):
    ethane = build_molecule_tokens(sp_record("CC"))
    block = stage_a_block([ethane])
    c1, c2 = (i for i, a in enumerate(ethane.atoms) if a.atomic_number == 6)
    donors = [o for o in ethane.occupied() if c1 in leading(o) and len(leading(o)) == 2]
    acceptors = [s for s in sigma_stars(ethane, {"C", "H"}) if c2 in leading(s)]
    assert donors and acceptors
    occ, virt = ethane.occupied(), ethane.virtual()
    cos = [
        block.z[occ.index(r), virt.index(s), A["dihedral_cos"]]
        for r in donors
        for s in acceptors
        if block.defined[occ.index(r), virt.index(s), A["dihedral_cos"]]
    ]
    assert cos and min(cos) < -0.95  # staggered: each C-H has an anti C-H on the other carbon
    assert all(-1.0 - 1e-9 <= c <= 1.0 + 1e-9 for c in cos)
