"""Open-shell plan P7: SOMO donors and their holes, the SOMO population, the occupation-weighted
gap factor, the two-level column g2, the own-pair channel and the recombination features."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from ksmech.arrows.reactants import ReactantTokens
from ksmech.arrows.steps import reactant_sample
from ksmech.model.config import ModelConfig
from ksmech.orbitals.classify import OrbitalType
from ksmech.orbitals.tokens import (
    CandidateConfig,
    open_shell_tokens,
    token_config,
    with_holes,
)
from ksmech.physics.ks_terms import (
    _two_level,
    gap_kernel_matrix,
    gap_kernel_two_level,
    gap_weights,
)
from ksmech.physics.pairs import valid_pairs
from ksmech.physics.stage_a import (
    OPEN_SHELL_BLOCK,
    SAMPLE_PAIR_FEATURES,
    STAGE_A_FEATURES,
    _betas,
    stage_a_block,
)
from tests.fixtures.molecules import MOLECULES


def _sample(smiles: str, loader):
    return reactant_sample("t", "t", ReactantTokens.from_smiles(smiles, loader))


def _col(s, name: str) -> int:
    return s.pair_features.index(name)


def _roles(s) -> np.ndarray:
    return s.orbital_features[:, s.orbital_columns.index("spin_role")].astype(int)


# ---- step 1 and 2: roles and holes


def test_a_token_file_keeps_its_somo_without_a_role(token_loader):
    """The SOMO of a token file has no role (a file is never changed); `with_holes` gives it
    its donor role and one VIRTUAL twin, its hole, an acceptor."""
    t = token_loader("[CH3]")
    somo = next(o for o in t.occupied() if o.descriptors.occupation < 1.5)
    assert somo.spin_role == 0 and somo not in t.donor_candidates() + t.acceptor_candidates()
    h = with_holes(t)
    assert h.holes == 1 and len(h.orbitals) == len(t.orbitals) + 1
    donor = next(o for o in h.occupied() if o.spin_role == 1)
    hole = h.virtual()[-1]
    assert hole.spin_role == -1 and hole.type == OrbitalType.SOMO_HOLE
    assert donor in h.donor_candidates() and hole in h.acceptor_candidates()
    assert np.array_equal(hole.coeff_vector, donor.coeff_vector)
    assert hole.descriptors is donor.descriptors
    assert hole not in h.acceptor_candidates(donor=donor)  # the token with its own hole


@pytest.mark.parametrize("name", sorted(MOLECULES))
def test_closed_shell_tokens_and_grids_are_unchanged(name, token_loader):
    t = token_loader(MOLECULES[name])
    assert with_holes(t) is t and not open_shell_tokens(t)
    off = CandidateConfig(somo_roles=False)
    assert np.array_equal(valid_pairs([t]), valid_pairs([t], off))


def test_methyl_has_one_donor_row_and_one_hole_column(token_loader):
    s = _sample("[CH3:1]", token_loader)
    roles = _roles(s)
    occ, virt = s.orbital_occupied, ~s.orbital_occupied
    assert (roles[occ] == 1).sum() == 1 and (roles[virt] == -1).sum() == 1
    r = int(np.flatnonzero(roles[occ] == 1)[0])
    c = int(np.flatnonzero(roles[virt] == -1)[0])
    assert s.donor_mask[np.flatnonzero(occ)[r]] and s.acceptor_mask[np.flatnonzero(virt)[c]]
    assert not s.pair_valid[r, c]  # the self-pair is masked
    assert not s.pair_defined[r, c, _col(s, "g")]  # weight 1 - 1 = 0


def test_a_bromine_atom_has_three_holes(token_loader):
    h = with_holes(token_loader("[Br]"))
    assert h.holes == 3


def test_a_cyclization_somo_to_pi_star_cell_is_valid(token_loader):
    """5-hexenyl: the SOMO row with the C=C pi* column of the same fragment stays a candidate
    (only open rows x hole columns are masked within a fragment)."""
    s = _sample("[CH2:1][CH2:2][CH2:3][CH2:4][CH:5]=[CH2:6]", token_loader)
    rt = ReactantTokens.from_smiles("[CH2:1][CH2:2][CH2:3][CH2:4][CH:5]=[CH2:6]", token_loader)
    occ = [o for o in rt.orbitals if o.kind == "occupied"]
    virt = [o for o in rt.orbitals if o.kind == "virtual"]
    r = next(k for k, o in enumerate(occ) if o.spin_role == 1)
    pi_star = [k for k, o in enumerate(virt) if o.type == OrbitalType.PI_STAR]
    assert pi_star and all(s.pair_valid[r, c] for c in pi_star)


def test_the_benzene_radical_anion_somo_reaches_its_partner_pi_star(qm_record):
    """C6H6.- at its optimized (Jahn-Teller distorted) geometry: a SOMO, not a smeared level;
    its row reaches a pi* column, and only its own hole is masked."""
    from ksmech.orbitals.tokens import build_molecule_tokens

    def loader(key: str):
        return build_molecule_tokens(qm_record(key))

    rt = ReactantTokens.from_smiles("[CH-:1]1[CH:2]=[CH:3][CH:4]=[CH:5][CH:6]1", loader)
    s = reactant_sample("t", "t", rt)
    occ = [o for o in rt.orbitals if o.kind == "occupied"]
    virt = [o for o in rt.orbitals if o.kind == "virtual"]
    r = next(k for k, o in enumerate(occ) if o.spin_role == 1)
    pi_star = [k for k, o in enumerate(virt) if o.spin_role == 0 and "pi_star" in o.type]
    assert any(s.pair_valid[r, c] for c in pi_star)
    hole = next(k for k, o in enumerate(virt) if o.spin_role == -1)
    assert not s.pair_valid[r, hole]


def test_metal_d_open_shells_take_no_role_by_default(qm_record):
    """FeBr4- (u = 5 under its spin tag, the record as P4 runs it): five open-shell d tokens,
    no role and no hole under the default rule, five holes with `metal_d_open_shell`."""
    from ksmech.orbitals.tokens import build_molecule_tokens

    t = build_molecule_tokens(qm_record("Br[Fe-](Br)(Br)Br"))
    assert len(open_shell_tokens(t)) == 5 and not open_shell_tokens(t, CandidateConfig())
    assert with_holes(t) is t
    assert with_holes(t, CandidateConfig(metal_d_open_shell=True)).holes == 5


def test_the_candidate_rule_never_enters_the_token_config():
    assert not {f for f in token_config() if "somo" in f or "metal_d" in f or "candidate" in f}


# ---- step 3: the SOMO population


@pytest.mark.parametrize(
    "smiles, u",
    [("[CH3:1]", 1), ("[Br:1]", 1), ("[OH:1]", 1), ("[O:1][O:2]", 2), ("[CH3:1][CH3:2]", 0)],
)
def test_the_somo_population_sums_to_u(smiles, u, token_loader):
    rt = ReactantTokens.from_smiles(smiles, token_loader)
    assert rt.somo_population.sum() == pytest.approx(u, abs=1e-6)
    if smiles == "[CH3:1]":
        assert rt.somo_population[rt.global_atom[0]] > 0.8


def test_the_somo_population_does_not_depend_on_the_drawing(token_loader):
    """The allyl radical's two drawings put u = 1 on either end (`lewis_unpaired`), while the
    SOMO population is the record's, one array for every drawing of the sample."""
    from ksmech.arrows.drawings import molecule_forms
    from ksmech.arrows.lewis import lewis_unpaired

    rt = ReactantTokens.from_smiles("[CH2:1]=[CH:2][CH2:3]", token_loader)
    forms = molecule_forms(rt.mol)
    us = [lewis_unpaired(rt, f) for f in forms]
    assert len(forms) == 2 and not np.array_equal(us[0], us[1])
    s = reactant_sample("t", "t", rt)
    rho = s.atom_features[:, s.atom_columns.index("somo_population")]
    assert np.allclose(rho, rt.somo_population) and rho.sum() == pytest.approx(1.0, abs=1e-5)
    ends = [rt.global_atom[0], rt.global_atom[2]]
    assert rho[ends].min() > 0.3  # both ends carry it, whichever is drawn


def test_an_old_checkpoint_reads_no_spin_input():
    from ksmech.model.inputs import token_columns

    old = ModelConfig.from_dict({"d": 32})
    assert old.atom_spin_input == "none" and not old.spin_role
    atoms, orbitals = token_columns(old)
    assert "somo_population" not in atoms and "spin_role" not in orbitals
    atoms, orbitals = token_columns(replace(old, atom_spin_input="somo_population", spin_role=True))
    assert atoms[-1] == "somo_population" and orbitals[-1] == "spin_role"


# ---- step 4: the occupation-weighted G


def test_somo_and_hole_cells_weigh_one(token_loader):
    rt = ReactantTokens.from_smiles("[CH3:1][CH2:2]", token_loader)
    occ = [o for o in rt.orbitals if o.kind == "occupied"]
    virt = [o for o in rt.orbitals if o.kind == "virtual"]
    w = gap_weights(occ, virt)
    r = [o.spin_role == 1 for o in occ]
    c = [o.spin_role == -1 for o in virt]
    closed_r, closed_c = ~np.array(r), ~np.array(c)
    assert (w[np.ix_(r, closed_c)] == 1).all() and (w[np.ix_(closed_r, c)] == 1).all()
    assert (w[np.ix_(closed_r, closed_c)] == 2).all() and (w[np.ix_(r, c)] == 0).all()


def test_the_bromine_atom_holes_pool_to_weight_one(token_loader):
    rt = ReactantTokens.from_smiles("[Br:1]", token_loader)
    occ = [o for o in rt.orbitals if o.kind == "occupied" and o.spin_role == 0]
    holes = [o for o in rt.orbitals if o.spin_role == -1]
    assert len(holes) == 3 and gap_weights(occ, holes).sum(1) == pytest.approx(1.0)


@pytest.mark.parametrize("name", sorted(MOLECULES))
def test_closed_shell_g_is_bitwise_the_v18_g(name, token_loader):
    t = token_loader(MOLECULES[name])
    occ, virt = t.occupied(), t.virtual()
    old = np.empty((len(occ), len(virt)))
    for i, r in enumerate(occ):  # the v18 product, one row at a time on its own block
        e_r, p_r = r.descriptors.full_energies, np.asarray(r.descriptors.full_weights)
        e_s = virt[0].descriptors.full_energies
        P_s = np.array([o.descriptors.full_weights for o in virt])
        gap = e_r[:, None] - e_s[None, :]
        gap = np.where(gap >= 0, 1.0, -1.0) * np.maximum(np.abs(gap), 1e-3)
        old[i] = (2.0 * p_r[None, :] @ (1.0 / gap) @ P_s.T)[0]
    assert np.allclose(gap_kernel_matrix(occ, virt), old, rtol=1e-13, atol=0)
    block = stage_a_block([t])
    assert block.defined[..., STAGE_A_FEATURES.index("g")].all()


# ---- step 5: the own-pair channel


def test_the_ethane_sigma_cc_own_cell_carries_own_g(token_loader):
    s = _sample("[CH3:1][CH3:2]", token_loader)
    own = s.pair_z[..., _col(s, "own_antibond")] > 0
    assert own.any()
    d = s.pair_defined[..., _col(s, "own_g")]
    assert np.array_equal(d, own)
    assert np.array_equal(s.pair_z[..., _col(s, "own_g")][own], s.pair_z[..., _col(s, "g")][own])


def test_own_cells_stay_out_of_the_valid_pooled_gap_and_b_own_sits_on_bonds(token_loader):
    from ksmech.arrows.batches import collate_samples
    from ksmech.arrows.lewis import reactant_lewis
    from ksmech.arrows.steps import with_lewis
    from ksmech.model.inputs import pooled_gap, pooled_own

    rt = ReactantTokens.from_smiles("[CH3:1][C:2](=[O:3])[CH3:4]", token_loader)
    s = with_lewis(reactant_sample("t", "t", rt), reactant_lewis(rt))
    own = s.pair_z[..., _col(s, "own_antibond")] > 0
    assert not (own & s.pair_valid).any()
    b = collate_samples([s])
    stripped = dict(b, pair_z=b["pair_z"][..., : len(SAMPLE_PAIR_FEATURES) - len(OPEN_SHELL_BLOCK)])
    stripped["pair_features"] = SAMPLE_PAIR_FEATURES[: -len(OPEN_SHELL_BLOCK)]
    assert torch.equal(pooled_gap(b), pooled_gap(stripped))
    b_own = pooled_own(b)[0]
    bonded = b["bond_order"][0] > 0
    assert b_own.abs().sum() > 0 and (b_own[~bonded] == 0).all()
    assert (pooled_own(stripped) == 0).all()


def test_the_gbdt_and_stage_a_feature_lists_are_unchanged():
    from ksmech.physics.baseline import FEATURE_NAMES

    assert STAGE_A_FEATURES[:2] == ("g", "log_abs_g") and len(STAGE_A_FEATURES) == 11
    assert not set(OPEN_SHELL_BLOCK) & set(FEATURE_NAMES)


# ---- step 6: g2


@pytest.mark.parametrize("name", sorted(MOLECULES))
def test_g2_at_beta_zero_is_g(name, token_loader):
    t = token_loader(MOLECULES[name])
    occ, virt = t.occupied(), t.virtual()
    zero = np.zeros((len(occ), len(virt)))
    assert np.allclose(gap_kernel_two_level(occ, virt, zero), gap_kernel_matrix(occ, virt),
                       rtol=1e-12, atol=0)  # fmt: skip


@pytest.mark.parametrize(
    "smiles",
    ["[CH3:1][CH2:2]", "[Br:1]", "[O:1][O:2]", "[CH-:1]1[CH:2]=[CH:3][CH:4]=[CH:5][CH:6]1",
     "[CH3:1][O:2][O:3]", "[O-:1][O:2]", "[CH3:1].[CH3:2][C:3](=[O:4])[CH3:5]"],
)  # fmt: skip
def test_g2_is_finite_and_bounded_on_open_shells(smiles, token_loader):
    rt = ReactantTokens.from_smiles(smiles, token_loader)
    tok = [f.tokens for f in rt.fragments]
    block = stage_a_block(tok)
    occ = [o for t in tok for o in t.occupied()]
    virt = [o for t in tok for o in t.virtual()]
    beta = _betas(tok, block, 1.0)
    g2 = gap_kernel_two_level(occ, virt, beta)
    assert np.isfinite(g2).all()
    w = np.abs(gap_weights(occ, virt))
    pr = np.array([np.sum(o.descriptors.full_weights) for o in occ])
    ps = np.array([np.sum(o.descriptors.full_weights) for o in virt])
    bound = w * pr[:, None] * ps[None, :] / np.maximum(np.abs(beta), 1e-300)
    assert (np.abs(g2) <= bound * (1 + 1e-9) + 1e-12).all()


def test_a_two_level_sweep_stays_bounded_and_flips_sign():
    beta = 0.5
    de = np.linspace(-2.0, 2.0, 401)
    v = _two_level(de, np.full_like(de, beta))
    assert np.isfinite(v).all() and (np.abs(v) <= 1 / beta + 1e-12).all()
    assert (np.sign(v[de > 0]) == 1).all() and (np.sign(v[de < 0]) == -1).all()
    assert v[200] == pytest.approx(-1 / beta)  # dE = 0: sign -1, D = |beta|
    big = _two_level(np.array([50.0]), np.array([beta]))
    assert big[0] == pytest.approx(1 / 50.0, rel=1e-3)


# ---- step 7: recombination


def test_two_methyl_radicals_have_one_somo_pair_cell_per_orientation(token_loader):
    s = _sample("[CH3:1].[CH3:2]", token_loader)
    k, j = _col(s, "somo_pair"), _col(s, "somo_level_diff")
    cells = s.pair_z[..., k] > 0
    assert cells.sum() == 2 and np.array_equal(s.pair_defined[..., k], cells)
    assert np.array_equal(s.pair_defined[..., j], cells)
    assert np.allclose(s.pair_z[..., j][cells], 0.0, atol=1e-6)  # one SOMO level, twice
    inter = s.pair_z[..., _col(s, "inter")] > 0
    assert not (cells & ~inter).any()
    assert not s.pair_defined[..., _col(s, "g")][cells].any()


# ---- step 8: one input path


def test_dense_and_cell_inputs_agree_on_a_radical_pool(token_loader):
    from ksmech.arrows.batches import collate_samples
    from ksmech.model.inputs import named_pair_z, pair_channels, pair_channels_at, pair_context

    s = _sample("[CH3:1].[CH2:2]=[CH2:3]", token_loader)
    b = collate_samples([replace(s, nonbonding=np.zeros(s.n_atoms, np.int8))])
    z, d = named_pair_z(b, SAMPLE_PAIR_FEATURES, "all")
    z = torch.where(d, z, 0.0)
    dense = pair_channels(b, z, d, 8, 8.0)
    ctx = pair_context(b, z, d)
    n = dense.shape[1]
    i, j = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    cells = pair_channels_at(ctx, b, torch.zeros_like(i), i, j, 8, 8.0)
    assert torch.allclose(cells, dense[0], atol=1e-6)


# ---- step 9 and 10: the tabulation and the own_break floor


def test_a_homolysis_is_reachable_only_under_own_break(token_loader):
    """HO-OH -> 2 HO. as a one-electron rmechdb step: its only admissible anchor breaks the
    O-O bond. The legacy floor proposes O-O as a form anchor (its order may rise), so the step
    is unreachable there; own_break proposes the break, scored by |B^own|."""
    from ksmech.arrows.batches import collate_samples
    from ksmech.arrows.steps import MechanismSample, sample_for_step
    from ksmech.data import chem
    from ksmech.data.schema import ElementaryStep, Source
    from ksmech.eval.floor import floor_ranks, floor_scores

    r, p = chem.parse_mapped("[OH:1][OH:2]"), chem.parse_mapped("[OH:1].[OH:2]")
    sides = (chem.canonical_mapped_smiles(r), chem.canonical_mapped_smiles(p))
    step = ElementaryStep("h2o2", Source.RMECHDB, None, None, *sides)
    s = sample_for_step(step, token_loader)
    assert isinstance(s, MechanismSample)
    assert all(k == "break" for a in s.edit_label.anchors for *_, k in a)
    b = collate_samples([s])
    _, legacy = floor_ranks(b, "valid", "legacy")
    rank, own = floor_ranks(b, "valid", "own_break")
    assert not legacy[0] and own[0]
    _, b_own = floor_scores(b, "valid", "g", False)
    ((i, j, _),) = s.edit_label.anchors[0]
    assert b_own[0, i, j] > 0
    _, own2 = floor_ranks(b, "valid", "own_break", kernel="g2")
    assert own2[0]


def test_the_tabulation_runs_on_small_species(token_loader):
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "probes" / "open_shell_g_table.py"
    spec = importlib.util.spec_from_file_location("open_shell_g_table", path)
    table = importlib.util.module_from_spec(spec)
    sys.modules["open_shell_g_table"] = table
    spec.loader.exec_module(table)
    rows = [table.cell_rows(s, token_loader) for s in ("[CH3:1].[CH2:2]=[CH2:3]", "[Br:1].[CH4:2]")]
    out = table.summarize(rows)
    kinds = {g.split("/")[0] for g in out["groups"]}
    assert {"somo_empty", "doubly_hole", "doubly_empty", "own_pair"} <= kinds
    assert out["whole"]["all"]["g2"]["max"] <= out["whole"]["all"]["g"]["max"] + 1e-9
    assert out["beta_by_graph_distance"]["0"]["n"] > 0


# ---- step 11: the radical selectivity suite


def test_the_radical_suite_runs_with_both_kernels(qm_record):
    from ksmech.orbitals.tokens import build_molecule_tokens
    from ksmech.physics.textbook import RADICAL_CASES, run_radical_suite

    memo: dict = {}

    def loader(key: str):
        if key not in memo:
            memo[key] = build_molecule_tokens(qm_record(key))
        return memo[key]

    for kernel in ("g", "g2"):
        out = run_radical_suite(loader, kernel, drop_intra_open=kernel == "g2")
        assert [r.case.name for r in out["results"]] == [c.name for c in RADICAL_CASES]
        assert all(r.expected_rank is not None for r in out["results"])
        assert isinstance(out["snap_order_i_br_cl"], bool)
