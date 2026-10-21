"""Textbook selectivity checks: case set-up, site matching, symmetry, token provisioning."""

import numpy as np

from ksmech.arrows.reactants import ReactantTokens
from ksmech.physics.baseline import main_centres
from ksmech.physics.ranking import ALL_SCORERS, pair_components
from ksmech.physics.textbook import (
    CASES,
    ensure_tokens,
    fragment_keys,
    run_suite,
    symmetry_classes,
)


def test_cases_cover_the_brief():
    names = [c.name for c in CASES]
    assert len(CASES) == 5 and names[0] == "MeO- + MeCl"
    assert [a.label[:2] for a in CASES[1].alternatives] == ["E2"]
    assert [a.label[:3] for a in CASES[2].alternatives] == ["1,2"]
    assert fragment_keys(CASES[4].smiles) == ["CC(C)=O", "[OH3+]"]


def test_hydronium_protons_are_one_symmetry_class(token_loader):
    rt = ReactantTokens.from_smiles(CASES[4].smiles, token_loader)
    classes = symmetry_classes(rt)
    h = [g for g in range(rt.n_atoms) if rt.atom_fragment[g] == 1 and g != rt.site_of_map(2)]
    assert len(h) == 3 and len({classes[g] for g in h}) == 1
    assert classes[rt.site_of_map(1)] != classes[h[0]]


def _oracle(sample, loader):
    """Scores 1 on pairs whose sites are the expected ones of the matching case, else 0."""
    case = next(c for c in CASES if sample.step_id.endswith(c.name))
    rt = ReactantTokens.from_smiles(case.smiles, loader)
    main = main_centres(sample)
    d = main[np.flatnonzero(sample.orbital_occupied)]
    a = main[np.flatnonzero(~sample.orbital_occupied)]
    classes = symmetry_classes(rt)
    want_d = classes[rt.site_of_map(case.expected.donor)]
    want_a = classes[rt.site_of_map(case.expected.acceptor)]
    return ((classes[d] == want_d)[:, None] & (classes[a] == want_a)[None, :]).astype(float)


def test_suite_runs_and_matches_sites(token_loader, pair_kernels):
    results = run_suite(_oracle, token_loader, pair_kernels)
    assert len(results) == 5 and all(r.passed and r.expected_rank == 0 for r in results)
    for r in results:
        assert len(r.top) == 3 and r.top[0]["score"] == 1.0

    def untrained(sample, loader):
        return ALL_SCORERS["ks3+ks2"](pair_components(sample, loader, pair_kernels.main))

    results = run_suite(untrained, token_loader, pair_kernels)
    assert all(r.expected_rank is not None and np.isfinite(r.expected_score) for r in results)
    sn2_vs_e2 = results[1].alternatives[0]
    assert sn2_vs_e2[1] is not None and np.isfinite(sn2_vs_e2[2])


def test_missing_tokens_are_built_once(qm_config, tmp_path):
    built = ensure_tokens(["[OH3+]"], tmp_path, qm_config)
    assert built == ["[OH3+]"] and len(list(tmp_path.rglob("*.npz"))) == 1
    assert ensure_tokens(["[OH3+]"], tmp_path, qm_config) == []


def test_textbook_cli(token_loader, monkeypatch, tmp_path, capsys):
    import ksmech.physics.cli as cli
    import ksmech.physics.textbook as tb

    monkeypatch.setattr(cli, "TokenCache", lambda *a, **k: token_loader)
    monkeypatch.setattr(tb, "ensure_tokens", lambda keys, tokens_dir, qm=None: [])
    code = cli.main(
        ["textbook", "--processed-dir", str(tmp_path), "--oriented-cache", str(tmp_path)]
    )
    out = capsys.readouterr().out
    assert code in (0, 1) and "untrained ks3+ks2" in out and "expected pairs in the top 3" in out
    assert "baseline` first" in out  # no GBDT model in this processed dir
