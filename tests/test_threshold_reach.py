"""scripts/probes/threshold_reach.py (open-shell plan P0a, step 9): on the fixed molecules the
recomputation at the current values reproduces the stored centre lists and weights, donor /
acceptor masks and spectral sets bit for bit (no token changes), an alternative spectral merge
moves acetone, methyl acetate and acrolein, each handler finds what its threshold moves, sample
hits land on the right fragment, and the report is stamped where --out-dir says."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ksmech.data import provenance as prov
from ksmech.orbitals.descriptors import DescriptorConfig
from ksmech.orbitals.localize import is_fermi_smeared
from ksmech.orbitals.tokens import CandidateConfig, OrbitalKind, build_molecule_tokens
from ksmech.qm.records import cache_path, save_record

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "threshold_reach", ROOT / "scripts" / "probes" / "threshold_reach.py"
)
tr = importlib.util.module_from_spec(_spec)
sys.modules["threshold_reach"] = tr  # dataclasses look their module up while being made
_spec.loader.exec_module(tr)

CURRENT = [v for v in tr.VARIANTS if v.current]
ALTERNATIVE = [v for v in tr.VARIANTS if not v.current]


@pytest.fixture(scope="module")
def fixed(sp_record, smiles):
    """name -> (tokens, QM record) of the ten fixed molecules (GFN2 single points)."""
    out = {}
    for name, smi in smiles.items():
        rec = sp_record(smi)
        out[name] = (build_molecule_tokens(rec), rec)
    return out


@pytest.fixture(scope="module")
def reach(fixed):
    return {name: tr.molecule_reach(t, rec) for name, (t, rec) in fixed.items()}


def test_the_variants_are_the_plan_values():
    names = [v.name for v in tr.VARIANTS]
    assert [v.name for v in CURRENT] == [
        "tau_atom=0.15",
        "fractional_band=0.2-1.8",
        "spectral_merge_ev=0.05",
        "smear_tol=0.05",
        "block_split=0.5-1.5",
    ]
    for name in (
        "tau_atom=0.1",
        "tau_atom=0.2",
        "fractional_band=0.1-1.9",  # the edges' distances to 0 and 2 halved
        "fractional_band=0.4-1.6",  # and doubled
        "spectral_merge_ev=0.0001",
        "spectral_merge_ev=0.01",
        "spectral_merge_ev=0.1",
        "smear_tol=0.025",
        "smear_tol=0.1",
        "block_split=0.25-1.75",
        "block_split=0.75-1.25",
    ):
        assert name in names


def test_current_values_rebuild_the_stored_tokens_bit_for_bit(fixed):
    """Centre lists and weights (with everything derived from them), spectral sets and the
    retained fraction, the donor / acceptor masks and the fermi_smeared flag."""
    for name, (t, rec) in fixed.items():
        m = tr.MoleculeView(t, rec)
        for k, o in enumerate(t.orbitals):
            d = o.descriptors
            new = tr.rebuilt_centres(m, k, DescriptorConfig())
            for f in tr.TAU_FIELDS:
                assert np.array_equal(getattr(new, f), getattr(d, f)), (name, k, f)
            e, w, retained = tr.rebuilt_spectrum(d, DescriptorConfig())
            assert np.array_equal(e, d.spectral_energies), (name, k)
            assert np.array_equal(w, d.spectral_weights), (name, k)
            assert retained == d.retained_fraction
            assert tr.spectral_shift(d, (e, w, retained)) == 0.0
        donors, acceptors = tr.mask_sets(t, CandidateConfig())
        assert [t.orbitals[k] for k in sorted(donors)] == t.donor_candidates()
        assert [t.orbitals[k] for k in sorted(acceptors)] == t.acceptor_candidates()
        assert is_fermi_smeared(rec, tr.CURRENT["smear_tol"]) == t.fermi_smeared


def test_no_token_changes_at_the_current_values(reach):
    for name, res in reach.items():
        for v in CURRENT:
            assert res["hits"][v.name] == {}, (name, v.name)


def test_an_alternative_merge_moves_spectral_sets_and_closed_shells_keep_their_masks(reach):
    """At 0.01 eV the merge splits near-degenerate levels of acetone, methyl acetate and
    acrolein; the band, SMEAR_TOL and the block split cannot move a closed shell whose
    occupations are 2 and 0."""
    moved = {
        v.name: sorted(n for n, res in reach.items() if res["hits"].get(v.name))
        for v in ALTERNATIVE
    }
    assert {"acetone", "methyl_acetate", "acrolein"} <= set(moved["spectral_merge_ev=0.01"])
    hit = reach["acrolein"]["hits"]["spectral_merge_ev=0.01"]
    assert hit["tokens"] == hit["input"] and sum(hit["shift"]) == len(hit["input"])
    for v in ALTERNATIVE:
        if v.threshold in ("fractional_band", "smear_tol", "block_split"):
            assert moved[v.name] == [], v.name


def _centre_rule(pops: np.ndarray, tau: float) -> set[int]:
    """The test's own oracle of the centre list: every atom >= tau, else the largest."""
    above = {int(a) for a in np.flatnonzero(pops >= tau)}
    return above or {int(np.argmax(pops))}


@pytest.mark.parametrize("tau", [0.05, 0.10, 0.20, 0.30])
def test_tau_hit_is_the_set_of_tokens_whose_centres_move(fixed, tau):
    for name, (t, rec) in fixed.items():
        m = tr.MoleculeView(t, rec)
        expected = {
            k for k, pops in enumerate(m.populations) if _centre_rule(pops, tau) != m.centres[k]
        }
        assert tr.tau_hit(m, tau)["input"] == expected, (name, tau)
    m = tr.MoleculeView(*fixed["benzene"])
    hit = tr.compact(tr.tau_hit(m, 0.30))  # the allyl-like pi LMOs drop their 0.22 centres
    assert len(hit["input"]) == 6 and hit["n_cells"] > 0


def test_band_moves_a_fractional_donor(fixed):
    t, rec = fixed["acetone"]
    k = max(tr.MoleculeView(t, rec).occ, key=lambda j: t.orbitals[j].descriptors.energy_ev)
    o = t.orbitals[k]
    smeared = replace(o, descriptors=replace(o.descriptors, occupation=1.85))
    t2 = replace(t, orbitals=[smeared if j == k else x for j, x in enumerate(t.orbitals)])
    m = tr.MoleculeView(t2, rec)
    assert k in m.donors  # 1.85 is a donor between 1.8 and 2
    assert tr.band_hit(m, (0.1, 1.9))["donor"] == {k}  # fractional once the edge is at 1.9
    assert tr.band_hit(m, (0.4, 1.6))["donor"] == set()
    assert tr.band_hit(m, (0.4, 1.6))["acceptor"] == set()


def test_smear_and_block_read_the_canonical_occupations(fixed):
    t, _ = fixed["ethane"]
    n_mo = len(t.orbitals)
    occ = np.zeros(n_mo)
    occ[: len([o for o in t.orbitals if o.kind == OrbitalKind.OCCUPIED])] = 2.0
    homo = int(np.flatnonzero(occ).max())
    occ[homo], occ[homo + 1] = 1.96, 0.04  # 0.04 from aufbau: smeared only below 0.04
    rec = SimpleNamespace(occupations=occ, orbital_energies_ev=np.arange(n_mo, dtype=float))
    rec.unpaired = 0
    m = tr.MoleculeView(t, rec)
    assert tr.smear_hit(m, 0.05)["flag"] == set()
    hit = tr.smear_hit(m, 0.025)
    assert hit["flag"] == set(range(n_mo)) and hit["strict"] == set(range(len(t.atoms)))
    assert tr.block_hit(m, (0.5, 1.5))["block"] == set()
    rec.occupations = occ.copy()
    rec.occupations[homo] = 1.6  # doubly at 1.5, singly at 1.75: the occupied block moves
    assert tr.block_hit(tr.MoleculeView(t, rec), (0.25, 1.75))["block"] == set(m.occ)
    rec.occupations[homo + 1] = 0.3  # crosses 0.25: both blocks move
    assert tr.block_hit(tr.MoleculeView(t, rec), (0.25, 1.75))["block"] == set(range(n_mo))


def _mol(n_atoms: int, n_orbitals: int, hits: dict, smeared: bool = False) -> dict:
    return {"n_atoms": n_atoms, "n_orbitals": n_orbitals, "smeared": smeared, "hits": hits}


def test_sample_hits_map_global_atoms_and_tokens_onto_fragments():
    """Two fragments of 3 and 4 atoms, 5 and 6 orbitals: global atom 4 is atom 1 of the
    second, global token 7 + 5 + 2 is its orbital 2."""
    name = "tau_atom=0.1"
    second = {"input": (2,), "tokens": (2,), "strict": (1,), "broad": (1, 2)}
    mols = {"a": _mol(3, 5, {name: {}}), "b": _mol(4, 6, {name: second})}
    ref = tr.SampleRef("s", ("a", "b"), (0, 4), (7 + 5 + 2,), (7,), True, False)
    hits = tr.sample_hits(ref, mols, name)
    assert hits["samples"] and hits["anchor_samples"] and hits["anchor_samples_strict"]
    assert hits["label_token_samples"] and not hits["label_mask_samples"]
    off = ref._replace(anchor_atoms=(0, 5), donors=(7 + 2,))  # atom 2 of b; orbital 2 of a
    hits = tr.sample_hits(off, mols, name)
    assert hits["anchor_samples"] and not hits["anchor_samples_strict"]
    assert not hits["label_token_samples"]
    flag = {"smear_tol=0.025": {"flag": (0, 1), "tokens": (0, 1), "molecule": True}}
    mols = {"a": _mol(3, 5, {}), "b": _mol(4, 6, flag)}
    assert tr.sample_hits(ref, mols, "smear_tol=0.025")["samples_flag_flipped"]


def test_occupation_record_reads_the_verified_record(fixed, tmp_path):
    t, rec = fixed["acrolein"]
    save_record(rec, cache_path(tmp_path, t.smiles, t.conformer_id, t.method))
    light = tr.occupation_record(t, tmp_path)
    assert np.array_equal(light.occupations, rec.occupations)
    assert np.array_equal(light.orbital_energies_ev, rec.orbital_energies_ev)
    assert light.unpaired == rec.unpaired
    with pytest.raises(FileNotFoundError):
        tr.occupation_record(t, tmp_path / "elsewhere")


def test_the_report_counts_the_fixed_samples_and_is_stamped(fixed, reach, tmp_path):
    names = ("ethanol", "acetone", "acrolein")
    mols = {fixed[n][0].smiles: {"key": n, "skip": "", **reach[n]} for n in names}
    keys = [fixed[n][0].smiles for n in names]
    parts = {
        "train": [tr.SampleRef("s1", (keys[0], keys[1]), (0, 3), (), (), False, False)],
        "val": [tr.SampleRef("s2", (keys[2],), (0, 1), (), (), False, False)],
        "test": [],
    }
    body = {"source": "fixed", "split": "molecule"} | tr.reach_report(parts, mols)
    assert tr.self_check_failures(body) == []
    merge = next(r for r in body["reach"] if r["value"] == 0.01)
    assert merge["parts"]["train"]["molecules"] == 1  # acetone
    assert merge["parts"]["all"]["molecules"] == 2  # and acrolein
    assert body["totals"]["all"]["samples"] == 2
    args = argparse.Namespace(split="molecule", limit=0)
    out = tr.write_report(tmp_path / "fixed" / tr.REPORT_NAME, body, {}, args)
    header = prov.read_header(out)
    assert header.kind == "threshold_reach" and header.schema_version == tr.REACH_VERSION
    assert json.loads(out.read_text())["reach"][0]["threshold"] == "tau_atom"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.realdata
def test_the_real_report_is_written_and_no_other_artifact_changes(tmp_path):
    from ksmech.arrows.steps import PARTS, samples_path

    views = [samples_path("pmechdb", f"molecule_{p}") for p in PARTS]
    if not all(v.exists() for v in views):
        pytest.skip("PMechDB molecule views missing")
    index = views[0].parent / "samples" / "index.json"
    before = {p: _sha(p) for p in (*views, index)}
    args = ["--source", "pmechdb", "--limit", "10", "--workers", "1", "--out-dir", str(tmp_path)]
    assert tr.main(args) == 0
    report = json.loads((tmp_path / "pmechdb" / tr.REPORT_NAME).read_text())
    assert report["provenance"]["kind"] == "threshold_reach"
    assert tr.self_check_failures(report) == [] and report["sample_flag_mismatches"] == 0
    assert {p: _sha(p) for p in before} == before
