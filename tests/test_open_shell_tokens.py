"""Real data: the open-shell token rule of P2b (open-shell plan) on the PMechDB records.

Before the rebuild: a random 500 of the closed-shell token files are rebuilt in memory with
today's identical content hash, and every open-shell record keeps its plateau reference for any
tolerance from 1e-13 to 1e-3 e. After `ksmech-orbitals build --source pmechdb` (the copy of the
summary taken before it is `p2b_reference/tokens_summary.json`): every open-shell token file is
current, none is smeared, and the smeared list lost exactly the open shells the rule made
integral.
"""

import json
import random
from pathlib import Path

import numpy as np
import pytest

from ksmech.data.io import DEFAULT_PROCESSED_DIR

pytestmark = pytest.mark.realdata

TOKENS = Path("data/cache/tokens")
SUMMARY = DEFAULT_PROCESSED_DIR / "pmechdb" / "tokens_summary.json"
BEFORE = DEFAULT_PROCESSED_DIR / "pmechdb" / "p2b_reference" / "tokens_summary.json"
MADE_INTEGRAL = {"[C]=O", "[CuH2]", "[O][O-]"}  # measured 2026-10-31 on the 44 open shells


def _molecules():
    from ksmech.orbitals.tokens import drawn_unpaired
    from ksmech.qm.batch import collect_molecules

    smiles, _ = collect_molecules("pmechdb", DEFAULT_PROCESSED_DIR)
    if not smiles:
        pytest.skip("no PMechDB steps")
    return [s for s in smiles if drawn_unpaired(s) == 0], [s for s in smiles if drawn_unpaired(s)]


def _paths(smiles: str):
    from ksmech.orbitals.build import tokens_path
    from ksmech.qm.records import DEFAULT_CACHE_DIR, cache_path
    from ksmech.qm.xtb_calc import QMConfig

    method = QMConfig().method
    return cache_path(DEFAULT_CACHE_DIR, smiles, 0, method), tokens_path(TOKENS, smiles, 0, method)


def same_closed_shell_payload(smiles: str) -> tuple[str, bool | None]:
    """(smiles, rebuilt digest == file digest), None without a record or a token file."""
    from ksmech.data import provenance as prov
    from ksmech.orbitals.tokens import build_molecule_tokens, token_digest
    from ksmech.qm.records import is_valid_cache_file, load_record

    record, tokens = _paths(smiles)
    if not (is_valid_cache_file(record) and tokens.is_file()):
        return smiles, None
    rebuilt = token_digest(build_molecule_tokens(load_record(record)))
    return smiles, rebuilt == prov.read_header(tokens).content_sha256


def test_closed_shell_token_payloads_are_unchanged(realdata_workers):
    from ksmech.processes import in_processes

    closed, _ = _molecules()
    sample = random.Random(0).sample(closed, 500)
    rows = list(in_processes(same_closed_shell_payload, sample, realdata_workers, chunksize=8))
    checked = [s for s, same in rows if same is not None]
    assert len(checked) >= 490, f"only {len(checked)} of 500 have a record and a token file"
    assert [s for s, same in rows if same is False] == []


def test_open_shell_references_hold_for_every_tolerance():
    from ksmech.orbitals.localize import reference_occupations
    from ksmech.orbitals.tokens import drawn_unpaired
    from ksmech.qm.records import is_valid_cache_file, load_record

    _, radicals = _molecules()
    tols = [10.0**k for k in range(-13, -2)]
    for smiles in radicals:
        path, _ = _paths(smiles)
        if not is_valid_cache_file(path):
            continue
        rec = load_record(path)
        assert rec.unpaired == drawn_unpaired(smiles), smiles
        ref = reference_occupations(rec)
        assert all(np.array_equal(reference_occupations(rec, t), ref) for t in tols), smiles


def test_the_rebuild_made_every_open_shell_file_current_and_integral():
    from ksmech.data import provenance as prov
    from ksmech.orbitals.tokens import OPEN_SHELL_RULE, load_tokens

    if not BEFORE.is_file():
        pytest.skip(f"{BEFORE} is missing (copy the summary before the P2b rebuild)")
    after, before = json.loads(SUMMARY.read_text()), json.loads(BEFORE.read_text())
    _, radicals = _molecules()
    for smiles in radicals:
        _, path = _paths(smiles)
        t = load_tokens(path)
        assert not t.fermi_smeared, smiles
        assert prov.read_header(path).config["open_shell_rule"] == OPEN_SHELL_RULE
    op = after["open_shell"]
    assert op["n_molecules"] == len(radicals) and op["n_fermi_smeared"] == 0
    old, new = set(before["fermi_smeared"]["smiles"]), set(after["fermi_smeared"]["smiles"])
    assert new <= old and old - new == MADE_INTEGRAL
