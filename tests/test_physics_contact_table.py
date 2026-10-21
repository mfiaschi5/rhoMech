"""The contact overlap table: samples, medians, backoff, persistence, kernel configuration."""

import numpy as np
import pytest

from ksmech.orbitals.tokens import OrbitalKind, build_molecule_tokens
from ksmech.physics.contact_kernel import ContactConfig, ContactKernel
from ksmech.physics.contact_table import (
    KINDS,
    LEVELS,
    collect_contacts,
    contact_key,
    load_tables,
    save_tables,
    vdw_sum,
)


def test_samples_are_non_bonded_vdw_contacts(sp_record):
    t = build_molecule_tokens(sp_record("OCCCCCl"))
    samples = collect_contacts(t)
    assert samples and {kind for kind, _, _ in samples} == set(KINDS)
    z_present = {a.atomic_number for a in t.atoms}
    for _kind, (za, la, zb, lb), value in samples:
        assert za in z_present and zb in z_present and la in (0, 1, 2) and lb in (0, 1, 2)
        assert value >= 0
    assert vdw_sum(6, 6) == pytest.approx(3.4, abs=0.01)


def test_key_is_main_centre_element_and_dominant_shell(sp_record):
    t = build_molecule_tokens(sp_record("CCl"))
    z = np.array([a.atomic_number for a in t.atoms])
    keys = {contact_key(o, z) for o in t.virtual()}
    assert (17, 2) in keys  # the chlorine d polarization functions
    assert all(ell == 0 for zz, ell in keys if zz == 1)  # hydrogen has only an s shell


def test_backoff_levels(contact_tables):
    table = contact_tables["occ_virt"]
    some_key = next(k for k, n in table.counts["element_shell"].items() if n >= table.min_count)
    assert table.lookup(some_key)[1] == "element_shell"
    za, la, zb, lb = some_key
    assert table.lookup((za, (la + 1) % 3, zb, lb))[1] in LEVELS  # an unseen shell combination
    value, level = table.lookup((86, 1, 86, 1))  # radon never occurs: shell or global level
    assert level in ("shell", "global") and np.isfinite(value) and value > 0
    assert sum(table.lookups.values()) >= 3


def test_round_trip(contact_tables, tmp_path):
    path = save_tables(contact_tables, tmp_path / "contact.json")
    back = load_tables(path)
    for kind in KINDS:
        assert back[kind].medians == contact_tables[kind].medians
        assert back[kind].counts == contact_tables[kind].counts
        assert back[kind].min_count == contact_tables[kind].min_count
    with pytest.raises(FileNotFoundError, match="contact-table"):
        load_tables(tmp_path / "missing.json")


def test_kernel_configuration(contact_tables, sp_record):
    with pytest.raises(ValueError, match="needs contact tables"):
        ContactKernel(None, ContactConfig(kernel="table"))
    with pytest.raises(ValueError, match="unknown contact kernel"):
        ContactKernel(contact_tables, ContactConfig(kernel="mlp"))  # deleted 2026-10-22
    assert ContactKernel().config.kernel == "oriented"  # the default needs no table
    with pytest.raises(ValueError, match="unknown"):
        ContactKernel(contact_tables, ContactConfig(kernel="nope"))
    analytic = ContactKernel(None, ContactConfig(kernel="analytic"))  # needs no table
    t = build_molecule_tokens(sp_record("CO"))
    occ = [o for o in t.orbitals if o.kind == OrbitalKind.OCCUPIED]
    assert analytic.amplitude(occ[0], t, t.virtual()[0], t) >= 0
    table = ContactKernel(contact_tables, ContactConfig(kernel="table"))
    assert table.amplitude(occ[0], t, occ[1], t) >= 0  # occupied-occupied uses its own table


def test_tables_from_token_files_count_what_they_skip(sp_record, tmp_path):
    from ksmech.orbitals.tokens import save_tokens
    from ksmech.physics.contact_table import tables_from_paths

    t = build_molecule_tokens(sp_record("OCCCCCl"))
    good = save_tokens(t, tmp_path / "a.npz")
    tables, skipped = tables_from_paths([good, tmp_path / "missing.npz"], t.method, min_count=1)
    assert skipped == {"missing_tokens": 1} and tables["occ_virt"].n_molecules == 1
    _, other = tables_from_paths([good], "another-method")
    assert other == {"other_method": 1}
    assert tables["occ_virt"].counts["global"][()] == sum(
        kind == "occ_virt" for kind, _, _ in collect_contacts(t)
    )
