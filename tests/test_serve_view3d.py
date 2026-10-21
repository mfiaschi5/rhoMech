"""view3d.py over the synthetic ApiContext (InlineScheduler): geometry, orbital cubes,
depictions (with and without tokens), the verified contact view and its cubes."""

import pytest

from ksmech.serve import schemas, view3d
from ksmech.serve.errors import Code, ServeError
from tests.test_serve_fixtures import (  # noqa: F401 (fixtures)
    no_xtb,
    sn2_state,
    synthetic_context,
    synthetic_root,
    synthetic_settings,
)

pytestmark = pytest.mark.serve


@pytest.fixture(scope="module")
def sn2(synthetic_context, sn2_state):  # noqa: F811
    """(state, rt, analysis, donor, donor_site, acceptor, acceptor_site) of the SN2 tuple."""
    an = synthetic_context.ensure_analysis(sn2_state)
    rt = synthetic_context.reactant_tokens(sn2_state)
    keys = [f.key for f in rt.fragments]
    f, g = keys.index("C[O-]"), keys.index("CCl")
    t_F, t_G = rt.fragments[f].tokens, rt.fragments[g].tokens

    def first(t, kind, elements, n=None):
        for k, o in enumerate(t.orbitals):
            atoms = list(o.descriptors.all_center_atoms)
            same = o.kind.value == kind and {t.symbol(int(a)) for a in atoms} == elements
            if same and (n is None or len(atoms) == n):
                return k
        raise AssertionError(elements)

    donor = rt.n_atoms + int(rt.orbital_offsets[f]) + first(t_F, "occupied", {"O"}, 1)
    acceptor = rt.n_atoms + int(rt.orbital_offsets[g]) + first(t_G, "virtual", {"C", "Cl"})
    i, j = an.cell(donor, acceptor)
    ds, as_ = (int(x) for x in an.contact_sites[i, j])
    return sn2_state, rt, an, donor, ds, acceptor, as_


def test_geometry_out(synthetic_context, sn2):  # noqa: F811
    state, rt, *_ = sn2
    out = view3d.geometry_out(synthetic_context, state.state_id)
    assert isinstance(out, schemas.GeometryOut) and out.state_id == state.state_id
    assert [s.species_key for s in out.species] == ["C[O-]", "CCl"]
    assert sum(len(s.atoms) for s in out.species) == rt.n_atoms
    assert out.species[1].display_offset[0] > 0.0
    with pytest.raises(ServeError) as err:
        view3d.geometry_out(synthetic_context, "s" + "0" * 20)
    assert err.value.code == Code.POOL_NOT_FOUND and err.value.status == 404


def test_orbital_cube_text(synthetic_context, sn2):  # noqa: F811
    state, rt, an, donor, *_ = sn2
    text = view3d.orbital_cube_text(synthetic_context, state.state_id, donor)
    assert text.startswith(f"ksmech-serve cube v1 state={state.state_id} token={donor}")
    again = view3d.orbital_cube_text(synthetic_context, state.state_id, donor)
    assert again == text  # read from the cache, no second task
    for bad in (0, rt.n_tokens):
        with pytest.raises(ServeError) as err:
            view3d.orbital_cube_text(synthetic_context, state.state_id, bad)
        assert err.value.code == Code.NOT_AN_ORBITAL_TOKEN
    with pytest.raises(ServeError) as err:
        view3d.orbital_cube_text(synthetic_context, state.state_id, donor, spacing=0.0)
    assert err.value.code == Code.REQUEST_INVALID
    with pytest.raises(ServeError) as err:
        view3d.orbital_cube_text(synthetic_context, state.state_id, donor, spacing=0.01)
    assert err.value.code == Code.CUBE_TOO_LARGE and err.value.status == 422


def test_depiction_out_with_and_without_tokens(synthetic_context, sn2):  # noqa: F811
    from ksmech.serve.pools import SpeciesInput, initial_state, save_state

    state, rt, *_ = sn2
    dep = view3d.depiction_out(synthetic_context, state.state_id, keep_maps=set(), server_svg=True)
    assert isinstance(dep, schemas.DepictionOut)
    assert all(a.global_site is not None for a in dep.atoms)
    assert dep.server_svg and len(dep.server_coords) == len(dep.atoms)
    bare = initial_state([SpeciesInput("CCCC", 1)], synthetic_context.settings.limits)
    save_state(bare, synthetic_context.settings)
    assert not synthetic_context.loader.has(bare.fragment_token_keys[0])
    out = view3d.depiction_out(synthetic_context, bare.state_id, keep_maps={1}, server_svg=False)
    assert len(out.atoms) == 4 and all(a.global_site is None for a in out.atoms)
    assert out.server_svg is None and out.server_coords is None


def test_contact_out_is_consistent_with_cube_urls(synthetic_context, sn2):  # noqa: F811
    state, rt, an, donor, ds, acceptor, as_ = sn2
    out = view3d.contact_out(synthetic_context, state.state_id, donor, ds, acceptor, as_)
    assert isinstance(out, schemas.ContactOut)
    assert out.consistent and out.is_contact_argmax and not out.same_fragment
    assert out.checks["consistent"] is True
    base = f"/api/contact/{state.state_id}/{donor}/{ds}/{acceptor}/{as_}/cube"
    assert out.cube_urls == {"donor": f"{base}/donor", "acceptor": f"{base}/acceptor"}
    assert out.cubes is None and out.iso_default == 0.05
    assert out.provenance.chemistry == "synthetic"
    i, j = an.cell(donor, acceptor)
    assert out.ks3_site == pytest.approx(float(an.ks3[i, j]), rel=1e-5)
    assert out.acceptor_context is not None
    inline = view3d.contact_out(
        synthetic_context, state.state_id, donor, ds, acceptor, as_, cubes="inline"
    )
    assert set(inline.cubes) == {"donor", "acceptor"}
    for which in ("donor", "acceptor"):
        text = view3d.contact_cube_text(
            synthetic_context, state.state_id, donor, ds, acceptor, as_, which
        )
        assert text == inline.cubes[which] and f"which={which}" in text.splitlines()[0]
    donor_cube = inline.cubes["donor"].splitlines()
    acceptor_cube = inline.cubes["acceptor"].splitlines()
    assert donor_cube[2:6] == acceptor_cube[2:6]  # one shared grid and atom count


def test_contact_errors(synthetic_context, sn2):  # noqa: F811
    state, rt, an, donor, ds, acceptor, as_ = sn2
    with pytest.raises(ServeError) as err:
        view3d.contact_cube_text(synthetic_context, state.state_id, donor, ds, acceptor, as_, "x")
    assert err.value.code == Code.POOL_NOT_FOUND and err.value.status == 404
    with pytest.raises(ServeError) as err:
        view3d.contact_out(synthetic_context, state.state_id, donor, ds, acceptor, as_, cubes="x")
    assert err.value.code == Code.REQUEST_INVALID
    non_centre = next(
        i for i in range(rt.n_atoms) if rt.atom_fragment[i] == 0 and i not in rt.centres(donor)
    )
    with pytest.raises(ServeError) as err:
        view3d.contact_out(synthetic_context, state.state_id, donor, non_centre, acceptor, as_)
    assert err.value.code == Code.SITE_NOT_A_CENTRE and err.value.status == 422
    with pytest.raises(ServeError) as err:
        view3d.contact_out(synthetic_context, state.state_id, acceptor, as_, donor, ds)
    assert err.value.code == Code.NOT_AN_ORBITAL_TOKEN


def test_contact_runs_through_the_scheduler(synthetic_context, sn2):  # noqa: F811
    """A non-contact site pair goes through the `contact` task of the InlineScheduler."""
    state, rt, an, donor, ds, acceptor, as_ = sn2
    other = next(int(x) for x in rt.centres(acceptor) if int(x) != as_)
    before = len(synthetic_context.scheduler.history)
    out = view3d.contact_out(synthetic_context, state.state_id, donor, ds, acceptor, other)
    assert out.consistent and not out.is_contact_argmax
    names = [name for name, _ in synthetic_context.scheduler.history[before:]]
    assert names == ["contact"]


def test_contact_version_pin_on_the_api_side(monkeypatch, synthetic_context, sn2):  # noqa: F811
    import ksmech.serve.versions as versions

    state, rt, an, donor, ds, acceptor, as_ = sn2
    monkeypatch.setattr(versions, "source_constant", lambda rel, name, *a: 6)
    with pytest.raises(ServeError) as err:
        view3d.contact_out(synthetic_context, state.state_id, donor, ds, acceptor, as_)
    assert err.value.code == Code.PHYSICS_VERSION_MISMATCH and err.value.status == 409
