"""pmindex.py and browse.py: the on-request PMechDB index over the mini fixture, curated steps
read live from the curation tables, exact and substructure search, filters and paging,
staleness, INDEX_NOT_READY, the build-job dedup, the step view and the export's pairs."""

import json

import pytest

from ksmech.serve.browse import pmindex
from ksmech.serve.browse.service import BrowseIndex, BrowseService, parse_query
from ksmech.serve.compute.worker import InlineScheduler
from ksmech.serve.core import db, keys, schemas
from ksmech.serve.core.contracts import LocalJobContext
from ksmech.serve.core.errors import Code, ServeError
from ksmech.serve.core.settings import TaskEnv
from tests.test_serve_fixtures import (  # noqa: F401 (fixtures)
    no_xtb,
    serve_mini_processed,
    serve_settings,
)

pytestmark = pytest.mark.serve

SN2_DOC = {
    "format": "ksmech-serve/step-v1",
    "job_id": "j0123456789abcdef",
    "edge_id": "e0123456789abcdef",
    "edge_kind": "predicted",
    "reactant_state_id": "s" + "1" * 20,
    "product_state_id": "s" + "2" * 20,
    "reactants_mapped": "[C:1]([O-:2])([H:3])([H:4])[H:5].[C:6]([Cl:7])([H:8])([H:9])[H:10]",
    "products_mapped": "[C:1]([O:2][C:6]([H:8])([H:9])[H:10])([H:3])([H:4])[H:5].[Cl-:7]",
    "arrows": [
        {
            "donor": 10,
            "donor_site": 1,
            "acceptor": 28,
            "acceptor_site": 5,
            "donor_site_map": 2,
            "acceptor_site_map": 6,
            "donor_element": "O",
            "acceptor_element": "C",
            "source_kind": "lone_pair",
            "source_partner_map": None,
            "sink_partner_map": 7,
            "donor_centres_map": {"2": 1.0},
            "acceptor_centres_map": {"6": 0.66, "7": 0.34},
        }
    ],
    "dataset_arrows": [
        {"source_kind": "lone_pair", "source_atoms": [2], "sink_kind": "atom", "sink_atoms": [6]},
        {"source_kind": "bond", "source_atoms": [6, 7], "sink_kind": "atom", "sink_atoms": [7]},
    ],
    "pmechdb_code": "2=6;6,7=7",
    "encode_reason": None,
    "score": 1.0,
    "probability": 0.5,
    "rank": 1,
    "predictor": "stub",
    "predictor_version": "stub/x",
    "spec_version": "v8+000000000000;tok9;smp15;ori6;qm1",
    "physics_fingerprint": "000000000000",
    "touched_maps": [2, 6, 7],
}
# an enolate drawn as its carbanion: one species (sigma) key, token key [CH2-]C(C)=O
ENOLATE = "[C:1](=[O:2])([C:3]([H:4])([H:5])[H:6])[C-:7]([H:8])[H:9]"
ENOLATE_DOC = SN2_DOC | {
    "reactants_mapped": ENOLATE,
    "products_mapped": ENOLATE,
    "arrows": [],
    "dataset_arrows": None,
    "pmechdb_code": None,
}


def _pk(smiles: str) -> str:
    """The pool key (sigma keys, `keys.side_keys`) of a reaction side."""
    return keys.side_keys(smiles).pool_key


def _sk(smiles: str) -> str:
    """The species key (the sigma key) of one species."""
    return keys.side_keys(smiles).species_keys[0]


def _mechanism(ses, status: str, steps: list[tuple[str, str, dict]]) -> int:
    m = db.Mechanism(
        reactants=json.dumps(["C[O-]", "CCl"]),
        stoichiometry=json.dumps([1, 1]),
        predictor="stub",
        checkpoint_hash="stub/x",
        spec_version="v8+000000000000;tok9;smp15;ori6;qm1",
        status=status,
    )
    ses.add(m)
    ses.flush()
    for i, (rpk, ppk, doc) in enumerate(steps):
        ses.add(
            db.CuratedStep(
                mechanism_id=m.id,
                step_index=i,
                order_group=i,
                reactant_pool_key=rpk,
                product_pool_key=ppk,
                arrows_json=json.dumps(doc),
            )
        )
    return int(m.id)


@pytest.fixture
def env(serve_settings, serve_mini_processed, tmp_path):  # noqa: F811
    """(settings, ctx, engine, mechanism ids, submitted calls, service) with an inline
    submitter that runs the index job before returning, as the inline JobRunner does."""
    from ksmech.serve.runtime.context import build_context

    s = serve_settings(chemistry="synthetic")
    ctx = build_context(s)
    engine = db.make_engine(tmp_path / "curation" / "curation.sqlite")
    db.init_db(engine)
    with db.session(engine) as ses:
        sn2 = (_pk("CCl.C[O-]"), _pk("COC.[Cl-]"), SN2_DOC)
        validated = _mechanism(ses, "validated", [sn2])
        enolate = (_pk("C=C(C)[O-]"), _pk("C=C(C)[O-]"), ENOLATE_DOC)
        rejected = _mechanism(ses, "rejected", [enolate])
    calls = []

    def submit(kind: str, request: dict) -> dict:
        job_id = f"j{len(calls) + 1:016x}"
        calls.append((kind, request))
        job = LocalJobContext(job_id, s, InlineScheduler(TaskEnv(s)))
        pmindex.run_index(job, request)
        return {"job_id": job_id, "kind": kind, "status": "done"} | {
            "status_url": f"/api/jobs/{job_id}",
            "parent_job_id": None,
        }

    service = BrowseService(ctx, engine, submit=submit)
    return s, ctx, engine, (validated, rejected), calls, service


def _refs(out: schemas.SearchOut) -> list[str]:
    return [r.ref for r in out.results]


def test_index_not_ready_and_search_never_builds(env):
    s, ctx, engine, (mid, _), calls, service = env
    status = service.status()
    assert status.pmechdb.state == "missing" and status.curated_steps == 2
    assert "POST /api/browse/index" in status.pmechdb.reason
    with pytest.raises(ServeError) as err:
        service.search(q="CCl")
    assert err.value.code == Code.INDEX_NOT_READY and err.value.status == 409
    assert err.value.detail == {"state": "missing", "job_id": None, "build": pmindex.BUILD_HINT}
    assert calls == [] and not s.index.exists()
    out = service.search(q="CCl", sources=["user_curated"])
    assert _refs(out) == [f"curated:{mid}:0"] and out.index.pmechdb.state == "missing"


def test_build_index_inline_and_status(env):
    s, ctx, engine, _, calls, service = env
    created = service.build_index()
    assert created.kind == "index" and calls == [("index", {})]
    path = pmindex.index_path(s, pmindex.source_path(s))
    assert path.is_file() and path.with_suffix(".json").is_file()
    assert path.parent == s.index and path.name.startswith("pmechdb-")
    status = service.status()
    assert status.pmechdb.state == "ready" and status.pmechdb.n_steps == 5
    assert status.pmechdb.built_at is not None and status.pmechdb.job_id is None
    pm = pmindex.load_index(path)
    assert pm.n_steps == 5 and [r["step_id"] for r in pm.rows] == sorted(
        r["step_id"] for r in pm.rows
    )
    badges = {r["split_molecule"] for r in pm.rows}
    assert "resonance" in badges and badges <= {"train", "val", "test", "resonance"}


def _built(env):
    service = env[-1]
    service.build_index()
    return service


def test_exact_search_with_a_resonance_equivalent_query(env):
    service = _built(env)
    pm = pmindex.load_index(service._loaded_index().path)
    resonance = next(r["step_id"] for r in pm.rows if r["resonance_step"])
    # the reactant was drawn as the carbanion; the query uses the enolate form
    out = service.search(q="C=C(C)[O-]", side="reactants", sources=["pmechdb"])
    assert _refs(out) == [f"pmechdb:{resonance}"]
    assert out.results[0].matched_species == [_sk("C=C(C)[O-]")]
    assert out.results[0].matched_forms == []
    as_carbanion = service.search(q="[CH2-]C(C)=O", side="reactants", sources=["pmechdb"])
    assert _refs(as_carbanion) == _refs(out)
    tokens = service.search(q="C=C(C)[O-]", side="reactants", sources=["pmechdb"], match="token")
    assert tokens.total == 0  # the stored token key is the carbanion
    hydroxide = service.search(q="[OH-]", sources=["pmechdb"])
    assert hydroxide.total == 2  # the SN2 and the proton transfer
    two = service.search(q="[OH-].CCl", sources=["pmechdb"])
    assert two.total == 1 and two.results[0].reactant_pool_key == _pk("CCl.[OH-]")


def test_substructure_through_a_non_key_token_form(env):
    service = _built(env)
    _, _, _, (_, rejected), _, _ = env
    out = service.search(q="[CH2-]C=O", mode="substructure", side="reactants")
    assert out.total == 2 and not out.truncated
    pm_hit, curated_hit = out.results
    assert pm_hit.source == "pmechdb" and pm_hit.resonance_step
    assert pm_hit.matched_species == [_sk("C=C(C)[O-]")]
    assert pm_hit.matched_forms == ["[CH2-]C(C)=O"]  # not the species key
    assert curated_hit.ref == f"curated:{rejected}:0" and curated_hit.status == "rejected"
    assert curated_hit.matched_forms == ["[CH2-]C(C)=O"]
    carbonyl = service.search(q="C=O", mode="substructure", sources=["pmechdb"], side="reactants")
    by_ref = {r.ref: r for r in carbonyl.results}
    assert len(by_ref) == 3  # acetaldehyde, acetic acid, and the enolate through its token form
    assert all(f in r.matched_forms[0] for r in by_ref.values() for f in ("=O",))
    two = service.search(q="[OH-].ClC", mode="substructure", sources=["pmechdb"])
    assert two.total == 1 and len(two.results[0].matched_species) == 2
    with pytest.raises(ServeError) as err:
        service.search(q="C1CC(", mode="substructure")
    assert err.value.code == Code.QUERY_INVALID and err.value.status == 422


def test_side_source_and_status_filters(env):
    service = _built(env)
    _, _, _, (validated, rejected), _, _ = env
    products = service.search(q="[Cl-]", side="products")
    assert {r.source for r in products.results} == {"pmechdb", "user_curated"}
    assert service.search(q="[Cl-]", side="reactants").total == 0
    curated = service.search(q="COC", sources=["user_curated"], status="validated")
    assert _refs(curated) == [f"curated:{validated}:0"]
    assert curated.results[0].status == "validated"
    assert curated.results[0].n_arrows == 2  # dataset arrows, as for PMechDB (2=6;6,7=7)
    assert service.search(q="COC", sources=["user_curated"], status="rejected").total == 0
    assert service.search(q="", sources=["user_curated"], status="dataset").total == 0
    dataset = service.search(q="", status="dataset")
    assert dataset.total == 5 and {r.source for r in dataset.results} == {"pmechdb"}
    resonant = service.search(q="", resonance=True)
    assert {r.ref for r in resonant.results} >= {f"curated:{rejected}:0"}
    assert all(r.resonance_step for r in resonant.results)
    assert service.search(q="", resonance=False).total == 5  # 4 PMechDB + the curated SN2
    for bad in ({"mode": "fuzzy"}, {"side": "left"}, {"sources": ["x"]}, {"status": "done"}):
        with pytest.raises(ServeError) as err:
            service.search(q="C", **bad)
        assert err.value.code == Code.QUERY_INVALID
    with pytest.raises(ServeError):
        service.search(q="C", limit=0)
    with pytest.raises(ServeError) as err:
        service.search(q="not a smiles ((")
    assert err.value.code == Code.QUERY_INVALID


def test_paging_and_order(env):
    service = _built(env)
    full = service.search(q="", limit=50)
    assert full.total == 7
    refs = _refs(full)
    pm_ids = [r.step_id for r in full.results if r.source == "pmechdb"]
    assert pm_ids == sorted(pm_ids) and refs[-2:][0].startswith("curated:")
    pages = [_refs(service.search(q="", limit=3, offset=o)) for o in (0, 3, 6)]
    assert [len(p) for p in pages] == [3, 3, 1] and sum(pages, []) == refs
    assert service.search(q="", limit=3, offset=9).results == []


def test_stale_on_a_content_change(env, mini_result):
    from ksmech.data.io import write_steps

    s, _, _, _, calls, service = env
    service = _built(env)
    write_steps(mini_result.steps[:4], pmindex.source_path(s))
    status = service.status()
    assert status.pmechdb.state == "stale" and status.pmechdb.built_at is not None
    with pytest.raises(ServeError) as err:
        service.search(q="CCl")
    assert err.value.detail["state"] == "stale"
    assert len(calls) == 1  # the search started nothing
    service.build_index()
    assert service.status().pmechdb.state == "ready" and service.status().pmechdb.n_steps == 4
    assert len(list(s.index.glob("pmechdb-*.parquet"))) == 1  # the old index was removed


def test_build_index_returns_the_queued_or_running_job(env):
    s, ctx, engine, *_ = env
    calls = []

    def submit(kind, request):
        calls.append(kind)
        return {"job_id": "j" + "a" * 16, "kind": kind, "status": "queued"} | {
            "status_url": "/api/jobs/j" + "a" * 16,
            "parent_job_id": None,
        }

    service = BrowseService(ctx, engine, submit=submit)
    first = service.build_index()
    second = service.build_index()
    assert first.job_id == second.job_id and calls == ["index"]
    status = service.status().pmechdb
    assert status.state == "building" and status.job_id == first.job_id
    key = pmindex.index_key(s)
    pmindex.JOBS.started(key, first.job_id)
    assert service.build_index().status == "running" and calls == ["index"]
    pmindex.JOBS.finished(key, first.job_id, {"code": "WORKER_CRASH", "detail": "x"})
    assert service.status().pmechdb.state == "missing"
    assert "WORKER_CRASH" in service.status().pmechdb.reason
    service.build_index()
    assert calls == ["index", "index"]
    pmindex.JOBS.finished(key, first.job_id)


def test_unavailable_without_the_parquet(serve_settings, tmp_path):  # noqa: F811
    from ksmech.serve.runtime.context import build_context

    s = serve_settings(chemistry="synthetic")
    engine = db.make_engine(tmp_path / "c.sqlite")
    db.init_db(engine)
    service = BrowseService(build_context(s), engine, submit=lambda k, r: pytest.fail("submit"))
    assert service.status().pmechdb.state == "unavailable"
    with pytest.raises(ServeError) as err:
        service.build_index()
    assert err.value.code == Code.INDEX_NOT_READY and err.value.detail["state"] == "unavailable"
    job = LocalJobContext("j" + "b" * 16, s, InlineScheduler(TaskEnv(s)))
    assert service.pmechdb_pairs(job) is None
    assert service.search(q="", sources=["user_curated"]).total == 0


def test_pmechdb_pairs_build_inside_the_calling_job(env, mini_steps):
    s, ctx, engine, _, calls, service = env
    job = LocalJobContext("j" + "c" * 16, s, InlineScheduler(TaskEnv(s)))
    pairs = service.pmechdb_pairs(job)
    assert calls == []  # built inside the job, no index job submitted
    assert pairs[(_pk("CCl.[OH-]"), _pk("CO.[Cl-]"))] == mini_steps["sn2"].step_id
    assert len(pairs) == 5
    assert any(p.get("message") == "indexing 5/5" for p in job.progress_log)
    assert pmindex.index_path(s, pmindex.source_path(s)).is_file()
    again = LocalJobContext("j" + "d" * 16, s, InlineScheduler(TaskEnv(s)))
    assert service.pmechdb_pairs(again) == pairs and again.progress_log == []


def test_skipped_steps_are_counted(serve_settings, mini_result):  # noqa: F811
    from dataclasses import replace

    from ksmech.data.io import write_steps

    s = serve_settings(chemistry="synthetic")
    bad = replace(mini_result.steps[0], step_id="pmechdb-" + "0" * 16, reactants_smiles="C1CC")
    write_steps([*mini_result.steps, bad], pmindex.source_path(s))
    job = LocalJobContext("j" + "e" * 16, s, InlineScheduler(TaskEnv(s)))
    out = pmindex.run_index(job, {})
    assert out["pmechdb"]["state"] == "ready" and out["pmechdb"]["n_steps"] == 5
    assert job.skipped["SPECIES_KEY_FAILED"] == 1
    assert job.skip_log[0]["step_id"] == bad.step_id
    meta = json.loads(
        pmindex.index_path(s, pmindex.source_path(s)).with_suffix(".json").read_text()
    )
    assert meta["skipped"] == {"SPECIES_KEY_FAILED": 1} and meta["n_steps"] == 6
    direct = pmindex.t_index_chunk(None, {"rows": [{"step_id": "x", "reactants_smiles": "C1CC"}]})
    assert direct["rows"] == [] and direct["skipped"] == {"SPECIES_KEY_FAILED": 1}


def test_step_view_pmechdb_and_curated(env, mini_steps):
    service = _built(env)
    _, _, _, (validated, rejected), _, _ = env
    sn2 = mini_steps["sn2"]
    view = service.step(f"pmechdb:{sn2.step_id}")
    assert isinstance(view, schemas.StepViewOut)
    assert view.step.ref == f"pmechdb:{sn2.step_id}" and view.step.status == "dataset"
    assert view.step.splits and view.mechanism is None and view.recheck is None
    assert [a.sink_atoms for a in view.dataset_arrows] == [[20], [21]]
    for dep in (view.reactants_depiction, view.products_depiction):
        assert dep.server_svg and len(dep.server_coords) == len(dep.atoms)
        assert {10, 20, 21} & {a.map for a in dep.atoms}
    curated = service.step(f"curated:{validated}:0")
    assert curated.step.source == "user_curated" and curated.step.status == "validated"
    assert curated.mechanism.status == "validated" and curated.mechanism.n_steps == 1
    assert curated.mechanism.reactants == ["C[O-]", "CCl"] and curated.mechanism.stale
    assert len(curated.dataset_arrows) == 2 and curated.recheck is None
    shown = {a.map for a in curated.reactants_depiction.atoms}
    assert {2, 6, 7} <= shown and not {3, 4, 5, 8, 9, 10} & shown
    no_arrows = service.step(f"curated:{rejected}:0")
    assert no_arrows.dataset_arrows == [] and no_arrows.step.resonance_step
    for bad in ("pmechdb:pmechdb-" + "f" * 16, f"curated:{validated}:9", "nonsense"):
        with pytest.raises(ServeError) as err:
            service.step(bad)
        assert err.value.code == Code.STEP_NOT_FOUND and err.value.status == 404


def test_step_view_before_the_index_and_with_a_recheck(env, mini_steps):
    s, ctx, engine, (validated, _), _, service = env
    view = service.step(f"pmechdb:{mini_steps['e2'].step_id}")  # no index yet
    assert view.step.n_arrows == 3 and view.step.splits is None
    with db.session(engine) as ses:
        ses.add(
            db.Recheck(
                mechanism_id=validated,
                step_index=0,
                predictor="stub",
                predictor_version="stub/x",
                spec_version="v3",
                physics_fingerprint="0" * 12,
                aligned=True,
                first_arrow_rank=1,
                n_valid=12,
                top1=True,
                top5=True,
                product_match="exact",
                top1_reproduces_product=True,
            )
        )
    curated = service.step(f"curated:{validated}:0")
    assert curated.recheck is not None and curated.recheck.first_arrow_rank == 1
    assert curated.mechanism.last_recheck is not None


def test_curated_and_pmechdb_count_dataset_arrows(env, mini_steps):
    """One SN2 shows the same arrow count from both sources: its dataset arrows."""
    service = _built(env)
    _, _, _, (validated, rejected), _, _ = env
    pm = service.search(q="", sources=["pmechdb"]).results
    (pm_sn2,) = [r for r in pm if r.step_id == mini_steps["sn2"].step_id]
    by_ref = {r.ref: r for r in service.search(q="", sources=["user_curated"]).results}
    curated = by_ref[f"curated:{validated}:0"]
    assert curated.n_arrows == pm_sn2.n_arrows == len(SN2_DOC["dataset_arrows"]) == 2
    assert by_ref[f"curated:{rejected}:0"].n_arrows == 0  # neither dataset nor token arrows
    assert service.step(f"curated:{validated}:0").step.n_arrows == 2


def test_substructure_cap_truncates():
    from ksmech.serve.browse.service import StepEntry

    entries = []
    for i in range(30):
        summary = {
            "ref": f"curated:{i + 1}:0",
            "source": "user_curated",
            "status": "validated",
            "mechanism_id": i + 1,
            "step_index": 0,
            "resonance_step": False,
        }
        species = {"reactants": [f"C{'C' * i}O"], "products": []}
        pairs = {"reactants": [(species["reactants"][0],) * 2], "products": []}
        entries.append(StepEntry(summary, species, {"reactants": [], "products": []}, pairs))
    index = BrowseIndex(None, entries, cap=10)
    res = index.search(parse_query("CO", mode="substructure", sources=["user_curated"]))
    assert res["truncated"] and res["total"] == 10 and res["calls"] == 10
    full = BrowseIndex(None, entries).search(parse_query("CO", mode="substructure"))
    assert not full["truncated"] and full["total"] == 30


def test_a_cancelled_queued_job_is_forgotten(env):
    s, ctx, engine, *_ = env
    states = {}

    def submit(kind, request):
        job_id = "j" + "9" * 16
        states[job_id] = "queued"
        return {"job_id": job_id, "kind": kind, "status": "queued"} | {
            "status_url": f"/api/jobs/{job_id}",
            "parent_job_id": None,
        }

    service = BrowseService(ctx, engine, submit=submit, job_status=states.get)
    job = service.build_index()
    assert service.status().pmechdb.state == "building"
    states[job.job_id] = "cancelled"
    status = service.status().pmechdb
    assert status.state == "missing" and "CANCELLED" in status.reason
    assert pmindex.JOBS.active(pmindex.index_key(s)) is None


def test_browse_index_step_detail(env, mini_steps):
    service = _built(env)
    _, _, engine, (validated, _), _, _ = env
    curated = service._curated.entries(engine)
    index = BrowseIndex(service._loaded_index(), curated)
    sn2 = mini_steps["sn2"]
    detail = index.step_detail(f"pmechdb:{sn2.step_id}")
    assert detail["step_id"] == sn2.step_id and detail["keep_maps"] == [10, 20, 21]
    assert [a["source_kind"] for a in detail["arrows"]] == ["lone_pair", "bond"]
    mine = index.step_detail(f"curated:{validated}:0")
    assert mine["status"] == "validated" and mine["keep_maps"] == [2, 6, 7]
    schemas.StepSummaryOut(**{k: v for k, v in mine.items() if k not in ("arrows", "keep_maps")})
    with pytest.raises(ServeError) as err:
        index.step_detail("curated:999:0")
    assert err.value.code == Code.STEP_NOT_FOUND
