"""JobService on StubChemistry with an InlineScheduler context and an inline runner:
predict -> get -> a fully typed NetworkOut with provenance, since_revision, list, cancel,
override (corrected and superseded edges, revision bump), pathways, network_of (404 / 409)."""

import re
import threading
import time

import pytest

from ksmech.serve.core.errors import ServeError
from ksmech.serve.core.schemas import (
    EdgeOut,
    JobOut,
    NetworkOut,
    PathwayOut,
    PredictIn,
)
from ksmech.serve.core.testing import requires
from ksmech.serve.predict.jobs import JobRunner, JobStore
from ksmech.serve.predict.service import JobService
from ksmech.serve.runtime.context import build_context
from tests.test_serve_fixtures_net import (  # noqa: F401 (fixtures)
    FakeScheduler,
    branching,
    no_xtb,
    serve_settings,
)

pytestmark = [pytest.mark.serve, requires("predict")]

SPEC_RE = re.compile(r"^v9\+[0-9a-f]{12};tok\d+;smp\d+;qm\d+$")
ENCODED = {
    "base_smiles": "[CH3:1][O-:2]",
    "products_smiles": "[CH3:1][OH:2]",
    "arrows": [
        {"source_kind": "LONE_PAIR", "source_atoms": [2], "sink_kind": "ATOM", "sink_atoms": [3]}
    ],
    "code": "2=3",
}


@pytest.fixture
def service(serve_settings):  # noqa: F811
    """factory make(chem=None, inline=True, scheduler=None) -> JobService."""
    made = []

    def make(chem=None, inline=True, scheduler=None):
        s = serve_settings(chemistry="synthetic")
        ctx = build_context(s, scheduler=scheduler)
        runner = JobRunner(JobStore(s.jobs, s), s, ctx.scheduler, inline=inline)
        made.append(runner)
        return JobService(ctx, runner, chem or branching())

    yield make
    for runner in made:
        runner.close()


def predict_body(**kw):
    kw = {"reactants_smiles": ["A"], "predictor": "floor", "beam": 2, "max_steps": 2} | kw
    return PredictIn(**kw)


def edge_by_donor(net: NetworkOut, donor: int) -> EdgeOut:
    return next(e for e in net.edges if e.matrix.bonds[0][2] == donor + 1)  # testing.toy_step


def test_predict_get_network_out(service):
    svc = service()
    created = svc.predict(predict_body(conditions="MeOH, 25 C"))
    assert created.kind == "predict" and created.status_url == f"/api/jobs/{created.job_id}"
    job = svc.get(created.job_id)
    assert isinstance(job, JobOut) and job.status == "done" and job.result_kind == "network"
    net = job.result
    assert isinstance(net, NetworkOut) and not job.partial and job.network_id == job.job_id
    assert net.network_id == job.job_id and net.complete and net.request.conditions
    prov = job.provenance
    assert prov.predictor == "floor" and prov.predictor_version.startswith("floor/")
    assert prov.chemistry == "stub" and SPEC_RE.match(prov.spec_version)
    assert net.provenance.predictor_version == prov.predictor_version
    assert created.provenance == prov  # the 202 already names predictor, version and spec
    assert net.params.temperature == 1.0 and net.params.max_scan == 64
    root = next(n for n in net.nodes if n.is_root)
    assert root.id == net.root and root.display_smiles == "A" and root.species[0].count == 1
    assert root.state_id.startswith("s") and root.status == "expanded"
    assert {n.status for n in net.nodes} == {"expanded", "terminal", "pruned"}
    e = edge_by_donor(net, 0)
    assert (e.source, e.target_pool_key, e.product_smiles) == (net.root, "B", ["B"])
    assert e.source_mapped_smiles == "A" and e.product_mapped_smiles == "B"
    assert e.arrows == [] and e.pmechdb_code is None and e.touched_maps == []
    assert net.pathways and net.pathways[0].rank == 1
    text = job.model_dump_json()
    assert "NaN" not in text and "Infinity" not in text


def test_since_revision_unchanged(service):
    svc = service()
    job_id = svc.predict(predict_body()).job_id
    rev = svc.get(job_id).result.revision
    same = svc.get(job_id, since_revision=rev)
    assert same.unchanged and same.result is None and same.result_kind == "network"
    assert same.provenance is not None
    older = svc.get(job_id, since_revision=rev - 1)
    assert not older.unchanged and older.result.revision == rev


def test_list_cancel_and_running(service):
    svc = service()
    a = svc.predict(predict_body()).job_id
    b = svc.predict(predict_body(beam=1)).job_id
    listed = svc.list()
    assert [j.job_id for j in listed] == [b, a]  # newest first
    assert all(j.kind == "predict" and j.label.startswith("floor: ") for j in listed)
    assert svc.list(kind="override") == [] and len(svc.list(limit=1)) == 1
    assert svc.list(status="done") and svc.list(status="queued") == []
    assert svc.cancel(a).status == "done"  # finished: unchanged
    assert svc.running() == 0


def test_pathways(service):
    svc = service()
    job_id = svc.predict(predict_body()).job_id
    paths = svc.pathways(job_id)
    assert paths and all(isinstance(p, PathwayOut) for p in paths)
    assert [p.rank for p in paths] == list(range(1, len(paths) + 1))
    assert len(svc.pathways(job_id, include_pruned=True)) == len(paths) + 1  # A, C, G
    assert len(svc.pathways(job_id, limit=1)) == 1


def test_network_of(service):
    svc = service()
    body = predict_body()
    job_id = svc.predict(body).job_id
    net, request = svc.network_of(job_id)
    assert isinstance(net, NetworkOut) and request == body.model_dump(mode="json")
    with pytest.raises(ServeError) as e:
        svc.network_of("j" + "5" * 16)
    assert (e.value.code, e.value.status) == ("JOB_NOT_FOUND", 404)
    failed = svc.runner.submit("predict", body.model_dump(mode="json") | {"predictor": "nope"})
    assert failed.status == "failed" and failed.error["code"] == "PREDICTOR_UNKNOWN"
    with pytest.raises(ServeError) as e:
        svc.network_of(failed.id)
    assert (e.value.code, e.value.status) == ("JOB_NOT_DONE", 409)
    with pytest.raises(ServeError) as e:
        svc.pathways(failed.id)
    assert e.value.code == "JOB_NOT_DONE"


def test_predict_preflight_errors(service):
    svc = service()
    cases = [
        (predict_body(predictor="network"), "PREDICTOR_UNAVAILABLE", 409),
        (predict_body(predictor="nope"), "PREDICTOR_UNKNOWN", 422),
        (predict_body(stoichiometry=[1, 2]), "STOICHIOMETRY_MISMATCH", 422),
        (predict_body(reactants_smiles=["."]), "EMPTY_INPUT", 422),
    ]
    for body, code, status in cases:
        with pytest.raises(ServeError) as e:
            svc.predict(body)
        assert (e.value.code, e.value.status) == (code, status)
    assert svc.list() == []  # nothing was queued


def test_current_task_of_a_running_job(service):
    gate = threading.Event()

    class Current(FakeScheduler):
        def current(self):
            return {
                "name": "expand",
                "label": "analysis and expansion of A",
                "elapsed_s": 1.5,
                "queue_depth": 2,
                "priority": "JOB",
                "owners": list(self.owners),
            }

    sched = Current()
    sched.owners = []
    svc = service(inline=False, scheduler=sched)
    svc.runner.register("index", lambda ctx, req: gate.wait(10.0) and {}, thread="background")
    job = svc.runner.submit("index", {})
    try:
        t0 = time.monotonic()
        while svc.runner.get(job.id).status != "running" and time.monotonic() - t0 < 5:
            time.sleep(0.01)
        sched.owners = [job.id]
        task = svc.get(job.id).progress.current_task
        assert task.name == "expand" and task.queue_depth == 2 and task.priority == "JOB"
        sched.owners = ["j" + "7" * 16]
        assert svc.get(job.id).progress.current_task is None  # another job's task
    finally:
        gate.set()
