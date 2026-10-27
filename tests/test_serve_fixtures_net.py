"""Fixtures of the network, chemistry and job tests (w2-runtime-jobs): toy chemistries, a
`LocalJobContext` factory and a fake scheduler that returns canned `TaskResult`s.

Import them with the repo's pattern:

    from tests.test_serve_fixtures_net import fake_scheduler, job_ctx  # noqa: F401 (fixtures)

Everything here is pure: no RDKit-heavy work, no worker process, never xTB.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable

import pytest

from ksmech.serve.compute.worker import TaskHandle, TaskResult, WorkerStatus
from ksmech.serve.core.contracts import LocalJobContext
from ksmech.serve.core.testing import StubChemistry, ToyRule, toy_catalytic_cycle, toy_keto_enol
from tests.test_serve_fixtures import no_xtb, serve_settings  # noqa: F401 (fixtures)

JOB_ID = "j" + "0" * 16

Response = TaskResult | Callable[[dict], TaskResult] | list


def ok(result: dict | None = None) -> TaskResult:
    return TaskResult(True, result or {}, None, None, "")


def fail(code: str, detail: str = "", upstream: str | None = None, result=None) -> TaskResult:
    return TaskResult(False, result, code, upstream, detail or f"{code} (canned)")


class FakeScheduler:
    """A `Scheduler` returning canned results by task name: a `TaskResult`, a callable
    payload -> TaskResult, or a list consumed in order. Records every call."""

    def __init__(self, responses: dict[str, Response] | None = None) -> None:
        self.responses: dict[str, Response] = dict(responses or {})
        self.calls: list[dict] = []
        self.killed: list[str] = []

    def submit(self, name, payload, *, priority, timeout_s, owner="", key=None, label=""):
        self.calls.append(
            dict(
                name=name,
                payload=payload,
                priority=priority,
                timeout_s=timeout_s,
                owner=owner,
                key=key,
                label=label,
            )
        )
        r = self.responses.get(name)
        if isinstance(r, list):
            r = r.pop(0) if r else None
        elif callable(r):
            r = r(payload)
        if r is None:
            r = fail("UNEXPECTED", f"no canned result for task {name}")
        handle = TaskHandle(f"fake-{len(self.calls)}", label)
        handle.set_result(r)
        return handle

    def run(self, name, payload, *, priority, timeout_s, owner="", label=""):
        h = self.submit(
            name, payload, priority=priority, timeout_s=timeout_s, owner=owner, label=label
        )
        return h.wait(None)

    def names(self) -> list[str]:
        return [c["name"] for c in self.calls]

    def find(self, key: str):
        return None

    def status(self) -> WorkerStatus:
        return WorkerStatus(True, 1, [], False, None, None, None, 0, 0, False)

    def kill_owner(self, owner: str) -> bool:
        self.killed.append(owner)
        return True

    def close(self) -> None:
        return None


def rule(name: str, consumes: str, produces: str, score: float, form: str = "") -> ToyRule:
    """A toy rule from dot-joined species lists ("" for nothing)."""

    def counter(s: str) -> Counter:
        return Counter(x for x in s.split(".") if x)

    return ToyRule(name, counter(consumes), counter(produces), score, form)


def branching(**kw) -> StubChemistry:
    """Two levels, so the global beam of the second layer prunes:

    - 0 A -> B (1.0), 1 A -> C (0.9), 7 A -> X (-3.0: a beam of 2 never scans it);
    - 2 B -> E (1.0), 3 B -> F (0.5); 4 C -> E in another Lewis form (0.6, an alias),
      5 C -> G (0.2); 6 G -> H (1.0).

    With beam 2 the second layer keeps E (via B) and F, and G is PRUNED."""
    return StubChemistry(
        [
            rule("ab", "A", "B", 1.0),
            rule("ac", "A", "C", 0.9),
            rule("be", "B", "E", 1.0),
            rule("bf", "B", "F", 0.5),
            rule("ce", "C", "E", 0.6, form="alt"),
            rule("cg", "C", "G", 0.2),
            rule("gh", "G", "H", 1.0),
            rule("ax", "A", "X", -3.0),
        ],
        **kw,
    )


@pytest.fixture
def fake_scheduler() -> FakeScheduler:
    return FakeScheduler()


@pytest.fixture
def job_ctx(serve_settings):  # noqa: F811
    """A factory make(job_id=JOB_ID, scheduler=None, **settings_overrides) -> LocalJobContext
    on a tmp data root."""

    def make(job_id: str = JOB_ID, scheduler=None, **overrides) -> LocalJobContext:
        settings = serve_settings(**overrides)
        return LocalJobContext(job_id, settings, scheduler or FakeScheduler())

    return make


@pytest.fixture
def toy_cycle() -> StubChemistry:
    return toy_catalytic_cycle()


@pytest.fixture
def keto_enol() -> StubChemistry:
    return toy_keto_enol()


# ---- sanity tests of the fixtures themselves


def test_fake_scheduler_records_and_replays():
    sched = FakeScheduler({"ping": ok({"pid": 1}), "seq": [ok({"n": 1}), fail("WORKER_CRASH")]})
    assert sched.run("ping", {}, priority=0, timeout_s=1.0).result == {"pid": 1}
    assert sched.run("seq", {}, priority=1, timeout_s=1.0, owner="j1").ok
    r = sched.run("seq", {}, priority=1, timeout_s=1.0)
    assert not r.ok and r.code == "WORKER_CRASH"
    assert sched.run("other", {}, priority=1, timeout_s=1.0).code == "UNEXPECTED"
    assert sched.names() == ["ping", "seq", "seq", "other"]
    assert sched.calls[1]["owner"] == "j1"
    assert sched.kill_owner("j1") and sched.killed == ["j1"]


def test_job_ctx_factory(job_ctx, tmp_path):
    ctx = job_ctx(chemistry="synthetic")
    assert ctx.job_id == JOB_ID and ctx.settings.data_root == tmp_path.resolve()
    ctx.progress(phase="expand", done=1)
    ctx.skip("RESONANCE_ONLY_STEP", "x", node="p1")
    assert ctx.progress_log == [{"phase": "expand", "done": 1}]
    assert ctx.skipped["RESONANCE_ONLY_STEP"] == 1 and not ctx.cancelled()


def test_toy_chemistries_are_well_formed(toy_cycle, keto_enol):
    from ksmech.serve.chem.pools import SpeciesInput

    root = toy_cycle.root([SpeciesInput("Cat", 1), SpeciesInput("S", 2)])
    assert root.pool_key == "Cat.S.S" and root.mapped_smiles == "Cat.S.S"
    assert root.state_id.startswith("s") and len(root.state_id) == 21
    assert [r.name for r in keto_enol.rules] == ["enolize", "ketonize"]
    assert branching().rules[4].form == "alt"


def test_toy_step_encodes_the_rule():
    from ksmech.serve.core.testing import toy_step

    step = toy_step(2, rule("x", "A.B", "C", 1.0))
    assert step["matrix"] == {"bonds": [[0, 1, 3]], "lone": []}
    assert step["sigma_keys"] == ["C"] and step["anchor"]["sign"] == "form"
