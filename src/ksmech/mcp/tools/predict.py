"""Prediction tools: the predictors, predict a mechanism (waits for the job), read a job, check
a reaction matrix, and the readings of one predicted step."""

from __future__ import annotations

from typing import Annotated, Any

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from ksmech.client import KsmechAPIError, KsmechClient, KsmechTimeout
from ksmech.client import summary as sm
from ksmech.mcp.tools import COMPUTE, READ, Enabled, add, blocking, dumps
from ksmech.mcp.tools.chem import Reactants, Stoichiometry
from ksmech.serve.core import schemas as S

JobId = Annotated[str, Field(description="A job id (j + 16 hex) from predict_mechanism.")]
NPathways = Annotated[int, Field(ge=1, le=20, description="Pathways in the summary.")]
StepDetail = Annotated[
    bool,
    Field(description="Per step also the matrix, the orbital readings and the state ids."),
]


def predict_body(fields: dict[str, Any]) -> dict[str, Any]:
    """The `PredictIn` body of the tool's arguments (unset ones left to the server)."""
    run, file, source = (fields.pop(k) for k in ("checkpoint_run", "checkpoint_file", "source"))
    body = {k: v for k, v in fields.items() if v is not None}
    if run:
        body["checkpoint"] = {
            k: v for k, v in (("run", run), ("file", file), ("source", source)) if v
        }
    return body


def _edge(job: S.JobOut, edge_id: str) -> tuple[S.EdgeOut, dict[str, S.NodeOut]]:
    net = job.result
    if not isinstance(net, S.NetworkOut):
        raise KsmechAPIError("JOB_NOT_DONE", f"job {job.job_id} has no network ({job.status})")
    for edge in net.edges:
        if edge.id == edge_id:
            return edge, {n.id: n for n in net.nodes}
    raise KsmechAPIError("EDGE_NOT_FOUND", f"no edge {edge_id} in job {job.job_id}")


class PredictTools:
    """The prediction tools over one client."""

    def __init__(self, client: KsmechClient) -> None:
        self.client = client

    async def list_predictors(self) -> str:
        """The mechanism predictors of this server and whether each is available now:
        'floor' (the physics floor: steps ranked by the Klopman-Salem gap factor, no learned
        parameter), 'gbdt' (a LightGBM ranker), 'network' (the trained transformer; the only
        one whose scores are probabilities). Unavailable ones carry the reason."""
        infos = await blocking(self.client.predictors)
        return dumps([sm.predictor_summary(p) for p in infos])

    async def predict_mechanism(
        self,
        reactants: Reactants,
        stoichiometry: Stoichiometry = None,
        predictor: Annotated[
            str | None, Field(description="floor | gbdt | network (server default: floor).")
        ] = None,
        beam: Annotated[int | None, Field(ge=1, le=10, description="Beam width.")] = None,
        max_steps: Annotated[
            int | None, Field(ge=1, le=12, description="Most elementary steps per pathway.")
        ] = None,
        temperature: Annotated[float | None, Field(gt=0, description="Softmax T.")] = None,
        checkpoint_run: Annotated[
            str | None, Field(description="network only: a run name (default: the best run).")
        ] = None,
        checkpoint_file: Annotated[str | None, Field(description="best | last.")] = None,
        source: Annotated[str | None, Field(description="The checkpoint's data source.")] = None,
        n_pathways: NPathways = 3,
        wait_s: Annotated[
            float, Field(ge=1, le=1800, description="Seconds to wait for the job.")
        ] = 300,
        detail: StepDetail = False,
    ) -> str:
        """Predict the reaction mechanism of reactants as sequences of elementary steps
        (each step: the bonds that form and break and where electron pairs go, read as
        curved arrows). Submits a prediction job (compute; nothing is changed or deleted),
        waits up to wait_s and returns the top pathways: per step the reactant and product
        SMILES, its rank, its probability p (network only; floor and gbdt give a score),
        the anchor bond, the arrows on pool atoms ('lp(3)->3-5': the lone pair of atom 3
        forms bond 3-5; '3-4->lp(4)': bond 3-4's pair becomes a lone pair on 4) and the
        PMechDB arrow code on atom maps. If the job is still running, returns its job_id:
        call get_job."""
        body = predict_body(
            {
                "reactants_smiles": reactants,
                "stoichiometry": stoichiometry,
                "predictor": predictor,
                "beam": beam,
                "max_steps": max_steps,
                "temperature": temperature,
                "checkpoint_run": checkpoint_run,
                "checkpoint_file": checkpoint_file,
                "source": source,
            }
        )
        created = await blocking(self.client.submit_predict, body)
        try:
            job = await blocking(self.client.wait_job, created.job_id, timeout=wait_s)
        except KsmechTimeout as exc:
            note = f"still running after {wait_s:g} s: call get_job with this job_id"
            last = exc.last if isinstance(exc.last, S.JobOut) else None
            status = last.status if last else created.status
            return dumps({"job_id": created.job_id, "status": status, "note": note})
        return dumps(sm.job_summary(job, n_pathways=n_pathways, detail=detail))

    async def get_job(
        self, job_id: JobId, n_pathways: NPathways = 3, detail: StepDetail = False
    ) -> str:
        """A job's status (queued, running, done, failed, cancelled), its progress or error,
        and once done the summary of its top pathways (as predict_mechanism answers)."""
        job = await blocking(self.client.job, job_id)
        return dumps(sm.job_summary(job, n_pathways=n_pathways, detail=detail))

    async def step_readings(
        self,
        job_id: JobId,
        edge_id: Annotated[str, Field(description="A step's 'edge' id from a pathway.")],
    ) -> str:
        """Everything about one predicted step: the reaction matrix (bond-order changes
        'i-j:+1' and lone-pair changes 'i:+2' on pool atoms of the source state), the anchor,
        the arrows, and per forming bond the orbital reading (its gap factor g_atom in 1/eV,
        the donor and acceptor orbitals, the dominant canonical pair such as HOMO->LUMO and
        its energy gap). Also the source state id, for check_step."""
        job = await blocking(self.client.job, job_id)
        edge, nodes = _edge(job, edge_id)
        out = sm.edge_summary(edge, nodes, detail=True)
        out["alternatives"] = [
            {"matrix": sm.matrix_text(a.matrix), "score": sm.rnd(a.score), "rank": a.rank}
            for a in edge.alternatives
        ]
        return dumps(out)

    async def check_step(
        self,
        state_id: Annotated[str, Field(description="A pool state id (s + 20 hex).")],
        bonds: Annotated[
            list[tuple[int, int, int]] | None,
            Field(description="Bond changes [i, j, delta] on pool atoms, i < j, delta +1/-1."),
        ] = None,
        lone: Annotated[
            list[tuple[int, int]] | None,
            Field(description="Lone-electron changes [i, delta], delta +2/-2."),
        ] = None,
        detail: Annotated[bool, Field(description="Also the sigma keys and charges.")] = False,
    ) -> str:
        """Validate a proposed elementary step (a reaction matrix R) on a pool state without
        a job: ok or the issues (CHARGE_NOT_CONSERVED, VALENCE_EXCEEDED, BREAK_UNBONDED,
        RESONANCE_ONLY_STEP, ...), the product SMILES, the arrows R reads as, and the
        orbital readings of its forming bonds. Pool atoms are the atom indices of the
        state's mapped SMILES (normalize_reactants or depict_svg give it)."""
        matrix = {"bonds": [list(b) for b in bonds or ()], "lone": [list(x) for x in lone or ()]}
        out = await blocking(self.client.step_check, state_id, matrix)
        return dumps(sm.step_check_summary(out, detail=detail))


def register(app: FastMCP, client: KsmechClient, enabled: Enabled) -> list[str]:
    t = PredictTools(client)
    names = [add(app, t.get_job, "Read a job", READ)]
    if enabled("predict"):
        names.append(add(app, t.list_predictors, "List predictors", READ))
        names.append(add(app, t.predict_mechanism, "Predict a mechanism", COMPUTE))
        names.append(add(app, t.step_readings, "Readings of a step", READ))
    if enabled("correct"):
        names.append(add(app, t.check_step, "Check a step", READ))
    return names
