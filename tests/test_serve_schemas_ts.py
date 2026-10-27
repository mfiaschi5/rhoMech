"""The API contract: the generated TypeScript mirror, NaN handling and strict request models."""

import inspect
import json
import math
import re
from datetime import UTC, datetime

import pytest
from pydantic import BaseModel, ValidationError

from ksmech.serve.core import schemas

pytestmark = pytest.mark.serve

TYPES_GEN = schemas.default_typescript_path()
INPUT_MODELS = [m for m in schemas.TS_MODELS if schemas.schema_mode(m) == "validation"]


def test_types_gen_has_no_drift():
    assert TYPES_GEN.is_file(), f"{TYPES_GEN} missing: run schemas.write_typescript()"
    assert TYPES_GEN.read_text(encoding="utf-8") == schemas.typescript(), (
        "apps/ui/src/api/types.gen.ts is stale: regenerate with "
        "uv run python -c 'from ksmech.serve.core.schemas import "
        "write_typescript; write_typescript()'"
    )


def test_ts_models_cover_every_model_of_the_module():
    defined = {
        name
        for name, obj in inspect.getmembers(schemas, inspect.isclass)
        if issubclass(obj, BaseModel) and obj.__module__ == schemas.__name__
    }
    public = {n for n in defined if not n.startswith("_")}
    assert {m.__name__ for m in schemas.TS_MODELS} == public
    assert len(public) == 104  # spec v9: -20 v3 arrow and contact models, +9 R and readings;
    # +4: point group; +4: checkpoints and embedding traces; +2: the network's checkpoint


def test_every_ts_model_is_an_interface():
    ts = schemas.typescript()
    names = re.findall(r"^export interface (\w+) \{$", ts, flags=re.M)
    assert names[: len(schemas.TS_MODELS)] == [m.__name__ for m in schemas.TS_MODELS]
    assert len(names) == len(set(names))
    referenced = set(re.findall(r"\b([A-Z][A-Za-z0-9]*(?:Out|In|Brief|Body|Provenance))\b", ts))
    assert referenced <= set(names)


def _interface(name: str) -> dict[str, str]:
    m = re.search(rf"^export interface {name} \{{\n(.*?)\n\}}$", schemas.typescript(), re.M | re.S)
    assert m, name
    return dict(line.strip().rstrip(";").split(": ", 1) for line in m.group(1).splitlines())


def test_typescript_mapping():
    edge = _interface("EdgeOut")
    assert edge["matrix"] == "MatrixIO" and edge["anchor"] == "AnchorOut | null"
    assert edge["arrows"] == "ArrowReadingOut[]" and edge["readings"] == "OrbitalReadingOut[]"
    assert edge["logp"] == "number | null" and edge["product_smiles"] == "string[]"
    assert edge["formal_charges"] == "[number, number][]"
    assert _interface("MatrixIO") == {
        "bonds": "[number, number, number][]",
        "lone": "[number, number][]",
    }
    reading = _interface("ArrowReadingOut")
    assert reading["kind"] == '"lone_pair_to_bond" | "bond_to_bond" | "bond_to_lone_pair"'
    assert _interface("OrbitalReadingOut")["canonical"] == "CanonicalPairOut | null"
    assert _interface("StepCheckOut")["readings_status"] == '"ready" | "pending" | "unavailable"'
    assert _interface("PredictorInfoOut")["kind"] == '"floor" | "gbdt" | "network"'
    assert _interface("PendingOut")["status"] == '"pending"'
    assert _interface("OrbitalDiagnosticOut")["diagnostic"] == "true"
    assert _interface("OrbitalModelInputOut")["spectral_set"] == "[number, number][]"
    assert _interface("OrbitalModelInputOut")["overlap_sign"] == "-1 | 0 | 1"
    assert _interface("ProgressOut")["updated_at"] == "string"
    assert _interface("ErrorBody")["detail"] == "Record<string, unknown> | null"
    assert _interface("SpeciesGeometryOut")["bonds"] == "[number, number, number][]"
    assert _interface("AtomTokenOut")["model_input"] == (
        "Record<string, number | (number | null)[] | null>"
    )
    job = _interface("JobOut")
    assert job["result"] == " | ".join(
        ["NetworkOut", "ExpandResultOut", "ExportReportOut", "RecheckReportOut"]
        + ["IndexStatusOut", "EmbeddingResultOut", "null"]
    )
    assert job["unchanged"] == "boolean"
    predict = _interface("PredictIn")  # request: defaulted fields are optional
    assert predict["reactants_smiles"] == "string[]"
    assert predict["beam?"] == "number" and predict["candidate_scope?"] == '"all" | "inter"'
    assert _interface("NetworkOut")["request"] == "PredictIn"
    assert _interface("DepictAtomOut")["hidden_h_atoms"] == "number[]"


def test_write_typescript(tmp_path):
    out = schemas.write_typescript(tmp_path / "x" / "types.gen.ts")
    assert out.read_text(encoding="utf-8") == schemas.typescript()
    assert TYPES_GEN.parts[-4:] == ("ui", "src", "api", "types.gen.ts")


def test_nan_and_inf_serialize_as_null():
    r = schemas.OrbitalReadingOut(
        i=0, j=1, g_atom=math.nan, donor_token=3, acceptor_token=4, g_rs=math.inf,
        term=-math.inf, donor_diag=None, acceptor_diag=None, canonical=None,
    )  # fmt: skip
    for dumped in (json.loads(r.model_dump_json()), r.model_dump(mode="json")):
        assert dumped["g_atom"] is None and dumped["g_rs"] is None and dumped["term"] is None
        assert dumped["donor_token"] == 3
    assert math.isnan(r.model_dump()["g_atom"])  # python mode keeps the float


def test_nan_inside_dicts_tuples_and_lists():
    canonical = schemas.CanonicalPairOut(
        k=1, m=2, e_k_ev=math.nan, e_m_ev=1.0, k_label="HOMO", m_label="LUMO", term=0.1,
        share=None, index_source="aufbau",
    )  # fmt: skip
    assert json.loads(canonical.model_dump_json())["e_k_ev"] is None
    phys = {"ks3": math.nan, "levels": [1.0, math.inf]}
    step = schemas.CuratedStepOut(
        step_index=0,
        order_group=0,
        reactant_pool_key="a",
        product_pool_key="b",
        step=phys,
        was_corrected=False,
        original=None,
        recheck=None,
    )
    assert step.model_dump(mode="json")["step"] == {"ks3": None, "levels": [1.0, None]}
    assert json.loads(step.model_dump_json())["step"] == {"ks3": None, "levels": [1.0, None]}
    cfg = schemas.CandidateConfigOut(
        allow_d_acceptors=False,
        full=1.8,
        empty=0.2,
        exclude_fractional=True,
        fractional_band=(0.2, math.nan),
        site_weight=0.15,
    )
    assert cfg.model_dump(mode="json")["fractional_band"] == [0.2, None]
    assert json.loads(cfg.model_dump_json())["fractional_band"] == [0.2, None]
    sp = schemas.SourceIndexStatusOut(
        state="ready",
        n_steps=1,
        built_at=datetime(2026, 10, 21, tzinfo=UTC),
        job_id=None,
        reason=None,
    )
    assert sp.model_dump(mode="json")["built_at"] == "2026-10-21T00:00:00Z"
    assert schemas.finite_or_none({"a": [math.nan, (1.0, math.inf)], "b": "x"}) == {
        "a": [None, [1.0, None]],
        "b": "x",
    }


@pytest.mark.parametrize("model", INPUT_MODELS, ids=lambda m: m.__name__)
def test_input_models_forbid_extra_fields(model):
    assert model.model_config.get("extra") == "forbid"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        model.model_validate({"definitely_not_a_field": 1})


def test_input_models_are_the_request_bodies():
    names = {m.__name__ for m in INPUT_MODELS}
    assert names == {
        "NormalizeIn",
        "PredictIn",
        "MatrixIO",
        "OverrideIn",
        "ExpandIn",
        "StepCheckIn",
        "CurateIn",
        "MechanismPatchIn",
        "RecheckIn",
        "RecheckOneIn",
        "MoleculeIn",
        "Depict2DIn",
        "Depict2DViewIn",
        "EmbeddingIn",
        "CheckpointRefIn",
    }


def test_predict_in_defaults_and_bounds():
    p = schemas.PredictIn(reactants_smiles=["C[O-]", "CCl"])
    assert (p.predictor, p.beam, p.max_steps, p.candidate_scope) == ("floor", 5, 8, "all")
    assert p.stoichiometry is None and p.temperature is None and p.stop_at_turnover
    for bad in ({"beam": 0}, {"max_steps": 13}, {"temperature": 0.0}, {"reactants_smiles": []}):
        with pytest.raises(ValidationError):
            schemas.PredictIn.model_validate({"reactants_smiles": ["C"], **bad})
    with pytest.raises(ValidationError):
        schemas.PredictIn(reactants_smiles=["C"] * 7)
    with pytest.raises(ValidationError):
        schemas.OverrideIn(parent_state_id="s" + "0" * 20, matrix={"bonds": []})


def test_output_models_ignore_unknown_fields_and_keep_defaults():
    job = schemas.JobCreatedOut(
        job_id="j0123456789abcdef", kind="predict", status="queued", status_url="/api/jobs/x", x=1
    )
    assert job.parent_job_id is None and "x" not in job.model_dump()
    assert schemas.PendingOut.model_fields["status"].default == "pending"
