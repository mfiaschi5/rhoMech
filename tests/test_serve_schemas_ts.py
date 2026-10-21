"""The API contract: the generated TypeScript mirror, NaN handling and strict request models."""

import inspect
import json
import math
import re
from datetime import UTC, datetime

import pytest
from pydantic import BaseModel, ValidationError

from ksmech.serve import schemas

pytestmark = pytest.mark.serve

TYPES_GEN = schemas.default_typescript_path()
INPUT_MODELS = [m for m in schemas.TS_MODELS if schemas.schema_mode(m) == "validation"]


def test_types_gen_has_no_drift():
    assert TYPES_GEN.is_file(), f"{TYPES_GEN} missing: run schemas.write_typescript()"
    assert TYPES_GEN.read_text(encoding="utf-8") == schemas.typescript(), (
        "apps/ui/src/api/types.gen.ts is stale: regenerate with "
        "uv run python -c 'from ksmech.serve.schemas import write_typescript; write_typescript()'"
    )


def test_ts_models_cover_every_model_of_the_module():
    defined = {
        name
        for name, obj in inspect.getmembers(schemas, inspect.isclass)
        if issubclass(obj, BaseModel) and obj.__module__ == schemas.__name__
    }
    public = {n for n in defined if not n.startswith("_")}
    assert {m.__name__ for m in schemas.TS_MODELS} == public
    assert len(public) == 83


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
    arrow = _interface("ArrowOut")
    assert arrow["electrons"] == "1 | 2"  # enum, and defaulted output fields are required
    assert arrow["site_source"] == '"contact" | "main_centre" | "rule" | "user"'
    assert arrow["donor_centres_map"] == "Record<string, number>"
    assert arrow["physics"] == "ArrowPhysicsBrief | null"
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
        ["NetworkOut", "OverrideResultOut", "ExportReportOut", "RecheckReportOut"]
        + ["IndexStatusOut", "null"]
    )
    assert job["unchanged"] == "boolean"
    predict = _interface("PredictIn")  # request: defaulted fields are optional
    assert predict["reactants_smiles"] == "string[]"
    assert predict["beam?"] == "number" and predict["candidate_scope?"] == '"all" | "inter"'
    assert _interface("ExportIn")["statuses?"] == '("validated" | "corrected")[]'
    assert _interface("NetworkOut")["request"] == "PredictIn"


def test_write_typescript(tmp_path):
    out = schemas.write_typescript(tmp_path / "x" / "types.gen.ts")
    assert out.read_text(encoding="utf-8") == schemas.typescript()
    assert TYPES_GEN.parts[-4:] == ("ui", "src", "api", "types.gen.ts")


def _brief(**kw) -> schemas.ArrowPhysicsBrief:
    base = dict(
        pair_valid=True,
        reason=None,
        ks1=None,
        ks2=0.1,
        ks2_form="contact",
        ks3=-0.5,
        G=0.2,
        beta2=1.0,
        amplitude_S=0.1,
        dominant=None,
        contact_distance_ang=2.3,
        contact_sites=(1, 5),
        clash=False,
        floor=0.4,
        floor_rank=1,
        n_valid=10,
        frozen_logit=0.3,
        frozen_weight=0.5,
        tau=1.0,
        frozen_source="untrained_kernel",
        flags=[],
    )
    return schemas.ArrowPhysicsBrief(**{**base, **kw})


def test_nan_and_inf_serialize_as_null():
    b = _brief(ks3=math.nan, G=math.inf, frozen_weight=-math.inf, ks2=math.nan)
    for dumped in (json.loads(b.model_dump_json()), b.model_dump(mode="json")):
        assert dumped["ks3"] is None and dumped["G"] is None
        assert dumped["frozen_weight"] is None and dumped["ks2"] is None
        assert dumped["contact_distance_ang"] == 2.3
    assert math.isnan(b.model_dump()["ks3"])  # python mode keeps the float
    levels = schemas.EnergyLevelsOut(
        donor_species="C[O-]",
        acceptor_species="CCl",
        donor_levels=[schemas.LevelOut(canonical=0, label="HOMO", energy=math.nan, weight=None)],
        acceptor_levels=[],
        donor_lmo_energy=-9.0,
        acceptor_lmo_energy=math.inf,
        homo_lumo_gap=1.0,
    )
    text = levels.model_dump_json()
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text)["donor_levels"][0]["energy"] is None
    assert levels.model_dump(mode="json")["acceptor_lmo_energy"] is None


def test_nan_inside_dicts_tuples_and_lists():
    sitepair = schemas.SitePairsOut(
        state_id="s" + "0" * 20, donor=3, acceptor=4, G=math.nan, site_pairs=[], note=""
    )
    assert json.loads(sitepair.model_dump_json())["G"] is None
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
        "ArrowIn",
        "OverrideIn",
        "CurateIn",
        "MechanismPatchIn",
        "ExportIn",
        "RecheckIn",
        "RecheckOneIn",
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
        schemas.OverrideIn(edge_id="e0123456789abcdef", arrows=[])


def test_output_models_ignore_unknown_fields_and_keep_defaults():
    job = schemas.JobCreatedOut(
        job_id="j0123456789abcdef", kind="predict", status="queued", status_url="/api/jobs/x", x=1
    )
    assert job.parent_job_id is None and "x" not in job.model_dump()
    arrow_fields = schemas.ArrowOut.model_fields
    assert arrow_fields["electrons"].default == 2
    assert schemas.PendingOut.model_fields["status"].default == "pending"
