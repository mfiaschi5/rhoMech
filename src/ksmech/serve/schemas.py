"""The HTTP contract of ksmech-serve: every pydantic model of the API, and its TypeScript mirror.

Output models share `_Out` (NaN and inf serialize as JSON null; defaulted fields are required in
the serialization schema, because a response always carries them). Input models share `_In`
(`extra="forbid"`, so a misspelt request field is a 422 instead of being ignored).
Cross-field rules marked "exactly one" are not enforced here: the services check them and raise
`REQUEST_INVALID`.

`typescript()` renders `TS_MODELS` to `apps/ui/src/api/types.gen.ts` from `model_json_schema()`
(output models in serialization mode, input models in validation mode); `write_typescript()`
writes the file, and `tests/test_serve_schemas_ts.py` fails on drift.

This module imports no other serve module.
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    model_serializer,
)

# ---- bases


def finite_or_none(value: Any) -> Any:
    """`value` with every NaN / inf float replaced by None, through lists, tuples and dicts."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, list | tuple):
        return [finite_or_none(v) for v in value]
    if isinstance(value, dict):
        return {k: finite_or_none(v) for k, v in value.items()}
    return value


def _holds_models(value: Any) -> bool:
    if isinstance(value, BaseModel):
        return True
    return isinstance(value, list | tuple) and bool(value) and isinstance(value[0], BaseModel)


class _Out(BaseModel):
    """Base of every response model.

    JSON never carries NaN or inf: `ser_json_inf_nan` covers `model_dump_json` (FastAPI's
    `response_model` fast path), and the wrap serializer covers `model_dump(mode="json")`
    (`jsonable_encoder`, hand-built `JSONResponse`s). Nested models clean their own fields.
    """

    model_config = ConfigDict(
        ser_json_inf_nan="null",
        json_schema_serialization_defaults_required=True,
        protected_namespaces=(),
    )

    @model_serializer(mode="wrap")
    def _null_non_finite(self, handler: SerializerFunctionWrapHandler, info: SerializationInfo):
        data = handler(self)
        if not info.mode_is_json() or not isinstance(data, dict):
            return data
        for key, value in data.items():
            if not _holds_models(getattr(self, key, None)):
                data[key] = finite_or_none(value)
        return data


class _In(BaseModel):
    """Base of every request model."""

    model_config = ConfigDict(
        extra="forbid",
        ser_json_inf_nan="null",
        protected_namespaces=(),
    )


# ---- common


class ErrorBody(_Out):
    code: str
    message: str
    upstream: str | None = None
    detail: dict | None = None
    request_id: str


class ErrorOut(_Out):
    error: ErrorBody


class WorkerStatusOut(_Out):
    alive: bool
    n_workers: int
    pids: list[int]
    busy: bool
    current_task: str | None
    current_label: str | None
    elapsed_s: float | None
    queue_depth: int
    restarts: int
    unstable: bool


class PendingOut(_Out):
    status: Literal["pending"] = "pending"
    task_id: str
    label: str
    retry_after_s: float
    worker: WorkerStatusOut


class KSParamsOut(_Out):
    K: float
    eps: float
    tau: float


class CandidateConfigOut(_Out):
    allow_d_acceptors: bool
    full: float
    empty: float
    exclude_fractional: bool
    fractional_band: tuple[float, float]
    site_weight: float


class Provenance(_Out):
    predictor: str
    predictor_version: str | None
    checkpoint_hash: str | None
    spec_version: str
    physics_fingerprint: str
    artifact_versions: dict[str, int | str]
    chemistry: Literal["real", "cached", "synthetic", "stub"]
    ks_params: KSParamsOut
    temperature: float | None
    candidate_scope: Literal["all", "inter"]
    contact_distance_ang: float
    candidate_config: CandidateConfigOut
    notes: list[str]


class LimitsOut(_Out):
    max_species: int
    max_count: int
    max_atoms: int
    max_beam: int
    max_steps: int
    max_nodes: int


class HealthOut(_Out):
    ok: bool
    chemistry: str
    worker: WorkerStatusOut
    jobs_running: int


class CapabilityOut(_Out):
    enabled: bool
    reason: str | None  # why it is off (None when on)


class MetaOut(_Out):
    api_schema: int
    app_version: str
    spec_label: str  # "v8"
    spec_sha256_12: str | None
    spec_version: str
    artifact_versions: dict[str, int | str]
    physics_fingerprint: str
    chemistry: str
    qm_method: str
    token_method: str
    ks_params: KSParamsOut
    primary_distance_ang: float
    feature_distances_ang: list[float]
    candidate_config: CandidateConfigOut
    limits: LimitsOut
    worker: WorkerStatusOut
    frontend_built: bool
    dev: bool
    data_root: str
    capabilities: dict[str, CapabilityOut]  # capabilities.CAPABILITIES (spec v8 port)


class PredictorInfoOut(_Out):
    name: str
    label: str
    kind: Literal["physics_floor", "baseline", "model", "stub"]
    available: bool
    code: str | None
    reason: str | None
    version: str | None
    checkpoint_hash: str | None
    score_unit: str
    default_temperature: float
    warnings: list[str]
    description: str
    spec_version: str  # the spec version the predictor runs under (brief item 1)


# ---- input and preflight


class NormalizeIn(_In):
    reactants_smiles: list[str] = Field(min_length=1, max_length=6)
    stoichiometry: list[int] | None = None  # same length, each 1..4; default all 1


class SpeciesCountOut(_Out):
    species_key: str
    token_keys: list[str]
    display_smiles: str
    count: int
    charge: int
    resonance_flag: str


class SpeciesPreflightOut(_Out):
    field: int
    input: str
    token_key: str | None
    species_key: str | None
    count: int
    charge: int | None
    multiplicity: int | None
    n_atoms: int | None
    tokens_cached: bool
    qm_cached: bool
    errors: list[ErrorBody]


class NormalizeOut(_Out):
    ok: bool
    species: list[SpeciesPreflightOut]
    pool_key: str | None
    pool_id: str | None
    state_id: str | None
    mapped_smiles: str | None
    n_atoms: int | None


class PredictIn(NormalizeIn):
    conditions: str | None = Field(None, max_length=500)  # recorded only; no predictor reads it
    predictor: str = "floor"
    beam: int = Field(5, ge=1, le=10)
    max_steps: int = Field(8, ge=1, le=12)
    temperature: float | None = Field(None, gt=0)
    stop_at_turnover: bool = True
    candidate_scope: Literal["all", "inter"] = "all"
    max_nodes: int = Field(200, ge=1, le=500)
    n_pathways: int = Field(20, ge=1, le=50)


class JobCreatedOut(_Out):
    job_id: str
    kind: str
    status: str
    status_url: str
    parent_job_id: str | None = None
    provenance: Provenance | None = None  # prediction and override jobs; None for the others


# ---- arrows, edges, nodes


class DominantBriefOut(_Out):
    """The dominant canonical MO pair (k, m) of G (design §7.5)."""

    k_canonical: int
    m_canonical: int
    k_label: str  # "HOMO-1"
    m_label: str  # "LUMO"
    E_k: float  # eV, canonical energy of k
    E_m: float  # eV, canonical energy of m
    term: float  # T[k, m] (1/eV)
    share: float  # T[k, m] / G
    index_source: str
    same_molecule: bool


class ArrowPhysicsBrief(_Out):
    """Inline on every network arrow (the spec §10 serving contract)."""

    pair_valid: bool
    reason: str | None
    ks1: float | None
    ks2: float | None
    ks2_form: Literal["difference", "contact", "undefined"]
    ks3: float
    G: float
    beta2: float
    amplitude_S: float
    dominant: DominantBriefOut | None  # null only with the CONSISTENCY_FAILED flag
    contact_distance_ang: float  # 2.30: the oriented placement distance
    contact_sites: tuple[int, int]
    clash: bool
    floor: float
    floor_rank: int | None
    n_valid: int
    frozen_logit: float
    frozen_weight: float | None
    tau: float
    frozen_source: Literal["untrained_kernel", "checkpoint"]
    flags: list[str]


class SitePairOut(_Out):
    donor_site: int
    acceptor_site: int
    donor_site_map: int
    acceptor_site_map: int
    S: float
    beta: float
    clash: bool
    ks3_site: float
    site_rule_excluded: bool
    is_contact: bool


class ArrowOut(_Out):
    donor: int
    donor_site: int
    acceptor: int
    acceptor_site: int
    donor_site_map: int
    acceptor_site_map: int
    donor_element: str
    acceptor_element: str
    donor_species: str
    acceptor_species: str
    electrons: Literal[1, 2] = 2  # 1 reserved for radicals, never emitted
    source_kind: Literal["lone_pair", "bond", "unknown"]
    source_partner: int | None  # c of a bond source
    source_partner_map: int | None
    sink_partner: int | None  # d of the implied sink heterolysis
    sink_partner_map: int | None
    site_source: Literal["contact", "main_centre", "rule", "user"]
    row: int
    col: int
    donor_centres_map: dict[str, float]  # map number -> centre weight
    acceptor_centres_map: dict[str, float]
    diagnostic: dict[str, str]  # {"donor_class", "acceptor_class"}: display only
    physics: ArrowPhysicsBrief | None
    # user sites that differ from the cell's contact sites: their site-pair row (design 7.4)
    site_pair: SitePairOut | None = None


class DatasetArrowOut(_Out):
    source_kind: str
    source_atoms: list[int]  # map numbers
    sink_kind: str
    sink_atoms: list[int]


class AlternativeOut(_Out):
    arrows: list[ArrowOut]
    score: float
    probability: float
    rank: int


class EdgeOriginalOut(_Out):
    edge_id: str
    target: str
    target_pool_key: str
    arrows: list[ArrowOut]
    score: float | None
    probability: float | None
    rank: int | None
    predictor: str | None
    predictor_version: str | None


class PredictorViewOut(_Out):
    score: float | None
    probability: float | None
    rank: int | None
    pair_valid: bool
    reason: str | None


class NodeOut(_Out):
    id: str
    pool_key: str
    state_id: str
    alias_state_ids: list[str]
    species: list[SpeciesCountOut]
    display_smiles: str
    depth: int
    best_logp: float
    best_parent_edge: str | None
    status: Literal["frontier", "expanded", "pruned", "terminal", "failed"]
    terminal_reason: str | None
    failure: ErrorBody | None
    is_root: bool
    n_valid: int | None
    scanned_mass: float | None
    unscanned_mass: float | None
    apply_failures: dict[str, int]
    fermi_smeared: bool
    warnings: list[str]


class EdgeOut(_Out):
    id: str
    source: str
    target: str
    target_pool_key: str
    kind: Literal["predicted", "corrected"]
    status: Literal["active", "superseded"]
    source_state_id: str
    product_state_id: str
    product_smiles: str  # display (unmapped)
    source_mapped_smiles: str
    product_mapped_smiles: str
    arrows: list[ArrowOut]
    dataset_arrows: list[DatasetArrowOut] | None
    pmechdb_code: str | None
    encoding_base_smiles: str | None  # the Kekulé base the encoding used (StepDoc reactants_mapped)
    encode_reason: str | None
    touched_maps: list[int]
    score: float | None
    probability: float | None
    logp: float
    rank: int | None
    n_alternatives: int
    alternatives: list[AlternativeOut]
    predictor: str | None
    predictor_version: str | None
    original: EdgeOriginalOut | None
    superseded_by: str | None
    predictor_view: list[PredictorViewOut] | None
    recurrence: bool
    closes_cycle: bool
    cycle_ids: list[str]
    revision: int


class CycleOut(_Out):
    id: str
    kind: Literal["recurrence", "turnover"]
    anchor: str
    closing_edge: str
    node_ids: list[str]
    edge_ids: list[str]
    catalyst: dict[str, int]
    consumed: dict[str, int]
    produced: dict[str, int]
    virtual_edge: dict[str, str] | None  # {"source", "target", "label"}


class PathwayOut(_Out):
    id: str
    rank: int
    node_ids: list[str]
    edge_ids: list[str]
    logp: float
    probability: float
    n_steps: int
    n_corrected: int
    ends_with: str
    closes_cycle: bool
    cycle_ids: list[str]


class FailedProposalOut(_Out):
    node: str
    rank: int
    arrows: list[ArrowOut]
    probability: float
    code: str
    detail: str


class NetParamsOut(_Out):
    beam: int
    max_steps: int
    temperature: float
    stop_at_turnover: bool
    candidate_scope: str
    max_scan: int
    max_nodes: int
    n_pathways: int


class NetworkOut(_Out):
    network_id: str
    revision: int
    complete: bool
    root: str
    nodes: list[NodeOut]
    edges: list[EdgeOut]
    cycles: list[CycleOut]
    pathways: list[PathwayOut]
    failed_proposals: list[FailedProposalOut]
    skipped: dict[str, int]
    params: NetParamsOut
    request: PredictIn
    provenance: Provenance


# ---- jobs


class TaskInfoOut(_Out):
    name: str
    label: str
    elapsed_s: float
    queue_depth: int
    priority: str


class ProgressOut(_Out):
    phase: str
    done: int = 0
    total: int | None = None
    message: str = ""
    depth: int | None = None
    max_steps: int | None = None
    n_nodes: int = 0
    n_edges: int = 0
    frontier: int = 0
    current_task: TaskInfoOut | None = None
    eta_s: float | None = None
    updated_at: datetime


class ArrowIn(_In):
    donor: int
    donor_site: int
    acceptor: int
    acceptor_site: int


class OverrideIn(_In):
    edge_id: str | None = None  # exactly one of edge_id / node_id
    node_id: str | None = None
    arrows: list[ArrowIn] = Field(min_length=1, max_length=6)  # applied in order
    allow_invalid: bool = False
    continue_prediction: bool = True
    continue_steps: int | None = Field(None, ge=0, le=12)


class OverrideResultOut(_Out):
    parent_job_id: str
    network_revision: int
    corrected_edge_id: str
    target_node: str
    continued_from: str | None
    issues: list[ErrorBody]


class JobOut(_Out):
    job_id: str
    kind: Literal["predict", "override", "recheck", "export", "index"]
    status: Literal["queued", "running", "done", "failed", "cancelled"]
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    progress: ProgressOut
    error: ErrorBody | None
    parent_job_id: str | None
    network_id: str | None
    partial: bool
    unchanged: bool = False
    provenance: Provenance | None
    result_kind: Literal["network", "override", "export", "recheck", "index"] | None
    result: (
        NetworkOut | OverrideResultOut | ExportReportOut | RecheckReportOut | IndexStatusOut | None
    )


class JobSummaryOut(_Out):
    job_id: str
    kind: str
    status: str
    created_at: datetime
    network_id: str | None
    label: str


# ---- pools and tokens


class PoolOut(_Out):
    state_id: str
    pool_id: str
    pool_key: str
    mapped_smiles: str
    species: list[SpeciesCountOut]
    n_atoms: int
    n_orbitals: int | None
    analysis: Literal["missing", "running", "ready", "failed", "disabled"]
    analysis_error: ErrorBody | None
    fermi_smeared: bool | None
    warnings: list[str]
    chemistry: str


class CentreOut(_Out):
    site: int
    map: int
    element: str
    weight: float


class OrbitalModelInputOut(_Out):
    """Exactly the model_input descriptors (orbitals/descriptors.py:48)."""

    occupation: float
    n_centers: int
    center_weights: list[float]
    center_spd: list[list[float]]
    cos_a: float
    cos_b: float
    overlap_population: float
    overlap_sign: Literal[-1, 0, 1]
    distance_ab: float
    radius_gyration: float
    spectral_set: list[tuple[float, float]]  # (E_k eV, weight) with weight > 0


class OrbitalBookkeepingOut(_Out):
    energy_ev: float
    centres: list[CentreOut]
    centroid: list[float]
    spectral_indices: list[int]
    retained_fraction: float
    n_block: int
    block: Literal["occupied", "virtual"]
    fragment: int
    local_position: int


class OrbitalMasksOut(_Out):
    donor_ok: bool
    acceptor_ok: bool
    d_type: bool
    fractional: bool
    somo: bool


class OrbitalDiagnosticOut(_Out):
    """model_input=False: the classifier's labels, display only."""

    diagnostic: Literal[True] = True
    type: str
    token_class: str
    parents: str
    s_share: float | None
    p_share: float | None
    d_share: float | None
    classifier_overlap_population: float | None
    metal_bound: bool


class OrbitalTokenOut(_Out):
    token_id: int
    kind: Literal["occupied", "virtual"]
    model_input: OrbitalModelInputOut
    bookkeeping: OrbitalBookkeepingOut
    masks: OrbitalMasksOut
    diagnostic: OrbitalDiagnosticOut


class AtomTokenOut(_Out):
    token_id: int
    fragment: int
    record_index: int
    map: int
    element: str
    # atomic_number, formal_charge, mulliken_charge, f_plus_fmo, f_minus_fmo, dual, fmo_weights,
    # vbur, chirality
    model_input: dict[str, float | int | list[float | None] | None]
    # f_plus, f_minus, hybridization, aromatic, fg_label
    diagnostic: dict[str, str | float | bool | None]


class FragmentOut(_Out):
    index: int
    species_key: str
    token_key: str
    atom_offset: int
    orbital_offset: int
    fermi_smeared: bool
    pm_converged: bool
    qm_warnings: str
    token_digest: str


class TokensOut(_Out):
    state_id: str
    n_atoms: int
    n_orbitals: int
    orbital_feature_names: list[str]
    atom_feature_names: list[str]
    fragments: list[FragmentOut]
    atoms: list[AtomTokenOut]
    orbitals: list[OrbitalTokenOut]
    provenance: Provenance


# ---- candidates and site pairs


class OrbitalDescBrief(_Out):
    occupation: float
    n_centers: int
    centres: list[CentreOut]
    spd0: list[float]
    cos_a: float
    cos_b: float
    overlap_sign: int
    spectral_set: list[tuple[float, float]]


class CandidateOut(_Out):
    rank: int | None
    pair_rank: int | None
    row: int
    col: int
    donor: int
    donor_site: int
    acceptor: int
    acceptor_site: int
    donor_site_map: int
    acceptor_site_map: int
    donor_site_element: str
    acceptor_site_element: str
    donor_species: str
    acceptor_species: str
    acceptor_centre_elements: list[str]  # sorted unique element symbols, e.g. ["C", "Cl"]
    is_contact_site_pair: bool
    contact_sites: tuple[int, int]
    score: float | None
    probability: float | None
    ks1: float | None
    ks2: float | None
    ks3: float
    clash: bool
    donor_desc: OrbitalDescBrief
    acceptor_desc: OrbitalDescBrief
    diagnostic: dict[str, str]  # donor_class / acceptor_class (greyed)
    valid: bool
    reason: str | None  # PairReason name, or SELF_SITE_PAIR / SCOPE_EXCLUDED


class CandidatesOut(_Out):
    state_id: str
    predictor: str
    temperature: float
    scope: str
    n_valid_pairs: int
    n_tuples: int
    total: int
    offset: int
    limit: int
    candidates: list[CandidateOut]
    excluded_counts: dict[str, int]
    provenance: Provenance


class SitePairsOut(_Out):
    state_id: str
    donor: int
    acceptor: int
    G: float
    site_pairs: list[SitePairOut]
    note: str
    provenance: Provenance | None = None


# ---- physics panel


class DominantPairOut(_Out):
    k: int
    m: int
    k_canonical: int
    m_canonical: int
    k_label: str
    m_label: str
    index_source: Literal[
        "qm_occupations", "aufbau", "qm_occupations_unchecked", "aufbau_unchecked"
    ]
    donor_species: str
    acceptor_species: str
    E_k: float
    E_m: float
    gap: float  # E_k - E_m (eV), clamped as in G
    p_rk: float
    p_sm: float
    term: float
    share: float
    ks3_part: float
    same_molecule: bool


class FrozenHeadOut(_Out):
    logit: float
    weight: float | None
    tau: float
    source: Literal["untrained_kernel", "checkpoint"]
    normalization: str


class Ks2DetailOut(_Out):
    form: Literal["difference", "contact", "undefined"]
    main_centres: tuple[int, int]
    q_donor: float
    q_acceptor: float
    R_contact: float
    R_geometric: float | None
    eps: float


class LevelOut(_Out):
    canonical: int
    label: str
    energy: float
    weight: float | None


class EnergyLevelsOut(_Out):
    donor_species: str
    acceptor_species: str
    donor_levels: list[LevelOut]
    acceptor_levels: list[LevelOut]
    donor_lmo_energy: float
    acceptor_lmo_energy: float
    homo_lumo_gap: float  # E_LUMO(acceptor) - E_HOMO(donor)


class PhysicsOut(_Out):
    state_id: str
    donor: int
    donor_site: int
    acceptor: int
    acceptor_site: int
    row: int
    col: int
    units: dict[str, str]
    pair_valid: bool
    reason: str | None
    ks1: float
    ks1_defined: bool
    ks2: float
    ks2_defined: bool
    ks2_detail: Ks2DetailOut
    ks3: float
    G: float
    G_exact: float
    beta: float
    beta2: float
    beta2_by_distance: dict[str, float]
    amplitude_S: float
    contact_sites: tuple[int, int]
    requested_sites_are_contact: bool
    site_pair: SitePairOut | None
    contact_distance_ang: float
    clash: bool
    floor: float
    floor_rank: int | None
    n_valid: int
    predictor: str
    score: float | None
    probability: float | None
    rank: int | None
    frozen_head: FrozenHeadOut
    dominant_pair: DominantPairOut
    energy_levels: EnergyLevelsOut
    flags: list[str]
    provenance: Provenance


# ---- geometry, depiction, 3D


class AtomXYZOut(_Out):
    global_site: int | None
    record_index: int
    map: int | None
    Z: int
    symbol: str
    x: float
    y: float
    z: float


class SpeciesGeometryOut(_Out):
    fragment: int
    token_key: str
    species_key: str
    atoms: list[AtomXYZOut]
    bonds: list[tuple[int, int, int]]  # (record i, record j, order)
    xyz: str
    molblock: str
    display_offset: tuple[float, float, float]
    frame_note: str


class GeometryOut(_Out):
    state_id: str
    species: list[SpeciesGeometryOut]


class DepictAtomOut(_Out):
    depiction_idx: int
    global_site: int | None
    map: int
    element: str
    fragment: int
    hidden_h: int


class DepictionOut(_Out):
    ref: str
    molblock: str
    atoms: list[DepictAtomOut]
    server_svg: str | None
    server_coords: list[tuple[float, float]] | None


class LobeOut(_Out):
    site: int
    origin: tuple[float, float, float]
    direction: tuple[float, float, float]
    tied: list[tuple[float, float, float]]
    opposite: tuple[float, float, float] | None


class CapOut(_Out):
    atoms: list[AtomXYZOut]
    bonds: list[tuple[int, int, int]]
    molblock: str
    local_ids: list[int]


class ContactOut(_Out):
    state_id: str
    donor: int
    donor_site: int
    acceptor: int
    acceptor_site: int
    same_fragment: bool
    is_contact_argmax: bool
    donor_fragment: SpeciesGeometryOut
    cap: CapOut
    acceptor_context: CapOut | None
    donor_lobe: LobeOut
    acceptor_lobe_placed: LobeOut
    acceptor_lobe_own_frame: LobeOut
    distance_ang: float
    clash_limit_ang: float
    S: float
    beta: float
    clash: bool
    tier: Literal[1, 2]
    roll_step: int
    roll_deg: float
    slots: tuple[int, int]
    ks3_site: float
    G: float
    consistent: bool
    checks: dict[str, float | bool]
    cube_urls: dict[str, str]
    cubes: dict[str, str] | None
    iso_default: float
    provenance: Provenance


# ---- curation, export, recheck


class CurateIn(_In):
    job_id: str
    pathway_id: str | None = None  # exactly one of pathway_id / edge_ids
    edge_ids: list[str] | None = None
    action: Literal["save", "validate", "reject"]
    notes: str = ""
    order_groups: list[int] | None = None
    conditions: str | None = None


class RecheckRowOut(_Out):
    mechanism_id: int
    step_index: int
    created_at: datetime
    predictor: str
    predictor_version: str
    spec_version: str
    physics_fingerprint: str
    aligned: bool
    first_arrow_rank: int | None
    n_valid: int | None
    top1: bool | None
    top5: bool | None
    product_match: str
    top1_reproduces_product: bool | None
    reason: str | None


class CuratedStepOut(_Out):
    step_index: int
    order_group: int
    reactant_pool_key: str
    product_pool_key: str
    step: dict  # StepDoc (design §9.2)
    was_corrected: bool
    original: dict | None
    recheck: RecheckRowOut | None


class MechanismSummaryOut(_Out):
    id: int
    created_at: datetime
    reactants: list[str]
    stoichiometry: list[int]
    status: str
    predictor: str
    checkpoint_hash: str
    spec_version: str
    n_steps: int
    stale: bool
    notes: str
    last_recheck: datetime | None


class MechanismOut(MechanismSummaryOut):
    conditions: str | None
    stale_reasons: list[str]
    steps: list[CuratedStepOut]
    rechecks: list[RecheckRowOut]
    warnings: list[str]


class MechanismPatchIn(_In):
    status: Literal["predicted", "validated", "corrected", "rejected"] | None = None
    notes: str | None = None
    order_groups: list[int] | None = None


class ExportIn(_In):
    statuses: list[Literal["validated", "corrected"]] = ["validated", "corrected"]


class ExportReportOut(_Out):
    export_id: str
    created_at: datetime
    spec_version: str
    paths: dict[str, str]
    n_mechanisms: int
    n_steps_considered: int
    n_steps_written: int
    skipped: dict[str, int]
    ingest_rejections: dict[str, int]
    duplicates_pmechdb: int
    duplicates_in_export: int
    withdrawn: int
    changed_ids: int
    # {mechanism_id, step_index, step_id | None, code | None, detail, provenance}
    steps: list[dict]
    sha256: str
    warnings: list[str] = []  # "PMECHDB_UNAVAILABLE: ...", "COMPACT_FALLBACK: ..."


class RecheckIn(_In):
    mechanism_ids: list[int] | None = None
    predictor: str | None = None
    stale_only: bool = True


class RecheckOneIn(_In):
    predictor: str | None = None


class RecheckReportOut(_Out):
    n_steps: int
    n_checked: int
    top1: float | None
    top5: float | None
    product_match: dict[str, int]
    skipped: dict[str, int]
    rows: list[RecheckRowOut]
    provenance: Provenance


# ---- browse


class SourceIndexStatusOut(_Out):
    state: Literal["ready", "building", "missing", "stale", "unavailable"]
    n_steps: int
    built_at: datetime | None
    job_id: str | None
    reason: str | None


class IndexStatusOut(_Out):
    pmechdb: SourceIndexStatusOut
    curated_steps: int


class StepSummaryOut(_Out):
    ref: str
    source: Literal["pmechdb", "user_curated"]
    status: str  # "dataset" for PMechDB
    step_id: str | None
    mechanism_id: int | None
    step_index: int | None
    reactants_smiles: str
    products_smiles: str
    reactant_pool_key: str
    product_pool_key: str
    n_arrows: int
    pmechdb_code: str | None
    resonance_step: bool
    matched_species: list[str]
    matched_forms: list[str]  # the token-key form each substructure hit matched (design §10)
    splits: dict[str, str] | None
    exported_step_id: str | None


class SearchOut(_Out):
    total: int
    truncated: bool
    index: IndexStatusOut
    results: list[StepSummaryOut]


class StepViewOut(_Out):
    step: StepSummaryOut
    reactants_depiction: DepictionOut
    products_depiction: DepictionOut
    dataset_arrows: list[DatasetArrowOut]
    mechanism: MechanismSummaryOut | None
    recheck: RecheckRowOut | None


# ---- molecule workbench (display only)


class MoleculeIn(_In):
    """Exactly one of `molblock` (V2000 / V3000, explicit H kept, stereo from the wedges) and
    `smiles`. `retry` submits featurize again after a recorded failure."""

    molblock: str | None = None
    smiles: str | None = None
    retry: bool = False


class SketchMappingOut(_Out):
    """Sketch atom i <-> record atom r (`prepare_mol(key)` order, H explicit)."""

    n_sketch_atoms: int
    n_record_atoms: int
    sketch_to_record: list[int]
    record_to_sketch: list[int | None]  # None: an implicit H of the drawing
    record_parent_sketch: list[int | None]  # an implicit H -> its drawn heavy atom
    chirality_matched: bool  # False: matched only with chirality ignored


class MoleculeOut(_Out):
    molecule_id: str
    key: str  # data.chem.unmapped_canonical: the QM / token cache key
    status: Literal["missing", "running", "ready", "qm_only", "failed"]
    error: ErrorBody | None  # the recorded featurize failure (status failed or qm_only)
    n_atoms: int | None
    charge: int | None
    multiplicity: int | None
    chemistry: str
    method: str
    qm_ready: bool
    tokens_ready: bool
    mapping: SketchMappingOut | None  # only in the response to the POST of a drawing
    mapping_error: str | None  # MAPPING_FAILED when the posted drawing did not match
    warnings: list[str]


class WorkbenchProvenanceOut(_Out):
    chemistry: str
    method: str
    key: str
    qm_digest: str  # sha256[:16] of the QM record file
    token_digest: str | None  # sha256[:16] of the model token file, when one was used
    versions: dict[str, int | str]
    notes: list[str]


class MoleculeGeometryOut(_Out):
    molecule_id: str
    key: str
    charge: int
    multiplicity: int
    atoms: list[AtomXYZOut]  # global_site = record index, map None
    bonds: list[tuple[int, int, int]]  # (record i, record j, order; 4 = aromatic)
    xyz: str
    molblock: str
    frame_note: str


class GlobalPropsOut(_Out):
    energy_ha: float
    energy_ev: float
    homo_ev: float | None
    lumo_ev: float | None
    gap_ev: float | None
    mu_ev: float | None  # chemical potential (E_HOMO + E_LUMO) / 2
    eta_ev: float | None  # hardness (E_LUMO - E_HOMO) / 2
    omega_ev: float | None  # electrophilicity mu^2 / (2 eta)
    charge: int
    multiplicity: int
    n_atoms: int
    n_ao: int
    n_mo: int
    n_occupied: int
    n_valence_electrons: float
    fermi_smeared: bool
    pm_converged: bool | None  # of the model token file (None: no token file)
    pm_sweeps_occ: int | None
    pm_sweeps_virt: int | None
    warnings: list[str]


class AtomDiagnosticOut(_Out):
    """model_input=False: chemist's labels, display only."""

    diagnostic: Literal[True] = True
    hybridization: str
    aromatic: bool
    fg_label: str


class AtomPropsOut(_Out):
    index: int  # record atom index
    element: str
    Z: int
    formal_charge: int
    mulliken_charge: float
    f_plus_fd: float | None  # finite-difference condensed Fukui (diagnostic in the tokens)
    f_minus_fd: float | None
    f_zero_fd: float | None
    dual_fd: float | None  # the dual descriptor the tokens carry (model input)
    f_plus_fmo: float | None  # frozen-orbital, LUMO block / its size (model input)
    f_minus_fmo: float | None
    fmo_weights: list[float | None]  # HOMO-1, HOMO, LUMO, LUMO+1 block populations
    vbur: float | None  # %Vbur, 3.5 Angstrom sphere
    chirality: float  # +1 / -1 on a specified stereocentre, else 0
    wiberg_valence: float  # sum of the atom's Wiberg indices
    diagnostic: AtomDiagnosticOut


class BondPropsOut(_Out):
    i: int
    j: int
    bonded: bool  # a Lewis bond of the record; False: a non-bonded pair with W >= threshold
    order: int | None  # Lewis order (4 = aromatic, not kekulized); None when not bonded
    wiberg: float
    length_ang: float


class MoleculePropertiesOut(_Out):
    molecule_id: str
    key: str
    global_props: GlobalPropsOut
    atoms: list[AtomPropsOut]
    bonds: list[BondPropsOut]
    fmo_levels: list[str]
    wiberg_threshold: float
    geometry: MoleculeGeometryOut
    provenance: WorkbenchProvenanceOut


class OrbitalCentreOut(_Out):
    atom: int  # record atom index
    element: str
    weight: float  # Mulliken (canonical) or the descriptor population (LMO)


class OrbitalRowDiagnosticOut(_Out):
    """model_input=False: the classifier's labels, display only."""

    diagnostic: Literal[True] = True
    type: str
    parents: str
    s_share: float | None
    p_share: float | None
    d_share: float | None
    classifier_overlap_population: float | None
    metal_bound: bool


class OrbitalRowOut(_Out):
    index: int  # the orbital's index in its set (the cube's `index`)
    kind: Literal["occupied", "virtual"]
    occupation: float
    energy_ev: float  # canonical eps_k, or the LMO's E_r = sum_k p_rk E_k
    label: str | None  # canonical: HOMO, HOMO-2, LUMO+1, ...
    frontier: str | None  # canonical: homo_minus_1 | homo | lumo | lumo_plus_1 block member
    level: int | None  # canonical: degenerate level (0.05 eV), ascending energy
    centres: list[OrbitalCentreOut]
    centroid: list[float] | None
    model_input: OrbitalModelInputOut | None  # LMO: exactly the model-input descriptors
    spectral: list[tuple[int, float, float]] | None  # LMO: (canonical index, E eV, weight)
    retained_fraction: float | None
    diagnostic: OrbitalRowDiagnosticOut | None


class OrbitalSetStatsOut(_Out):
    n_occupied: int
    n_virtual: int
    converged: bool | None  # localization convergence (None: canonical)
    sweeps_occ: int | None
    sweeps_virt: int | None
    orthonormality_error: float  # max |C^T S C - I|
    occupied_projector_error: float  # max |C_occ C_occ^T - (canonical)|, AO basis


class OrbitalSetOut(_Out):
    molecule_id: str
    key: str
    set: str  # canonical | ibo | pm_mulliken
    start: str | None  # cholesky | canonical (None for the canonical set)
    is_model_set: bool  # the localization the model's tokens use
    source: Literal["canonical", "token_file", "workbench"]
    homo_index: int | None
    orbitals: list[OrbitalRowOut]
    stats: OrbitalSetStatsOut
    sets: list[str]
    starts: list[str]
    model_set: tuple[str, str]
    notes: list[str]
    provenance: WorkbenchProvenanceOut


class Depict2DViewIn(_In):
    """The 3D viewer's orientation: `quaternion` = 3Dmol getView()[4..7] [x, y, z, w]; the view
    frame is R(q) p (x right, y up, z towards the viewer), as sketch/projection.ts."""

    quaternion: list[float] = Field(min_length=4, max_length=4)
    convention: Literal["3dmol-rotationGroup"] = "3dmol-rotationGroup"


class Depict2DIn(_In):
    """POST /depict2d: a 2D drawing of the sketch (V2000 molblock). match3d needs the computed
    molecule's id and the view."""

    molblock: str
    mode: Literal["clean", "match3d"] = "clean"
    style: Literal["textbook", "perspective"] = "textbook"
    molecule_id: str | None = None
    view: Depict2DViewIn | None = None


class Depict2DWedgeOut(_Out):
    bond: int  # bond index of the input molblock
    narrow: int  # atom index at the narrow end (the stereocentre)
    kind: Literal["wedge", "hash"]


class Depict2DDefectsOut(_Out):
    close_contacts: int
    coincident: int
    label_overlaps: int
    atom_on_bond: int
    crossings: int
    small_angles: int
    inside_ring: int
    squashed_rings: int
    bond_length_rms: float
    score: float


class Depict2DCandidateOut(_Out):
    method: str
    total: float  # defect score + 20 x view residual (+ 50 when the stereo differs)
    score: float | None
    residual: float | None
    stereo: str


class Depict2DOut(_Out):
    """The drawing in EXACTLY the input atom and bond order: coords[i] of atom i (bond 1.5),
    depth[i] (+ towards the viewer) when the drawing has one, wedges by bond index."""

    mode: Literal["clean", "match3d"]
    style: Literal["textbook", "perspective"]
    method: str  # rdkit-templates | coordgen | perspective:<t> | match3d-<rdkit|projected|...>
    template: str | None  # the perspective template used
    molblock: str
    coords: list[tuple[float, float]]
    depth: list[float] | None
    wedges: list[Depict2DWedgeOut]
    smiles: str  # canonical isomeric SMILES RDKit reads from the molblock
    stereo: Literal["none", "kept", "mismatch"]
    view_residual: float | None
    defects: Depict2DDefectsOut
    candidates: list[Depict2DCandidateOut]
    notes: list[str]
    version: int
    molecule_id: str | None


class SymmetryElementOut(_Out):
    """One symmetry element of the point group, in the QM record's frame."""

    id: str  # e0, e1, ... (stable for one response)
    kind: Literal["C", "S", "sigma", "i"]  # proper axis, improper axis, mirror, inversion
    label: str  # C3, C2′, C∞, S4, σh, σv, σd, σ, ∞σv, i
    order: int | None  # n of C_n / S_n; None for a mirror, i and C∞
    vector: tuple[float, float, float] | None  # unit axis or plane normal; None for i
    point: tuple[float, float, float]  # the centre of mass, on every element
    atoms: list[int]  # record atoms on the element within the tolerance
    deviation_ang: float  # the largest atom residual of its operations
    principal: bool
    n_operations: int  # group operations about this element (0: an infinite family)


class SymmetryClassOut(_Out):
    element: str
    atoms: list[int]  # symmetry-equivalent record atoms (an orbit of the group)


class PointGroupLooserOut(_Out):
    point_group: str
    order: int | None
    tolerance: float  # the smallest hint tolerance that gives it


class PointGroupOut(_Out):
    """GET /molecule/{id}/symmetry: the Schoenflies point group of the record geometry."""

    molecule_id: str
    key: str
    tolerance: float  # Angstrom: every atom maps within it onto an atom of the same element
    point_group: str  # Schoenflies symbol: C1, Cs, C2v, D6h, Td, Ih, C∞v, D∞h, ...
    order: int | None  # None: an infinite group (linear, atom)
    top: Literal["atom", "linear", "spherical", "symmetric", "asymmetric"]
    moments: tuple[float, float, float]  # principal moments of inertia, amu Å², ascending
    n_atoms: int
    centre: tuple[float, float, float]  # centre of mass, record frame
    elements: list[SymmetryElementOut]
    classes: list[SymmetryClassOut]
    rms_ang: float  # RMS displacement of the symmetrized structure from the record
    max_deviation_ang: float
    symmetrized_xyz: list[tuple[float, float, float]]  # record atom order
    looser: PointGroupLooserOut | None  # a higher group at a looser tolerance, if any
    notes: list[str]
    qm_digest: str


# ---- the TypeScript mirror

TS_MODELS: list[type[BaseModel]] = [
    obj
    for obj in list(globals().values())
    if isinstance(obj, type)
    and issubclass(obj, BaseModel)
    and obj.__module__ == __name__
    and not obj.__name__.startswith("_")
]
for _model in TS_MODELS:
    _model.model_rebuild()

TS_HEADER = (
    "// Generated by ksmech.serve.schemas.write_typescript() from the pydantic models in\n"
    "// src/ksmech/serve/schemas.py. Do not edit: regenerate with\n"
    "//   uv run python -c 'from ksmech.serve.schemas import write_typescript; "
    "write_typescript()'\n"
    "// (or `uv run ksmech-serve schema-ts`). tests/test_serve_schemas_ts.py fails on drift."
)
_IDENT = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*$")
_REF_SUFFIX = re.compile(r"-(Input|Output)$")


def schema_mode(model: type[BaseModel]) -> Literal["validation", "serialization"]:
    """Request models describe what the client sends; responses what the server returns."""
    return "validation" if issubclass(model, _In) else "serialization"


def _ref_name(ref: str) -> str:
    return _REF_SUFFIX.sub("", ref.rsplit("/", 1)[-1])


def _literal(value: Any) -> str:
    return json.dumps(value)


def _union(parts: list[str]) -> str:
    out: list[str] = []
    for p in parts:
        if p not in out:
            out.append(p)
    return " | ".join(out) if out else "never"


def _array_of(item: str) -> str:
    return f"({item})[]" if " | " in item else f"{item}[]"


def _ts_object(s: dict) -> str:
    props = s.get("properties")
    if props:
        required = set(s.get("required", []))
        fields = [_ts_property(k, v, k in required) for k, v in props.items()]
        return "{ " + " ".join(f + ";" for f in fields) + " }"
    extra = s.get("additionalProperties", True)
    value = "unknown" if extra is True or extra == {} else _ts_type(extra)
    return f"Record<string, {value}>"


def _ts_array(s: dict) -> str:
    if "prefixItems" in s:
        return "[" + ", ".join(_ts_type(x) for x in s["prefixItems"]) + "]"
    items = s.get("items")
    return _array_of("unknown" if not items else _ts_type(items))


_SCALARS = {"string": "string", "integer": "number", "number": "number", "boolean": "boolean"}


def _ts_type(s: dict | bool) -> str:
    """One JSON-schema node as a TypeScript type (the mapping of design §6.2)."""
    if s is True or s == {}:
        return "unknown"
    if s is False:
        return "never"
    if "$ref" in s:
        return _ref_name(s["$ref"])
    if "const" in s:
        return _literal(s["const"])
    if "enum" in s:
        return _union([_literal(v) for v in s["enum"]])
    for key in ("anyOf", "oneOf"):
        if key in s:
            return _union([_ts_type(x) for x in s[key]])
    if "allOf" in s and len(s["allOf"]) == 1:
        return _ts_type(s["allOf"][0])
    kind = s.get("type")
    if isinstance(kind, list):
        return _union([_ts_type({**s, "type": k}) for k in kind])
    if kind == "null":
        return "null"
    if kind == "array":
        return _ts_array(s)
    if kind == "object":
        return _ts_object(s)
    if kind in _SCALARS:
        return _SCALARS[kind]
    return "unknown"


def _ts_property(name: str, s: dict, required: bool) -> str:
    key = name if _IDENT.match(name) else json.dumps(name)
    return f"{key}{'' if required else '?'}: {_ts_type(s)}"


def _interface(name: str, schema: dict) -> str:
    required = set(schema.get("required", []))
    lines = [f"export interface {name} {{"]
    for prop, s in schema.get("properties", {}).items():
        lines.append(f"  {_ts_property(prop, s, prop in required)};")
    lines.append("}")
    return "\n".join(lines)


def typescript() -> str:
    """The TypeScript interfaces of every model in `TS_MODELS`, in definition order.

    Each model becomes one `export interface`; a `$ref` names the referenced model's interface.
    A `$defs` entry that is not itself in `TS_MODELS` is emitted after them, by name.
    """
    blocks = [TS_HEADER]
    names = {m.__name__ for m in TS_MODELS}
    extra: dict[str, dict] = {}
    for model in TS_MODELS:
        schema = model.model_json_schema(mode=schema_mode(model))
        for def_name, d in schema.get("$defs", {}).items():
            if _ref_name(def_name) not in names:
                extra.setdefault(_ref_name(def_name), d)
        blocks.append(_interface(model.__name__, schema))
    blocks.extend(_interface(n, extra[n]) for n in sorted(extra))
    return "\n\n".join(blocks) + "\n"


def default_typescript_path() -> Path:
    import ksmech

    repo = Path(ksmech.__file__).resolve().parents[2]
    return repo / "apps" / "ui" / "src" / "api" / "types.gen.ts"


def write_typescript(out: Path | None = None) -> Path:
    """Write `typescript()` to `out` (default `<repo>/apps/ui/src/api/types.gen.ts`)."""
    path = Path(out) if out is not None else default_typescript_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(typescript(), encoding="utf-8")
    return path
