"""Browse: exact and substructure search over PMechDB and curated steps, and the step view
(design 10, 6.4 `BrowseService`). API side.

- **Sources.** PMechDB steps come from the index (`pmindex`), which is built only on request:
  a search that asks for PMechDB while the index is not ready is 409 `INDEX_NOT_READY` with
  detail {state, job_id, build}; a search never starts a job. Curated steps are read live from
  the curation database (`Mechanism`, `CuratedStep`): species from splitting the pool keys on
  ".", token keys from `keys.side_keys` of the StepDoc's mapped sides; status = the
  mechanism's status.
- **exact**: the query (SMILES) is reduced to species keys per fragment, a Counter Q; a step
  matches when Q is a sub-multiset of the chosen side (`either`: reactants or products).
  `match="token"` compares token keys instead.
- **substructure**: the query is parsed as SMARTS (falling back to SMILES); every query
  fragment must match (`HasSubstructMatch`) some recorded form of some species of the side.
  The forms of a species are its key and every token key recorded for it anywhere in the
  index or the curated steps, because the species key is only the lexicographically smallest
  resonance form (an enolate stored as its carbanion would miss C=O). `matched_species` and
  `matched_forms` name, per query fragment, the species and the form that matched. One search
  runs at a time (a lock); at most 20,000 `HasSubstructMatch` calls are made, after which the
  result is `truncated`.
- **filters**: source, status (dataset | predicted | validated | corrected | rejected),
  resonance; paging by limit / offset over the order source, then step_id or
  (mechanism_id, step_index).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rdkit import Chem, rdBase

from ksmech.serve import db, schemas
from ksmech.serve.contracts import LRU, JobContext, JobSubmitter
from ksmech.serve.depict import step_depiction
from ksmech.serve.errors import Code, ServeError
from ksmech.serve.keys import side_keys, species_key_of
from ksmech.serve.pmindex import (
    JOBS,
    SIDES,
    PMIndex,
    index_key,
    index_path,
    index_row,
    load_index,
    not_ready,
    run_index,
    side_columns,
    source_path,
    source_status,
)
from ksmech.serve.settings import append_failure

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine
    from sqlmodel import Session

    from ksmech.serve.context import ApiContext

log = logging.getLogger(__name__)

SOURCES = ("pmechdb", "user_curated")
STATUSES = ("dataset", "predicted", "validated", "corrected", "rejected")
MODES = ("exact", "substructure")
SIDE_CHOICES = ("reactants", "products", "either")
MATCHES = ("species", "token")
TERMINAL = ("done", "failed", "cancelled")
MAX_LIMIT = 500
SUBSTRUCTURE_CAP = 20_000
MOL_CACHE = 20_000
REF_RE = re.compile(r"^(?:pmechdb:(?P<step>[^:\s]+)|curated:(?P<mid>[0-9]+):(?P<idx>[0-9]+))$")
_BAD_MOL = Chem.Mol()  # LRU marker of a form RDKit cannot parse


@dataclass(frozen=True)
class SearchQuery:
    q: str
    mode: str = "exact"
    side: str = "either"
    sources: tuple[str, ...] = SOURCES
    statuses: frozenset[str] | None = None
    resonance: bool | None = None
    limit: int = 50
    offset: int = 0
    match: str = "species"

    @property
    def sides(self) -> tuple[str, ...]:
        return SIDES if self.side == "either" else (self.side,)


def _invalid(msg: str) -> ServeError:
    return ServeError(Code.QUERY_INVALID, msg, status=422)


def _one_of(value: str, choices: Sequence[str], name: str) -> str:
    if value not in choices:
        raise _invalid(f"{name} must be one of {list(choices)}, got {value!r}")
    return value


def parse_query(
    q: str | None,
    *,
    mode: str = "exact",
    side: str = "either",
    sources: Iterable[str] | None = SOURCES,
    status: str | Iterable[str] | None = None,
    resonance: bool | None = None,
    limit: int = 50,
    offset: int = 0,
    match: str = "species",
) -> SearchQuery:
    """A validated `SearchQuery`; QUERY_INVALID (422) for a bad parameter."""
    srcs = tuple(dict.fromkeys(sources or SOURCES))
    for src in srcs:
        _one_of(src, SOURCES, "source")
    statuses = None
    if status:
        parts = status.split(",") if isinstance(status, str) else list(status)
        statuses = frozenset(_one_of(p.strip(), STATUSES, "status") for p in parts if p.strip())
    if not 1 <= int(limit) <= MAX_LIMIT or int(offset) < 0:
        raise _invalid(f"limit must lie in 1..{MAX_LIMIT} and offset be >= 0")
    return SearchQuery(
        (q or "").strip(),
        _one_of(mode, MODES, "mode"),
        _one_of(side, SIDE_CHOICES, "side"),
        srcs,
        statuses or None,
        resonance,
        int(limit),
        int(offset),
        _one_of(match, MATCHES, "match"),
    )


@dataclass
class StepEntry:
    """One searchable step: its `StepSummaryOut` fields (without the matched lists), the
    species and token keys of each side, and the dataset arrows of the step view."""

    summary: dict
    species: dict[str, list[str]]
    tokens: dict[str, list[str]]
    pairs: dict[str, list[tuple[str, str]]]  # side -> (species key, token key) per fragment
    arrows: list[dict] = field(default_factory=list)
    keep_maps: set[int] = field(default_factory=set)

    @property
    def order(self) -> tuple:
        s = self.summary
        if s["source"] == "pmechdb":
            return (0, s["step_id"], 0)
        return (1, int(s["mechanism_id"]), int(s["step_index"]))


def arrow_maps(arrows: Iterable[dict]) -> set[int]:
    """Every map number an arrow touches (dataset arrows and StepDoc token arrows)."""
    out: set[int] = set()
    for a in arrows:
        out.update(int(x) for x in a.get("source_atoms", []) or [])
        out.update(int(x) for x in a.get("sink_atoms", []) or [])
        for k in ("donor_site_map", "acceptor_site_map", "source_partner_map", "sink_partner_map"):
            if a.get(k) is not None:
                out.add(int(a[k]))
    return out


def pmechdb_entry(row: dict) -> StepEntry:
    """A `StepEntry` from one index row."""
    splits = {"random": row.get("split_random"), "scaffold": row.get("split_scaffold")}
    summary = {
        "ref": f"pmechdb:{row['step_id']}",
        "source": "pmechdb",
        "status": "dataset",
        "step_id": row["step_id"],
        "mechanism_id": None,
        "step_index": None,
        "reactants_smiles": row["reactants_smiles"],
        "products_smiles": row["products_smiles"],
        "reactant_pool_key": row["reactant_pool_key"],
        "product_pool_key": row["product_pool_key"],
        "n_arrows": int(row["n_arrows"]),
        "pmechdb_code": row.get("pmechdb_code"),
        "resonance_step": bool(row["resonance_step"]),
        "splits": {k: v for k, v in splits.items() if v} or None,
        "exported_step_id": None,
    }
    species, tokens, pairs = {}, {}, {}
    for side in SIDES:
        sp_col, tok_col = side_columns(side)
        species[side], tokens[side] = list(row[sp_col]), list(row[tok_col])
        pairs[side] = list(zip(species[side], tokens[side], strict=True))
    arrows = json.loads(row.get("arrows_json") or "[]")
    return StepEntry(summary, species, tokens, pairs, arrows, arrow_maps(arrows))


# ---- matching


class _CapReached(Exception):  # noqa: N818 - control flow, not an error
    pass


def _query_fragments(q: str, mode: str, match: str) -> list:
    """Counter of query keys (exact) or the query fragment mols (substructure)."""
    from ksmech.data import chem

    with rdBase.BlockLogs():
        if mode == "exact":
            mol = Chem.MolFromSmiles(q)
        else:
            mol = Chem.MolFromSmarts(q) or Chem.MolFromSmiles(q)
    if mol is None or mol.GetNumAtoms() == 0:
        raise _invalid(f"RDKit cannot parse the query {q!r}")
    if mode == "substructure":
        return list(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False))
    out = []
    for frag in chem.fragments(mol):
        try:
            token = chem.unmapped_canonical(frag)
            out.append(token if match == "token" else species_key_of(token)[0])
        except (chem.SanitizeError, ValueError, ServeError) as exc:
            raise _invalid(
                f"the query fragment {Chem.MolToSmiles(frag)} has no key: {exc}"
            ) from exc
    return out


class BrowseIndex:
    """The searchable steps of one search: the loaded PMechDB index (or None) and the curated
    entries, with the shared mol LRU."""

    def __init__(
        self,
        pm: PMIndex | None,
        curated: list[StepEntry],
        mols: LRU[str, Chem.Mol] | None = None,
        cap: int = SUBSTRUCTURE_CAP,
    ) -> None:
        self.pm = pm
        self.curated = sorted(curated, key=lambda e: e.order)
        self.mols = mols if mols is not None else LRU(MOL_CACHE)
        self.cap = int(cap)
        self.forms: dict[str, set[str]] = {k: set(v) for k, v in (pm.forms if pm else {}).items()}
        for entry in self.curated:
            for pairs in entry.pairs.values():
                for species, token in pairs:
                    self.forms.setdefault(species, set()).add(token)

    # -- candidates and filters

    def _pm_positions(self, q: SearchQuery, keys: list[str] | None) -> list[int]:
        if self.pm is None:
            return []
        if not keys:
            return list(range(self.pm.n_steps))
        maps = self.pm.by_token if q.match == "token" else self.pm.by_species
        hits: set[int] = set()
        for side in q.sides:
            sets = [maps[side].get(k, set()) for k in set(keys)]
            hits |= set.intersection(*sets) if sets else set()
        return sorted(hits)

    def _entries(self, q: SearchQuery, keys: list[str] | None) -> Iterable[StepEntry]:
        if "pmechdb" in q.sources and self.pm is not None:
            for pos in self._pm_positions(q, keys):
                entry = pmechdb_entry(self.pm.rows[pos])
                if self._keep(q, entry):
                    yield entry
        if "user_curated" in q.sources:
            yield from (e for e in self.curated if self._keep(q, e))

    @staticmethod
    def _keep(q: SearchQuery, entry: StepEntry) -> bool:
        s = entry.summary
        if q.statuses is not None and s["status"] not in q.statuses:
            return False
        return q.resonance is None or bool(s["resonance_step"]) == bool(q.resonance)

    # -- exact

    @staticmethod
    def _exact(
        q: SearchQuery, entry: StepEntry, keys: list[str]
    ) -> tuple[list[str], list[str]] | None:
        want = Counter(keys)
        for side in q.sides:
            have = Counter(entry.tokens[side] if q.match == "token" else entry.species[side])
            if all(have[k] >= n for k, n in want.items()):
                return list(dict.fromkeys(keys)), []
        return None

    # -- substructure

    def _mol(self, smiles: str) -> Chem.Mol | None:
        mol = self.mols.get(smiles)
        if mol is None:
            with rdBase.BlockLogs():
                mol = Chem.MolFromSmiles(smiles) or _BAD_MOL
            self.mols.put(smiles, mol)
        return None if mol.GetNumAtoms() == 0 else mol

    def _form_of(self, species: str, fi: int, frag, memo: dict, calls: list[int]) -> str | None:
        key = (fi, species)
        if key not in memo:
            memo[key] = None
            for form in [species, *sorted(self.forms.get(species, set()) - {species})]:
                mol = self._mol(form)
                if mol is None:
                    continue
                if calls[0] >= self.cap:
                    del memo[key]
                    raise _CapReached
                calls[0] += 1
                if mol.HasSubstructMatch(frag):
                    memo[key] = form
                    break
        return memo[key]

    def _substructure(self, q, entry, frags, memo, calls) -> tuple[list[str], list[str]] | None:
        for side in q.sides:
            found = []
            for fi, frag in enumerate(frags):
                hit = None
                for species in dict.fromkeys(entry.species[side]):
                    form = self._form_of(species, fi, frag, memo, calls)
                    if form is not None:
                        hit = (species, form)
                        break
                if hit is None:
                    break
                found.append(hit)
            if len(found) == len(frags):
                pairs = list(dict.fromkeys(found))
                return [p[0] for p in pairs], [p[1] for p in pairs]
        return None

    def search(self, q: SearchQuery) -> dict:
        """{"total", "truncated", "results": [StepSummaryOut fields]} of one query."""
        parsed = _query_fragments(q.q, q.mode, q.match) if q.q else None
        keys = parsed if q.mode == "exact" else None
        matches, truncated, memo, calls = [], False, {}, [0]
        for entry in self._entries(q, keys):
            if parsed is None:
                hit = ([], [])
            elif q.mode == "exact":
                hit = self._exact(q, entry, parsed)
            else:
                try:
                    hit = self._substructure(q, entry, parsed, memo, calls)
                except _CapReached:
                    truncated = True
                    break
            if hit is not None:
                matches.append(entry.summary | {"matched_species": hit[0], "matched_forms": hit[1]})
        page = matches[q.offset : q.offset + q.limit]
        return {"total": len(matches), "truncated": truncated, "results": page, "calls": calls[0]}

    def step_detail(self, ref: str) -> dict:
        """The `StepSummaryOut` fields of one indexed step, with its arrows and the maps they
        touch; STEP_NOT_FOUND (404) when it is not here."""
        entry = self.entry(ref)
        return entry.summary | {
            "matched_species": [],
            "matched_forms": [],
            "arrows": entry.arrows,
            "keep_maps": sorted(entry.keep_maps),
        }

    def entry(self, ref: str) -> StepEntry:
        m = REF_RE.match(str(ref))
        if m and m["step"] and self.pm is not None and m["step"] in self.pm.by_step:
            return pmechdb_entry(self.pm.rows[self.pm.by_step[m["step"]]])
        for entry in self.curated:
            if entry.summary["ref"] == ref:
                return entry
        raise ServeError(Code.STEP_NOT_FOUND, f"no step {ref!r}", status=404)


# ---- curated steps (live from the curation database)


def _doc_digest(st: db.CuratedStep) -> str:
    return hashlib.sha256(st.arrows_json.encode()).hexdigest()[:16]


def _exported_ids(ses: Session, mechanism_id: int | None = None) -> dict[tuple[int, int], str]:
    """(mechanism id, step index) -> step id of the curated steps in the newest export.

    Only the ExportedStep rows whose `last_export_id` is the newest ExportRecord's: a row a
    later export withdrew from steps.parquet keeps its old export id and is left out. Empty
    before the first export. `mechanism_id` limits the rows to one mechanism.
    """
    from sqlmodel import col, select

    newest = ses.exec(
        select(db.ExportRecord).order_by(col(db.ExportRecord.created_at).desc()).limit(1)
    ).first()
    if newest is None:
        return {}
    q = select(db.ExportedStep).where(col(db.ExportedStep.last_export_id) == newest.export_id)
    if mechanism_id is not None:
        q = q.where(col(db.ExportedStep.mechanism_id) == mechanism_id)
    return {(e.mechanism_id, e.step_index): e.step_id for e in ses.exec(q).all()}


class _Curated:
    """Curated entries from the database rows; StepDoc parsing and side keys are cached per
    (mechanism, step, StepDoc digest), and each failure is reported once per process."""

    def __init__(self, settings) -> None:
        self.settings = settings
        self._cache: LRU[tuple, tuple] = LRU(4096)
        self._reported: set[tuple] = set()

    def _report(self, key: tuple, code: str, detail: str) -> None:
        if key in self._reported:
            return
        self._reported.add(key)
        log.warning("browse: curated step %s skipped part [%s]: %s", key[:2], code, detail)
        append_failure(self.settings, code, detail, mechanism_id=key[0], step_index=key[1])

    def _parsed(self, st: db.CuratedStep) -> tuple[dict, dict]:
        key = (st.mechanism_id, st.step_index, _doc_digest(st))
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        try:
            doc = json.loads(st.arrows_json)
            doc = doc if isinstance(doc, dict) else {}
        except ValueError as exc:
            self._report(key, "STEPDOC_INVALID", str(exc))
            doc = {}
        sides = {}
        for side, mapped in (("reactants", "reactants_mapped"), ("products", "products_mapped")):
            try:
                sides[side] = side_keys(doc[mapped]) if doc.get(mapped) else None
            except ServeError as exc:
                self._report(key + (side,), str(exc.code), exc.message)
                sides[side] = None
        self._cache.put(key, (doc, sides))
        return doc, sides

    def entry(self, st: db.CuratedStep, m: db.Mechanism, exported: dict) -> StepEntry:
        doc, sides = self._parsed(st)
        pools = {"reactants": st.reactant_pool_key, "products": st.product_pool_key}
        species = {side: [k for k in pools[side].split(".") if k] for side in SIDES}
        tokens = {side: list(sides[side].token_keys) if sides[side] else [] for side in SIDES}
        pairs = {
            side: list(zip(sides[side].species_keys, sides[side].token_keys, strict=True))
            if sides[side]
            else [(k, k) for k in species[side]]
            for side in SIDES
        }
        arrows = [a for a in doc.get("dataset_arrows") or [] if _is_dataset_arrow(a)]
        summary = {
            "ref": f"curated:{st.mechanism_id}:{st.step_index}",
            "source": "user_curated",
            "status": m.status,
            "step_id": None,
            "mechanism_id": int(st.mechanism_id),
            "step_index": int(st.step_index),
            "reactants_smiles": doc.get("reactants_mapped") or st.reactant_pool_key,
            "products_smiles": doc.get("products_mapped") or st.product_pool_key,
            "reactant_pool_key": st.reactant_pool_key,
            "product_pool_key": st.product_pool_key,
            "n_arrows": len(doc.get("dataset_arrows") or doc.get("arrows") or []),
            "pmechdb_code": doc.get("pmechdb_code"),
            "resonance_step": st.reactant_pool_key == st.product_pool_key,
            "splits": None,
            "exported_step_id": exported.get((st.mechanism_id, st.step_index)),
        }
        keep = arrow_maps(arrows) | arrow_maps(
            a for a in doc.get("arrows") or [] if isinstance(a, dict)
        )
        return StepEntry(summary, species, tokens, pairs, arrows, keep)

    def entries(self, engine: Engine) -> list[StepEntry]:
        from sqlmodel import select

        with db.session(engine) as ses:
            mechs = {m.id: m for m in ses.exec(select(db.Mechanism)).all()}
            steps = list(ses.exec(select(db.CuratedStep)).all())
            exported = _exported_ids(ses)
        out = []
        for st in sorted(steps, key=lambda x: (x.mechanism_id, x.step_index)):
            m = mechs.get(st.mechanism_id)
            if m is None:
                key = (st.mechanism_id, st.step_index, "mechanism")
                self._report(key, str(Code.MECHANISM_NOT_FOUND), "a curated step without its row")
                continue
            out.append(self.entry(st, m, exported))
        return out


# ---- the service


def _staleness(ctx: ApiContext, m: db.Mechanism, steps: list[db.CuratedStep]) -> list[str]:
    """`curation.staleness` when it is installed, else the minimal rule: spec, physics
    fingerprint or predictor version differ from the current ones."""
    try:
        from ksmech.serve.curation import staleness
    except ImportError:
        staleness = None
    if staleness is not None:
        return list(staleness(m, steps, ctx.registry, ctx.spec_version, ctx.physics_fp))
    out = []
    if m.spec_version != ctx.spec_version:
        out.append("spec_version")
    for st in steps:
        try:
            fp = json.loads(st.arrows_json).get("physics_fingerprint")
        except (ValueError, AttributeError):
            fp = None
        if fp is not None and fp != ctx.physics_fp:
            out.append(f"physics_fingerprint:{st.step_index}")
    try:
        version = ctx.registry.info_of(m.predictor).version
    except ServeError:
        version = None
    if version is not None and version != m.checkpoint_hash:
        out.append("predictor_version")
    return out


class BrowseService:
    """GET /browse/status, POST /browse/index, GET /browse/search, GET /browse/step/{ref}, and
    the export's PMechDB pairs provider."""

    def __init__(
        self,
        ctx: ApiContext,
        engine: Engine,
        *,
        submit: JobSubmitter,
        job_status: Callable[[str], str | None] | None = None,
    ) -> None:
        """`job_status(job_id)` (optional, e.g. the JobStore's status of a job) lets a queued
        index job that ended without running (cancelled) be forgotten at once."""
        self.ctx = ctx
        self.engine = engine
        self.submit = submit
        self.job_status = job_status
        self._search_lock = threading.Lock()
        self._build_lock = threading.Lock()
        self._index_lock = threading.Lock()
        self._index: PMIndex | None = None
        self._mols: LRU[str, Chem.Mol] = LRU(MOL_CACHE)
        self._curated = _Curated(ctx.settings)

    # -- status and build

    def _curated_count(self) -> int:
        from sqlalchemy import func
        from sqlmodel import select

        with db.session(self.engine) as ses:
            return int(ses.exec(select(func.count()).select_from(db.CuratedStep)).one())

    def _sync_jobs(self) -> None:
        """Forget the tracked index job when the runner reports it ended without finishing."""
        key = index_key(self.ctx.settings)
        entry = JOBS.active(key)
        if entry is None or self.job_status is None:
            return
        status = self.job_status(entry.job_id)
        if status in TERMINAL:
            failure = None if status == "done" else {"code": status.upper(), "detail": status}
            JOBS.finished(key, entry.job_id, failure)

    def _status(self) -> dict:
        self._sync_jobs()
        return {"pmechdb": source_status(self.ctx.settings), "curated_steps": self._curated_count()}

    def status(self) -> schemas.IndexStatusOut:
        return schemas.IndexStatusOut(**self._status())

    def build_index(self) -> schemas.JobCreatedOut:
        """Submit the `index` job, or return the one already queued or running."""
        s = self.ctx.settings
        with self._build_lock:
            self._sync_jobs()
            st = source_status(s)
            if st["state"] == "unavailable":
                raise not_ready(st)
            active = JOBS.active(index_key(s))
            if active is not None:
                return schemas.JobCreatedOut(**active.created)
            created = self.submit("index", {})
            JOBS.note_queued(index_key(s), created)
        return schemas.JobCreatedOut(**created)

    # -- search

    def _loaded_index(self) -> PMIndex:
        s = self.ctx.settings
        path = index_path(s, source_path(s))
        with self._index_lock:
            if self._index is None or self._index.path != path:
                self._index = load_index(path)
            if self._index is None:
                raise not_ready(source_status(s))
            return self._index

    def _browse_index(self, sources: Sequence[str], status: dict) -> BrowseIndex:
        pm = None
        if "pmechdb" in sources:
            if status["pmechdb"]["state"] != "ready":
                raise not_ready(status["pmechdb"])
            pm = self._loaded_index()
        curated = self._curated.entries(self.engine) if "user_curated" in sources else []
        return BrowseIndex(pm, curated, self._mols)

    def search(
        self,
        *,
        q: str = "",
        mode: str = "exact",
        side: str = "either",
        sources: Sequence[str] = SOURCES,
        status: str | Sequence[str] | None = None,
        resonance: bool | None = None,
        limit: int = 50,
        offset: int = 0,
        match: str = "species",
    ) -> schemas.SearchOut:
        """Exact or substructure search (design 10); never starts the index job."""
        query = parse_query(
            q,
            mode=mode,
            side=side,
            sources=sources,
            status=status,
            resonance=resonance,
            limit=limit,
            offset=offset,
            match=match,
        )
        index_status = self._status()
        with self._search_lock:
            res = self._browse_index(query.sources, index_status).search(query)
        return schemas.SearchOut(
            total=res["total"],
            truncated=res["truncated"],
            index=schemas.IndexStatusOut(**index_status),
            results=[schemas.StepSummaryOut(**r) for r in res["results"]],
        )

    # -- the step view

    def _pmechdb_entry(self, step_id: str) -> StepEntry:
        s = self.ctx.settings
        src = source_path(s)
        if not src.is_file():
            raise ServeError(Code.STEP_NOT_FOUND, f"no PMechDB steps for {step_id}", status=404)
        if index_path(s, src).is_file():
            pm = self._loaded_index()
            if step_id in pm.by_step:
                return pmechdb_entry(pm.rows[pm.by_step[step_id]])
            raise ServeError(Code.STEP_NOT_FOUND, f"no PMechDB step {step_id}", status=404)
        return pmechdb_entry(_unindexed_row(src, step_id))

    def _curated_entry(self, mid: int, idx: int) -> tuple[StepEntry, db.Mechanism]:
        with db.session(self.engine) as ses:
            st = ses.get(db.CuratedStep, (mid, idx))
            m = ses.get(db.Mechanism, mid)
            exported = _exported_ids(ses, mid)
        if st is None or m is None:
            raise ServeError(Code.STEP_NOT_FOUND, f"no curated step {mid}:{idx}", status=404)
        return self._curated.entry(st, m, exported), m

    def _mechanism(self, m: db.Mechanism) -> tuple[dict, dict[int, db.Recheck]]:
        from sqlmodel import select

        with db.session(self.engine) as ses:
            steps = list(
                ses.exec(select(db.CuratedStep).where(db.CuratedStep.mechanism_id == m.id))
            )
            rechecks = list(ses.exec(select(db.Recheck).where(db.Recheck.mechanism_id == m.id)))
        latest: dict[int, db.Recheck] = {}
        for r in sorted(rechecks, key=lambda r: (r.created_at, r.id or 0)):
            latest[r.step_index] = r
        summary = {
            "id": int(m.id),
            "created_at": m.created_at,
            "reactants": json.loads(m.reactants),
            "stoichiometry": json.loads(m.stoichiometry),
            "status": m.status,
            "predictor": m.predictor,
            "checkpoint_hash": m.checkpoint_hash,
            "spec_version": m.spec_version,
            "n_steps": len(steps),
            "stale": bool(_staleness(self.ctx, m, steps)),
            "notes": m.notes,
            "last_recheck": max((r.created_at for r in rechecks), default=None),
        }
        return summary, latest

    def step(self, ref: str) -> schemas.StepViewOut:
        """One step with both depictions (server SVG and coordinates included), its dataset
        arrows, and, for a curated step, the mechanism and its latest re-check."""
        m = REF_RE.match(str(ref))
        if m is None:
            raise ServeError(Code.STEP_NOT_FOUND, f"malformed step ref {ref!r}", status=404)
        mechanism, recheck = None, None
        if m["step"]:
            entry = self._pmechdb_entry(m["step"])
        else:
            entry, mech = self._curated_entry(int(m["mid"]), int(m["idx"]))
            mechanism, latest = self._mechanism(mech)
            row = latest.get(int(m["idx"]))
            recheck = None if row is None else _recheck_out(row)
        keep, summary = entry.keep_maps, entry.summary
        return schemas.StepViewOut(
            step=schemas.StepSummaryOut(**summary, matched_species=[], matched_forms=[]),
            reactants_depiction=step_depiction(
                summary["reactants_smiles"], keep, True, ref=f"{ref}:reactants"
            ),
            products_depiction=step_depiction(
                summary["products_smiles"], keep, True, ref=f"{ref}:products"
            ),
            dataset_arrows=[schemas.DatasetArrowOut(**_dataset_arrow(a)) for a in entry.arrows],
            mechanism=mechanism,
            recheck=recheck,
        )

    # -- the export's pairs provider

    def pmechdb_pairs(self, ctx: JobContext) -> dict[tuple[str, str], str] | None:
        """(reactant pool key, product pool key) -> PMechDB step_id; the index is built inside
        the calling job (progress "indexing k/N") when it is missing or stale. None without
        the PMechDB parquet."""
        s = ctx.settings
        src = source_path(s)
        if not src.is_file():
            return None
        path = index_path(s, src)
        if not path.is_file():
            ctx.progress(phase="index", done=0, total=None, message="building the PMechDB index")
            run_index(ctx, {"job_kind": "export"}, engine=None)
        pm = load_index(path)
        if pm is None:
            raise not_ready(source_status(s))
        return pm.pairs()


def _is_dataset_arrow(a: object) -> bool:
    keys = ("source_kind", "source_atoms", "sink_kind", "sink_atoms")
    return isinstance(a, dict) and all(k in a for k in keys)


def _dataset_arrow(a: dict) -> dict:
    return {
        "source_kind": str(a["source_kind"]),
        "source_atoms": [int(x) for x in a["source_atoms"]],
        "sink_kind": str(a["sink_kind"]),
        "sink_atoms": [int(x) for x in a["sink_atoms"]],
    }


def _recheck_out(r: db.Recheck) -> dict:
    names = schemas.RecheckRowOut.model_fields
    return {k: getattr(r, k) for k in names}


def _unindexed_row(src, step_id: str) -> dict:
    """The index row of one PMechDB step read straight from the source parquet (the step view
    works before the index is built)."""
    import pyarrow.parquet as pq

    from ksmech.data.schema import STEP_SCHEMA, ElementaryStep

    table = pq.read_table(src, schema=STEP_SCHEMA, filters=[("step_id", "=", step_id)])
    rows = table.to_pylist()
    if not rows:
        raise ServeError(Code.STEP_NOT_FOUND, f"no PMechDB step {step_id}", status=404)
    st = ElementaryStep.from_dict(rows[0])
    raw = {
        "step_id": st.step_id,
        "reactants_smiles": st.reactants_smiles,
        "products_smiles": st.products_smiles,
        "arrows": [a.to_dict() for a in st.arrows],
        "resonance_step": bool(st.resonance_step),
    }
    return index_row(raw) | {"split_random": None, "split_scaffold": None}
