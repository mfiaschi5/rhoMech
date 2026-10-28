"""Source-independent ingest bookkeeping: collect accepted steps and rejections, write outputs."""

from __future__ import annotations

import csv
import json
import logging
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ksmech.data.io import source_dir, write_resonance_pairs, write_steps
from ksmech.data.schema import ElementaryStep, Source
from ksmech.data.validation import Rejection, RejectReason, StepRejected

log = logging.getLogger(__name__)

FROZEN_SAMPLE_SCHEMA = 1  # `sample_frozen.json`: a pilot's step ids, independent of every version


@dataclass
class IngestResult:
    source: Source
    inputs: list[str] = field(default_factory=list)
    n_records: int = 0
    steps: list[ElementaryStep] = field(default_factory=list)
    rejections: list[Rejection] = field(default_factory=list)
    notes: Counter[str] = field(default_factory=Counter)  # non-fatal normalizations, counted
    _seen: set[str] = field(default_factory=set, repr=False)
    sample: dict | None = None  # (`keep_sample`) n, seed and the accepted steps drawn from

    def add(self, step_id: str, raw: str, make: Callable[[], ElementaryStep]) -> None:
        """Run one record through `make` and record either the step or the rejection."""
        self.n_records += 1
        if step_id in self._seen:
            self.reject(step_id, raw, StepRejected(RejectReason.DUPLICATE_STEP, "repeat record"))
            return
        self._seen.add(step_id)
        try:
            step = make()
        except StepRejected as exc:
            self.reject(step_id, raw, exc)
            return
        self.steps.append(step)

    def reject(self, step_id: str, raw: str, exc: StepRejected) -> None:
        log.warning("rejected %s [%s] %s", step_id, exc.reason, exc.detail)
        self.rejections.append(Rejection(step_id, exc.reason, exc.detail, raw))

    def keep_sample(self, n: int, seed: int, frozen: set[str] | None = None) -> None:
        """Keep `n` accepted steps drawn uniformly at random (seeded; independent of file order:
        the draw is over the sorted step ids); the rest are counted in `sample`, not rejected.
        Rejections stay those of every record read. With `frozen` (a pilot's
        `sample_frozen.json`) the kept steps are exactly the frozen ids that are still
        accepted, so a rule that changes the accepted set never redraws the pilot: accepted
        steps outside it count as not sampled, frozen ids no longer accepted as missing."""
        if frozen is not None:
            have = {s.step_id for s in self.steps}
            self.sample = {"n": n, "seed": seed, "of_accepted": len(self.steps)}
            self.sample |= {"frozen": len(frozen), "frozen_missing": len(frozen - have)}
            self.steps = [s for s in self.steps if s.step_id in frozen]
            return
        if n >= len(self.steps):
            return
        ids = sorted(s.step_id for s in self.steps)
        keep = set(np.random.default_rng(seed).choice(ids, size=n, replace=False).tolist())
        self.sample = {"n": n, "seed": seed, "of_accepted": len(self.steps)}
        self.steps = [s for s in self.steps if s.step_id in keep]

    def reason_counts(self) -> Counter[str]:
        return Counter(str(r.reason) for r in self.rejections)

    @property
    def rejected_fraction(self) -> float:
        return len(self.rejections) / self.n_records if self.n_records else 0.0

    def report(self) -> dict:
        return {
            "source": str(self.source),
            "inputs": self.inputs,
            "n_records": self.n_records,
            "n_accepted": len(self.steps),
            "n_resonance_steps": sum(s.resonance_step for s in self.steps),
            "n_resonance_pairs": sum(
                s.resonance_step and s.domain_exclusion is None for s in self.steps
            ),
            "domain_exclusions": dict(
                Counter(s.domain_exclusion for s in self.steps if s.domain_exclusion).most_common()
            ),
            "n_charge_repaired": sum(s.charge_repair is not None for s in self.steps),
            "n_rejected": len(self.rejections),
            "rejected_fraction": self.rejected_fraction,
            "rejections_by_reason": dict(self.reason_counts().most_common()),
            "notes": dict(self.notes.most_common()),
            "sample": self.sample,
        }


def write_outputs(result: IngestResult, processed_dir: Path) -> dict[str, Path]:
    """Write steps.parquet, resonance_pairs.parquet, rejections.csv and ingest_report.json under
    <processed>/<source>/."""
    out = source_dir(result.source, processed_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "steps": write_steps(result.steps, out / "steps.parquet"),
        "resonance_pairs": write_resonance_pairs(result.steps, out / "resonance_pairs.parquet"),
        "rejections": out / "rejections.csv",
        "report": out / "ingest_report.json",
        "repairs": out / "charge_repairs.csv",
    }
    with open(paths["rejections"], "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step_id", "reason", "detail", "raw"])
        for r in result.rejections:
            writer.writerow([r.step_id, str(r.reason), r.detail, r.raw])
    with open(paths["repairs"], "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step_id", "side", "rule", "before", "after", "domain_exclusion"])
        for s in result.steps:
            for r in json.loads(s.charge_repair)["repairs"] if s.charge_repair else []:
                row = [r["side"], r["rule"], r["before"], r["after"], s.domain_exclusion or ""]
                writer.writerow([s.step_id, *row])
    paths["report"].write_text(json.dumps(result.report(), indent=2) + "\n")
    return paths


def frozen_sample_path(source: Source | str, processed_dir: Path) -> Path:
    return source_dir(source, processed_dir) / "sample_frozen.json"


def freeze_sample(source: Source | str, processed_dir: Path) -> Path:
    """Snapshot the step ids of a sampled source's steps.parquet (a pilot drawn by
    `ingest --sample`) as `sample_frozen.json`, stamped with no file input (a content hash),
    so no later re-ingest makes it stale. From then on `ingest --sample` keeps exactly these
    ids. An existing snapshot is never overwritten: a new pilot is an explicit new version."""
    import pandas as pd

    from ksmech.data import provenance as prov

    folder = source_dir(source, processed_dir)
    out = frozen_sample_path(source, processed_dir)
    if out.exists():
        raise FileExistsError(f"{out} exists: a frozen sample is replaced only by hand")
    sample = json.loads((folder / "ingest_report.json").read_text()).get("sample")
    if not sample:
        raise ValueError(f"{source} was not ingested with --sample: nothing to freeze")
    ids = sorted(pd.read_parquet(folder / "steps.parquet", columns=["step_id"])["step_id"])
    content = prov.sha256_text(json.dumps(ids))
    header = prov.stamp("sample_frozen", FROZEN_SAMPLE_SCHEMA, {}, sample, content)
    return prov.write_json(out, {"ids": ids}, header)


def load_frozen_sample(source: Source | str, processed_dir: Path) -> set[str] | None:
    """The frozen pilot ids, None when the source has no snapshot; kind, schema, clean build
    and content hash checked."""
    from ksmech.data import provenance as prov

    path = frozen_sample_path(source, processed_dir)
    if not path.exists():
        return None
    header = prov.verify_clean(path, "sample_frozen", FROZEN_SAMPLE_SCHEMA)
    ids = json.loads(path.read_text())["ids"]
    if prov.sha256_text(json.dumps(ids)) != header.content_sha256:
        raise prov.StaleArtifact(path, ["its ids differ from its content hash"])
    return set(ids)
