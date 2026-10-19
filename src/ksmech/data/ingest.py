"""Source-independent ingest bookkeeping: collect accepted steps and rejections, write outputs."""

from __future__ import annotations

import csv
import json
import logging
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ksmech.data.io import source_dir, write_steps
from ksmech.data.schema import ElementaryStep, Source
from ksmech.data.validation import Rejection, RejectReason, StepRejected

log = logging.getLogger(__name__)


@dataclass
class IngestResult:
    source: Source
    inputs: list[str] = field(default_factory=list)
    n_records: int = 0
    steps: list[ElementaryStep] = field(default_factory=list)
    rejections: list[Rejection] = field(default_factory=list)
    notes: Counter[str] = field(default_factory=Counter)  # non-fatal normalizations, counted
    sections: dict[str, dict] = field(default_factory=dict)  # extra report blocks, by name
    _seen: set[str] = field(default_factory=set, repr=False)

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
            "n_rejected": len(self.rejections),
            "rejected_fraction": self.rejected_fraction,
            "rejections_by_reason": dict(self.reason_counts().most_common()),
            "notes": dict(self.notes.most_common()),
            **self.sections,
        }


def write_outputs(result: IngestResult, processed_dir: Path) -> dict[str, Path]:
    """Write steps.parquet, rejections.csv and ingest_report.json under <processed>/<source>/."""
    out = source_dir(result.source, processed_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "steps": write_steps(result.steps, out / "steps.parquet"),
        "rejections": out / "rejections.csv",
        "report": out / "ingest_report.json",
    }
    with open(paths["rejections"], "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step_id", "reason", "detail", "raw"])
        for r in result.rejections:
            writer.writerow([r.step_id, str(r.reason), r.detail, r.raw])
    paths["report"].write_text(json.dumps(result.report(), indent=2) + "\n")
    return paths
