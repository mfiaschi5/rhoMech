"""A fake spec-v9 checkpoint for the serve embedding tests and the e2e: a tiny randomly
initialised `KSMech` with identity input statistics (mean 0, std 1: no sample is needed),
saved as `ksmech-train fit` saves one (`train.loop.checkpoint_payload`), with the run's
`run.json` and `metrics.jsonl` beside it.

    uv run --no-sync python -m tests.serve_fake_model <data root>/data/models/e2e/tiny

The provenance header is built by hand (clean unless `dirty`), so the checkpoint loads
whatever the state of the working tree.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

TINY = {
    "d": 32,
    "d_pair": 16,
    "layers": 2,
    "heads": 4,
    "decoder_hidden": 16,
    "physics_hidden": 8,
    "checkpoint": False,
}
METRICS = [
    {"epoch": 0, "step": 1, "val_top1": 0.25, "val_top5": 0.5, "val_loss": 3.0},
    {"epoch": 1, "step": 2, "val_top1": 0.5, "val_top5": 0.75, "val_loss": 2.0},
    {"epoch": 2, "step": 3, "val_top1": 0.4, "val_top5": 0.7, "val_loss": 2.5},
]


def fake_model(encoder: str = "dense", seed: int = 0):
    """A tiny KSMech with identity feature and pair statistics."""
    import torch

    from ksmech.model.config import ModelConfig
    from ksmech.model.inputs import FeatureStats
    from ksmech.model.net import KSMech
    from ksmech.orbitals.tokens import atom_feature_names, feature_names
    from ksmech.physics.pairs import PairStats
    from ksmech.physics.stage_a import STAGE_A_FEATURES

    torch.manual_seed(seed)
    n_atom, n_orb, n_pair = len(atom_feature_names()), len(feature_names()), len(STAGE_A_FEATURES)
    stats = FeatureStats(
        np.zeros(n_atom), np.ones(n_atom), np.zeros(n_orb), np.ones(n_orb), 0.0, 10.0
    )
    pairs = PairStats(np.zeros(n_pair), np.ones(n_pair), np.ones(n_pair), STAGE_A_FEATURES)
    cfg = replace(ModelConfig(**TINY), encoder=encoder)
    return KSMech(cfg, stats, pairs).eval()


def _header(model, dirty: bool):
    from ksmech.data import provenance as prov

    return prov.Provenance(
        kind="train_run",
        schema_version=1,
        code_git_hash="0" * 40,
        spec_sha256="0" * 64,
        inputs={},
        config={"model": model.cfg.to_dict(), "train": {"source": "fake"}},
        built_at=datetime.now(UTC).isoformat(timespec="seconds"),
        code_dirty=dirty,
        code_diff_sha256="f" * 64 if dirty else "",
    )


def write_fake_run(
    run_dir: Path, encoder: str = "dense", dirty: bool = False, files=("best", "last")
) -> list[Path]:
    """`<run_dir>/{best,last}.pt`, `run.json` and `metrics.jsonl` of a fake run."""
    import torch

    from ksmech.train.loop import checkpoint_payload

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    model = fake_model(encoder)
    header = _header(model, dirty)
    out = []
    for file in files:
        best = file == "best"
        m = METRICS[1] if best else METRICS[-1]
        payload = checkpoint_payload(model, header, epoch=m["epoch"], metrics=m)
        torch.save(payload, run_dir / f"{file}.pt")
        out.append(run_dir / f"{file}.pt")
    run_json = {"provenance": asdict(header), "device": "cpu", "amp": "None"}
    (run_dir / "run.json").write_text(json.dumps(run_json, indent=1))
    (run_dir / "metrics.jsonl").write_text("".join(json.dumps(m) + "\n" for m in METRICS))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--encoder", default="dense", choices=("dense", "flex", "sparse"))
    parser.add_argument("--dirty", action="store_true")
    args = parser.parse_args(argv)
    for path in write_fake_run(args.run_dir, args.encoder, args.dirty):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
