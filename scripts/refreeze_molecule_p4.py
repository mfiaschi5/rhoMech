"""One-off (open-shell plan P4, user decision 2026-11-03): extend PMechDB's frozen molecule split
with the steps placed since the P0 snapshot (P1's repairs, P3's radicals), so that P4's new
steps, which bridge groups frozen in two parts, cannot send them to SPLIT_CONFLICT.

The pre-P4 split is rebuilt from the pre-P4 steps (`p4_reference/steps.parquet`) with the P0
snapshot, exactly as the P3 build placed it (checked against the pre-P4 views); the new snapshot
is the P0 snapshot plus every step that split placed and the snapshot lacked. The P0 snapshot is
kept as `p4_reference/split_frozen_molecule_p0.json`.

    uv run python scripts/refreeze_molecule_p4.py
"""

import json
import shutil

from ksmech.arrows.steps import (
    FROZEN_SPLIT_SCHEMA,
    frozen_split_path,
    load_frozen_split,
    split_ids,
)
from ksmech.data import provenance as prov
from ksmech.data.io import DEFAULT_PROCESSED_DIR, read_steps
from ksmech.data.resonance import training_steps

P = DEFAULT_PROCESSED_DIR / "pmechdb"


def main() -> None:
    path = frozen_split_path("pmechdb", "molecule")
    backup = P / "p4_reference" / "split_frozen_molecule_p0.json"
    if backup.exists():
        raise SystemExit(f"{backup} exists: the snapshot was already extended")
    old_header = prov.read_header(path)
    frozen0 = load_frozen_split("pmechdb", "molecule")
    pre = split_ids(training_steps(read_steps(P / "p4_reference" / "steps.parquet")), "molecule",
                    frozen=frozen0)  # fmt: skip
    for part in ("train", "val", "test"):
        view_path = P / "p4_reference" / f"samples_molecule_{part}.json"
        view = set(json.loads(view_path.read_text())["ids"])
        assert view <= set(pre[f"molecule_{part}"]), part  # the P3 build's placement
    ids = {k: sorted(set(frozen0[k]) | set(pre[k])) for k in frozen0}
    added = {k: len(ids[k]) - len(frozen0[k]) for k in ids}
    config = {"strategy": "molecule", "seed": old_header.config.get("seed")}
    trace = {
        "from": "split_frozen_molecule.json (P0) + the steps the pre-P4 split placed",
        "p0_content_sha256": old_header.content_sha256,
        "added": added,
        "decided": "2026-11-03, open-shell plan P4 (docs/decisions.md)",
    }
    shutil.copy2(path, backup)
    content = prov.sha256_text(json.dumps(ids, sort_keys=True))
    header = prov.stamp("split_frozen", FROZEN_SPLIT_SCHEMA, {}, config, content, trace)
    prov.write_json(path, {"splits": ids}, header)
    print(f"wrote {path}: added {added}; P0 snapshot kept as {backup}")


if __name__ == "__main__":
    main()
