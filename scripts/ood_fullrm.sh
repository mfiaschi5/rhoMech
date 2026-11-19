#!/usr/bin/env bash
# The next step of docs/todo.md, "A model that holds out of distribution" (2026-11-19): the lever
# the two-seed reading found is more and more varied training data, so phase 1's arm e (the
# general preset without the fragment embedding, PMechDB + ReactMech) trains on the full
# ReactMech v19 store (~85k training samples, scaffold split: the molecule split is degenerate
# there) instead of the 10k-step pilot it read (split locoall-acid-halide). Read on the same
# three sources as e and k: PMechDB closed_base val, the ReactMech scaffold test part, and the
# human benchmark (a source neither trains on).
#
# The store must be at data/processed/reactmech: a worktree whose data/processed/reactmech links
# to the main checkout's data/processed_full/reactmech (the provenance of the store then
# verifies as built, without --processed-dir). Checked below.
#
# Usage: scripts/ood_fullrm.sh fit|ood|walk   (runs fullrm_e_s<seed> under
#   data/models/pmechdb+reactmech/)
#   SEEDS="0 1" (default), RUN and EXTRA as in scripts/ood_phase1.sh; the memory and speed
#   options of 2026-11-19 (deflated cells, packed samples) are on by default.
set -euo pipefail

STAGE=${1:?fit, ood or walk}
SEEDS=(${SEEDS:-0 1})
RUN=(${RUN-uv run})
EXTRA=(${EXTRA:-})
SRC=pmechdb+reactmech
BASE=(--preset general --epochs 30 --encoder dense --no-resonance-draws --no-fragment-embedding)
JOINT=(--source "$SRC" --split "pmechdb=molecule,reactmech=scaffold"
       --val-cohort pmechdb=closed_base)

view=data/processed/reactmech/samples_scaffold_train.json
n=$(python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))['ids']))" "$view")
if [ "$n" -lt 50000 ]; then
  echo "$view has $n samples: not the full ReactMech store" >&2
  exit 2
fi

for s in "${SEEDS[@]}"; do
  run="fullrm_e_s${s}"
  case "$STAGE" in
    fit)
      "${RUN[@]}" ksmech-train fit --run "$run" --seed "$s" "${BASE[@]}" "${JOINT[@]}" "${EXTRA[@]}"
      ;;
    ood)
      "${RUN[@]}" ksmech-eval ood --run "$run" --run-source "$SRC" --source pmechdb \
        --split molecule --cohort closed_base
      "${RUN[@]}" ksmech-eval ood --run "$run" --run-source "$SRC" --source reactmech \
        --split scaffold --part test
      ;;
    walk)
      "${RUN[@]}" ksmech-network evaluate --set human_benchmark --run "$run" --run-source "$SRC" \
        --walk
      ;;
    *) echo "stage: fit, ood or walk" >&2; exit 2 ;;
  esac
done
