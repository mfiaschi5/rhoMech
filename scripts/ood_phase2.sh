#!/usr/bin/env bash
# Phase 2 of docs/todo.md, "A model that holds out of distribution": the local selector
# (docs/decisions.md, 2026-11-16; preset `local`) and the arms that take its parts away, each
# read with `ksmech-eval ood` on the sets of phase 1. Compared with phase 1's arm b
# (`ood1_b_s0`: the model as it was, without the fragment embedding), so the same GPU, torch
# build and options (scripts/ood_phase1.sh).
#
# Trained on PMechDB (molecule split), ReactMech val scored every epoch (--probe):
#   g  the local selector: each molecule encoded alone, the anchor from its orbital pairs, no
#      fragment index, no E_Z, no formal-charge embedding
#   h  g with the pool's attention (--attention-scope pool): what the locality costs or buys
#   i  g with the pointer of spec v9 (--anchor-head pointer): what the orbital-pair head does
#   j  g with E_Z and the formal-charge embedding (--element-dropout 0 --charge-embedding)
# Trained on PMechDB + ReactMech as phase 1's arm e (split locoall-<CLASS>), i's flags:
#   k  arm i's encoder on the joint data: does the local encoder keep e's transfer
#
# Usage: scripts/ood_phase2.sh fit|ood [ARM ...]   (default: every arm)
#   SEEDS, EXTRA and RUN as in scripts/ood_phase1.sh; `--last-update-atoms` in EXTRA has no
#   effect on the arms with the orbital-pair head.
set -euo pipefail

STAGE=${1:?fit or ood}
shift || true
ARMS=("${@:-g h i j k}")
ARMS=(${ARMS[*]})
SEEDS=(${SEEDS:-0})
CLASS=${CLASS:-acid-halide}
RUN=(${RUN-uv run})
EXTRA=(${EXTRA:-})
BASE=(--preset local --epochs 30 --encoder dense --no-resonance-draws)
PM=(--source pmechdb --split molecule --val-cohort pmechdb=closed_base --probe reactmech=molecule)
JOINT=(--source pmechdb+reactmech --split "pmechdb=molecule,reactmech=locoall-$CLASS"
       --val-cohort pmechdb=closed_base)

flags() {
  case $1 in
    g) echo "" ;;
    h) echo "--attention-scope pool" ;;
    i | k) echo "--anchor-head pointer" ;;
    j) echo "--element-dropout 0 --charge-embedding" ;;
    *) echo "unknown arm $1" >&2; exit 2 ;;
  esac
}

for arm in "${ARMS[@]}"; do
  for s in "${SEEDS[@]}"; do
    run="ood2_${arm}_s${s}"
    case "$STAGE:$arm" in
      fit:[ghij])
        # shellcheck disable=SC2046
        "${RUN[@]}" ksmech-train fit --run "$run" --seed "$s" "${BASE[@]}" "${PM[@]}" \
          "${EXTRA[@]}" $(flags "$arm")
        ;;
      fit:k)
        # shellcheck disable=SC2046
        "${RUN[@]}" ksmech-train fit --run "$run" --seed "$s" "${BASE[@]}" "${JOINT[@]}" \
          "${EXTRA[@]}" $(flags "$arm")
        ;;
      ood:[ghij])
        "${RUN[@]}" ksmech-eval ood --run "$run" --source pmechdb --split molecule --cohort closed_base
        "${RUN[@]}" ksmech-eval ood --run "$run" --run-source pmechdb --source reactmech --split molecule
        ;;
      ood:k)
        src=pmechdb+reactmech
        "${RUN[@]}" ksmech-eval ood --run "$run" --run-source "$src" --source pmechdb \
          --split molecule --cohort closed_base
        "${RUN[@]}" ksmech-eval ood --run "$run" --run-source "$src" --source reactmech \
          --split "locoall-$CLASS" --part test
        ;;
      *) echo "stage: fit or ood" >&2; exit 2 ;;
    esac
  done
done
