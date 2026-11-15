#!/usr/bin/env bash
# Phase 1 of docs/todo.md, "A model that holds out of distribution" (2026-11-15): the cheap arms
# on the model as it is, each read with `ksmech-eval ood` on the same sets. Every arm, the
# control included, must train on the same GPU and torch build (decisions.md, P9 step 9: the
# hardware confound).
#
# Trained on PMechDB (molecule split, as pm_v19_s0), ReactMech val scored every epoch (--probe):
#   a  control
#   b  no fragment-index embedding (--no-fragment-embedding)
#   c  b and --element-dropout 0.5
#   d  b and --element-dropout 1 (the element is never read)
# Trained on PMechDB + ReactMech with one ReactMech class held out (split locoall-<CLASS>: the
# class is the test part, every other step trains and validates), b's switch:
#   e  every sample once an epoch
#   f  --balance-power 0.5 (an epoch drawn by edit signature)
#
# Usage: scripts/ood_phase1.sh fit|ood [ARM ...]   (default: every arm)
#   SEEDS="0 1 2" CLASS=acid-halide scripts/ood_phase1.sh fit a b
#   EXTRA="--compile --half-pair-norm" adds options to every fit (the speed options: the same
#   for every arm); RUN= (empty) calls ksmech-train and ksmech-eval from PATH instead of
#   `uv run` (a worktree run with the main checkout's venv and PYTHONPATH=<worktree>/src).
# On the RX 6600 XT export HSA_OVERRIDE_GFX_VERSION=10.3.0 first; on a CUDA pod UV_NO_SYNC=1.
set -euo pipefail

STAGE=${1:?fit or ood}
shift || true
ARMS=("${@:-a b c d e f}")
ARMS=(${ARMS[*]})
SEEDS=(${SEEDS:-0})
CLASS=${CLASS:-acid-halide}
RUN=(${RUN-uv run})
EXTRA=(${EXTRA:-})
BASE=(--preset general --epochs 30 --encoder dense --no-resonance-draws)
PM=(--source pmechdb --split molecule --val-cohort pmechdb=closed_base --probe reactmech=molecule)
JOINT=(--source pmechdb+reactmech --split "pmechdb=molecule,reactmech=locoall-$CLASS"
       --val-cohort pmechdb=closed_base)

flags() {
  case $1 in
    a) echo "" ;;
    b | e) echo "--no-fragment-embedding" ;;
    c) echo "--no-fragment-embedding --element-dropout 0.5" ;;
    d) echo "--no-fragment-embedding --element-dropout 1" ;;
    f) echo "--no-fragment-embedding --balance-power 0.5" ;;
    *) echo "unknown arm $1" >&2; exit 2 ;;
  esac
}

for arm in "${ARMS[@]}"; do
  for s in "${SEEDS[@]}"; do
    run="ood1_${arm}_s${s}"
    case "$STAGE:$arm" in
      fit:[abcd])
        # shellcheck disable=SC2046
        "${RUN[@]}" ksmech-train fit --run "$run" --seed "$s" "${BASE[@]}" "${PM[@]}" \
          "${EXTRA[@]}" $(flags "$arm")
        ;;
      fit:[ef])
        # shellcheck disable=SC2046
        "${RUN[@]}" ksmech-train fit --run "$run" --seed "$s" "${BASE[@]}" "${JOINT[@]}" \
          "${EXTRA[@]}" $(flags "$arm")
        ;;
      ood:[abcd])
        "${RUN[@]}" ksmech-eval ood --run "$run" --source pmechdb --split molecule --cohort closed_base
        "${RUN[@]}" ksmech-eval ood --run "$run" --run-source pmechdb --source reactmech --split molecule
        ;;
      ood:[ef])
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
