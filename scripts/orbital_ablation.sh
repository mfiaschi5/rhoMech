#!/usr/bin/env bash
# The orbital-column ablation of docs/todo.md (2026-11-09): four arms x 3 seeds on the PMechDB
# molecule split, selected on closed_base val, general preset (as pm_v19_s0), then the decodes
# and the paired comparisons against arm a. Every arm, the reference included, must train on the
# same GPU and torch build (decisions.md, P9 step 9: the hardware confound).
#
#   a  as pm_v19_s0
#   b  no orbital columns (--drop-token-features orbital)
#   c  b and no spectral set (--no-spectral-set)
#   d  c and no inherited atom embeddings (--no-inherit-atoms)
#
# Usage: scripts/orbital_ablation.sh fit|decode|compare [ARM ...]   (default: every arm)
# On the RX 6600 XT export HSA_OVERRIDE_GFX_VERSION=10.3.0 first; on a CUDA pod UV_NO_SYNC=1.
set -euo pipefail

STAGE=${1:?fit, decode or compare}
shift || true
ARMS=("${@:-a b c d}")
ARMS=(${ARMS[*]})
SEEDS=(0 1 2)
COMMON=(--source pmechdb --split molecule --preset general --epochs 30
        --val-cohort pmechdb=closed_base)

flags() {
  case $1 in
    a) echo "" ;;
    b) echo "--drop-token-features orbital" ;;
    c) echo "--drop-token-features orbital --no-spectral-set" ;;
    d) echo "--drop-token-features orbital --no-spectral-set --no-inherit-atoms" ;;
    *) echo "unknown arm $1" >&2; exit 2 ;;
  esac
}

hits() {  # the closed_base val hit files of one arm
  for s in "${SEEDS[@]}"; do
    echo "data/models/pmechdb/orb_$1_s$s/decode_hits_molecule_val_closed_base.json"
  done
}

for arm in "${ARMS[@]}"; do
  for s in "${SEEDS[@]}"; do
    run="orb_${arm}_s${s}"
    case $STAGE in
      fit)
        # shellcheck disable=SC2046
        uv run ksmech-train fit --run "$run" --seed "$s" "${COMMON[@]}" $(flags "$arm")
        ;;
      decode)
        OMP_NUM_THREADS=8 uv run ksmech-eval decode --run "$run" --source pmechdb \
          --split molecule --part val --cohort closed_base
        ;;
      compare) ;;
      *) echo "stage: fit, decode or compare" >&2; exit 2 ;;
    esac
  done
done

if [[ $STAGE == compare ]]; then
  out=data/models/pmechdb/orbital_ablation_compare
  mkdir -p "$out"
  for arm in "${ARMS[@]}"; do
    [[ $arm == a ]] && continue
    for metric in own any; do
      for k in 1 3; do
        # shellcheck disable=SC2046
        uv run ksmech-eval compare --a $(hits "$arm") --b $(hits a) --k "$k" --metric "$metric" \
          --out "$out/${arm}_vs_a_top${k}_${metric}.json"
      done
    done
  done
fi
