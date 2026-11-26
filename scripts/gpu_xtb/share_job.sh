#!/usr/bin/env bash
# gpu-xtb work sharing (after stage 6): gpu-anc with and without idle CPU workers taking the GPU
# queue's species (`--gpu-share-max-atoms`), against tblite-anc; --workers 12, a fresh cache and
# processed dir per (set, arm), outputs under $IN/share/. The inputs are stage 6's
# (stage6_sample.py). First the GPU tests of the driver and the optimizer, then the arms.
# Resumable: an arm with a DONE marker is skipped, a partial one is wiped and rerun.
# Foreign work = a process whose command line starts with a python running pytest or a ksmech fit
# (anchored, so a shell that only mentions the words does not count).
FOREIGN_RE='^[^ ]*python[^ ]* ([^ ]*/pytest|-m pytest|[^ ]*/ksmech-train|-m ksmech[.]train)( |$)'
# Env: BENCH (output root), WORKERS, CAP_S (per-arm cap), IDLE_PCT, SKIP_TESTS.
set -uo pipefail
W=/home/jacopo/Documents/ksmech/.claude/worktrees/gpu-xtb
IN=/home/jacopo/Documents/ksmech/data/processed/gpu_qm_benchmark
B=${BENCH:-$IN/share}
WORKERS=${WORKERS:-12}
CAP_S=${CAP_S:-7200}
IDLE_PCT=${IDLE_PCT:-85}
PY=$W/.venv/bin/python
export OMP_NUM_THREADS=1 LC_NUMERIC=C
mkdir -p "$B/runs"
PG=$(ps -o pgid= -p $$ | tr -d ' ')  # the job's process group: the arms' own processes
cd "$W"
echo "commit $(git -C "$W" rev-parse --short HEAD) $(git -C "$W" status --porcelain | wc -l) dirty files"

wait_idle() {  # up to ~60 min for the CPU to be IDLE_PCT idle and no pytest or fit to run
  for _ in $(seq 360); do
    idle=$(vmstat 5 2 | tail -1 | awk '{print $15}')  # CPU idle % over the last 5 s
    if [ "$idle" -ge "$IDLE_PCT" ] && ! pgrep -f "$FOREIGN_RE" >/dev/null; then
      echo "$(date '+%F %T') idle: ${idle}% CPU idle, load $(cut -d' ' -f1 /proc/loadavg)"; return 0
    fi
    sleep 5
  done
  echo "$(date '+%F %T') WARNING: not idle after ~60 min (load $(cut -d' ' -f1 /proc/loadavg)); going on"
}

state() {  # $1 label, $2 file
  {
    echo "$1 $(date '+%F %T') loadavg $(cut -d' ' -f1-3 /proc/loadavg)"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null
    pgrep -af "$FOREIGN_RE" | cut -c1-160
  } >> "$2"
}

arm() {  # $1 set, $2 arm, $3 optimizer, $4 extra options (commas for spaces)
  local set=$1 name=$2 opt=$3 extra=${4:-} d="$B/runs/$1__$2"
  if [ -f "$d/DONE" ]; then echo "skip $set $name (done)"; return 0; fi
  rm -rf "$d"; mkdir -p "$d"
  wait_idle
  state start "$d/state.txt"
  vmstat -n -t 5 > "$d/vmstat.txt" 2>&1 & local vm=$!
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv,noheader -l 1 \
    > "$d/gpu.csv" 2>&1 & local gs=$!
  (while true; do  # foreign work during the arm: pytest, fits, GPU processes not ours
     f=$(pgrep -af "$FOREIGN_RE" | cut -c1-160)
     [ -n "$f" ] && echo "$(date '+%T') $f"
     ps -eo pgid=,pcpu=,etimes=,args= | awk -v g="$PG" '$1 != g && $2 >= 30 && $3 >= 20' |
       grep -vE "firefox|kwin|Xwayland" | cut -c1-140 | sed "s/^/$(date '+%T') /" >> "$d/busy.txt"
     sleep 30
   done) > "$d/foreign.txt" 2>&1 & local fs=$!
  local t0 t1 rc
  t0=$(date +%s.%N)
  # shellcheck disable=SC2086
  timeout --kill-after=120 "$CAP_S" "$PY" -m ksmech.qm.cli run --species-from "$IN/$set.json" \
    --name "$set" --optimizer "$opt" --workers "$WORKERS" --timeout 900 \
    --processed-dir "$d/proc" --cache-dir "$d/cache" --min-success 0 ${extra//,/ } > "$d/run.log" 2>&1
  rc=$?
  t1=$(date +%s.%N)
  kill "$vm" "$gs" "$fs" 2>/dev/null; wait "$vm" "$gs" "$fs" 2>/dev/null
  state end "$d/state.txt"
  python3 -c "import json; json.dump({'set': '$set', 'arm': '$name', 'optimizer': '$opt', \
'extra': '$extra', 'wall_s': $t1 - $t0, 'exit': $rc, 'workers': $WORKERS, 'start': $t0}, \
open('$d/wall.json', 'w'))"
  grep -v UFFTYPER "$d/run.log" | tail -6
  printf '%s %s exit %s wall %.1f s\n' "$set" "$name" "$rc" "$(python3 -c "print($t1-$t0)")"
  [ "$rc" -eq 0 ] && touch "$d/DONE"
}

run_set() {  # $1 set, then arms as name:optimizer[:extra]
  local set=$1; shift
  for a in "$@"; do
    IFS=: read -r name opt extra <<< "$a"
    arm "$set" "$name" "$opt" "$extra"
  done
}

if [ -z "${SKIP_TESTS:-}" ]; then
  "$PY" -m pytest -m gpu -q -p no:cacheprovider tests/test_qm_gpu_run.py tests/test_qm_gpu_optimize.py \
    2>&1 | tail -5 | tee "$B/gpu_tests.txt"
  grep -qE "^[0-9]+ passed" "$B/gpu_tests.txt" && ! grep -q "failed\|error" "$B/gpu_tests.txt" ||
    { echo "GPU TESTS FAILED"; exit 1; }
fi
OFF=gpu_noshare:gpu-anc:--gpu-share-max-atoms,0
ON=gpu_share:gpu-anc
run_set reactmech_1000 $OFF $ON anc:tblite-anc gpu_share_r2:gpu-anc \
  gpu_share120:gpu-anc:--gpu-share-max-atoms,120
for s in spike_31-50 spike_51-80; do run_set "$s" anc:tblite-anc $OFF $ON; done
"$PY" "$W/scripts/gpu_xtb/share_compare.py" "$B" > "$B/compare.txt" 2>&1
tail -80 "$B/compare.txt"
echo ALL DONE
