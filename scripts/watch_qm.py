"""Follow the ReactMech pilot's QM records from the terminal: the local run and the pod run
together, one line per tick (records done of the pending list, per-source counts, the rate over
the last minutes, the pod's failure share and the ETA). A record counts as done when its file is in
this checkout's cache or in the pod's; the pod is read over ssh ("SSH over exposed TCP").

Needs `ksmech` (the cache key), so run it with a venv that has it, e.g.:

    .claude/worktrees/reactmech/.venv/bin/python scripts/watch_qm.py --host root@H --port P  # 60 s
    .claude/worktrees/reactmech/.venv/bin/python scripts/watch_qm.py --host root@H --port P --once
    .claude/worktrees/reactmech/.venv/bin/python scripts/watch_qm.py --no-pod --interval 30
"""

from __future__ import annotations

import argparse
import subprocess
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from ksmech.qm.records import cache_path
from ksmech.qm.xtb_calc import QMConfig

WORKTREE = Path(__file__).resolve().parents[1] / ".claude/worktrees/reactmech"
WINDOW_S = 600  # the rate is taken over this much history


@dataclass
class PodState:
    names: set[str]
    rows: int
    failed: int
    tmux: bool


def pending_names(path: Path) -> dict[str, str]:
    """Cache file name -> SMILES for every pending molecule."""
    method = QMConfig().method
    smiles = [s for s in path.read_text().split("\n") if s]
    return {cache_path(Path("."), s, 0, method).name: s for s in smiles}


def local_done(cache: Path, names: dict[str, str]) -> set[str]:
    return {n for n in names if (cache / n).exists()}


def read_pod(args: argparse.Namespace) -> PodState | None:
    """The pod's record names and row counts, or None when it does not answer."""
    remote = (
        f"cd {args.pod_dir} && ls data/cache/qm 2>/dev/null; echo @@; wc -l < rows.jsonl; "
        'grep -c \'"status": "failed"\' rows.jsonl; tmux has-session -t qm 2>/dev/null '
        "&& echo tmux_up"
    )
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-i", args.key]
    cmd += ["-p", str(args.port), args.host, remote]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=True).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    head, _, tail = out.partition("@@\n")
    nums = tail.split()
    names = {n for n in head.split() if n.endswith(".npz")}
    if len(nums) < 2:
        return None
    return PodState(names, int(nums[0]), int(nums[1]), "tmux_up" in nums)


def local_running() -> bool:
    out = subprocess.run(["pgrep", "-f", "ksmech-qm run"], capture_output=True, text=True)
    return bool(out.stdout.strip())


def fmt_eta(minutes: float) -> str:
    if not minutes < 1e6:
        return "?"
    hours, mins = divmod(int(minutes), 60)
    return f"{hours} h {mins:02d} min"


def tick(args, names, history: deque) -> str:
    now = time.time()
    local = local_done(args.cache, names)
    pod = None if args.no_pod else read_pod(args)
    pod_names = (pod.names & set(names)) if pod else set()
    done = local | pod_names
    history.append((now, len(done)))
    while history and now - history[0][0] > WINDOW_S and len(history) > 2:
        history.popleft()
    t0, n0 = history[0]
    rate = (len(done) - n0) / max((now - t0) / 60, 1e-9) if len(history) > 1 else float("nan")
    fail = (pod.failed / pod.rows) if pod and pod.rows else 0.0
    expected = len(names) * (1 - fail)  # the failures never write a record
    left = max(expected - len(done), 0)
    eta = fmt_eta(left / rate) if rate > 0 else "?"
    pod_txt = "pod: not answering" if pod is None and not args.no_pod else ""
    if pod:
        pod_txt = (
            f"pod {len(pod_names)} ({pod.rows} rows, {fail:.0%} failed"
            f"{'' if pod.tmux else ', TMUX DOWN'})"
        )
    run = "up" if local_running() else "DOWN"
    return (
        f"{time.strftime('%H:%M:%S')}  {len(done)}/{len(names)} records "
        f"({len(done) / len(names):.1%})  local run {len(local - pod_names)} [{run}]  {pod_txt}  "
        f"{rate:.1f} mol/min  ~{left:.0f} left, ETA {eta}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--pending", type=Path, default=WORKTREE / "data/processed/reactmech/pending_2026-10-27.txt"
    )
    p.add_argument("--cache", type=Path, default=WORKTREE / "data/cache/qm")
    p.add_argument("--host", help="the pod's user@address (required unless --no-pod)")
    p.add_argument("--port", type=int, help="the pod's exposed SSH port (required unless --no-pod)")
    p.add_argument("--key", default=str(Path.home() / ".ssh/id_ed25519_pod"))
    p.add_argument("--pod-dir", default="/root/ksmech")
    p.add_argument("--interval", type=int, default=60, help="seconds between lines")
    p.add_argument("--once", action="store_true")
    p.add_argument("--no-pod", action="store_true")
    args = p.parse_args()
    if not args.no_pod and (args.host is None or args.port is None):
        p.error("--host and --port name the pod; pass both, or --no-pod")
    names = pending_names(args.pending)
    history: deque = deque()
    while True:
        print(tick(args, names, history), flush=True)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
