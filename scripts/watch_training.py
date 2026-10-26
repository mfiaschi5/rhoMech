"""Stream a `ksmech-train fit` run to the terminal: one row per epoch as each validation lands
(the best marked) and a live status line (step, loss, samples/s, ETA, time since the last log
line). The run is read over ssh from a pod (default) or from this checkout (--local); the
stream reconnects when the connection drops and ends with the run. Only the standard library,
so `python3` runs it anywhere (on a CUDA host `uv run` would re-sync to the rocm group).

    python3 scripts/watch_training.py                        # flex_first on the pod
    python3 scripts/watch_training.py --run NAME --host root@IP --port N
    python3 scripts/watch_training.py --local --run NAME     # a run of this checkout
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import select
import shlex
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
POD_HOST, POD_PORT, POD_KEY, POD_REPO = (
    "root@174.94.157.109",
    36836,
    "~/.ssh/id_ed25519_pod",
    "/workspace/ksmech",
)
STAMP = r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ \w+ [\w.]+: "
STEP = re.compile(
    STAMP + r"step (\d+) / (\d+) epoch (\d+) loss (\S+) anchor_nll (\S+) grad \S+ lr (\S+) "
    r"(\S+) samples/s"
)
EPOCH = re.compile(STAMP + r"epoch \d+ of (\d+) done")
DONE = re.compile(STAMP + r"best epoch")
FAIL = re.compile(r"Traceback|\bERROR\b|Error:")
HEADER = " ep   top-1   top-3   top-5  R exact  val loss  anchor NLL  train loss     s"
GREEN, RED, YELLOW, RESET = "\x1b[32m", "\x1b[31m", "\x1b[33m", "\x1b[0m"


@dataclass
class Watch:
    seen: set[int] = field(default_factory=set)  # epochs printed, 1-based
    best: tuple[float, float] | None = None  # (top-1, -val loss), the trainer's best.pt key
    best_epoch: int = 0
    n_epochs: int = 0
    points: deque = field(default_factory=lambda: deque(maxlen=60))  # (log time, step)
    step: re.Match | None = None  # the last step line
    received: float = field(default_factory=time.monotonic)
    done: bool = False


class Screen:
    """Permanent lines above one status line that is redrawn in place (on a terminal)."""

    def __init__(self, stream=None):
        self.out = stream or sys.stdout
        self.tty = self.out.isatty()
        self.status = ""

    def color(self, text: str, code: str) -> str:
        return f"{code}{text}{RESET}" if self.tty else text

    def line(self, text: str) -> None:
        if self.tty:
            self.out.write("\r\x1b[K")
        self.out.write(text + "\n")
        if self.tty and self.status:
            self.out.write(self.status)
        self.out.flush()

    def set_status(self, text: str) -> None:
        if text == self.status:
            return
        self.status = text
        self.out.write(("\r\x1b[K" + text) if self.tty else text + "\n")
        self.out.flush()


def tail_command(args) -> list[str]:
    """The two tails (every metrics row, the end of the log), stopped when this side closes
    their stdin, so none lingers on the pod after ssh goes."""
    root = REPO if args.local else args.repo
    run = shlex.quote(f"{root}/data/models/{args.source}/{args.run}")
    sh = (
        f"tail -q -n +1 -F {run}/metrics.jsonl 2>/dev/null & "
        f"tail -q -n 200 -F {run}/train.log 2>/dev/null & read -r _; kill 0"
    )
    if args.local:
        return ["bash", "-c", sh]
    opts = ["BatchMode=yes", "ConnectTimeout=20", "ServerAliveInterval=15", "ServerAliveCountMax=4"]
    ssh = ["ssh", "-p", str(args.port), "-i", os.path.expanduser(args.key)]
    return ssh + [x for o in opts for x in ("-o", o)] + [args.host, sh]


def row(r: dict, w: Watch, screen: Screen) -> None:
    """One validation row; a new best (by top-1, then loss, as best.pt) in green."""
    ep = r["epoch"] + 1
    if ep in w.seen:
        return
    if not w.seen:
        screen.line(HEADER)
    w.seen.add(ep)
    key = (r["val_top1"], -r["val_loss"])
    new_best = w.best is None or key > w.best
    if new_best:
        w.best, w.best_epoch = key, ep
    text = (
        f"{ep:>3}  {100 * r['val_top1']:6.1f}% {100 * r['val_top3']:6.1f}% "
        f"{100 * r['val_top5']:6.1f}%  {100 * r['val_r_exact']:6.1f}%  {r['val_loss']:8.3f}"
        f"  {r['val_anchor_nll']:10.3f}  {r['train_loss']:10.3f}  {r['seconds']:4.0f}"
    )
    screen.line(screen.color(text + "  * best", GREEN) if new_best else text)


def eta(w: Watch) -> str:
    """Remaining time from the step rate over the last log lines (validation pauses included)."""
    if len(w.points) < 2 or w.step is None:
        return "ETA ?"
    (t0, s0), (t1, s1) = w.points[0], w.points[-1]
    seconds = (t1 - t0).total_seconds()
    if s1 <= s0 or seconds <= 0:
        return "ETA ?"
    left = (int(w.step[3]) - s1) * seconds / (s1 - s0)
    end = datetime.now() + timedelta(seconds=left)
    return f"ETA {left / 60:.0f} min ({end:%H:%M})"


def render(w: Watch) -> str:
    m = w.step
    if m is None:
        return "waiting for the first step line..."
    step, total, epoch = int(m[2]), int(m[3]), int(m[4]) + 1
    of = f"/{w.n_epochs}" if w.n_epochs else ""
    age = time.monotonic() - w.received
    stale = f"  last line {age / 60:.0f} min ago" if age > 120 else ""
    return (
        f"step {step}/{total} {100 * step / total:5.1f}%  epoch {epoch}{of}  loss {m[5]}  "
        f"nll {m[6]}  lr {m[7]}  {float(m[8]):.0f} samples/s  {eta(w)}{stale}"
    )


def handle(line: str, w: Watch, screen: Screen) -> None:
    w.received = time.monotonic()
    if line.startswith("{"):
        with contextlib.suppress(ValueError, KeyError):
            row(json.loads(line), w, screen)
    elif m := STEP.match(line):
        t = datetime.strptime(m[1], "%Y-%m-%d %H:%M:%S")
        if not w.points or int(m[2]) > w.points[-1][1]:
            w.points.append((t, int(m[2])))
        w.step = m
        screen.set_status(render(w))
    elif m := EPOCH.match(line):
        w.n_epochs = int(m[2])
    elif DONE.match(line):
        w.done = True
    elif FAIL.search(line):
        screen.line(screen.color(line, RED))


def pump(proc: subprocess.Popen, w: Watch, screen: Screen) -> None:
    """Lines from the tails until the run ends or the stream does; the status line is
    redrawn every 15 s without input, so a stalled run shows its age."""
    fd, buf = proc.stdout.fileno(), b""
    while not w.done:
        ready, _, _ = select.select([fd], [], [], 15)
        if not ready:
            if proc.poll() is not None:
                return
            screen.set_status(render(w))
            continue
        data = os.read(fd, 1 << 16)
        if not data:
            return
        *lines, buf = (buf + data).split(b"\n")
        for line in lines:
            handle(line.decode(errors="replace"), w, screen)


def stop(proc: subprocess.Popen) -> None:
    with contextlib.suppress(OSError):
        proc.stdin.close()  # ends the tails at the other end
    proc.terminate()
    try:
        proc.wait(5)
    except subprocess.TimeoutExpired:
        proc.kill()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--run", default="flex_first")
    p.add_argument("--source", default="pmechdb")
    p.add_argument("--local", action="store_true", help="a run under this checkout's data/models")
    p.add_argument("--host", default=POD_HOST)
    p.add_argument("--port", type=int, default=POD_PORT)
    p.add_argument("--key", default=POD_KEY)
    p.add_argument("--repo", default=POD_REPO, help="the checkout on the pod")
    args = p.parse_args(argv)
    w, screen = Watch(), Screen()
    where = "this checkout" if args.local else f"{args.host}:{args.port}"
    screen.line(f"{args.source}/{args.run} on {where}")
    try:
        while not w.done:
            cmd = tail_command(args)
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            try:
                pump(proc, w, screen)
            finally:
                stop(proc)
            if not w.done:
                screen.line(screen.color("stream lost, reconnecting in 10 s", YELLOW))
                time.sleep(10)
    except KeyboardInterrupt:
        screen.line("")
        return 130
    best = f"best epoch {w.best_epoch}, top-1 {100 * w.best[0]:.1f}%" if w.best else "no epoch"
    screen.line(screen.color(f"run finished: {best} (best.pt)", GREEN))
    return 0


if __name__ == "__main__":
    sys.exit(main())
