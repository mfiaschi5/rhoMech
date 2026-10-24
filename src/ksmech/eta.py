"""ksmech-eta: progress and estimated time left of the running ksmech commands.

Each running `ksmech-*` process is found under /proc and its log is the file its stderr (or
stdout) is redirected to. Two progress formats are read from the end of that log:

- tqdm bars, `18%|##   | 48/263 [28:01<2:21:12, 39.41s/mol]`;
- logging lines with a timestamp and a count, `2026-10-22 13:19:23,039 INFO ...: 20 / 401 chunks`.

Two estimates are given: `recent` (tqdm's smoothed rate, or the last few logging points) and
`mean` (the average rate since the bar or the series started). Standard library only, so it
also runs outside the venv while a build owns it: `python3 -I src/ksmech/eta.py`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

PROC = Path("/proc")
TAIL_BYTES = 256 * 1024
RECENT_POINTS = 5  # logging points behind the `recent` rate

TQDM_RE = re.compile(r"(\d+)%\|[^|]*\|\s*(\d+)/(\d+)\s*\[([\d:]+)<([\d:?]+),\s*([^\]]*)\]")
TQDM_UNIT_RE = re.compile(r"[\d.?]+\s*(?:s/(\w+)|(\w+)/s)")
LOG_TS_RE = re.compile(r"^(\d{4}-\d\d-\d\d[ T]\d\d:\d\d:\d\d)(?:[,.]\d+)?")
LOG_COUNT_RE = re.compile(r"(\d+)\s*/\s*(\d+)\s*([A-Za-z]\w*)?")


@dataclass
class Progress:
    done: int
    total: int
    unit: str
    source: str  # "tqdm" or "log"
    at: float  # epoch of the reading the estimates refer to
    elapsed_s: float | None
    recent_left_s: float | None
    mean_left_s: float | None


@dataclass
class Proc:
    pid: int
    command: str
    started: float
    workers: int
    log: str | None
    stream: str  # where stderr points when it is not a file
    driver: str | None
    unit: str | None  # systemd scope or service


# ---- parsing


def parse_interval(text: str) -> float | None:
    """tqdm's `MM:SS` or `H:MM:SS` in seconds; None for `?`."""
    if "?" in text or not text:
        return None
    seconds = 0.0
    for part in text.split(":"):
        seconds = 60 * seconds + int(part)
    return seconds


def last_tqdm(text: str, mtime: float) -> Progress | None:
    """The last tqdm bar of a log; `mtime` dates it (tqdm writes no clock time)."""
    matches = list(TQDM_RE.finditer(text))
    if not matches:
        return None
    m = matches[-1]
    done, total = int(m.group(2)), int(m.group(3))
    elapsed, left = parse_interval(m.group(4)), parse_interval(m.group(5))
    unit_m = TQDM_UNIT_RE.search(m.group(6))
    unit = (unit_m.group(1) or unit_m.group(2)) if unit_m else "it"
    mean = elapsed / done * (total - done) if done and elapsed is not None else None
    return Progress(done, total, unit, "tqdm", mtime, elapsed, left, mean)


def _log_points(text: str) -> list[tuple[float, int, int, str, str]]:
    """(epoch, done, total, unit, message key) of every timestamped line with a count."""
    points = []
    for line in text.splitlines():
        ts = LOG_TS_RE.match(line)
        count = LOG_COUNT_RE.search(line, ts.end()) if ts else None
        if not count:
            continue
        when = datetime.fromisoformat(ts.group(1).replace("T", " ")).timestamp()
        key = line[ts.end() : count.start()].strip()
        done, total = int(count.group(1)), int(count.group(2))
        points.append((when, done, total, count.group(3) or "", key))
    return points


def _rate(points: list[tuple[float, int, int, str, str]], since: float | None) -> float | None:
    """Items per second over `points`; one point is measured from `since` (process start)."""
    (t0, n0, *_), (t1, n1, *_) = points[0], points[-1]
    if len(points) == 1 and since is not None:
        t0, n0 = since, 0
    return (n1 - n0) / (t1 - t0) if t1 > t0 and n1 > n0 else None


def last_log_series(text: str, since: float | None = None) -> Progress | None:
    """The last count series of a log (same message, same total), after `since` if given.

    A restarted run appends to the same log; `since` (the process start) drops the old run."""
    points = [p for p in _log_points(text) if since is None or p[0] >= since - 1]
    points = [p for p in points if p[1] <= p[2]]
    if not points:
        return None
    _, done, total, unit, key = points[-1]
    series = [p for p in points if p[2] == total and p[4] == key]
    first, last = series[0][0], series[-1][0]
    mean_rate = _rate(series, since)
    recent_rate = _rate(series[-RECENT_POINTS:], since if len(series) == 1 else None)
    left = total - done
    start = since if since is not None and len(series) == 1 else first
    return Progress(
        done,
        total,
        unit or "it",
        "log",
        last,
        last - start,
        left / recent_rate if recent_rate else None,
        left / mean_rate if mean_rate else None,
    )


def read_tail(path: Path, nbytes: int = TAIL_BYTES) -> str:
    with path.open("rb") as f:
        f.seek(max(0, path.stat().st_size - nbytes))
        return f.read().decode(errors="replace").replace("\r", "\n")


def log_progress(path: Path, since: float | None = None) -> Progress | None:
    """The progress a log shows: its last tqdm bar, else its last logging count series."""
    text = read_tail(path)
    return last_tqdm(text, path.stat().st_mtime) or last_log_series(text, since)


# ---- processes


def _read(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""


def _argv(pid: int) -> list[str]:
    return [a for a in _read(PROC / str(pid) / "cmdline").split("\0") if a]


def _ppid_map() -> dict[int, int]:
    """pid -> parent pid of every process."""
    out = {}
    for d in PROC.iterdir():
        stat = _read(d / "stat") if d.name.isdigit() else ""
        if stat:
            out[int(d.name)] = int(stat.rsplit(")", 1)[1].split()[1])
    return out


def _start_time(pid: int) -> float:
    """Epoch at which `pid` started (field 22 of /proc/<pid>/stat, in ticks after boot)."""
    ticks = int(_read(PROC / str(pid) / "stat").rsplit(")", 1)[1].split()[19])
    lines = _read(PROC / "stat").splitlines()
    btime = next(int(line.split()[1]) for line in lines if line.startswith("btime"))
    return btime + ticks / os.sysconf("SC_CLK_TCK")


def ksmech_command(argv: list[str]) -> str | None:
    """`ksmech-qm run ...` for the Python process of a ksmech entry point, else None.

    `uv run ksmech-qm` and the multiprocessing workers are not matched, only the interpreter
    running the entry-point script; ksmech-eta itself is left out."""
    if len(argv) < 2 or not Path(argv[0]).name.startswith("python"):
        return None
    name = Path(argv[1]).name
    if not name.startswith("ksmech-") or name == "ksmech-eta":
        return None
    return " ".join([name, *argv[2:]])


def _descendants(pid: int, ppids: dict[int, int]) -> int:
    children: dict[int, list[int]] = {}
    for p, pp in ppids.items():
        children.setdefault(pp, []).append(p)
    todo, n = list(children.get(pid, [])), 0
    while todo:
        n += 1
        todo.extend(children.get(todo.pop(), []))
    return n


def _driver(pid: int, ppids: dict[int, int]) -> str | None:
    """The shell script among the ancestors of `pid` (a pipeline driver), if any."""
    p = ppids.get(pid, 0)
    while p > 1:
        argv = _argv(p)
        if argv and Path(argv[0]).name in {"bash", "sh", "zsh"} and len(argv) > 1:
            script = next((a for a in argv[1:] if not a.startswith("-")), None)
            if script:
                return script
        p = ppids.get(p, 0)
    return None


def _systemd_unit(pid: int) -> str | None:
    for part in reversed(_read(PROC / str(pid) / "cgroup").strip().split("/")):
        if part.endswith((".scope", ".service")) and "ksmech" in part:
            return part
    return None


def _log_target(pid: int) -> tuple[str | None, str]:
    """(log file, stream description): the file stderr or stdout is redirected to."""
    stream = ""
    for fd in (2, 1):
        try:
            target = os.readlink(PROC / str(pid) / "fd" / str(fd))
        except OSError:
            continue
        if Path(target).is_file():
            return target, target
        stream = stream or target
    return None, stream or "unknown"


def find_processes() -> list[Proc]:
    """Every running ksmech command, oldest first."""
    ppids = _ppid_map()
    out = []
    for pid in sorted(ppids):
        command = ksmech_command(_argv(pid))
        if command is None:
            continue
        try:
            started = _start_time(pid)
        except (IndexError, ValueError, StopIteration):
            continue  # exited while being read
        log, stream = _log_target(pid)
        workers, driver, unit = _descendants(pid, ppids), _driver(pid, ppids), _systemd_unit(pid)
        out.append(Proc(pid, command, started, workers, log, stream, driver, unit))
    return sorted(out, key=lambda p: p.started)


# ---- report


def fmt_duration(seconds: float | None) -> str:
    """`2d03h`, `2h21m`, `5m03s`, `42s`; `?` when unknown."""
    if seconds is None:
        return "?"
    s = max(0, round(seconds))
    days, rem = divmod(s, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}d{hours:02d}h"
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _clock(epoch: float, now: float) -> str:
    when = datetime.fromtimestamp(epoch)
    if when.date() == datetime.fromtimestamp(now).date():
        return when.strftime("%H:%M")
    return when.strftime("%a %d %H:%M")


def _short(path: str, root: Path) -> str:
    try:
        return str(Path(path).relative_to(root))
    except ValueError:
        return path


def progress_lines(prog: Progress | None, now: float) -> list[str]:
    """The progress and ETA lines of one log."""
    if prog is None:
        return ["  progress  none found in the log"]
    pct = 100 * prog.done / prog.total if prog.total else 0.0
    lines = [
        f"  progress  {prog.done}/{prog.total} {prog.unit} ({pct:.1f}%), "
        f"elapsed {fmt_duration(prog.elapsed_s)}  [{prog.source}]"
    ]
    if prog.done >= prog.total:
        return [*lines, "  ETA       done"]
    parts = []
    for name, left in (("recent", prog.recent_left_s), ("mean", prog.mean_left_s)):
        if left is None:
            parts.append(f"{name} ?")
            continue
        end = prog.at + left
        parts.append(f"{name} {fmt_duration(end - now)} (~{_clock(end, now)})")
    return [*lines, "  ETA       " + "   ".join(parts)]


def proc_report(proc: Proc, root: Path, now: float) -> tuple[list[str], dict]:
    """Text lines and JSON record of one running command."""
    lines = [
        proc.command,
        f"  pid {proc.pid}, {proc.workers} child processes, up {fmt_duration(now - proc.started)}",
    ]
    if proc.driver or proc.unit:
        via = [x for x in (proc.driver and _short(proc.driver, root), proc.unit) if x]
        lines.append("  via       " + ", ".join(via))
    prog = None
    if proc.log:
        prog = log_progress(Path(proc.log), since=proc.started)
        quiet = now - Path(proc.log).stat().st_mtime
        lines.append(f"  log       {_short(proc.log, root)} (written {fmt_duration(quiet)} ago)")
        lines += progress_lines(prog, now)
    else:
        lines.append(f"  log       stderr is {proc.stream}, no file to read progress from")
    record = asdict(proc) | {"progress": asdict(prog) if prog else None}
    return lines, record


def file_report(path: Path, root: Path, now: float) -> tuple[list[str], dict]:
    """Text lines and JSON record of a log file given on the command line."""
    if not path.is_file():
        return [f"{path}: no such file"], {"log": str(path), "progress": None}
    prog = log_progress(path)
    quiet = now - path.stat().st_mtime
    writers = [p.pid for p in find_processes() if p.log and Path(p.log) == path.resolve()]
    lines = [f"{_short(str(path.resolve()), root)} (written {fmt_duration(quiet)} ago)"]
    lines += progress_lines(prog, now)
    if not writers and prog is not None and prog.done < prog.total:
        lines.append("  note      no running ksmech process writes this log: the run stopped")
    record = {"log": str(path), "writers": writers, "progress": asdict(prog) if prog else None}
    return lines, record


def report(logs: list[Path], as_json: bool) -> str:
    root, now = Path.cwd(), time.time()
    if logs:
        results = [file_report(p, root, now) for p in logs]
    else:
        results = [proc_report(p, root, now) for p in find_processes()]
    if as_json:
        return json.dumps([rec for _, rec in results], indent=2)
    if not results:
        return "no ksmech command is running"
    return "\n\n".join("\n".join(lines) for lines, _ in results)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ksmech-eta",
        description="Progress and estimated time left of the running ksmech commands, "
        "read from their logs (tqdm bars or timestamped 'n / total' lines).",
    )
    parser.add_argument("logs", nargs="*", type=Path, help="read these logs instead")
    parser.add_argument("--watch", type=float, metavar="S", help="refresh every S seconds")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)
    if not PROC.is_dir() and not args.logs:
        parser.error("needs /proc (Linux) to find processes; pass log files instead")
    if args.watch is None:
        print(report(args.logs, args.json))
        return 0
    try:
        while True:
            text = report(args.logs, args.json)
            sys.stdout.write("\033[H\033[2J" + time.strftime("%H:%M:%S") + "\n\n" + text + "\n")
            sys.stdout.flush()
            time.sleep(args.watch)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
