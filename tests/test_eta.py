"""ksmech-eta: tqdm and logging progress parsing, durations, process discovery and the CLI."""

import os
import sys
from datetime import datetime

import pytest

from ksmech import eta

TQDM_LOG = (
    " 15%|#4        | 39/263 [21:03<4:18:54, 69.35s/mol][15:13:51] UFFTYPER: atom type\n"
    "WARNING ksmech.qm.xtb_calc: QM failed: SCF_NOT_CONVERGED\r"
    " 18%|#8        | 48/263 [28:01<2:21:12, 39.41s/mol]\r"
)

SAMPLES_LOG = """\
2026-10-22 13:19:23,039 INFO ksmech.arrows.steps: samples: 20 / 401 chunks, 593 samples
2026-10-22 14:30:32,676 INFO ksmech.arrows.steps: samples: 40 / 401 chunks, 1193 samples
  molecule_train: 9525/10264 samples, skipped {'UNALIGNED_ARROW': 559}
2026-10-22 15:40:31,191 INFO ksmech.arrows.steps: samples: 60 / 401 chunks, 1791 samples
"""


def _epoch(text):
    return datetime.fromisoformat(text).timestamp()


def test_parse_interval():
    assert eta.parse_interval("28:01") == 28 * 60 + 1
    assert eta.parse_interval("2:21:12") == 2 * 3600 + 21 * 60 + 12
    assert eta.parse_interval("?") is None


def test_last_tqdm_reads_the_last_bar():
    text = TQDM_LOG.replace("\r", "\n")
    prog = eta.last_tqdm(text, mtime=1000.0)
    assert (prog.done, prog.total, prog.unit, prog.source) == (48, 263, "mol", "tqdm")
    assert prog.elapsed_s == 28 * 60 + 1
    assert prog.recent_left_s == 2 * 3600 + 21 * 60 + 12
    assert prog.mean_left_s == pytest.approx((28 * 60 + 1) / 48 * (263 - 48))
    assert prog.at == 1000.0


def test_last_tqdm_unknown_rate_and_items_per_second():
    prog = eta.last_tqdm("  0%|          | 0/50 [00:00<?, ?mol/s]", 0.0)
    assert (prog.done, prog.recent_left_s, prog.mean_left_s, prog.unit) == (0, None, None, "mol")
    prog = eta.last_tqdm(" 40%|####      | 4/10 [00:02<00:03, 2.00it/s]", 0.0)
    assert prog.unit == "it"
    assert eta.last_tqdm("no bar here", 0.0) is None


def test_last_log_series_rates():
    prog = eta.last_log_series(SAMPLES_LOG)
    assert (prog.done, prog.total, prog.unit, prog.source) == (60, 401, "chunks", "log")
    t20, t60 = _epoch("2026-10-22 13:19:23"), _epoch("2026-10-22 15:40:31")
    assert prog.at == t60
    assert prog.elapsed_s == pytest.approx(t60 - t20)
    rate = 40 / (t60 - t20)
    assert prog.mean_left_s == pytest.approx(341 / rate)
    assert prog.recent_left_s == pytest.approx(341 / rate)  # fewer points than RECENT_POINTS


def test_last_log_series_drops_a_previous_run():
    since = _epoch("2026-10-22 15:00:00")  # the process restarted between 40 and 60
    prog = eta.last_log_series(SAMPLES_LOG, since=since)
    t60 = _epoch("2026-10-22 15:40:31")
    assert prog.done == 60
    assert prog.elapsed_s == pytest.approx(t60 - since)
    assert prog.mean_left_s == pytest.approx(341 / (60 / (t60 - since)))


def test_log_progress_prefers_tqdm(tmp_path):
    path = tmp_path / "run.log"
    path.write_text(SAMPLES_LOG + TQDM_LOG)
    assert eta.log_progress(path).source == "tqdm"
    path.write_text(SAMPLES_LOG)
    assert eta.log_progress(path).source == "log"
    path.write_text("exit 0\n")
    assert eta.log_progress(path) is None


def test_fmt_duration():
    assert eta.fmt_duration(None) == "?"
    assert eta.fmt_duration(42) == "42s"
    assert eta.fmt_duration(303) == "5m03s"
    assert eta.fmt_duration(2 * 3600 + 21 * 60) == "2h21m"
    assert eta.fmt_duration(2 * 86400 + 3 * 3600) == "2d03h"


def test_ksmech_command_matches_only_the_entry_point_interpreter():
    py, bin_ = "/repo/.venv/bin/python", "/repo/.venv/bin/"
    assert eta.ksmech_command([py, bin_ + "ksmech-qm", "run", "--workers", "4"]) == (
        "ksmech-qm run --workers 4"
    )
    assert eta.ksmech_command(["uv", "run", "ksmech-qm", "run"]) is None
    assert eta.ksmech_command([py, "-B", "-c", "from multiprocessing.spawn import x"]) is None
    assert eta.ksmech_command([py, bin_ + "ksmech-eta"]) is None


def test_progress_lines():
    prog = eta.Progress(48, 263, "mol", "tqdm", 0.0, 1681.0, 8472.0, 7530.0)
    lines = eta.progress_lines(prog, now=0.0)
    assert "48/263 mol (18.3%)" in lines[0]
    assert "recent 2h21m" in lines[1] and "mean 2h05m" in lines[1]
    done = eta.Progress(401, 401, "chunks", "log", 0.0, 10.0, None, None)
    assert eta.progress_lines(done, 0.0)[-1].endswith("done")


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs /proc")
def test_find_processes_and_cli(tmp_path, capsys):
    procs = eta.find_processes()
    assert all(p.command.startswith("ksmech-") for p in procs)
    assert os.getpid() not in {p.pid for p in procs}
    path = tmp_path / "run.log"
    path.write_text(SAMPLES_LOG)
    assert eta.main([str(path)]) == 0
    out = capsys.readouterr().out
    assert "60/401 chunks" in out and "the run stopped" in out
