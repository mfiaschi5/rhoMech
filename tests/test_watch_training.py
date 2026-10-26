"""scripts/watch_training.py: the metrics rows and log lines it reads, the ETA, the best row."""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "watch_training", Path(__file__).resolve().parents[1] / "scripts" / "watch_training.py"
)
wt = importlib.util.module_from_spec(_spec)
sys.modules["watch_training"] = wt  # dataclasses look their module up while being made
_spec.loader.exec_module(wt)

STEP = (
    "2026-10-26 00:{m}:00,534 INFO ksmech.train.loop: step {s} / 12750 epoch 6 loss 2.5889 "
    "anchor_nll 1.0390 grad 10.54 lr 2.98e-04 194.9 samples/s"
)


def _metrics(epoch: int, top1: float, loss: float) -> str:
    keys = ("val_top3", "val_top5", "val_r_exact", "val_anchor_nll", "train_loss", "seconds")
    row = {"epoch": epoch, "val_top1": top1, "val_loss": loss}
    return json.dumps(row | dict.fromkeys(keys, 0.5))


def _watch():
    out = io.StringIO()  # not a terminal: every status is its own line
    return wt.Watch(), wt.Screen(out), out


def test_rows_mark_the_best_epoch_and_skip_repeats():
    w, screen, out = _watch()
    for line in (
        _metrics(0, 0.3, 5.0),
        _metrics(1, 0.5, 4.0),
        _metrics(1, 0.5, 4.0),  # read again after a reconnect
        _metrics(2, 0.5, 3.0),  # same top-1, lower loss: the trainer's best.pt key
        _metrics(3, 0.4, 2.0),
    ):
        wt.handle(line, w, screen)
    lines = out.getvalue().splitlines()
    assert lines[0] == wt.HEADER
    assert [x.endswith("* best") for x in lines[1:]] == [True, True, True, False]
    assert w.best_epoch == 3


def test_step_lines_give_the_status_and_the_eta():
    w, screen, out = _watch()
    wt.handle(STEP.format(m="28", s=5400), w, screen)
    wt.handle(STEP.format(m="29", s=5600), w, screen)
    wt.handle("2026-10-26 00:29:30,000 INFO ksmech.train.loop: epoch 7 of 30 done: val", w, screen)
    status = wt.render(w)
    assert status.startswith("step 5600/12750  43.9%  epoch 7/30  loss 2.5889")
    assert "ETA 36 min" in status  # 7150 steps left at 200 steps a minute
    assert out.getvalue().splitlines()[0].startswith("step 5400/12750")


def test_the_best_epoch_line_ends_the_watch_and_errors_are_shown():
    w, screen, out = _watch()
    wt.handle("Traceback (most recent call last):", w, screen)
    assert not w.done
    wt.handle("2026-10-26 00:50:00,000 INFO ksmech.train.cli: best epoch 9: {}", w, screen)
    assert w.done
    assert out.getvalue() == "Traceback (most recent call last):\n"


def test_the_tails_run_over_ssh_or_locally():
    args = argparse.Namespace(
        run="r", source="pmechdb", local=False, host="root@h", port=22, key="~/k", repo="/w"
    )
    cmd = wt.tail_command(args)
    assert cmd[:3] == ["ssh", "-p", "22"] and cmd[-2] == "root@h"
    assert "/w/data/models/pmechdb/r/metrics.jsonl" in cmd[-1] and "read -r _; kill 0" in cmd[-1]
    args.local = True
    assert wt.tail_command(args)[:2] == ["bash", "-c"]
