// @ts-nocheck -- run by `node --experimental-strip-types --test` (see sketch/model.test.ts).
// The network checkpoint on the Predict page: the default (the server's rule), the labels, and
// an expansion that keeps the network's predictor and checkpoint.
import test from "node:test";
import assert from "node:assert/strict";
import { chooseCheckpoint, expandBody, pickerLabel, predictable, recordedLabel, refOf } from "./checkpoints.ts";

const ck = (run, file, top1, extra = {}) => ({ source: "pmechdb", run, file, usable: true, smoke: false, encoder: "flex", epoch: 19, metrics: top1 === null ? {} : { val_top1: top1 }, reason: null, ...extra });

test("the choice is the user's when the predictor can run it, else the server's default", () => {
  const list = [ck("orb_a_s0", "best", 0.65), ck("orb_b_s0", "best", 0.7), ck("bwd_s0", "best", 0.63, { direction: "backward" }), ck("dirty", "best", 0.95, { usable: false })];
  assert.equal(chooseCheckpoint(list, null, "pmechdb/orb_b_s0/best").run, "orb_b_s0");
  assert.equal(chooseCheckpoint(list, "pmechdb/orb_a_s0/best", "pmechdb/orb_b_s0/best").run, "orb_a_s0");
  assert.equal(chooseCheckpoint(list, "pmechdb/bwd_s0/best", "pmechdb/orb_b_s0/best").run, "orb_b_s0"); // backward: not runnable
  assert.equal(chooseCheckpoint(list, "pmechdb/dirty/best", null), null);
  assert.equal(chooseCheckpoint(list, null, "pmechdb/gone/best"), null);
  assert.equal(chooseCheckpoint([], null, null), null);
  assert.equal(predictable(list[2]), false);
  assert.equal(predictable(ck("old", "best", 0.5, { direction: undefined })), true); // absent: forward
});

test("labels say the run, encoder, epoch, val top-1, smoke and refusal", () => {
  assert.equal(pickerLabel(ck("flex_first", "best", 0.60855)), "flex_first/best.pt · flex · epoch 19 · val top-1 60.9%");
  assert.equal(pickerLabel(ck("smoke_x", "best", 0.03, { smoke: true, epoch: 1 })), "smoke_x/best.pt · flex · epoch 1 · val top-1 3.0% · smoke");
  assert.match(pickerLabel(ck("d", "best", null, { usable: false })), /\(refused\)$/);
  assert.match(pickerLabel(ck("bwd_s0", "best", 0.63, { direction: "backward" })), / · backward \(previous step\)$/);
  const rec = { source: "pmechdb", run: "flex_first", file: "best", sha256: "0123456789abcdef", epoch: 19, val_top1: 0.609, encoder: "flex", smoke: false };
  assert.equal(recordedLabel(rec), "flex_first/best.pt · flex · epoch 19 · val top-1 60.9% · sha 01234567");
  assert.deepEqual(refOf(rec), { source: "pmechdb", run: "flex_first", file: "best" });
});

test("an expansion keeps the network's predictor and checkpoint", () => {
  const rec = { source: "pmechdb", run: "flex_first", file: "best", sha256: "ab", epoch: 1, val_top1: 0.6, encoder: "flex", smoke: false };
  assert.deepEqual(expandBody({ predictor: "network", checkpoint: rec }, "p1", 1), { node_id: "p1", steps: 1, predictor: "network", checkpoint: { source: "pmechdb", run: "flex_first", file: "best" } });
  assert.deepEqual(expandBody({ predictor: "floor", checkpoint: null }, "p1", 3), { node_id: "p1", steps: 3, predictor: "floor" });
  assert.deepEqual(expandBody({ predictor: "user", checkpoint: null }, "p1", 3), { node_id: "p1", steps: 3 });
  assert.deepEqual(expandBody(null, "p1", 1), { node_id: "p1", steps: 1 });
});
