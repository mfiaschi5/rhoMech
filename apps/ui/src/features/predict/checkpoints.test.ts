// @ts-nocheck -- run by `node --experimental-strip-types --test` (see sketch/model.test.ts).
// The network checkpoint on the Predict page: the default (the server's rule), the labels, and
// an expansion that keeps the network's predictor and checkpoint.
import test from "node:test";
import assert from "node:assert/strict";
import { checkpointId, defaultNetworkCheckpoint, expandBody, pickerLabel, recordedLabel, refOf } from "./checkpoints.ts";

const ck = (run, file, top1, extra = {}) => ({ source: "pmechdb", run, file, usable: true, smoke: false, encoder: "flex", epoch: 19, metrics: top1 === null ? {} : { val_top1: top1 }, reason: null, ...extra });

test("the default is the best usable real run, best.pt before last.pt", () => {
  const list = [ck("flex_first", "last", 0.587), ck("flex_first", "best", 0.609), ck("smoke_x", "best", 0.9, { smoke: true }), ck("dirty", "best", 0.95, { usable: false })];
  assert.equal(checkpointId(defaultNetworkCheckpoint(list)), "pmechdb/flex_first/best");
  const tie = [ck("a", "last", 0.5), ck("a", "best", 0.5)];
  assert.equal(defaultNetworkCheckpoint(tie).file, "best");
});

test("a smoke run is the default only when nothing else is usable", () => {
  const list = [ck("smoke_x", "best", 0.1, { smoke: true }), ck("dirty", "best", 0.9, { usable: false })];
  assert.equal(defaultNetworkCheckpoint(list).run, "smoke_x");
  assert.equal(defaultNetworkCheckpoint([ck("dirty", "best", 0.9, { usable: false })]), null);
  assert.equal(defaultNetworkCheckpoint([]), null);
});

test("labels say the run, encoder, epoch, val top-1, smoke and refusal", () => {
  assert.equal(pickerLabel(ck("flex_first", "best", 0.60855)), "flex_first/best.pt · flex · epoch 19 · val top-1 60.9%");
  assert.equal(pickerLabel(ck("smoke_x", "best", 0.03, { smoke: true, epoch: 1 })), "smoke_x/best.pt · flex · epoch 1 · val top-1 3.0% · smoke");
  assert.match(pickerLabel(ck("d", "best", null, { usable: false })), /\(refused\)$/);
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
