// @ts-nocheck -- run by `node --experimental-strip-types --test` (see sketch/model.test.ts).
// The embedding view's pure logic: the checkpoint keys, the preselected checkpoint, the
// picker labels, the page URL with its theme and view hash, and storage that never throws.
import test from "node:test";
import assert from "node:assert/strict";
import { checkpointKey, checkpointLabel, cleanViewHash, defaultCheckpoint, parseCheckpointKey, readStored, themeOf, tracePageUrl, viewKey, writeStored } from "./embedding.ts";

const ck = (run, file, top1, usable = true, extra = {}) => ({ source: "pmechdb", run, file, usable, encoder: "flex", epoch: 3, metrics: top1 === null ? {} : { val_top1: top1 }, reason: usable ? null : "STALE_ARTIFACT: dirty", ...extra });

test("checkpoint keys round-trip and refuse paths", () => {
  assert.equal(checkpointKey(ck("flex_first", "best", 0.5)), "pmechdb/flex_first/best");
  assert.deepEqual(parseCheckpointKey("pmechdb/flex_first/best"), { source: "pmechdb", run: "flex_first", file: "best" });
  for (const bad of [null, "", "a/b", "a/../b", "a/b/c/d", "../x/y", "a/b c/d"]) assert.equal(parseCheckpointKey(bad), null, String(bad));
});

test("the default checkpoint: remembered if usable, else the best usable top-1, best.pt on a tie", () => {
  const list = [ck("smoke", "best", 0.1), ck("flex_first", "last", 0.58), ck("flex_first", "best", 0.58), ck("dirty", "best", 0.9, false), ck("nometrics", "best", null)];
  assert.equal(checkpointKey(defaultCheckpoint(list, null)), "pmechdb/flex_first/best");
  assert.equal(checkpointKey(defaultCheckpoint(list, "pmechdb/smoke/best")), "pmechdb/smoke/best");
  assert.equal(checkpointKey(defaultCheckpoint(list, "pmechdb/dirty/best")), "pmechdb/flex_first/best");
  assert.equal(defaultCheckpoint([ck("dirty", "best", 0.9, false)], null), null);
  assert.equal(defaultCheckpoint([], "x"), null);
});

test("picker labels name the file, the encoder, the epoch and the top-1; refused ones say so", () => {
  assert.equal(checkpointLabel(ck("flex_first", "best", 0.5867)), "flex_first/best.pt · flex · epoch 3 · val top-1 58.7%");
  assert.equal(checkpointLabel(ck("d", "last", null, false, { encoder: null, epoch: null })), "d/last.pt (refused)");
});

test("the view hash keeps only the page's own well-formed parameters", () => {
  assert.equal(cleanViewHash("#sel=5&mode=cos&layer=3&head=-1"), "#sel=5&mode=cos&layer=3&head=-1");
  assert.equal(cleanViewHash("sel=5&evil=<x>&mode=<script>&layer=-2&head=-1"), "#sel=5&head=-1");
  assert.equal(cleanViewHash(""), "");
  assert.equal(cleanViewHash(null), "");
  assert.equal(cleanViewHash("#sel=99999"), "");
});

test("the page URL carries the theme override and the view", () => {
  const page = "/api/embedding/b0123/page";
  assert.equal(tracePageUrl(page, null, ""), page);
  assert.equal(tracePageUrl(page, "dark", "#layer=2"), `${page}?theme=dark#layer=2`);
  assert.equal(themeOf("light"), "light");
  assert.equal(themeOf("auto"), null);
  assert.equal(themeOf(undefined), null);
});

test("storage reads and writes never throw", () => {
  const blocked = {
    getItem() {
      throw new Error("blocked");
    },
    setItem() {
      throw new Error("blocked");
    },
  };
  assert.equal(readStored(blocked, "k"), null);
  writeStored(blocked, "k", "v");
  const mem = new Map();
  const store = { getItem: (k) => mem.get(k) ?? null, setItem: (k, v) => mem.set(k, v) };
  writeStored(store, viewKey("b1"), "#sel=1");
  assert.equal(readStored(store, "ksmech.embedding.view.b1"), "#sel=1");
  assert.equal(readStored(null, "k"), null);
});
