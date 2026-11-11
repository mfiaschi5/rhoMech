// @ts-nocheck -- run by `node --experimental-strip-types --test` (see sketch/model.test.ts).
// The target product of the Predict form: no fields without one (the beam expansion), else the
// search fields of PredictIn; and the #/predict?t= prefill.
import test from "node:test";
import assert from "node:assert/strict";
import { DEFAULT_TARGET, targetRequest } from "./target.ts";
import { parseHash, parsePredictTarget, predictPrefillHash } from "../../router.ts";

test("an empty target asks for no search", () => {
  assert.deepEqual(targetRequest(DEFAULT_TARGET), {});
  assert.deepEqual(targetRequest({ ...DEFAULT_TARGET, smiles: "   " }), {});
});

test("a target asks for the search with its levels, k and budget", () => {
  assert.deepEqual(targetRequest({ ...DEFAULT_TARGET, smiles: " CC(C)([O-])OC " }), {
    target_smiles: "CC(C)([O-])OC",
    target_levels: ["EXACT", "SKELETON"],
    n_pathways: 5,
    max_expansions: 50,
  });
  const exact = targetRequest({ smiles: "CCO", skeleton: false, k: null, maxExpansions: 120 });
  assert.deepEqual(exact.target_levels, ["EXACT"]);
  assert.equal(exact.n_pathways, 5);
  assert.equal(exact.max_expansions, 120);
});

test("the prefill link keeps the target", () => {
  const hash = predictPrefillHash([{ smiles: "C[O-]", count: 1 }, { smiles: "CC(C)=O", count: 1 }], "CC(C)([O-])OC");
  assert.equal(parsePredictTarget(parseHash(hash)), "CC(C)([O-])OC");
  assert.equal(parsePredictTarget(parseHash(predictPrefillHash([{ smiles: "C", count: 1 }]))), "");
});
