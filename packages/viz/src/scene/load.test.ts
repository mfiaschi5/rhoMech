// @ts-nocheck -- run by `npm --prefix apps/ui test` (node --experimental-strip-types --test).
// loadScene and the asset loader over a fake fetch: validation errors, HTTP errors, the byte
// size and sha256 checks, one fetch per asset.
import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { loadScene, NO_ASSETS, SceneError } from "./load.ts";

const CUBE = "cube text\n";
const sha = (s) => createHash("sha256").update(Buffer.from(s)).digest("hex");
const molecule = (asset) => ({ type: "molecule", smiles: "C", molecule: { molecule_id: "m" }, geometry: { molblock: "M  END", atoms: [] }, orbitals: { set: "canonical", homo_index: 1, orbitals: [{ index: 1, kind: "occupied", occupation: 2, energy_ev: -9 }] }, cubes: [{ index: 1, asset }] });
const doc = (asset) => ({ kind: "ksmech-scene", version: 1, scene: molecule(asset), provenance: { exporter: "t", exported_at: "2026", server: {}, request: {}, notes: [] } });
const ASSET = { path: "m/mo-1.cube", bytes: Buffer.byteLength(CUBE), sha256: sha(CUBE), media_type: "chemical/x-cube" };

function site(files) {
  const calls = [];
  const fetch = async (url) => {
    calls.push(url);
    const body = files[url];
    return body === undefined ? new Response("missing", { status: 404 }) : new Response(typeof body === "string" ? body : JSON.stringify(body));
  };
  return { fetch, calls };
}

test("a valid scene loads and its asset is read once, relative to the scene", async () => {
  const s = site({ "http://h/s/a.json": doc(ASSET), "http://h/s/m/mo-1.cube": CUBE });
  const loaded = await loadScene("http://h/s/a.json", { fetch: s.fetch });
  assert.equal(loaded.doc.scene.type, "molecule");
  assert.equal(await loaded.assets(ASSET), CUBE);
  assert.equal(await loaded.assets(ASSET), CUBE);
  assert.deepEqual(s.calls, ["http://h/s/a.json", "http://h/s/m/mo-1.cube"]);
});

test("HTTP errors, JSON errors and invalid documents are SceneErrors", async () => {
  const s = site({ "http://h/bad.json": "{nope", "http://h/v2.json": { ...doc(ASSET), version: 2 } });
  await assert.rejects(loadScene("http://h/none.json", { fetch: s.fetch }), (e) => e instanceof SceneError && /HTTP 404/.test(e.message));
  await assert.rejects(loadScene("http://h/bad.json", { fetch: s.fetch }), /is not JSON/);
  await assert.rejects(loadScene("http://h/v2.json", { fetch: s.fetch }), (e) => e instanceof SceneError && e.errors[0] === "version: expected version 1, got 2");
});

test("an asset that is not the file the scene describes is refused", async () => {
  const s = site({ "http://h/a.json": doc(ASSET), "http://h/m/mo-1.cube": "cube text, edited\n" });
  const loaded = await loadScene("http://h/a.json", { fetch: s.fetch });
  await assert.rejects(loaded.assets(ASSET), /bytes, the scene says 10/);
  await assert.rejects(loaded.assets({ ...ASSET, bytes: Buffer.byteLength("cube text, edited\n") }), /sha256 .*, the scene says/);
  await assert.rejects(loaded.assets({ ...ASSET, path: "../x.cube" }), /not a relative path/);
  await assert.rejects(NO_ASSETS(ASSET), /no asset loader/);
});
