// @ts-nocheck -- run by `npm test` (node --experimental-strip-types --test, @ksmech/viz through
// scripts/viz-node-hooks.mjs). The example scenes committed under public/scenes: every one is a
// valid scene whose assets are the files it describes (size and sha256), index.json lists them
// all, and the #/scene URL check.
import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { validateScene } from "@ksmech/viz/scene/validate";
import { sameOriginUrl } from "./sceneUrl.ts";

const DIR = fileURLToPath(new URL("../../../public/scenes/", import.meta.url));
const FILES = readdirSync(DIR).filter((f) => f.endsWith(".json") && f !== "index.json");

function assetsOf(scene) {
  if (scene.type === "molecule") return scene.cubes.map((c) => c.asset);
  if (scene.type === "step") return scene.source.cubes.map((c) => c.asset);
  return [];
}

test("there are example scenes of all three types", () => {
  const types = FILES.map((f) => JSON.parse(readFileSync(join(DIR, f), "utf8")).scene?.type).sort();
  assert.deepEqual(types, ["molecule", "network", "step"]);
});

for (const file of FILES) {
  test(`${file} is a valid scene and its assets match`, () => {
    const doc = JSON.parse(readFileSync(join(DIR, file), "utf8"));
    const v = validateScene(doc);
    assert.deepEqual(v.ok ? [] : v.errors, []);
    for (const a of assetsOf(doc.scene)) {
      const path = join(DIR, a.path);
      assert.ok(existsSync(path), `${a.path} is missing`);
      const bytes = readFileSync(path);
      assert.equal(bytes.length, a.bytes, `${a.path} size`);
      assert.equal(createHash("sha256").update(bytes).digest("hex"), a.sha256, `${a.path} sha256`);
      assert.ok(bytes.length < 300_000, `${a.path} is ${bytes.length} bytes: keep example assets small`);
    }
  });
}

test("index.json lists every scene with its type", () => {
  const index = JSON.parse(readFileSync(join(DIR, "index.json"), "utf8"));
  assert.deepEqual(index.map((e) => e.file).sort(), [...FILES].sort());
  for (const e of index) assert.equal(e.type, JSON.parse(readFileSync(join(DIR, e.file), "utf8")).scene.type);
});

test("#/scene draws files of its own origin only", () => {
  const page = "http://127.0.0.1:8000/#/scene?src=x";
  assert.equal(sameOriginUrl("scenes/a.json", page), "http://127.0.0.1:8000/scenes/a.json");
  assert.equal(sameOriginUrl("/scenes/a.json", page), "http://127.0.0.1:8000/scenes/a.json");
  assert.equal(sameOriginUrl("https://evil.example/a.json", page), null);
  assert.equal(sameOriginUrl("//evil.example/a.json", page), null);
});
