// @ts-nocheck -- run by `node --experimental-strip-types --test` (see model.test.ts).
// The pure part of the image import (OCR): file type and size checks, the image a drop or a
// paste carries (files first, then file items; text beats an image on paste) and the notice line.
import test from "node:test";
import assert from "node:assert/strict";
import { imageProblem, imageTypeOf, isImageFile, OCR_IMAGE_TYPES, OCR_MAX_BYTES, ocrSummary, pickImage, pickPaste } from "./imageImport.ts";

const file = (name, type, size = 1000) => ({ name, type, size });
const item = (f, kind = "file") => ({ kind, type: f.type, getAsFile: () => f });
const transfer = ({ files = [], items = [], text = "" } = {}) => ({ files, items, getData: (fmt) => (fmt.startsWith("text") ? text : "") });

test("image types: MIME type first, the extension when the browser gives none", () => {
  assert.equal(imageTypeOf(file("a.png", "image/png")), "image/png");
  assert.equal(imageTypeOf(file("scan.TIF", "")), "image/tiff");
  assert.equal(imageTypeOf(file("x.jpeg", "application/octet-stream")), "image/jpeg");
  assert.equal(imageTypeOf(file("x.mol", "")), "");
  assert.equal(imageTypeOf(file("x.png", "text/plain")), "", "a declared non-image type wins");
  assert.ok(isImageFile(file("a.webp", "image/webp")));
  assert.ok(OCR_IMAGE_TYPES.includes("image/bmp"));
});

test("imageProblem: unsupported type, too large, empty; fine otherwise", () => {
  assert.equal(imageProblem(file("a.png", "image/png")), null);
  assert.match(imageProblem(file("a.svg", "image/svg+xml")), /not supported/);
  assert.match(imageProblem(file("a.mol", "chemical/x-mdl-molfile")), /not an image/);
  assert.match(imageProblem(file("big.png", "image/png", OCR_MAX_BYTES + 1)), /limit is 10 MB/);
  assert.match(imageProblem(file("e.png", "image/png", 0)), /empty/);
});

test("pickImage: the first image file, else the first image item; null without one", () => {
  const png = file("a.png", "image/png");
  const mol = file("a.mol", "");
  assert.equal(pickImage(transfer({ files: [mol, png] })), png);
  const shot = file("image.png", "image/png");
  assert.equal(pickImage(transfer({ items: [item(file("t", "text/plain"), "string"), item(shot)] })), shot);
  assert.equal(pickImage(transfer({ files: [mol] })), null);
  assert.equal(pickImage(null), null);
});

test("pickPaste: plain text beats an image; an image alone goes to OCR", () => {
  const shot = file("image.png", "image/png");
  assert.deepEqual(pickPaste(transfer({ text: "CCO", items: [item(shot)] })), { kind: "text", text: "CCO" });
  assert.deepEqual(pickPaste(transfer({ text: "  ", items: [item(shot)] })), { kind: "image", file: shot });
  assert.equal(pickPaste(transfer({ text: "" })), null);
});

test("ocrSummary: engine, SMILES, confidence in percent, time", () => {
  const r = { smiles: "CC(=O)O", molfile: "", confidence: 0.934, engine: "molscribe", elapsed_s: 1.24, warnings: [] };
  assert.equal(ocrSummary(r), "Image read by molscribe: CC(=O)O · confidence 93 % · 1.2 s");
  assert.equal(ocrSummary({ ...r, confidence: null }), "Image read by molscribe: CC(=O)O · 1.2 s");
});
