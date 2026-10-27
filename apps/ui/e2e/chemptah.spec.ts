// Playwright test of ChemPtah's sketcher additions (#/workbench), with POST /api/ocr mocked by
// page.route (the engine is the backend's; the UI's part is the three ways in and the import):
//   OCR: an image dropped on the canvas, pasted (a synthetic paste event whose DataTransfer
//   holds only a PNG File) and picked in the import panel's "Image (OCR)" each draw acetic acid
//   from the mocked molfile (/api/meta's ocr capability mocked on) (data-smiles CC(=O)O, one undo step back to the empty sketch); the
//   notice shows engine, confidence and warnings; with /api/meta's ocr capability turned off
//   (mocked) a drop shows the reason instead.
//   Shortcuts: Shift+6 on an atom (a spiro cyclohexane), w / h on a bond (wedge, hash), Enter on
//   an atom (the label editor: "N+"), Space on an atom (back to carbon), "?" (the cheat sheet),
//   Ctrl+A Ctrl+D (a duplicate fragment), Ctrl+Shift+H (a flip, same SMILES).
//   Fixed lengths and angles: a bond-tool drag at an odd angle ends one standard bond away at a
//   multiple of 30 degrees; dragging that bond's end with the select tool snaps the same way;
//   with Alt held it lands where the pointer is; the toggles turn snapping off and are kept per
//   tab across a reload.
// Skipped, with the server's reason, while the workbench capability is off.
import { expect, test, type Locator, type Page } from "@playwright/test";

import { browserPath, NO_BROWSER_REASON } from "./browser";
import { skipUnless, waitRdkit, type XY } from "./helpers";

test.skip(!browserPath, NO_BROWSER_REASON);

const ACETIC_ACID = [
  "",
  "  ChemRho OCR mock",
  "",
  "  4  3  0  0  0  0  0  0  0  0999 V2000",
  "    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0",
  "    1.2990    0.7500    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0",
  "    2.5981    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0",
  "    1.2990    2.2500    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0",
  "  1  2  1  0",
  "  2  3  1  0",
  "  2  4  2  0",
  "M  END",
  "",
].join("\n");

const OCR_OUT = { smiles: "CC(=O)O", molfile: ACETIC_ACID, confidence: 0.91, engine: "mock-ocr", elapsed_s: 0.4, warnings: ["one stereo bond unread"] };

/** A 1x1 PNG (the mock never looks at it). */
const PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";

/** Mock POST /api/ocr; the number of calls so far. */
/** Mock POST /api/ocr, and turn the capability on in /api/meta (the e2e server has no engine). */
async function mockOcr(page: Page): Promise<{ calls: () => number }> {
  let calls = 0;
  await mockOcrCapability(page, { enabled: true, reason: null });
  await page.route("**/api/ocr", async (route) => {
    if (route.request().method() !== "POST") return route.fallback();
    calls += 1;
    expect(route.request().headers()["content-type"] ?? "").toMatch(/^multipart\/form-data/);
    await route.fulfill({ json: OCR_OUT });
  });
  return { calls: () => calls };
}

async function mockOcrCapability(page: Page, ocr: { enabled: boolean; reason: string | null }): Promise<void> {
  await page.route("**/api/meta", async (route) => {
    const meta = await (await route.fetch()).json();
    meta.capabilities = { ...meta.capabilities, ocr };
    await route.fulfill({ json: meta });
  });
}

async function openSketch(page: Page): Promise<Locator> {
  await page.goto("/#/workbench");
  await waitRdkit(page);
  const sketch = page.getByTestId("sketch");
  await expect(sketch).toHaveAttribute("data-status", /^(empty|ready)$/);
  return sketch;
}

/** Dispatch a drop / paste with a PNG File in a DataTransfer, inside the page. */
async function sendImage(page: Page, how: "drop" | "paste"): Promise<void> {
  await page.evaluate(
    ({ b64, how }) => {
      const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
      const dt = new DataTransfer();
      dt.items.add(new File([bytes], "structure.png", { type: "image/png" }));
      if (how === "drop") {
        const stage = document.querySelector(".sk-stage");
        stage?.dispatchEvent(new DragEvent("dragover", { dataTransfer: dt, bubbles: true, cancelable: true }));
        stage?.dispatchEvent(new DragEvent("drop", { dataTransfer: dt, bubbles: true, cancelable: true }));
      } else {
        document.dispatchEvent(new ClipboardEvent("paste", { clipboardData: dt, bubbles: true, cancelable: true }));
      }
    },
    { b64: PNG_B64, how },
  );
}

async function expectAceticAcid(page: Page, sketch: Locator): Promise<void> {
  await expect(sketch).toHaveAttribute("data-smiles", "CC(=O)O");
  const notice = page.getByTestId("sketch-ocr");
  await expect(notice).toHaveAttribute("data-state", "done");
  await expect(notice).toHaveAttribute("data-engine", "mock-ocr");
  await expect(page.getByTestId("sketch-ocr-text")).toContainText("confidence 91 %");
  await expect(page.getByTestId("sketch-ocr-warning")).toHaveText("one stereo bond unread");
  await page.getByTestId("sketch-undo").click();
  await expect(sketch).toHaveAttribute("data-atoms", "0");
  await page.getByTestId("sketch-ocr-dismiss").click();
}

test("OCR: drop, paste and the file picker each draw the recognized structure", async ({ page, request }) => {
  await skipUnless(request, "workbench");
  const ocr = await mockOcr(page);
  const sketch = await openSketch(page);
  await sendImage(page, "drop");
  await expectAceticAcid(page, sketch);
  await page.getByTestId("sketch-canvas").hover();
  await sendImage(page, "paste");
  await expectAceticAcid(page, sketch);
  await page.getByTestId("sketch-import").click();
  await page.getByTestId("sketch-import-image").setInputFiles({ name: "scan.png", mimeType: "image/png", buffer: Buffer.from(PNG_B64, "base64") });
  await expectAceticAcid(page, sketch);
  expect(ocr.calls()).toBe(3);
  // a text paste still imports the text (no OCR call)
  await page.getByTestId("sketch-canvas").hover();
  await page.evaluate(() => {
    const dt = new DataTransfer();
    dt.setData("text/plain", "CCO");
    document.dispatchEvent(new ClipboardEvent("paste", { clipboardData: dt, bubbles: true, cancelable: true }));
  });
  await expect(sketch).toHaveAttribute("data-smiles", "CCO");
  expect(ocr.calls()).toBe(3);
});

test("OCR off: a dropped image shows the reason", async ({ page, request }) => {
  await skipUnless(request, "workbench");
  await mockOcrCapability(page, { enabled: false, reason: "no OCR engine installed (mocked)" });
  const sketch = await openSketch(page);
  await sendImage(page, "drop");
  await expect(page.getByTestId("sketch-ocr")).toHaveAttribute("data-state", "off");
  await expect(page.getByTestId("sketch-ocr-text")).toContainText("no OCR engine installed (mocked)");
  await expect(sketch).toHaveAttribute("data-atoms", "0");
  await page.getByTestId("sketch-import").click();
  await expect(page.getByTestId("sketch-import-image")).toBeDisabled();
  await expect(page.getByTestId("sketch-import-image-off")).toContainText("no OCR engine installed");
});

/** Ethane by one click of the bond tool: atoms 1 and 2, bond 3. */
async function drawEthane(page: Page, sketch: Locator): Promise<void> {
  const canvas = page.getByTestId("sketch-canvas");
  const box = await canvas.boundingBox();
  if (!box) throw new Error("the sketch canvas has no box");
  await canvas.click({ position: { x: box.width / 2 - 40, y: box.height / 2 } });
  await expect(sketch).toHaveAttribute("data-smiles", "CC");
}

test("ChemDraw hotkeys: rings, bond stereo, label editor, Space, cheat sheet, duplicate, flip", async ({ page, request }) => {
  await skipUnless(request, "workbench");
  const sketch = await openSketch(page);
  await drawEthane(page, sketch);
  await page.getByTestId("sketch-atom-2").hover();
  await page.keyboard.press("Shift+Digit6");
  await expect(sketch).toHaveAttribute("data-smiles", "CC1CCCCC1");
  await page.keyboard.press("Control+z");
  await expect(sketch).toHaveAttribute("data-smiles", "CC");
  const bond = page.getByTestId("sketch-bond-3");
  await bond.hover();
  await page.keyboard.press("w");
  await expect(bond).toHaveAttribute("data-stereo", "wedge");
  await page.keyboard.press("h");
  await expect(bond).toHaveAttribute("data-stereo", "hash");
  await page.getByTestId("sketch-atom-1").hover();
  await page.keyboard.press("Enter");
  const input = page.getByTestId("sketch-label-input");
  await input.fill("N+");
  await input.press("Enter");
  await expect(sketch).toHaveAttribute("data-smiles", "C[NH3+]");
  await page.getByTestId("sketch-atom-1").hover();
  await page.keyboard.press("Space");
  await expect(sketch).toHaveAttribute("data-smiles", "CC");
  await page.keyboard.press("?");
  await expect(page.getByTestId("sketch-keys-panel")).toBeVisible();
  expect(await page.getByTestId("sketch-key-row").count()).toBeGreaterThan(20);
  await page.keyboard.press("Escape");
  await expect(page.getByTestId("sketch-keys-panel")).toHaveCount(0);
  await page.keyboard.press("Control+a");
  await page.keyboard.press("Control+d");
  await expect(sketch).toHaveAttribute("data-fragments", "2");
  await expect(sketch).toHaveAttribute("data-smiles", "CC.CC");
  await page.keyboard.press("Escape");
  await page.getByTestId("sketch-atom-1").hover();
  await page.keyboard.press("o");
  const smiles = await sketch.getAttribute("data-smiles");
  const before = await atomXY(page, 1);
  await page.getByTestId("sketch-canvas").hover({ position: { x: 5, y: 5 } });
  await page.keyboard.press("Control+Shift+H");
  await expect.poll(() => atomXY(page, 1)).not.toEqual(before);
  await expect(sketch).toHaveAttribute("data-smiles", smiles ?? "");
});

/** Screen position (svg px) of sketch atom `id`. */
async function atomXY(page: Page, id: number): Promise<XY> {
  const c = page.getByTestId(`sketch-atom-${id}`).locator("circle.sk-hit");
  return [Number(await c.getAttribute("cx")), Number(await c.getAttribute("cy"))];
}

/** Angle (degrees, counter-clockwise on screen, in [0, 360)) and length of p -> q in svg px. */
function polar(p: XY, q: XY): { deg: number; len: number } {
  const deg = (Math.atan2(-(q[1] - p[1]), q[0] - p[0]) * 180) / Math.PI;
  return { deg: (deg + 360) % 360, len: Math.hypot(q[0] - p[0], q[1] - p[1]) };
}

async function dragOnCanvas(page: Page, from: XY, to: XY, alt = false): Promise<void> {
  const box = await page.getByTestId("sketch-canvas").boundingBox();
  if (!box) throw new Error("the sketch canvas has no box");
  await page.mouse.move(box.x + from[0], box.y + from[1]);
  if (alt) await page.keyboard.down("Alt");
  await page.mouse.down();
  await page.mouse.move(box.x + (from[0] + to[0]) / 2, box.y + (from[1] + to[1]) / 2, { steps: 4 });
  await page.mouse.move(box.x + to[0], box.y + to[1], { steps: 4 });
  await page.mouse.up();
  if (alt) await page.keyboard.up("Alt");
}

function expectSnapped(p: XY, q: XY, bondPx: number): void {
  const { deg, len } = polar(p, q);
  expect(Math.abs(deg / 30 - Math.round(deg / 30))).toBeLessThan(0.01);
  expect(Math.abs(len - bondPx)).toBeLessThan(0.5);
}

test("fixed lengths and angles: drawn and dragged bonds snap; Alt draws freely; the toggles persist", async ({ page, request }) => {
  await skipUnless(request, "workbench");
  const sketch = await openSketch(page);
  await expect(sketch).toHaveAttribute("data-fixed-lengths", "true");
  await expect(sketch).toHaveAttribute("data-fixed-angles", "true");
  const box = await page.getByTestId("sketch-canvas").boundingBox();
  if (!box) throw new Error("the sketch canvas has no box");
  const bondPx = Number(await page.getByTestId("sketch-canvas").getAttribute("data-bond-px"));
  const start: XY = [box.width / 2 - 60, box.height / 2];
  await dragOnCanvas(page, start, [start[0] + 70, start[1] - 23]); // 18 degrees, 74 px
  await expect(sketch).toHaveAttribute("data-bonds", "1");
  expectSnapped(await atomXY(page, 1), await atomXY(page, 2), bondPx);
  expect(Math.round(polar(await atomXY(page, 1), await atomXY(page, 2)).deg)).toBe(30);
  // the select tool drags the bond end: it snaps around atom 1
  await page.getByTestId("tool-select").click();
  const end = await atomXY(page, 2);
  await dragOnCanvas(page, end, [end[0] - 10, end[1] - 60]);
  expectSnapped(await atomXY(page, 1), await atomXY(page, 2), bondPx);
  // Alt: the end lands where the pointer is
  const snapped = await atomXY(page, 2);
  const free: XY = [snapped[0] + 17, snapped[1] + 9];
  await dragOnCanvas(page, snapped, free, true);
  const at = await atomXY(page, 2);
  expect(Math.hypot(at[0] - free[0], at[1] - free[1])).toBeLessThan(1.5);
  // the toggles: off, kept after a reload, and a drag with fixed lengths off follows the pointer
  await page.getByTestId("sketch-fixed-lengths").click();
  await expect(sketch).toHaveAttribute("data-fixed-lengths", "false");
  await page.reload();
  await waitRdkit(page);
  await expect(page.getByTestId("sketch")).toHaveAttribute("data-fixed-lengths", "false");
  await expect(page.getByTestId("sketch")).toHaveAttribute("data-fixed-angles", "true");
  await page.getByTestId("sketch-fixed-lengths").click();
  await expect(page.getByTestId("sketch")).toHaveAttribute("data-fixed-lengths", "true");
});
