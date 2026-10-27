// Image import (OCR) of the sketcher, the pure part: which image a drop, a paste or a file pick
// carries, whether the server will take it, and what the sketcher says about the answer. The
// recognition itself is the page's OcrSource (POST /api/ocr), handed to the Sketcher like the
// Depictor, so the sketcher never fetches.
//
// Paste rule: a clipboard with plain text imports the text as before (a SMILES or molfile is the
// exact structure; programs that copy a picture often add a text version); only a clipboard
// with an image and no text runs OCR. A drop prefers an image file over any other file or text.

/** The image types POST /api/ocr accepts (the file picker's accept list). */
export const OCR_IMAGE_TYPES = ["image/png", "image/jpeg", "image/webp", "image/gif", "image/bmp", "image/tiff"] as const;
/** The server's upload limit. */
export const OCR_MAX_BYTES = 10 * 1024 * 1024;

const EXTENSION_TYPES: Readonly<Record<string, string>> = {
  png: "image/png",
  jpg: "image/jpeg",
  jpeg: "image/jpeg",
  webp: "image/webp",
  gif: "image/gif",
  bmp: "image/bmp",
  tif: "image/tiff",
  tiff: "image/tiff",
};

/** What the OCR result carries (api/ocr.ts OcrOut, structurally). */
export interface OcrResult {
  smiles: string;
  molfile: string;
  confidence: number | null;
  engine: string;
  elapsed_s: number;
  warnings: string[];
}

/** The page's recognizer: whether the server offers it (capability "ocr") and the call. */
export interface OcrSource {
  enabled: boolean;
  /** Why it is off; null when on. */
  reason: string | null;
  recognize(image: Blob, signal?: AbortSignal): Promise<OcrResult>;
}

/** A file as the browser gives it (File, or a test double). */
export interface FileLike {
  name: string;
  type: string;
  size: number;
}

/** The image type of a file: its MIME type, or the one its extension implies when the browser
 *  gives none (some systems drop TIFF and BMP types); "" when it is not an image. */
export function imageTypeOf(f: Pick<FileLike, "name" | "type">): string {
  if (f.type.startsWith("image/")) return f.type.toLowerCase();
  if (f.type && f.type !== "application/octet-stream") return "";
  const ext = f.name.toLowerCase().split(".").pop() ?? "";
  return EXTENSION_TYPES[ext] ?? "";
}

export const isImageFile = (f: Pick<FileLike, "name" | "type">): boolean => imageTypeOf(f) !== "";

/** Why the server would refuse this image, or null when it is fine. */
export function imageProblem(f: FileLike): string | null {
  const type = imageTypeOf(f);
  if (!type) return `${f.name || "the file"} is not an image`;
  if (!(OCR_IMAGE_TYPES as readonly string[]).includes(type)) return `${type} images are not supported (PNG, JPEG, WebP, GIF, BMP or TIFF)`;
  if (f.size > OCR_MAX_BYTES) return `the image is ${(f.size / 1024 / 1024).toFixed(1)} MB; the limit is 10 MB`;
  if (f.size === 0) return "the image is empty";
  return null;
}

/** DataTransfer / ClipboardEvent data, structurally. */
export interface TransferLike<F extends FileLike = FileLike> {
  files?: ArrayLike<F> | null;
  items?: ArrayLike<{ kind: string; type: string; getAsFile(): F | null }> | null;
  getData(format: string): string;
}

/** The first image file of a transfer: its files, else its file items (a pasted screenshot is
 *  an item only); null when it carries none. */
export function pickImage<F extends FileLike>(data: TransferLike<F> | null | undefined): F | null {
  if (!data) return null;
  for (const f of Array.from(data.files ?? [])) if (isImageFile(f)) return f;
  for (const item of Array.from(data.items ?? [])) {
    if (item.kind !== "file" || !item.type.startsWith("image/")) continue;
    const f = item.getAsFile();
    if (f) return f;
  }
  return null;
}

export type PastePick<F extends FileLike = FileLike> = { kind: "text"; text: string } | { kind: "image"; file: F } | null;

/** What a paste imports (see the paste rule above). */
export function pickPaste<F extends FileLike>(data: TransferLike<F> | null | undefined): PastePick<F> {
  if (!data) return null;
  const text = data.getData("text/plain") || data.getData("text");
  if (text.trim()) return { kind: "text", text };
  const file = pickImage(data);
  return file ? { kind: "image", file } : null;
}

/** The notice line of an OCR result: engine, SMILES, confidence and time. */
export function ocrSummary(r: OcrResult): string {
  const conf = r.confidence === null ? "" : ` · confidence ${Math.round(r.confidence * 100)} %`;
  return `Image read by ${r.engine}: ${r.smiles || "(no SMILES)"}${conf} · ${r.elapsed_s.toFixed(1)} s`;
}

/** A low confidence the notice flags (below 0.5). */
export const LOW_CONFIDENCE = 0.5;
