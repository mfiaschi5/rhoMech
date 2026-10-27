// The OCR router (serve/ocr): a picture of a structure (a scan, a screenshot, a figure of a paper)
// to SMILES and a V2000 molfile whose 2D coordinates follow the drawing in the image. The
// capability "ocr" of /api/meta is off, with its reason, when the recognition engine is not
// installed. The types are written by hand to the backend's contract until types.gen.ts carries
// OcrOut and OcrStatusOut; then they become re-exports of the generated ones.
// Errors (ApiError): 413 (larger than 10 MB), 415 (not png, jpeg, webp, gif, bmp or tiff),
// 422 OCR_FAILED (no structure recognized), 429 QUOTA_EXCEEDED, 503 when the capability is off.
import { request, type RequestOptions } from "./http";

export interface OcrOut {
  smiles: string;
  /** V2000, 2D coordinates as laid out in the image. */
  molfile: string;
  /** The engine's confidence in [0, 1], null when it gives none. */
  confidence: number | null;
  engine: string;
  elapsed_s: number;
  warnings: string[];
}

export interface OcrStatusOut {
  enabled: boolean;
  reason: string | null;
  engine: string | null;
}

export const getOcrStatus = (o?: RequestOptions) => request<OcrStatusOut>({ path: "/ocr" }, o);

/** POST /ocr with the image as the multipart field `image`. */
export function recognizeImage(image: Blob, o?: RequestOptions): Promise<OcrOut> {
  const form = new FormData();
  const name = image instanceof File && image.name ? image.name : "image";
  form.append("image", image, name);
  return request<OcrOut>({ method: "POST", path: "/ocr", body: form }, o);
}

/** What a page hands the sketcher (its OcrSource): the capability and the call. */
export function ocrSource(cap: { enabled: boolean; reason: string | null }) {
  return {
    enabled: cap.enabled,
    reason: cap.reason,
    recognize: (image: Blob, signal?: AbortSignal) => recognizeImage(image, { signal }),
  };
}
