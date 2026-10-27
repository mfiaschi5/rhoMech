// The OCR router (serve/ocr): a picture of a structure (a scan, a screenshot, a figure of a paper)
// to SMILES and a V2000 molfile whose 2D coordinates follow the drawing in the image. The
// capability "ocr" of /api/meta is off, with its reason, when the recognition engine is not
// installed. The types are the generated ones (types.gen.ts).
// Errors (ApiError): 413 (larger than 10 MB), 415 (not png, jpeg, webp, gif, bmp or tiff),
// 422 OCR_FAILED (no structure recognized), 429 QUOTA_EXCEEDED, 503 when the capability is off.
import { request, type RequestOptions } from "./http";
import type * as T from "./types.gen";

export type OcrOut = T.OcrOut;
export type OcrStatusOut = T.OcrStatusOut;

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
