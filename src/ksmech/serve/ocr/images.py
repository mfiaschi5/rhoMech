"""Uploads for the OCR engine: check what the bytes are, then hand the engine one plain PNG.

The type is what Pillow reads in the bytes, never the file name or the client's content type:
PNG, JPEG (and MPO, the multi-picture JPEG of phone cameras), WebP, GIF, BMP and TIFF. Anything
Pillow does not recognize, or recognizes as another format, is IMAGE_TYPE_UNSUPPORTED (415); an
accepted format that does not decode is IMAGE_INVALID (422); more than `MAX_BYTES` or
`MAX_PIXELS` is IMAGE_TOO_LARGE (413). The engine gets the first frame, turned upright by its
EXIF orientation, flattened onto white (a transparent background would otherwise turn black:
drawing programs export transparent PNGs) and scaled down to at most `MAX_SIDE` pixels on its
long side (MolScribe reads at 384 x 384 anyway).
"""

from __future__ import annotations

import io
import warnings
from dataclasses import dataclass, field

from ksmech.serve.core.errors import Code, ServeError

MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 40_000_000
MAX_SIDE = 2048
ACCEPTED = {
    "PNG": "png",
    "JPEG": "jpeg",
    "MPO": "jpeg",
    "WEBP": "webp",
    "GIF": "gif",
    "BMP": "bmp",
    "TIFF": "tiff",
}
ACCEPTED_NAMES = ("png", "jpeg", "webp", "gif", "bmp", "tiff")


@dataclass(frozen=True)
class PreparedImage:
    png: bytes
    format: str  # one of ACCEPTED_NAMES: what the upload was
    width: int  # of `png`
    height: int
    warnings: list[str] = field(default_factory=list)


def too_large(message: str, **detail) -> ServeError:
    return ServeError(Code.IMAGE_TOO_LARGE, message, detail={"max_bytes": MAX_BYTES} | detail)


def check_size(n_bytes: int) -> None:
    if n_bytes == 0:
        raise ServeError(Code.IMAGE_INVALID, "the uploaded image is empty")
    if n_bytes > MAX_BYTES:
        raise too_large(f"the image is larger than {MAX_BYTES // (1024 * 1024)} MB")


def _open(data: bytes):
    from PIL import Image, UnidentifiedImageError

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            img = Image.open(io.BytesIO(data))
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise too_large(f"the image has too many pixels: {exc}") from exc
    except UnidentifiedImageError as exc:
        msg = f"not an image of an accepted type ({', '.join(ACCEPTED_NAMES)})"
        raise ServeError(Code.IMAGE_TYPE_UNSUPPORTED, msg) from exc
    if img.format not in ACCEPTED:
        msg = f"{img.format} images are not accepted: send one of {', '.join(ACCEPTED_NAMES)}"
        raise ServeError(Code.IMAGE_TYPE_UNSUPPORTED, msg, detail={"format": img.format})
    width, height = img.size
    if width < 1 or height < 1 or width * height > MAX_PIXELS:
        msg = f"the image is {width} x {height} pixels: at most {MAX_PIXELS} pixels"
        raise too_large(msg, width=width, height=height, max_pixels=MAX_PIXELS)
    return img


def _decoded(img):
    from PIL import ImageOps

    try:
        img.load()
        return ImageOps.exif_transpose(img)
    except (OSError, SyntaxError, ValueError, EOFError) as exc:
        msg = f"the {ACCEPTED[img.format]} image does not decode: {exc}"
        raise ServeError(Code.IMAGE_INVALID, msg) from exc


def _flatten(img):
    """RGB on a white background, whatever the mode (palette, alpha, 16-bit, CMYK)."""
    from PIL import Image

    if img.mode.startswith("I;16"):
        img = img.convert("I")
    if img.mode in ("I", "F"):
        lo, hi = img.getextrema()
        scale = 255.0 / (hi - lo) if hi > lo else 1.0
        img = img.point(lambda v: (v - lo) * scale).convert("L")
    alpha = img.mode in ("RGBA", "LA", "PA", "La", "RGBa") or "transparency" in img.info
    if alpha:
        white = Image.new("RGBA", img.size, (255, 255, 255, 255))
        return Image.alpha_composite(white, img.convert("RGBA")).convert("RGB")
    return img.convert("RGB")


def to_png(data: bytes) -> PreparedImage:
    """The upload as the PNG the engine reads (module docstring), or a reason-coded error."""
    from PIL import Image

    check_size(len(data))
    img = _open(data)
    kind = ACCEPTED[img.format]
    notes = []
    frames = int(getattr(img, "n_frames", 1) or 1)
    if frames > 1:
        notes.append(f"the {kind} image has {frames} frames: only the first was read")
    img = _flatten(_decoded(img))
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
    out = io.BytesIO()
    img.save(out, format="PNG")
    return PreparedImage(out.getvalue(), kind, img.size[0], img.size[1], notes)
