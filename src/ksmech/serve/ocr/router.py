"""Routes of the OCR engine: its status, and one image to a structure.

The router is admitted by the profile (`ocr` is in `public`); `POST /ocr` needs the capability
itself (the engine installed: FEATURE_DISABLED otherwise) and the user role, and is charged to
the job quota, all before a byte of the upload is read. The upload is parsed here rather than by
a FastAPI `File` parameter, so that the body is refused at `MAX_BYTES` (plus the multipart
framing) while it streams in, instead of being spooled to disk whole first.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import APIRouter, Request
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser

from ksmech.serve.core import schemas as S
from ksmech.serve.core.capabilities import admit, require
from ksmech.serve.core.errors import Code, ServeError
from ksmech.serve.core.http import Services, Svc
from ksmech.serve.ocr import images
from ksmech.serve.ocr.service import OcrService

FIELD = "image"
FRAMING = 64 * 1024  # multipart boundaries and part headers
UPLOAD_BODY = {
    "required": True,
    "content": {
        "multipart/form-data": {
            "schema": {
                "type": "object",
                "required": [FIELD],
                "properties": {FIELD: {"type": "string", "format": "binary"}},
            }
        }
    },
}

router = APIRouter(prefix="/api", dependencies=[admit("ocr")])


def _ocr(svc: Services) -> OcrService:
    if svc.ocr is None:
        raise ServeError(Code.FEATURE_DISABLED, "ocr is not set up in this app")
    return svc.ocr


async def _limited(stream: AsyncIterator[bytes], limit: int) -> AsyncIterator[bytes]:
    total = 0
    async for chunk in stream:
        total += len(chunk)
        if total > limit:
            raise images.too_large(f"the upload is larger than {images.MAX_BYTES} bytes")
        yield chunk


async def _parse(request: Request, limit: int):
    parser = MultiPartParser(
        request.headers, _limited(request.stream(), limit), max_files=1, max_fields=8
    )
    try:
        return await parser.parse()
    except MultiPartException as exc:
        raise ServeError(Code.REQUEST_INVALID, f"malformed multipart body: {exc.message}") from exc


async def read_upload(request: Request) -> bytes:
    """The bytes of the multipart field `image`, refused past `images.MAX_BYTES`."""
    if not request.headers.get("content-type", "").lower().startswith("multipart/form-data"):
        msg = f"send the image as multipart/form-data in the field {FIELD!r}"
        raise ServeError(Code.REQUEST_INVALID, msg)
    limit = images.MAX_BYTES + FRAMING
    length = request.headers.get("content-length", "")
    if length.isdigit() and int(length) > limit:
        raise images.too_large(f"the upload is {length} bytes: at most {images.MAX_BYTES}")
    form = await _parse(request, limit)
    try:
        upload = form.get(FIELD)
        if not isinstance(upload, UploadFile):
            raise ServeError(Code.REQUEST_INVALID, f"no file in the multipart field {FIELD!r}")
        data = await upload.read(images.MAX_BYTES + 1)
    finally:
        await form.close()
    images.check_size(len(data))
    return data


@router.get("/ocr", response_model=S.OcrStatusOut)
def ocr_status(svc: Svc) -> S.OcrStatusOut:
    """Whether the OCR engine is installed on this server, and which one."""
    return _ocr(svc).status_out()


@router.post(
    "/ocr",
    response_model=S.OcrOut,
    dependencies=[require("ocr")],
    openapi_extra={"requestBody": UPLOAD_BODY},
)
async def ocr_image(request: Request, svc: Svc) -> S.OcrOut:
    """One image of a drawn structure (multipart field `image`: png, jpeg, webp, gif, bmp or
    tiff, at most 10 MB) to its SMILES and a 2D molfile laid out as drawn."""
    data = await read_upload(request)
    return await run_in_threadpool(_ocr(svc).recognize, data)
