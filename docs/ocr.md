# OCR: an image of a structure to a molfile (`serve.ocr`)

`POST /api/ocr` takes an image of a drawn structure and returns its SMILES and a 2D molfile laid
out as drawn. The engine is MolScribe 1.1.1. It runs as one persistent worker process in an
isolated environment, never inside the server's own environment.

**Status: written, not installed.** The setup has not been run yet. No MolScribe code, no
dependency of it and no weights have been downloaded. Until `ksmech-serve ocr-setup` has run,
the `ocr` capability is off with the reason "OCR engine not installed: run `ksmech-serve
ocr-setup`". `GET /api/ocr` says so, and `POST /api/ocr` answers 503 FEATURE_DISABLED.

## Install

```bash
uv run ksmech-serve ocr-setup
```

The command has two steps (`serve.ocr.provision`, CLI `serve.app.ocr_cli`):

1. **The environment.** `uv lock --project tools/ocr` runs when `tools/ocr/uv.lock` does not
   exist yet. Then `uv sync --project tools/ocr --locked` runs with
   `UV_PROJECT_ENVIRONMENT=data/ocr/venv`. It never touches the root `pyproject.toml`,
   `uv.lock` or `.venv`.
2. **The weights.** `swin_base_char_aux_1m680k.pth` (1,134,173,494 bytes) is downloaded from
   the Hugging Face repository `yujieq/MolScribe` to `data/ocr/weights/`.
   - The download goes to `<file>.part` and resumes with an HTTP Range request after an
     interruption: up to 5 tries per run, and across runs.
   - The file is then checked: its size, then its sha256
     `32821fa4ba8d17f9a92e374601438c5e757947dbf1141e9ad0ef51cccfa1c501`.
   - A file that passes is renamed into place and gets the marker `<file>.verified` (JSON:
     file, sha256, size, url, time). A file that fails is deleted.

Options:

| Option | Effect |
|---|---|
| `--relock` | re-resolve `tools/ocr/uv.lock` first |
| `--uv PATH` | another uv executable |
| `--skip-env` | skip the environment step |
| `--skip-weights` | skip the weights step |

Run it again after a failure: what is done is kept. Exit 0 means installed, 1 means a step
failed. A running server needs no restart: the capability is a file check made per request.

After the first successful lock, commit `tools/ocr/uv.lock` so that the next machine installs
the same versions.

## Layout (all under `data/ocr/`, gitignored)

| Path | What |
|---|---|
| `data/ocr/venv/` | the isolated environment (Python 3.11, CPU torch) |
| `data/ocr/weights/swin_base_char_aux_1m680k.pth` | the checkpoint |
| `data/ocr/weights/swin_base_char_aux_1m680k.pth.verified` | the marker written after the sha256 check |
| `data/ocr/worker.log` | the worker's stderr: library output and tracebacks |

The `ocr` capability is on when the venv's python, the worker script
`tools/ocr/ocr_worker.py`, the checkpoint and a marker naming the expected sha256 and the
checkpoint's current size all exist (`serve.ocr.install.engine_status`). It is a few file
stats; the sha256 is computed only by the setup.

## Pins (`tools/ocr/pyproject.toml`)

Every direct dependency is pinned exactly. `exclude-newer = "2026-10-13T00:00:00Z"` holds every
transitive dependency to a release uploaded at least two weeks before the pins were chosen.

| Package | Pin | Why |
|---|---|---|
| MolScribe | 1.1.1 | the engine |
| torch | 2.5.1 (CPU index) | below 2.6, where `torch.load` started defaulting to `weights_only=True`; MolScribe calls it without that argument |
| torchvision | 0.20.1 (CPU index) | needed by timm 0.4.12; the build that matches torch 2.5.1 |
| numpy | 1.26.4 | the last 1.x: the old pins below predate numpy 2 |
| OpenNMT-py | 2.2.0 | MolScribe's pin |
| albumentations | 1.1.0 | MolScribe's pin |
| timm | 0.4.12 | MolScribe's pin |
| SmilesPE | 0.0.3 | MolScribe's pin |
| opencv-python, opencv-python-headless | 4.10.0.84 | MolScribe asks for one, albumentations for the other: one version for both |
| Pillow | 10.4.0 | the worker reads the image size |

Python is 3.11 (`requires-python = "==3.11.*"`). The resolution is limited to Linux x86_64.

### Expected trouble at lock time (not verified: nothing has been locked)

- **torchtext.** OpenNMT-py 2.2.0 pins an old torchtext (0.5.0 in its setup.py) built for the
  torch of its time. If the lock fails on it, or if torchtext does not import under torch
  2.5.1, add an override to `tools/ocr/pyproject.toml`:

  ```toml
  [tool.uv]
  override-dependencies = ["torchtext==<a version that resolves>"]
  ```

  MolScribe imports only OpenNMT-py's transformer modules. As a last resort the override can
  point torchtext at an empty stub.
- **RDKit.** MolScribe declares RDKit as `rdkit-pypi` (older releases) or `rdkit`. If the
  resolver picks an RDKit without a cp311 wheel, constrain it with `constraint-dependencies`.
- **Two OpenCV wheels.** Both install the `cv2` module. They are pinned to one version so the
  files agree; if one is enough at lock time, drop the other.

## The worker

`tools/ocr/ocr_worker.py` runs as `data/ocr/venv/bin/python -I tools/ocr/ocr_worker.py
--weights CKPT --threads N`. It loads the model once, then answers JSON lines on stdin/stdout:

| Direction | Line |
|---|---|
| out, first | `{"ready": true, "engine": "molscribe 1.1.1", "threads": 2}`, or `{"ready": false, "error": "..."}` and exit 1 |
| in | `{"id": 7, "image": "/abs/path.png"}` |
| out | `{"id": 7, "ok": true, "smiles": "...", "molfile": "...", "confidence": 0.98, "width": 640, "height": 480}` |
| out, on failure | `{"id": 7, "ok": false, "error": "..."}` |
| in | `{"op": "quit"}`, or the end of stdin: exit 0 |

Torch threads come from `--threads`, else `KSMECH_OCR_THREADS`, else 2. The server passes its
own `KSMECH_OCR_THREADS` (default 2). File descriptor 1 is pointed at stderr before the
libraries load, so stdout carries only protocol lines.

The server side is `serve.ocr.engine.JsonLinesWorker`:

- it spawns the worker at the first image and keeps it running;
- it waits up to 300 s for the ready line (loading the checkpoint);
- it runs one image at a time under a lock: a request that waits more than 120 s for the lock
  is 503 OCR_BUSY;
- an image without an answer in 60 s kills the worker: 504 OCR_TIMEOUT;
- a worker that exits or breaks its pipe is 500 OCR_ENGINE_FAILED;
- after either failure the next request spawns a new worker.

The engine is a small protocol (`OcrEngine`: `name`, `recognize(png, timeout_s)`, `close()`),
so OSRA or DECIMER could be added beside MolScribe. Only MolScribe is implemented.

## The API

- `GET /api/ocr` returns `OcrStatusOut {enabled, reason, engine}`. Anyone may read it.
- `POST /api/ocr` takes the multipart field `image` and returns `OcrOut {smiles, molfile,
  confidence, engine, elapsed_s, warnings}`. It needs the `ocr` capability and the user role,
  and counts as one job against the role's quota (admins are unlimited). The quota is charged
  before the upload is read, so a refused upload counts too.

### Limits and errors

- Accepted types: PNG, JPEG, WebP, GIF, BMP and TIFF. Pillow reads the type from the bytes,
  never from the file name or the content type.
- At most 10 MB and 40 megapixels.
- The body is refused while it streams in, so an oversized upload never lands on disk whole.
- The engine gets the first frame, upright by its EXIF orientation, flattened onto white (a
  transparent background would turn black), scaled to at most 2048 px on its long side, as a
  PNG.

| Code | Status | When |
|---|---|---|
| `REQUEST_INVALID` | 422 | not multipart, no file in `image`, more than one file |
| `IMAGE_INVALID` | 422 | an empty upload, or an accepted type that does not decode |
| `IMAGE_TOO_LARGE` | 413 | over 10 MB, or over 40 megapixels |
| `IMAGE_TYPE_UNSUPPORTED` | 415 | not one of the accepted types |
| `OCR_FAILED` | 422 | the engine read nothing, or RDKit parses nothing it returned |
| `OCR_ENGINE_FAILED` | 500 | the worker could not start, exited or broke its pipe |
| `OCR_BUSY` | 503 | the engine stayed busy with other images |
| `OCR_TIMEOUT` | 504 | no answer in time |
| `FEATURE_DISABLED` | 503 | the engine is not installed |

### What RDKit does with the answer (`serve.ocr.structure`)

MolScribe returns two things:

- a SMILES, with abbreviations such as Ph and OMe expanded;
- a molfile of the atoms as drawn, written before the expansion. Its coordinates are fractions
  of the image width and height, with y up.

RDKit then takes the first case that applies:

1. **The molfile has the SMILES's structure.** The drawn layout is kept: scaled back by the
   image's width and height, then to a median bond of 1.5 Å, centred on the origin.
2. **Only the SMILES parses, or the drawing holds abbreviations.** The SMILES's structure is
   returned with RDKit's own layout and a warning.
3. **Only the molfile parses.** It is returned with a warning.
4. **The molfile reads only without sanitization.** It is returned as read, with a warning that
   names RDKit's reason.
5. **Nothing parses.** 422 OCR_FAILED.

The SMILES out is RDKit's canonical SMILES. The molfile is V2000.

## Licence

- MolScribe: MIT (the LICENSE of https://github.com/thomas0809/MolScribe).
- The checkpoint is published with the model on Hugging Face (`yujieq/MolScribe`); check its
  model card before redistributing it.
- Neither is vendored in this repository: the setup downloads both on the machine that runs it.
- Its dependencies keep their own licences: OpenNMT-py MIT, timm Apache-2.0, albumentations MIT,
  PyTorch BSD-3-Clause, OpenCV Apache-2.0.

## Tests

`tests/test_serve_ocr.py` runs everything on fakes:

- a fake engine for the service and the routes;
- a stub worker script, written to a temporary directory and run with the current python, for
  the protocol, the restart after a crash, the timeout and the lock;
- the real worker's loop with a fake model, and its `ready: false` path under an interpreter
  without MolScribe;
- mocked `uv` runs and a mocked URL opener for the setup: resume, Range, sha256, the marker.

`test_real_molscribe_reads_aspirin` renders aspirin with RDKit and checks the round trip of its
canonical SMILES through the real engine. It is skipped until `ocr-setup` has run on the
repository's data root.
