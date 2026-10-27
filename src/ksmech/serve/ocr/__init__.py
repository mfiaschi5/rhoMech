"""Optical chemical structure recognition: an image of a drawn structure to a molfile and SMILES.

The engine (MolScribe 1.1.1, `engine.py`) runs in its own isolated environment `data/ocr/venv`
(`tools/ocr/pyproject.toml`), as one persistent worker process that answers JSON lines
(`tools/ocr/ocr_worker.py`); nothing of it is importable in the server's environment. The
`ocr` capability is on only once `ksmech-serve ocr-setup` has built that environment and
verified the checkpoint (`install.py`, `provision.py`). Uploads are validated and converted to
PNG here (`images.py`), and the engine's answer is checked with RDKit (`structure.py`).
"""
