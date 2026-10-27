"""Routes of the molecule workbench (display only, gated by the `workbench` capability)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query
from fastapi.responses import PlainTextResponse

from ksmech.serve.core import schemas as S
from ksmech.serve.core.capabilities import require
from ksmech.serve.core.http import (
    MoleculeId,
    Svc,
    blank,
    check_id,
)
from ksmech.serve.workbench import molecule
from ksmech.serve.workbench import service as workbench_service

router = APIRouter(prefix="/api")


_WB = [require("workbench")]


@router.post("/molecule", response_model=S.MoleculeOut, dependencies=_WB)
def create_molecule(svc: Svc, body: S.MoleculeIn) -> S.MoleculeOut:
    return molecule.create_molecule(svc.ctx, body)


@router.get("/molecule/{molecule_id}", response_model=S.MoleculeOut, dependencies=_WB)
def get_molecule(svc: Svc, molecule_id: MoleculeId) -> S.MoleculeOut:
    return molecule.molecule_out(svc.ctx, check_id("molecule_id", molecule_id))


@router.get(
    "/molecule/{molecule_id}/geometry", response_model=S.MoleculeGeometryOut, dependencies=_WB
)
def molecule_geometry(svc: Svc, molecule_id: MoleculeId) -> S.MoleculeGeometryOut:
    return workbench_service.geometry_out(svc.ctx, check_id("molecule_id", molecule_id))


@router.get(
    "/molecule/{molecule_id}/properties",
    response_model=S.MoleculePropertiesOut,
    dependencies=_WB,
)
def molecule_properties(svc: Svc, molecule_id: MoleculeId) -> S.MoleculePropertiesOut:
    return workbench_service.properties_out(svc.ctx, check_id("molecule_id", molecule_id))


@router.get("/molecule/{molecule_id}/symmetry", response_model=S.PointGroupOut, dependencies=_WB)
def molecule_symmetry(
    svc: Svc,
    molecule_id: MoleculeId,
    tolerance: Annotated[float | None, Query(ge=0.01, le=0.5)] = None,
) -> S.PointGroupOut:
    mid = check_id("molecule_id", molecule_id)
    return workbench_service.symmetry_out(svc.ctx, mid, tolerance)


@router.get("/molecule/{molecule_id}/orbitals", response_model=S.OrbitalSetOut, dependencies=_WB)
def molecule_orbitals(
    svc: Svc,
    molecule_id: MoleculeId,
    set_: Annotated[str | None, Query(alias="set")] = None,
    start: str | None = None,
) -> S.OrbitalSetOut:
    mid = check_id("molecule_id", molecule_id)
    return workbench_service.orbitals_out(svc.ctx, mid, blank(set_), blank(start))


@router.get("/molecule/{molecule_id}/cube", response_class=PlainTextResponse, dependencies=_WB)
def molecule_cube(
    svc: Svc,
    molecule_id: MoleculeId,
    kind: str,
    set_: Annotated[str | None, Query(alias="set")] = None,
    start: str | None = None,
    index: Annotated[int | None, Query(ge=0)] = None,
    quality: str | None = None,
    spacing: float | None = None,
    padding: float | None = None,
    download: bool = False,
) -> PlainTextResponse:
    mid = check_id("molecule_id", molecule_id)
    cube = workbench_service.cube_out(
        svc.ctx,
        mid,
        kind,
        set_=blank(set_),
        start=blank(start),
        index=index,
        quality=blank(quality),
        spacing=spacing,
        padding=padding,
    )
    headers = workbench_service.cube_headers(cube, download)
    return PlainTextResponse(cube.text, headers=headers)


@router.post("/depict2d", response_model=S.Depict2DOut, dependencies=_WB)
def depict2d(svc: Svc, body: S.Depict2DIn) -> S.Depict2DOut:
    if body.molecule_id is not None:
        check_id("molecule_id", body.molecule_id)
    return workbench_service.depict2d_out(svc.ctx, body)
