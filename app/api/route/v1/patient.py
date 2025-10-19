from typing import Annotated, cast

from fastapi import APIRouter, Query, status
from fastapi_pagination import Page
from fastapi_pagination.ext.sqlmodel import apaginate

from app.core.exceptions import PatientNotFoundError
from app.model import Patient, PatientBrief, PatientCreate, PatientUpdate
from app.service import PatientService

from ..dependencies import DatabaseReadonlySession
from ..util import build_responses

patient_router = APIRouter(prefix="/patients", tags=["patients"])


@patient_router.get(
    "",
    status_code=status.HTTP_200_OK,
    summary="Get paginated list of all patients",
)
async def get_all_patients(session: DatabaseReadonlySession) -> Page[PatientBrief]:
    return cast(
        Page[PatientBrief],
        await apaginate(
            session=session,
            query=PatientService.get_all_statement(),
            transformer=PatientService.transform_all_for_pagination,
        ),
    )


@patient_router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    summary="Create a new patient",
)
async def create_patient(patient_create: PatientCreate) -> Patient:
    return await PatientService.create(patient_create)


@patient_router.get(
    "/{patient_id}",
    status_code=status.HTTP_200_OK,
    responses=build_responses(PatientNotFoundError),
    summary="Get detailed information about a patient",
)
async def get_patient(
    patient_id: int,
    *,
    zip_: Annotated[bool, Query(alias="zip")] = False,  # noqa: ARG001
) -> Patient:
    return await PatientService.get_by_id(patient_id)


@patient_router.patch(
    "/{patient_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses=build_responses(PatientNotFoundError),
    summary="Partially update information about a patient",
)
async def update_patient(patient_id: int, patient_update: PatientUpdate) -> None:
    await PatientService.update_by_id(patient_id, patient_update)
