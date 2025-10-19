from typing import Annotated

from fastapi import APIRouter, File, UploadFile, status

from app.core.exceptions import (
    ExaminationNotFoundError,
    ExaminationPartNotFoundError,
    PatientNotFoundError,
    UnknownFileTypeError,
)
from app.model import Examination, ExaminationBrief, ExaminationPart
from app.service.examination import ExaminationService

from ..util import build_responses

examination_router = APIRouter(prefix="/patients/{patient_id}/examinations", tags=["examinations"])


@examination_router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    responses=build_responses(PatientNotFoundError, UnknownFileTypeError),
    summary="Upload a new examination for a patient (and create new empty patient if needed)",
)
async def create_examination(
    patient_id: int, file: Annotated[UploadFile, File(media_type="application/zip")]
) -> ExaminationBrief:
    if (
        file.content_type
        and file.content_type != "application/octet-stream"
        and "zip" not in file.content_type.split("/")[-1]
    ):
        raise UnknownFileTypeError(file)

    return await ExaminationService.create(patient_id, file)


@examination_router.get(
    "/{examination_id}",
    status_code=status.HTTP_200_OK,
    responses=build_responses(PatientNotFoundError, ExaminationNotFoundError),
    summary="Get information about an examination",
)
async def get_examination(patient_id: int, examination_id: int) -> Examination:
    return await ExaminationService.get_by_id(patient_id, examination_id)


@examination_router.get(
    "/{examination_id}/part/{part_index}",
    status_code=status.HTTP_200_OK,
    responses=build_responses(
        PatientNotFoundError, ExaminationNotFoundError, ExaminationPartNotFoundError
    ),
    summary="Retrieve data of a specific examination part",
)
async def get_examination_part(
    patient_id: int, examination_id: int, part_index: int
) -> ExaminationPart:
    return await ExaminationService.get_part_data(patient_id, examination_id, part_index)
