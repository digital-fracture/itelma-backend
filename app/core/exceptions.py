from typing import Any, ClassVar

from fastapi import UploadFile, status


class AppBaseError(Exception):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    ws_status_code = 4000
    message = "Unknown error"
    headers: ClassVar[dict[str, str] | None] = None

    def __init__(self, details: dict[str, Any] | None = None) -> None:
        self.details = details


class UnknownFileTypeError(AppBaseError):
    status_code = status.HTTP_400_BAD_REQUEST
    message = "Unknown file type"

    def __init__(self, file: UploadFile | None) -> None:
        super().__init__(
            {"filename": file.filename, "content_type": file.content_type} if file else None
        )


class PatientNotFoundError(AppBaseError):
    status_code = status.HTTP_404_NOT_FOUND
    ws_status_code = 4001
    message = "Patient not found"

    def __init__(self, patient_id: int) -> None:
        super().__init__(details={"patient_id": patient_id})


class ExaminationNotFoundError(AppBaseError):
    status_code = status.HTTP_404_NOT_FOUND
    ws_status_code = 4002
    message = "Examination not found"

    def __init__(self, patient_id: int, examination_id: int) -> None:
        super().__init__(details={"patient_id": patient_id, "examination_id": examination_id})


class ExaminationPartNotFoundError(AppBaseError):
    status_code = status.HTTP_404_NOT_FOUND
    ws_status_code = 4003
    message = "Examination part not found"

    def __init__(self, patient_id: int, examination_id: int, part_index: int) -> None:
        super().__init__(
            details={
                "patient_id": patient_id,
                "examination_id": examination_id,
                "part_index": part_index,
            }
        )


class EmulationAlreadyStartedError(AppBaseError):
    ws_status_code = 4100
    message = "Emulation of this examination already started"

    def __init__(self, patient_id: int, examination_id: int) -> None:
        super().__init__(details={"patient_id": patient_id, "examination_id": examination_id})


class EmulationNotFoundError(AppBaseError):
    ws_status_code = 4101
    message = "Emulation of this examination not found"

    def __init__(self, patient_id: int, examination_id: int) -> None:
        super().__init__(details={"patient_id": patient_id, "examination_id": examination_id})
