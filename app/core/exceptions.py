from typing import Any, ClassVar

from fastapi import UploadFile, status


class AppBaseError(Exception):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    message = "Unknown error"
    headers: ClassVar[dict[str, str] | None] = None

    def __init__(self, details: dict[str, Any] | None = None) -> None:
        self.details = details


class UnknownFileTypeError(AppBaseError):
    status_code = status.HTTP_400_BAD_REQUEST
    message = "Unknown file type"

    def __init__(self, file: UploadFile) -> None:
        super().__init__({"filename": file.filename, "content_type": file.content_type})


class SessionNotFoundError(AppBaseError):
    status_code = status.HTTP_404_NOT_FOUND
    message = "Session not found"

    def __init__(self, session_id: str) -> None:
        super().__init__({"session_id": session_id})
