import logfire
from fastapi import FastAPI, Request
from fastapi.responses import ORJSONResponse

from app.core.exceptions import AppBaseError


async def _app_exception_handler(request: Request, exc: Exception) -> ORJSONResponse:
    if not isinstance(exc, AppBaseError):
        raise exc

    logfire.exception(
        "Exception raised while handling request",
        request=request,
        _exc_info=exc,
    )

    return ORJSONResponse(
        status_code=exc.status_code,
        content={"message": exc.message} | ({"details": exc.details} if exc.details else {}),
        headers=exc.headers,
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppBaseError, _app_exception_handler)
