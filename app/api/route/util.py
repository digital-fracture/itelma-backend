from functools import cache
from typing import Any

from pydantic import create_model

from app.core.exceptions import AppBaseError


@cache
def _build_responses_internal(
    exceptions: tuple[type[AppBaseError]],
) -> dict[int | str, dict[str, Any]]:
    return {
        exception.status_code: {
            "model": create_model(
                exception.__name__,
                message=(str, exception.message),
            ),
        }
        for exception in exceptions
    }


@cache
def build_responses(
    *exceptions: type[AppBaseError],
    # include_dev_auth: bool = False,
) -> dict[int | str, dict[str, Any]]:
    exception_list: list[type[AppBaseError]] = []

    # if include_dev_auth:
    #     exception_list.extend((ApiKeyForbiddenException, ApiKeyUnauthorizedException))

    exception_list.extend(exceptions)

    return _build_responses_internal(tuple(exception_list))
