from functools import cache
from typing import Any

from pydantic import create_model

from app.core.exceptions import AppBaseError


@cache
def build_responses(*exceptions: type[AppBaseError]) -> dict[int | str, dict[str, Any]]:
    return {
        exception.status_code: {
            "model": create_model(
                exception.__name__,
                message=(str, exception.message),
            )
        }
        for exception in exceptions
    }
