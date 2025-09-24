__all__ = ["app"]

import logfire

from app.core import config

from .api import app

logfire.configure(
    service_name=config.logfire.service_name,
    environment=config.logfire.environment,
    send_to_logfire="if-token-present",
    distributed_tracing=False,
)

logfire.instrument_pydantic()
logfire.instrument_fastapi(app)
