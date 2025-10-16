__all__ = ["app"]

import logfire

from app.core import Config

from .api import app

logfire.configure(
    service_name=Config.logfire.service_name,
    environment=Config.logfire.environment,
    send_to_logfire="if-token-present",
    distributed_tracing=False,
    console=logfire.ConsoleOptions(min_log_level="debug"),
)

logfire.instrument_pydantic()
logfire.instrument_fastapi(app)
