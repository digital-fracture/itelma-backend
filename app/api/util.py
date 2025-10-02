from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import logfire
from fastapi import FastAPI

from app.storage.lifespan import build_storage


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None]:
    with logfire.span("App startup"):
        await build_storage()

    yield
