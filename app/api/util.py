from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import logfire
from fastapi import FastAPI

from app.core import Paths
from app.storage.lifespan import dispose_storage, initialize_storage


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None]:
    with logfire.span("App startup"):
        Paths.temp_dir.mkdir(parents=True, exist_ok=True)

        await initialize_storage()

    yield

    with logfire.span("App shutdown"):
        await dispose_storage()
