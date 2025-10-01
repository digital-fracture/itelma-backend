__all__ = ["create_empty", "dump_yaml", "load_yaml", "save_temp_file"]

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any
from uuid import uuid4

import aiofiles
import logfire
import yaml
from fastapi import UploadFile

from app.core import Paths

CHUNK_SIZE = 64 * 1024  # 64KB

executor = ThreadPoolExecutor()


@logfire.instrument
async def load_yaml(path: Path) -> dict[Any, Any] | None:
    """Read YAML from file asynchronously."""
    async with aiofiles.open(path) as file:
        yaml_str = await file.read()

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, yaml.safe_load, yaml_str)


@logfire.instrument
async def dump_yaml(data: dict[Any, Any], path: Path) -> None:
    """Write YAML to file asynchronously."""
    loop = asyncio.get_running_loop()
    yaml_str: str = await loop.run_in_executor(
        executor,
        partial(yaml.safe_dump, default_flow_style=False),  # type: ignore
        data,
    )

    async with aiofiles.open(path, "w") as f:
        await f.write(yaml_str)


@logfire.instrument
async def create_empty(path: Path) -> None:
    """Create an empty file asynchronously."""
    async with aiofiles.open(path, "w") as file:
        await file.write("")


@logfire.instrument
async def save_temp_file(file: UploadFile) -> Path:
    """Save an UploadFile to a temporary file asynchronously and return the path."""
    suffix = file.filename.rsplit(".", 1)[-1] if file.filename and "." in file.filename else "tmp"
    temp_path = Paths.temp_dir / f"{uuid4().hex}.{suffix}"

    async with aiofiles.open(temp_path, "wb") as out_file:
        while chunk := await file.read(CHUNK_SIZE):
            await out_file.write(chunk)

    return temp_path
