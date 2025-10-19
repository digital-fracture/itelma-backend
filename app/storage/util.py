__all__ = ["create_empty", "dump_yaml", "load_yaml", "save_temp_file"]

import asyncio
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import Any
from uuid import uuid4

import aiofiles
import aiofiles.os
import logfire
import yaml
from fastapi import UploadFile

from app.core import Paths

CHUNK_SIZE = 64 * 1024  # 64KB

executor = ThreadPoolExecutor()


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


@logfire.instrument
async def unzip(source: Path, destination: Path) -> None:
    """Extract a zip file asynchronously."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(executor, _unzip_sync, source, destination)


@logfire.instrument
async def read_text(path: Path) -> str:
    """Read text from a file asynchronously."""
    async with aiofiles.open(path) as file:
        return await file.read()


@logfire.instrument
async def write_text(contents: str, path: Path) -> None:
    """Write text to a file asynchronously."""
    async with aiofiles.open(path, "w") as file:
        await file.write(contents)


@logfire.instrument
async def load_yaml(path: Path) -> dict[Any, Any] | list[Any]:
    """Read YAML from file asynchronously."""
    async with aiofiles.open(path) as file:
        yaml_str = await file.read()

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(executor, yaml.safe_load, yaml_str)

    return result or {}


@logfire.instrument
async def dump_yaml(data: dict[Any, Any] | list[Any], path: Path) -> None:
    """Write YAML to file asynchronously."""
    loop = asyncio.get_running_loop()
    yaml_str: str = await loop.run_in_executor(
        executor,
        partial(yaml.safe_dump, allow_unicode=True),  # type: ignore
        data,
    )

    async with aiofiles.open(path, "w") as f:
        await f.write(yaml_str)


async def rename_to_numbers(directory: Path) -> None:
    """Rename all files in a directory to sequential numbers asynchronously."""
    file_names = sorted(await aiofiles.os.listdir(directory))

    async with asyncio.TaskGroup() as tg:
        for index, old_name in enumerate(file_names, start=1):
            old_path = directory / old_name
            new_path = directory / f"{index}{old_path.suffix}"
            tg.create_task(aiofiles.os.rename(old_path, new_path))


def _unzip_sync(source: Path, destination: Path) -> None:
    try:
        with zipfile.ZipFile(source, "r") as z:
            for member in z.infolist():
                if member.is_dir():
                    continue

                safe = _safe_name(member.filename)

                if not safe:
                    continue

                out_path = Path.joinpath(destination, safe)
                out_path.parent.mkdir(parents=True, exist_ok=True)

                with z.open(member, "r") as source_file, out_path.open("wb") as destination_file:
                    shutil.copyfileobj(source_file, destination_file, CHUNK_SIZE)
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def _safe_name(name: str) -> Path | None:
    path = Path(name)
    parts = [part for part in path.parts if part not in ("", ".", "..")]
    return Path(*parts) if parts else None


def _represent_strenum(dumper: yaml.SafeDumper, data: StrEnum) -> yaml.ScalarNode:
    return dumper.represent_str(data.value)


yaml.SafeDumper.add_multi_representer(StrEnum, _represent_strenum)
