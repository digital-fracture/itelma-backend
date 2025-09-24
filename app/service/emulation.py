import asyncio
import contextlib
from pathlib import Path
from typing import ClassVar
from uuid import uuid4

import aiofiles
import logfire
from fastapi import UploadFile

from app.core import config
from app.core.exceptions import SessionNotFoundError, UnknownFileTypeError

from .model import GraphPoint, QueueMessage, QueueType, Session

CHUNK_SIZE = 64 * 1024  # 64KB
ALLOWED_TYPE_TO_EXTENSION = {"text/csv": "csv"}


class EmulationService:
    sessions: ClassVar[dict[str, Session]] = {}

    @classmethod
    @logfire.instrument(record_return=True)
    async def create_session(cls, file: UploadFile) -> str:
        session_id = uuid4().hex

        saved_path = await cls._save_temp_file(file, session_id)
        cls.sessions[session_id] = Session(session_id=session_id, file_path=saved_path)

        return session_id

    @classmethod
    @logfire.instrument
    async def subscribe_to_session(cls, session_id: str) -> QueueType:
        if not (session := cls.sessions.get(session_id)):
            raise SessionNotFoundError(session_id)

        if session.queue is None:
            session.queue = asyncio.Queue()
            session.emulation_task = asyncio.create_task(
                cls._emulate_streaming(session.file_path, session.queue),
            )

        return session.queue

    @classmethod
    @logfire.instrument
    async def close_session(cls, session_id: str) -> None:
        if not (session := cls.sessions.get(session_id)):
            raise SessionNotFoundError(session_id)

        if session.emulation_task:
            session.emulation_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await session.emulation_task

        if session.queue:
            await session.queue.put(None)  # the end

        if session.file_path.exists():
            session.file_path.unlink(missing_ok=True)

        cls.sessions.pop(session_id)

    @classmethod
    @logfire.instrument
    async def _emulate_streaming(cls, file_path: Path, queue: QueueType) -> None:
        current_loop = asyncio.get_event_loop()
        start_time = current_loop.time()

        async with aiofiles.open(file_path) as file:
            async for line in file:
                time, value = map(float, line.strip().split(","))
                message = QueueMessage(graph_point=GraphPoint(time=time, value=value))

                current_time = current_loop.time()
                wait_time = start_time + time - current_time

                await asyncio.sleep(wait_time)
                await queue.put(message)

        await queue.put(None)  # the end

    @classmethod
    async def _save_temp_file(cls, uploaded_file: UploadFile, name: str) -> Path:
        extension = cls._get_file_extension(uploaded_file)
        path = config.app.file_storage_dir / f"{name}.{extension}"

        async with aiofiles.open(path, "wb") as out_file:
            while True:
                chunk = await uploaded_file.read(CHUNK_SIZE)

                if not chunk:
                    break

                await out_file.write(chunk)

        return path

    @classmethod
    def _get_file_extension(cls, file: UploadFile) -> str:
        if file.content_type in ALLOWED_TYPE_TO_EXTENSION:
            return ALLOWED_TYPE_TO_EXTENSION[file.content_type]

        if file.filename and (
            (file_extension := file.filename.rsplit(".", 1)[-1])
            in ALLOWED_TYPE_TO_EXTENSION.values()
        ):
            return file_extension

        raise UnknownFileTypeError(file)
