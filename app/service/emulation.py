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

from .model import Plot, Point, QueueMessage, QueueType, Session

CHUNK_SIZE = 64 * 1024  # 64KB


class EmulationService:
    sessions: ClassVar[dict[str, Session]] = {}

    @classmethod
    @logfire.instrument(record_return=True)
    async def create_session(cls, files: list[UploadFile]) -> str:
        session_id = uuid4().hex

        saved_paths = await asyncio.gather(*(cls._save_temp_file(file) for file in files))
        cls.sessions[session_id] = Session(session_id=session_id, files=saved_paths)

        return session_id

    @classmethod
    @logfire.instrument
    async def subscribe_to_session(cls, session_id: str) -> QueueType:
        if not (session := cls.sessions.get(session_id)):
            raise SessionNotFoundError(session_id)

        if session.queue is None:
            queue = session.queue = asyncio.Queue()
            start_time = asyncio.get_event_loop().time()

            session.emulation_task_gather = asyncio.gather(
                *(
                    asyncio.create_task(
                        cls._emulate_streaming(
                            index=index,
                            file_path=file,
                            queue=queue,
                            start_time=start_time,
                        )
                    )
                    for index, file in enumerate(session.files)
                )
            )
            session.emulation_task_gather.add_done_callback(
                lambda _: asyncio.create_task(queue.put(None))
            )

        return session.queue

    @classmethod
    @logfire.instrument
    async def close_session(cls, session_id: str) -> None:
        if not (session := cls.sessions.get(session_id)):
            raise SessionNotFoundError(session_id)

        if session.emulation_task_gather:
            session.emulation_task_gather.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await session.emulation_task_gather

        if session.queue:
            await session.queue.put(None)  # the end

        for file in session.files:
            file.unlink(missing_ok=True)

        del cls.sessions[session_id]

    @classmethod
    @logfire.instrument
    async def _emulate_streaming(
        cls, index: int, file_path: Path, queue: QueueType, start_time: float
    ) -> None:
        current_loop = asyncio.get_event_loop()

        async with aiofiles.open(file_path) as file:
            await file.readline()  # skip csv header

            async for line in file:
                time, value = map(float, line.strip().split(","))
                message = QueueMessage(
                    plot=Plot(
                        index=index,
                        point=Point(time=time, value=value),
                    )
                )

                current_time = current_loop.time()
                wait_time = start_time + time - current_time

                await asyncio.sleep(wait_time)
                await queue.put(message)

    @classmethod
    async def _save_temp_file(cls, uploaded_file: UploadFile) -> Path:
        suffix = cls._get_file_extension(uploaded_file)
        path = config.app.file_storage_dir / f"{uuid4().hex}.{suffix}"

        async with aiofiles.open(path, "wb") as out_file:
            while True:
                chunk = await uploaded_file.read(CHUNK_SIZE)

                if not chunk:
                    break

                await out_file.write(chunk)

        return path

    @classmethod
    def _get_file_extension(cls, file: UploadFile) -> str:
        if file.content_type in config.app.allowed_file_types:
            return config.app.allowed_file_types[file.content_type]

        if file.filename and (
            (suffix := file.filename.rsplit(".", 1)[-1]) in config.app.allowed_file_types.values()
        ):
            return suffix

        raise UnknownFileTypeError(file)
