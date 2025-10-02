from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import ClassVar

import logfire

from app.core.exceptions import EmulationAlreadyStartedError, EmulationNotFoundError
from app.model import EmulationQueueIn, EmulationQueueOut
from app.storage import ExaminationStorage

from .session import EmulationSession


class EmulationService:
    _sessions: ClassVar[dict[tuple[int, int], EmulationSession]] = {}

    @classmethod
    @asynccontextmanager
    @logfire.instrument(allow_generator=True)
    async def new(
        cls, patient_id: int, examination_id: int
    ) -> AsyncGenerator[tuple[EmulationQueueOut, EmulationQueueIn]]:
        if (patient_id, examination_id) in cls._sessions:
            raise EmulationAlreadyStartedError(patient_id, examination_id)

        await ExaminationStorage.check_exists(patient_id, examination_id)

        session = EmulationSession(patient_id, examination_id)
        cls._sessions[(patient_id, examination_id)] = session

        queue_out = await session.subscribe()
        session.start()

        try:
            yield queue_out, session.queue_in
        except Exception:
            logfire.exception(
                "Exception in main session listener, aborting forcefully...",
                patient_id=patient_id,
                examination_id=examination_id,
            )
            session.force_abort()
        finally:
            del cls._sessions[(patient_id, examination_id)]

    @classmethod
    @asynccontextmanager
    @logfire.instrument(allow_generator=True)
    async def attach(
        cls, patient_id: int, examination_id: int
    ) -> AsyncGenerator[EmulationQueueOut]:
        await cls._assert_exists(patient_id, examination_id)

        session = cls._sessions[(patient_id, examination_id)]
        queue = await session.subscribe()

        try:
            yield queue
        finally:
            await session.unsubscribe(queue)

    @classmethod
    async def _assert_exists(cls, patient_id: int, examination_id: int) -> None:
        if (patient_id, examination_id) not in cls._sessions:
            raise EmulationNotFoundError(patient_id, examination_id)
