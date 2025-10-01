import asyncio
import contextlib
from pathlib import Path
from typing import Any

import aiofiles
from pydantic import BaseModel, Field

from app.core import Paths
from app.model import (
    Channel,
    EmulationMessageInCommand,
    EmulationMessageInitial,
    EmulationMessageOut,
    EmulationMessageOutStatus,
    EmulationMessageOutUnion,
    EmulationPlot,
    EmulationPrediction,
    EmulationQueueIn,
    EmulationQueueOut,
    ExaminationPartData,
)
from app.util import AsyncRWLock

from ..examination import ExaminationService


class SessionMemory(BaseModel):
    last_status: EmulationMessageOutStatus = EmulationMessageOutStatus.SENDING
    current_part_index: int = 1
    sent_part_data: ExaminationPartData = Field(default_factory=ExaminationPartData)
    sent_predictions: list[EmulationPrediction] = Field(default_factory=list)

    def rotate(self) -> None:
        self.last_status = EmulationMessageOutStatus.SENDING
        self.current_part_index += 1
        self.sent_part_data = ExaminationPartData()
        self.sent_predictions = []


class EmulationSession:
    def __init__(self, patient_id: int, examination_id: int) -> None:
        self._patient_id = patient_id
        self._examination_id = examination_id

        self._global_lock = AsyncRWLock()
        self._bpm_log_lock = AsyncRWLock()
        self._uterus_log_lock = AsyncRWLock()
        self._log_lock_by_channel = {
            Channel.BPM: self._bpm_log_lock,
            Channel.UTERUS: self._uterus_log_lock,
        }

        self._memory = SessionMemory()
        self._queue_in = EmulationQueueIn()
        self._queues_out: list[EmulationQueueOut] = []

        self._task: asyncio.Future[Any] | None = None
        self._next_part_command_event = asyncio.Event()

    @property
    def patient_id(self) -> int:
        return self._patient_id

    @property
    def examination_id(self) -> int:
        return self._examination_id

    @property
    def queue_in(self) -> EmulationQueueIn:
        return self._queue_in

    async def start(self) -> None:
        self._task = asyncio.gather(
            self._consumer(),
            self._producer(),
            self._predictor(),
        )

    async def abort(self) -> None:
        async with self._global_lock.write():
            self._memory.last_status = EmulationMessageOutStatus.ABORTED
        await self._broadcast()

        await self._dispose()

    async def _dispose(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def subscribe(self) -> EmulationQueueOut:
        """Get new output queue populated with initial message."""
        new_queue = EmulationQueueOut()

        async with self._global_lock.read():
            initial_message_needed = len(self._queues_out) > 0

        if initial_message_needed:
            async with (
                self._global_lock.write(),
                self._bpm_log_lock.read(),
                self._uterus_log_lock.read(),
            ):
                initial_message = EmulationMessageInitial.model_validate(
                    self._memory, from_attributes=True
                )
                self._queues_out.append(new_queue)
            await new_queue.put_item(initial_message)
        else:
            async with self._global_lock.write():
                self._queues_out.append(new_queue)

        return new_queue

    async def unsubscribe(self, queue: EmulationQueueOut) -> None:
        """Make given queue no longer receive messages fron this session."""
        async with self._global_lock.write():
            self._queues_out.remove(queue)

    async def _broadcast(self, message: EmulationMessageOutUnion | None = None) -> None:
        async with self._global_lock.read():
            if message is None:
                message = EmulationMessageOut(status=self._memory.last_status)

            queues_to_send = self._queues_out.copy()

        async with asyncio.TaskGroup() as tg:
            for queue in queues_to_send:
                tg.create_task(queue.put_item(message))

    async def _consumer(self) -> None:
        async for message in self._queue_in:
            match message.command:
                case EmulationMessageInCommand.ABORT:
                    await self.abort()

                case EmulationMessageInCommand.NEXT_PART:
                    async with self._global_lock.read():
                        if (
                            self._memory.last_status
                            is not EmulationMessageOutStatus.WAITING_FOR_NEXT_COMMAND
                        ):
                            continue

                    async with self._global_lock.write():
                        self._next_part_command_event.set()

    async def _producer(self) -> None:
        patient_id = self._patient_id
        examination_id = self._examination_id
        examination = await ExaminationService.get_by_id(patient_id, examination_id)

        for part_index in range(1, examination.metadata.part_count + 1):
            bpm_path = Paths.examination_part_bpm_file(patient_id, examination_id, part_index)
            uterus_path = Paths.examination_part_uterus_file(patient_id, examination_id, part_index)

            start_time = asyncio.get_running_loop().time()
            await asyncio.gather(
                self._process_file(
                    channel=Channel.BPM,
                    path=bpm_path,
                    start_time=start_time,
                ),
                self._process_file(
                    channel=Channel.UTERUS,
                    path=uterus_path,
                    start_time=start_time,
                ),
            )

            async with self._global_lock.write():
                self._memory.last_status = EmulationMessageOutStatus.WAITING_FOR_NEXT_COMMAND
            await self._broadcast()

            await self._next_part_command_event.wait()

            self._next_part_command_event.clear()

            async with self._global_lock.write():
                self._memory.rotate()
            await self._broadcast()

        await self._broadcast(None)  # indicate emulation end
        await self._dispose()

    async def _process_file(self, channel: Channel, path: Path, start_time: float) -> None:
        log = self._memory.sent_part_data.by_channel(channel)
        lock = self._log_lock_by_channel[channel]
        current_loop = asyncio.get_running_loop()

        async with aiofiles.open(path) as file:
            await file.readline()  # skip csv header

            async for line in file:
                timestamp, value = map(float, line.strip().split(","))
                plot_point = (timestamp, value)

                elapsed = current_loop.time() - start_time
                wait_time = timestamp - elapsed
                await asyncio.sleep(wait_time)

                async with lock.write():
                    log.append(plot_point)

                await self._broadcast(
                    EmulationMessageOut(
                        plot=EmulationPlot(
                            channel=channel,
                            point=plot_point,
                        )
                    )
                )

    async def _predictor(self) -> None:
        pass  # TODO: WIP
