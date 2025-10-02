import asyncio
from pathlib import Path
from typing import Any

import aiofiles
import logfire

from app.core import Paths
from app.model import (
    Channel,
    EmulationCommand,
    EmulationMessageClose,
    EmulationMessageOut,
    EmulationMessagePlot,
    EmulationMessageState,
    EmulationMessageStatus,
    EmulationPlot,
    EmulationQueueIn,
    EmulationQueueOut,
    EmulationState,
    EmulationStatus,
    ExaminationPartData,
)
from app.util import AsyncRWLock

from ..examination import ExaminationService


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

        self._memory = EmulationState(
            last_status=EmulationStatus.SENDING,
            current_part_index=1,
            sent_part_data=ExaminationPartData(),
            sent_predictions=[],
        )
        self._queue_in = EmulationQueueIn()
        self._queues_out: list[EmulationQueueOut] = []

        self._task: asyncio.Task[Any] | None = None
        self._shutdown_event = asyncio.Event()
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

    @logfire.instrument("Starting session")
    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    @logfire.instrument("Forcefully aborting session")
    def force_abort(self) -> None:
        self._shutdown_event.set()

    @logfire.instrument("Subscribing to session")
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
                initial_state = self._memory.model_copy()
                self._queues_out.append(new_queue)
            await new_queue.put(EmulationMessageState(state=initial_state))
        else:
            async with self._global_lock.write():
                self._queues_out.append(new_queue)

        return new_queue

    @logfire.instrument("Unsubscribing from session")
    async def unsubscribe(self, queue: EmulationQueueOut) -> None:
        """Make given queue no longer receive messages fron this session."""
        async with self._global_lock.write():
            self._queues_out.remove(queue)

    async def _run(self) -> None:
        tasks = [
            asyncio.create_task(self._consumer(), name="emulation:consumer"),
            asyncio.create_task(self._producer(), name="emulation:producer"),
            asyncio.create_task(self._predictor(), name="emulation:predictor"),
        ]

        await self._shutdown_event.wait()

        for task in tasks:
            if not task.done():
                task.cancel()

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(tasks, results, strict=True):
            if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
                logfire.warn(f"Exception in task '{task.get_name()}'", _exc_info=result)

        await self._broadcast(EmulationMessageClose())

    async def _broadcast(self, message: EmulationMessageOut) -> None:
        async with self._global_lock.read():
            queues_to_send = self._queues_out.copy()

        async with asyncio.TaskGroup() as tg:
            for queue in queues_to_send:
                tg.create_task(queue.put(message))

    async def _change_status_and_broadcast(self, new_status: EmulationStatus) -> None:
        async with self._global_lock.write():
            self._memory.last_status = new_status

        await self._broadcast(EmulationMessageStatus(status=new_status))

    async def _consumer(self) -> None:
        while message := await self._queue_in.get():
            match message.command:
                case EmulationCommand.NEXT_PART:
                    logfire.info("Moving to the next part")

                    async with self._global_lock.read():
                        if self._memory.last_status is not EmulationStatus.WAITING_FOR_NEXT_COMMAND:
                            continue

                    self._next_part_command_event.set()

                case EmulationCommand.ABORT:
                    logfire.info("Aborting emulation by user request")

                    await self._change_status_and_broadcast(EmulationStatus.ABORTED)
                    self._shutdown_event.set()

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

            if part_index == examination.metadata.part_count:
                await self._change_status_and_broadcast(EmulationStatus.FINISHED)

                self._shutdown_event.set()
                break

            await self._change_status_and_broadcast(EmulationStatus.WAITING_FOR_NEXT_COMMAND)

            await self._next_part_command_event.wait()

            async with self._global_lock.write():
                self._memory.current_part_index += 1
                self._memory.sent_part_data = ExaminationPartData()
                self._memory.sent_predictions = []
            await self._change_status_and_broadcast(EmulationStatus.SENDING)

            self._next_part_command_event.clear()

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
                if wait_time > 0:
                    await asyncio.sleep(wait_time)

                # manually check for shutdown, task doesn't want to cancel form outside
                if self._shutdown_event.is_set():
                    return

                async with lock.write():
                    log.append(plot_point)

                await self._broadcast(
                    EmulationMessagePlot(
                        plot=EmulationPlot(
                            channel=channel,
                            point=plot_point,
                        )
                    )
                )

    async def _predictor(self) -> None:
        pass  # TODO: WIP
