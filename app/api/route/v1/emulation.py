import asyncio
import contextlib

from fastapi import APIRouter, WebSocket
from pydantic import ValidationError

from app.core.exceptions import EmulationAlreadyStartedError, EmulationNotFoundError
from app.model import EmulationMessageIn, EmulationQueueIn, EmulationQueueOut
from app.service.emulation import EmulationService

emulation_router = APIRouter(
    prefix="/patients/{patient_id}/examinations/{examination_id}/emulation",
    tags=["emulation"],
)


@emulation_router.websocket("/start")
async def emulation_start(websocket: WebSocket, patient_id: int, examination_id: int) -> None:
    try:
        queue_out: EmulationQueueOut
        queue_in: EmulationQueueIn
        async with EmulationService.new(patient_id, examination_id) as (queue_out, queue_in):
            await websocket.accept()

            send_task = asyncio.create_task(_sender(websocket, queue_out))
            receiver_task = asyncio.create_task(_receiver(websocket, queue_in))

            done, pending = await asyncio.wait(
                (send_task, receiver_task), return_when=asyncio.FIRST_COMPLETED
            )

            for task in pending:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            for task in done:
                task.result()  # can raise exceptions - this is intentional

    except EmulationAlreadyStartedError as e:
        await websocket.close(e.status_code, e.message)


@emulation_router.websocket("/attach")
async def emulation_attach(websocket: WebSocket, patient_id: int, examination_id: int) -> None:
    try:
        async with EmulationService.attach(patient_id, examination_id) as queue_out:
            await websocket.accept()

            await _sender(websocket, queue_out)

    except EmulationNotFoundError as e:
        await websocket.close(e.status_code, e.message)


async def _sender(websocket: WebSocket, queue_out: EmulationQueueOut) -> None:
    async for message in queue_out:
        if message is None:
            await websocket.close(code=1000, reason="Emulation finished")
            return

        await websocket.send_json(message.model_dump_json(exclude_unset=True))


async def _receiver(websocket: WebSocket, queue_in: EmulationQueueIn) -> None:
    while raw_message := await websocket.receive_json():
        with contextlib.suppress(ValidationError):
            await queue_in.put_item(EmulationMessageIn.model_validate(raw_message))
