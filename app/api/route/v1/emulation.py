import asyncio
import contextlib

import logfire
from fastapi import APIRouter, WebSocket, WebSocketException
from pydantic import ValidationError

from app.core.exceptions import AppBaseError
from app.model import (
    EmulationMessageClose,
    EmulationMessageCommand,
    EmulationQueueIn,
    EmulationQueueOut,
)
from app.service.emulation import EmulationService

emulation_router = APIRouter(
    prefix="/patients/{patient_id}/examinations/{examination_id}/emulation",
    tags=["emulation"],
)


@emulation_router.websocket("/start")
async def emulation_start(websocket: WebSocket, patient_id: int, examination_id: int) -> None:
    await websocket.accept()

    try:
        queue_out: EmulationQueueOut
        queue_in: EmulationQueueIn
        async with EmulationService.new(patient_id, examination_id) as (queue_out, queue_in):
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

    except AppBaseError as exc:
        raise WebSocketException(code=exc.ws_status_code, reason=exc.message) from exc


@emulation_router.websocket("/attach")
async def emulation_attach(websocket: WebSocket, patient_id: int, examination_id: int) -> None:
    await websocket.accept()

    try:
        async with EmulationService.attach(patient_id, examination_id) as queue_out:
            await _sender(websocket, queue_out)

    except AppBaseError as exc:
        raise WebSocketException(code=exc.ws_status_code, reason=exc.message) from exc


async def _sender(websocket: WebSocket, queue_out: EmulationQueueOut) -> None:
    while message := await queue_out.get():
        if isinstance(message, EmulationMessageClose):
            await websocket.close(code=1000, reason="Emulation finished")
            return

        await websocket.send_json(message.model_dump())


async def _receiver(websocket: WebSocket, queue_in: EmulationQueueIn) -> None:
    while raw_message := await websocket.receive_json():
        logfire.info("Received message", message=raw_message)

        with contextlib.suppress(ValidationError):
            validated_message = EmulationMessageCommand.model_validate(raw_message)
            await queue_in.put(validated_message)

            logfire.info("Validated received message", message=validated_message)
