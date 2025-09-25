import asyncio
import contextlib

import logfire
from fastapi import (
    APIRouter,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    WebSocketException,
    status,
)

from app.api.schema.emulation import EmulationUploadResponse
from app.core import config
from app.core.exceptions import SessionNotFoundError, UnknownFileTypeError
from app.service import EmulationService

from ..util import build_responses

emulation_router = APIRouter(prefix="/emulation", tags=["emulation"])


@emulation_router.post(
    "/upload",
    status_code=status.HTTP_201_CREATED,
    responses=build_responses(UnknownFileTypeError),
    summary="Upload file with data for emulation",
)
async def emulation_upload(file: UploadFile) -> EmulationUploadResponse:
    session_id = await EmulationService.create_session(file)

    return EmulationUploadResponse(session_id=session_id)


@emulation_router.websocket("/connect/{session_id}")
async def emulation_connect(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()

    try:
        queue = await EmulationService.subscribe_to_session(session_id)
    except SessionNotFoundError as exc:
        raise WebSocketException(code=4001, reason="Session not found") from exc

    close_task = asyncio.create_task(_close_on_client_request(websocket))

    # noinspection PyBroadException
    try:
        while True:
            message = await queue.get()

            if message is None:  # the end
                await websocket.close(code=1000, reason="Emulation finished")
                break

            await websocket.send_json(message.model_dump_json())

    except WebSocketDisconnect:
        logfire.exception("WebSocket disconnected", session_id=session_id)

    except Exception:
        logfire.exception("Internal error in WebSocket", session_id=session_id)
        await websocket.close(code=1011, reason="Internal error")

    finally:
        close_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await close_task

        await EmulationService.close_session(session_id)


async def _close_on_client_request(websocket: WebSocket) -> None:
    while True:
        message = await websocket.receive_text()

        if message == config.server.ws_stop_message:
            await websocket.close(code=1000, reason="Client requested to close the connection")
            break
