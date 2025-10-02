import asyncio
from enum import StrEnum

from pydantic import BaseModel

from .examination import ExaminationPartData
from .misc import Channel, MessageModel, PlotPoint


class EmulationStatus(StrEnum):
    SENDING = "sending"
    WAITING_FOR_NEXT_COMMAND = "waiting-for-next-command"
    FINISHED = "finished"
    ABORTED = "aborted"


class EmulationPlot(BaseModel):
    channel: Channel
    point: PlotPoint


class EmulationPrediction(BaseModel):  # TODO: WIP
    dummy: str = "placeholder"


class EmulationState(BaseModel):
    status: EmulationStatus
    part_index: int
    sent_part_data: ExaminationPartData
    sent_predictions: list[EmulationPrediction]


class EmulationMessageState(MessageModel):
    message_priority = 0
    state: EmulationState


class EmulationMessageStatus(MessageModel):
    message_priority = 1
    status: EmulationStatus


class EmulationMessageClose(MessageModel):  # signals the end of emulation
    message_priority = 2


class EmulationMessagePrediction(MessageModel):
    message_priority = 3
    prediction: EmulationPrediction


class EmulationMessagePlot(MessageModel):
    message_priority = 4
    plot: EmulationPlot


class EmulationCommand(StrEnum):
    ABORT = "abort"
    NEXT_PART = "next-part"


class EmulationMessageCommand(MessageModel):
    command: EmulationCommand


EmulationMessageOut = (
    EmulationMessageState
    | EmulationMessageStatus
    | EmulationMessageClose
    | EmulationMessagePrediction
    | EmulationMessagePlot
)
EmulationMessageIn = EmulationMessageCommand


EmulationQueueOut = asyncio.PriorityQueue[EmulationMessageOut]
EmulationQueueIn = asyncio.PriorityQueue[EmulationMessageIn]
